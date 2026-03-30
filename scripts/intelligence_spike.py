import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

# mlx-lm 0.31+: generate_step uses sampler= instead of temp=.
_SPIKE_SAMPLER = make_sampler(temp=0.0)
FATAL_LINE = (
    "pheromoned is not running under launchd. From repo root: "
    "`cargo build --release -p hiveclaw-daemon` then `make daemon-load`. See scripts/README.md."
)

# XPC-backed slab offsets for Phase 4A (must match hiveclaw-core math.rs)
SLOT0_SCENT_BYTE_OFFSET = 384
SLOT0_SCENT_ELEMS = 4096


class CaptureWrapper(nn.Module):
    """
    Captures the pre-norm hidden state tensor at the model's final transformer layer.
    Contract: captured_h has shape exactly (batch, seq, 4096).
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.captured_h = None

    def __getattr__(self, name: str):
        # mlx_lm LlamaModel reads e.g. layer.use_sliding when building masks; forward to the block.
        try:
            return getattr(self.layer, name)
        except AttributeError:
            return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        self.captured_h = out[0] if isinstance(out, tuple) else out
        return out


class ActiveSteeringWrapper(nn.Module):
    """
    Wraps the final transformer layer.
    On every forward call: runs the real layer, reads the VRAM scent,
    adds alpha * scent to the last token's hidden state, returns modified output.
    """

    def __init__(self, layer, slab_client, alpha=0.1):
        super().__init__()
        self.layer = layer
        self.slab_client = slab_client
        self.alpha = alpha

    def __getattr__(self, name: str):
        try:
            return getattr(self.layer, name)
        except AttributeError:
            return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out  # (batch, seq, 4096)

        h_step = h[:, -1:, :]  # (batch, 1, 4096) — only steer last position

        retrieved = self.slab_client.read_bf16_at(
            SLOT0_SCENT_BYTE_OFFSET, SLOT0_SCENT_ELEMS
        )
        scent = mx.array(retrieved, dtype=mx.bfloat16).reshape(1, 1, 4096)

        h_modified = h_step + (scent * self.alpha)

        if h.shape[1] > 1:
            h_final = mx.concatenate([h[:, :-1, :], h_modified], axis=1)
        else:
            h_final = h_modified

        return (h_final,) + out[1:] if isinstance(out, tuple) else h_final


def _encode_prompt(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return mx.array(tokenizer.encode(prompt))


def main():
    try:
        import hiveclaw_python  # provided by PyO3 / maturin

        slab_client = hiveclaw_python.SlabClient()
    except Exception:
        print(FATAL_LINE, file=sys.stderr)
        sys.exit(1)

    model, tokenizer = load(MODEL_ID)
    original_layer = model.model.layers[-1]

    try:
        # ── Agent A: writer ──────────────────────────────
        print("=== AGENT A: WRITING SCENT (cat) ===")
        prompt_a = _encode_prompt(tokenizer, "Write a short story about a cat.")

        wrapper_a = CaptureWrapper(original_layer)
        model.model.layers[-1] = wrapper_a
        _ = model(prompt_a[None])
        mx.eval(wrapper_a.captured_h)

        h_prompt_a = wrapper_a.captured_h[:, -1:, :]  # (1, 1, 4096)

        eps = mx.array(1e-7, dtype=mx.bfloat16)
        norm = mx.linalg.norm(
            h_prompt_a.astype(mx.float32), ord=2, axis=-1, keepdims=True
        ).astype(mx.bfloat16)
        normalized_scent = h_prompt_a / (norm + eps)

        scent_list = mx.flatten(normalized_scent).tolist()  # exactly 4096 scalars
        assert len(scent_list) == 4096
        slab_client.write_bf16_at(SLOT0_SCENT_BYTE_OFFSET, scent_list)
        print("[INFO] Agent A wrote 4096-D 'cat' scent to IOSurface.")

        # ── Agent B: reader ──────────────────────────────
        print("\n=== AGENT B: GENERATING (dog + cat latent pressure) ===")
        prompt_b = _encode_prompt(tokenizer, "Write a short story about a dog.")

        alpha = 0.1  # conservative start; increase if tokens remain unaffected
        wrapper_b = ActiveSteeringWrapper(original_layer, slab_client, alpha=alpha)
        model.model.layers[-1] = wrapper_b

        print("[Output]: ", end="", flush=True)
        for token, _logprob in generate_step(
            prompt_b, model, sampler=_SPIKE_SAMPLER, max_tokens=150
        ):
            tok_id = int(token.item()) if hasattr(token, "item") else int(token)
            print(tokenizer.decode([tok_id]), end="", flush=True)
            if tok_id == int(tokenizer.eos_token_id):
                break
        print("\n")

    finally:
        model.model.layers[-1] = original_layer


if __name__ == "__main__":
    main()
