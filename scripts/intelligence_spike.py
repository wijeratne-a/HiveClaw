import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# mlx-lm 0.31+: generate_step uses sampler= instead of temp=.
_SPIKE_SAMPLER = make_sampler(temp=0.0)
FATAL_LINE = (
    "pheromoned is not running under launchd. From repo root: "
    "`cargo build --release -p hiveclaw-daemon` then `make daemon-load`. See scripts/README.md."
)

# Phase C: slot-indexed scent (layout in hiveclaw-core math.rs)
SLOT_INDEX = 0


class CaptureWrapper(nn.Module):
    """
    Captures the hidden state tensor at the model's final transformer layer.
    """

    def __init__(self, layer):
        super().__init__()
        # Bypass nn.Module.__setattr__: it uses hasattr(self, key), which triggers __getattr__
        # before self.layer exists → infinite recursion.
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "captured_h", None)

    def __getattr__(self, name: str):
        # mlx_lm LlamaModel reads e.g. layer.use_sliding when building masks; forward to the block.
        if name in ("layer", "captured_h"):
            return super().__getattr__(name)
        try:
            return getattr(self.layer, name)
        except AttributeError:
            return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        object.__setattr__(
            self, "captured_h", out[0] if isinstance(out, tuple) else out
        )
        return out


class ActiveSteeringWrapper(nn.Module):
    """
    Wraps the final transformer layer.
    On every forward call: runs the real layer, reads the VRAM scent,
    adds alpha * scent to the last token's hidden state, returns modified output.
    """

    def __init__(self, layer, slab_client, scent_dim: int, alpha=0.1):
        super().__init__()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "slab_client", slab_client)
        object.__setattr__(self, "scent_dim", scent_dim)
        object.__setattr__(self, "alpha", alpha)

    def __getattr__(self, name: str):
        if name in ("layer", "slab_client", "scent_dim", "alpha"):
            return super().__getattr__(name)
        try:
            return getattr(self.layer, name)
        except AttributeError:
            return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        d = self.scent_dim

        h_step = h[:, -1:, :]  # (batch, 1, D) — only steer last position

        scent = self.slab_client.read_scent(
            SLOT_INDEX,
            [1, 1, d],
            like=h_step,
            depends=h_step,
        )

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

    d = slab_client.get_scent_dim()
    model, tokenizer = load(MODEL_ID)
    hidden_size = int(model.model.embed_tokens.weight.shape[-1])
    if hidden_size != d:
        print(
            f"[ERROR] Model hidden size {hidden_size} does not match HiveClaw slab "
            f"configuration {d}. Recompile the C++ extension.",
            file=sys.stderr,
        )
        sys.exit(1)

    original_layer = model.model.layers[-1]

    try:
        # ── Agent A: writer ──────────────────────────────
        print("=== AGENT A: WRITING SCENT (cat) ===")
        prompt_a = _encode_prompt(tokenizer, "Write a short story about a cat.")

        wrapper_a = CaptureWrapper(original_layer)
        model.model.layers[-1] = wrapper_a
        _ = model(prompt_a[None])
        mx.eval(wrapper_a.captured_h)

        h_prompt_a = wrapper_a.captured_h[:, -1:, :]  # (1, 1, D)

        eps = mx.array(1e-7, dtype=mx.bfloat16)
        norm = mx.linalg.norm(
            h_prompt_a.astype(mx.float32), ord=2, axis=-1, keepdims=True
        ).astype(mx.bfloat16)
        normalized_scent = (h_prompt_a / (norm + eps)).astype(mx.bfloat16)

        write_node = slab_client.write_scent(SLOT_INDEX, normalized_scent.reshape(d))
        mx.eval(write_node)
        print(f"[INFO] Agent A wrote {d}-D 'cat' scent to IOSurface (GPU path).")

        # ── Agent B: reader ──────────────────────────────
        print("\n=== AGENT B: GENERATING (dog + cat latent pressure) ===")
        prompt_b = _encode_prompt(tokenizer, "Write a short story about a dog.")

        alpha = 0.1  # conservative start; increase if tokens remain unaffected
        wrapper_b = ActiveSteeringWrapper(
            original_layer, slab_client, scent_dim=d, alpha=alpha
        )
        model.model.layers[-1] = wrapper_b

        print("[Output]: ", end="", flush=True)
        eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1
        for token, _logprob in generate_step(
            prompt_b, model, sampler=_SPIKE_SAMPLER, max_tokens=150
        ):
            tok_id = int(token.item()) if hasattr(token, "item") else int(token)
            print(tokenizer.decode([tok_id]), end="", flush=True)
            if eos_id >= 0 and tok_id == eos_id:
                break
        print("\n")

    finally:
        model.model.layers[-1] = original_layer


if __name__ == "__main__":
    main()
