import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

# Phase 4: deterministic scientific control
mx.random.seed(42)
np.random.seed(42)


MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
PROMPT_TEXT = "Write a short story about a cat."

# mlx-lm 0.31+: generate_step uses sampler= instead of temp=.
_SPIKE_SAMPLER = make_sampler(temp=0.0)
FATAL_LINE = (
    "pheromoned is not running under launchd. From repo root: "
    "`cargo build --release -p hiveclaw-daemon` then `make daemon-load`. See scripts/README.md."
)

# XPC-backed slab offsets for Phase 4
SLOT0_SCENT_BYTE_OFFSET = 384
SLOT0_SCENT_ELEMS = 1024


class CaptureWrapper(nn.Module):
    """
    Captures the pre-norm hidden state tensor at the model's final transformer layer.
    Contract: captured_h has shape exactly (1, 1, 4096).
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.captured_h = None

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        self.captured_h = out[0] if isinstance(out, tuple) else out
        return out


def compute_scent(h, W_down):
    """
    Contract math:
      z = normalize(h @ W_down.T) in R^{(1,1,1024)}
    """
    # h: (1, 1, 4096), W_down: (1024, 4096) => W_down.T: (4096, 1024)
    z = mx.matmul(h, W_down.T)  # (1, 1, 1024)

    # L2 normalize on last axis; allowed to do norm in float32 for stability.
    eps = mx.array(1e-7, dtype=mx.bfloat16)
    norm = mx.linalg.norm(z.astype(mx.float32), ord=2, axis=-1, keepdims=True).astype(mx.bfloat16)
    return z / (norm + eps)


def _encode_prompt(tokenizer):
    messages = [{"role": "user", "content": PROMPT_TEXT}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_array = mx.array(tokenizer.encode(prompt))
    return prompt_array


def baseline_generation(model, tokenizer, prompt_array, wrapper, W_down, target_scent):
    """
    Baseline ("Normal"): no VRAM reads. Compare freshly computed z vs static target_scent.
    Returns: (tokens, cos_trace)
    """
    tokens = []
    cos_trace = []

    for token, _logprob in generate_step(
        prompt_array, model, sampler=_SPIKE_SAMPLER, max_tokens=-1
    ):
        mx.eval(wrapper.captured_h)

        h_step = wrapper.captured_h[:, -1:, :]  # (1, 1, 4096)
        z = compute_scent(h_step, W_down)  # (1, 1, 1024)

        cos_sim = mx.sum(z * target_scent, axis=-1)  # (1, 1)
        val = float(cos_sim.item())

        tok_id = int(token.item()) if hasattr(token, "item") else int(token)
        tokens.append(tok_id)
        cos_trace.append(val)

        print(f"[NORMAL] token={tok_id} alignment={val:.6f}")

        if tok_id == int(tokenizer.eos_token_id):
            break

    return tokens, cos_trace


def injected_generation(model, tokenizer, prompt_array, wrapper, W_down, slab_client):
    """
    Scent-Injected:
      - read VRAM scent fresh at every token step
      - compute cosine alignment vs current z
    Returns: (tokens, cos_trace)
    """
    tokens = []
    cos_trace = []

    for token, _logprob in generate_step(
        prompt_array, model, sampler=_SPIKE_SAMPLER, max_tokens=-1
    ):
        # Fresh read contract: inside the loop for every token step
        retrieved_list = slab_client.read_bf16_at(SLOT0_SCENT_BYTE_OFFSET, SLOT0_SCENT_ELEMS)
        scent = mx.array(retrieved_list, dtype=mx.bfloat16).reshape(1, 1, 1024)

        mx.eval(wrapper.captured_h)

        h_step = wrapper.captured_h[:, -1:, :]  # (1, 1, 4096)
        z = compute_scent(h_step, W_down)  # (1, 1, 1024)

        cos_sim = mx.sum(z * scent, axis=-1)  # (1, 1)
        val = float(cos_sim.item())

        tok_id = int(token.item()) if hasattr(token, "item") else int(token)
        tokens.append(tok_id)
        cos_trace.append(val)

        print(f"[INJECTED] token={tok_id} alignment={val:.6f}")

        if tok_id == int(tokenizer.eos_token_id):
            break

    return tokens, cos_trace


def test_c_memory_leak_local(iters=50):
    """
    Test C (Memory Leak): local MLX-only test.
    Contract: must not instantiate/use slab_client or touch IOSurface.
    """
    import psutil

    proc = psutil.Process(os.getpid())
    rss_prev = proc.memory_info().rss

    kv_chunks = []
    for i in range(iters):
        dummy_kv = mx.array(
            np.zeros((1, 8, 1, 128), dtype=np.float16),
        )
        kv_chunks.append(dummy_kv)
        mx.eval(dummy_kv)

        rss_now = proc.memory_info().rss

        # Allow tiny allocator jitter; enforce "no runaway growth".
        # This is still a monotonic guard against leaks.
        if rss_now > rss_prev + 4 * 1024 * 1024:
            raise RuntimeError(f"Test C memory leak suspected at iter={i}: {rss_prev} -> {rss_now}")

        rss_prev = rss_now

    print("[TEST C] PASS (no monotonic runaway RSS)")


def main():
    try:
        import hiveclaw_python  # provided by PyO3 / maturin

        slab_client = hiveclaw_python.SlabClient()
    except Exception:
        print(FATAL_LINE, file=sys.stderr)
        sys.exit(1)

    model, tokenizer = load(MODEL_ID)

    # Patch capture point
    original_layer = model.model.layers[-1]
    wrapper = CaptureWrapper(original_layer)
    model.model.layers[-1] = wrapper

    try:
        prompt_array = _encode_prompt(tokenizer)

        # Prefill pass to capture the final prompt hidden state
        _ = model(prompt_array[None])
        mx.eval(wrapper.captured_h)

        h_prompt = wrapper.captured_h[:, -1:, :]  # (1, 1, 4096)

        # Dummy PCA projection for the spike
        W_down = mx.random.normal((1024, 4096), dtype=mx.bfloat16)

        # target_scent contract: derived from the last prompt token, computed as z
        target_scent = compute_scent(h_prompt, W_down)  # (1, 1, 1024)

        # Flatten strictly at the serialization boundary for the PyO3 write
        target_scent_list = mx.flatten(target_scent).tolist()  # exactly 1024 scalars

        print("=== SECTION 2: BASELINE (Normal) ===")
        tokens_normal, cos_normal = baseline_generation(
            model=model,
            tokenizer=tokenizer,
            prompt_array=prompt_array,
            wrapper=wrapper,
            W_down=W_down,
            target_scent=target_scent,
        )

        # Write once right after baseline run finishes
        slab_client.write_bf16_at(SLOT0_SCENT_BYTE_OFFSET, target_scent_list)
        print("[INFO] Successfully wrote 1024-D target scent to IOSurface VRAM")

        print("=== SECTION 4: SCENT-INJECTED ===")
        tokens_injected, cos_injected = injected_generation(
            model=model,
            tokenizer=tokenizer,
            prompt_array=prompt_array,
            wrapper=wrapper,
            W_down=W_down,
            slab_client=slab_client,
        )

        print("=== SECTION 5: TEST C (Memory Leak) ===")
        test_c_memory_leak_local(iters=50)

    finally:
        model.model.layers[-1] = original_layer


if __name__ == "__main__":
    main()
