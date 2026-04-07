import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from hiveclaw_python.steering import (
    ActiveSteeringWrapper,
    CaptureWrapper,
    check_latent_dim,
    load_sae,
)

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
SAE_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "hiveclaw_sae_v1.safetensors"
)

_SPIKE_SAMPLER = make_sampler(temp=0.0)
FATAL_LINE = (
    "pheromoned is not running under launchd. From repo root: "
    "`cargo build --release -p hiveclaw-daemon` then `make daemon-load`. See scripts/README.md."
)

SLOT_INDEX = 0


def _config_hidden_size(config: dict) -> int:
    if "hidden_size" in config:
        return int(config["hidden_size"])
    tc = config.get("text_config")
    if isinstance(tc, dict) and "hidden_size" in tc:
        return int(tc["hidden_size"])
    raise ValueError("Could not read hidden_size from model config.")


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
        import hiveclaw_python

        slab_client = hiveclaw_python.SlabClient()
    except Exception:
        print(FATAL_LINE, file=sys.stderr)
        sys.exit(1)

    check_latent_dim(slab_client)
    sae = load_sae(SAE_PATH)
    W_enc = sae["encoder.weight"]
    b_enc = sae["encoder.bias"]
    b_dec = sae["decoder.bias"]

    model, tokenizer, hf_config = load(MODEL_ID, return_config=True)
    try:
        hidden_size = _config_hidden_size(hf_config)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    if hidden_size != 2048:
        print(
            f"[ERROR] Model hidden_size={hidden_size}; SAE expects 2048 (Llama 3.2 1B).",
            file=sys.stderr,
        )
        sys.exit(1)

    original_layer = model.model.layers[-1]

    try:
        print("=== AGENT A: WRITING LATENT (cat) ===")
        prompt_a = _encode_prompt(tokenizer, "Write a short story about a cat.")

        wrapper_a = CaptureWrapper(original_layer)
        model.model.layers[-1] = wrapper_a
        _ = model(prompt_a[None])
        mx.eval(wrapper_a.captured_h)

        d_latent = slab_client.get_latent_dim()
        h_prompt_a = wrapper_a.captured_h[:, -1:, :].astype(mx.float32)
        latent = mx.maximum(mx.matmul(h_prompt_a, mx.transpose(W_enc)) + b_enc, 0.0).astype(
            mx.bfloat16
        )
        latent = latent.reshape(1, 1, d_latent)
        write_node = slab_client.write_slot_v5(SLOT_INDEX, latent)
        mx.eval(write_node)
        print(f"[INFO] Agent A wrote {d_latent}-D SAE latent to IOSurface (GPU path).")

        print("\n=== AGENT B: GENERATING (dog + latent pressure) ===")
        prompt_b = _encode_prompt(tokenizer, "Write a short story about a dog.")

        alpha = 0.1
        wrapper_b = ActiveSteeringWrapper(
            original_layer,
            slab_client,
            W_enc,
            b_dec,
            alpha=alpha,
            slot_index=SLOT_INDEX,
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
