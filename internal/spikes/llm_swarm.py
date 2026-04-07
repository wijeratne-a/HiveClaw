#!/usr/bin/env python3
"""
Phase C LLM swarm: stigmergic slab contention + mlx_lm generation with SAE latent steering.
Sense → claim → generate (≤10 tokens) → encode → write_slot_v5 → release → repeat.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mlx.core as mx
import numpy as np
from hiveclaw_steering import ActiveSteeringWrapper, CaptureWrapper, check_latent_dim, load_sae
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
SAE_PATH = Path(__file__).resolve().parent.parent / "models/hiveclaw_sae_v1.safetensors"
MAX_TOKENS_PER_HOLD = 10
ALPHA_DEFAULT = 0.1
_SPIKE_SAMPLER = make_sampler(temp=0.8)

FATAL_LINE = (
    "pheromoned is not running under launchd. From repo root: "
    "`cargo build --release -p hiveclaw-daemon` then `make daemon-load`. "
    "See scripts/README.md."
)

PROMPT_POOL = [
    "Describe a peaceful forest.",
    "Explain how a combustion engine works.",
    "Write a short poem about the ocean.",
    "Describe the life cycle of a star.",
    "Explain the concept of recursion.",
    "Write a haiku about winter.",
    "Describe what it feels like to travel at the speed of light.",
    "Explain how bees make honey.",
    "Write a story opening about a mysterious door.",
    "Describe the architecture of the internet.",
    "Explain photosynthesis to a child.",
    "Describe a city two hundred years from now.",
    "What does silence sound like in space?",
    "Explain why the sky is blue.",
    "Write a lullaby for a robot.",
]


def _encode_prompt(tokenizer, text: str) -> mx.array:
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return mx.array(tokenizer.encode(prompt))


def _goal_latent(
    model, tokenizer, prompt_text: str, original_layer, W_enc: mx.array, b_enc: mx.array
) -> np.ndarray:
    wrapper = CaptureWrapper(original_layer)
    model.model.layers[-1] = wrapper
    toks = _encode_prompt(tokenizer, prompt_text)
    _ = model(toks[None])
    mx.eval(wrapper.captured_h)
    model.model.layers[-1] = original_layer
    h = wrapper.captured_h[:, -1:, :].astype(mx.float32)
    norm = mx.linalg.norm(h, ord=2, axis=-1, keepdims=True)
    h_n = h / (norm + 1e-7)
    z = mx.maximum(mx.matmul(h_n, mx.transpose(W_enc)) + b_enc, 0.0).reshape(-1)
    mx.eval(z)
    return np.array(z, dtype=np.float32)


def _config_hidden_size(config: dict) -> int:
    if "hidden_size" in config:
        return int(config["hidden_size"])
    tc = config.get("text_config")
    if isinstance(tc, dict) and "hidden_size" in tc:
        return int(tc["hidden_size"])
    raise ValueError("Could not read hidden_size from model config.")


def _cosine_np(vec_bf16: mx.array, goal_f32_1d: np.ndarray) -> float:
    v = np.array(vec_bf16.astype(mx.float32), dtype=np.float64).reshape(-1)
    g = goal_f32_1d.astype(np.float64).reshape(-1)
    nv = np.linalg.norm(v)
    ng = np.linalg.norm(g)
    if nv < 1e-12 or ng < 1e-12:
        return 0.0
    return float(np.dot(v, g) / (nv * ng))


def main() -> None:
    p = argparse.ArgumentParser(description="HiveClaw LLM swarm (Phase C v5)")
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Initial user prompt (default: random from built-in PROMPT_POOL)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=ALPHA_DEFAULT,
        help=f"Steering blend weight (default {ALPHA_DEFAULT})",
    )
    args = p.parse_args()
    alpha = float(args.alpha)
    if alpha < 0.0:
        print("--alpha must be >= 0", file=sys.stderr)
        sys.exit(2)

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
        print(f"[ERROR] hidden_size={hidden_size}, SAE expects 2048.", file=sys.stderr)
        sys.exit(1)

    original_layer = model.model.layers[-1]
    current_prompt_text = args.prompt.strip() or random.choice(PROMPT_POOL)
    goal_np = _goal_latent(
        model, tokenizer, current_prompt_text, original_layer, W_enc, b_enc
    )

    steering = ActiveSteeringWrapper(
        original_layer, slab_client, W_enc, b_dec, alpha=alpha, slot_index=0
    )
    model.model.layers[-1] = steering

    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1

    d_latent = slab_client.get_latent_dim()
    print(
        f"[llm_swarm] pid={os.getpid()} latent_dim={d_latent} alpha={alpha} model={MODEL_ID}",
        flush=True,
    )

    try:
        while True:
            states = slab_client.get_slot_states()
            unclaimed = [i for i, s in enumerate(states) if not s["claimed"]]
            if not unclaimed:
                time.sleep(random.uniform(0.001, 0.010))
                continue

            scored: list[tuple[float, int]] = []
            for slot in unclaimed:
                scent = slab_client.read_slot_v5(slot)
                mx.eval(scent)
                scored.append((_cosine_np(scent, goal_np), slot))

            if not scored:
                time.sleep(random.uniform(0.001, 0.010))
                continue

            scored.sort(key=lambda t: t[0], reverse=True)
            order = [s for _, s in scored]
            candidates = mx.array(order, dtype=mx.int32)
            claim_res = slab_client.claim_task(candidates)
            mx.eval(claim_res)
            slot = int(np.asarray(claim_res).reshape(-1)[0])
            if slot < 0:
                time.sleep(random.uniform(0.001, 0.010))
                continue

            object.__setattr__(steering, "current_slot", slot)
            encoded = _encode_prompt(tokenizer, current_prompt_text)
            eos_hit = False
            for token, _ in generate_step(
                encoded,
                model,
                sampler=_SPIKE_SAMPLER,
                max_tokens=MAX_TOKENS_PER_HOLD,
            ):
                tok_id = int(token.item()) if hasattr(token, "item") else int(token)
                print(tokenizer.decode([tok_id]), end="", flush=True)
                if eos_id >= 0 and tok_id == eos_id:
                    eos_hit = True
                    break

            last_h = steering.last_steered_h
            if last_h is None:
                slab_client.release_task(slot)
                continue

            h_f = last_h.astype(mx.float32)
            latent = mx.maximum(mx.matmul(h_f, mx.transpose(W_enc)) + b_enc, 0.0).astype(
                mx.bfloat16
            )
            latent = latent.reshape(1, 1, d_latent)
            write_res = slab_client.write_slot_v5(slot, latent)
            mx.eval(write_res)
            slab_client.release_task(slot)

            if eos_hit:
                current_prompt_text = random.choice(PROMPT_POOL)
                goal_np = _goal_latent(
                    model, tokenizer, current_prompt_text, original_layer, W_enc, b_enc
                )

    except KeyboardInterrupt:
        print("\n[llm_swarm] stopped.", flush=True)
    finally:
        model.model.layers[-1] = original_layer


if __name__ == "__main__":
    main()
