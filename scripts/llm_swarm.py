#!/usr/bin/env python3
"""
Phase C LLM swarm: stigmergic slab contention + mlx_lm generation with active steering.
Sense → claim → generate (≤10 tokens) → write post-steer scent → release → repeat.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
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


class CaptureWrapper(nn.Module):
    """Captures final-layer hidden state in `captured_h`."""

    def __init__(self, layer):
        super().__init__()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "captured_h", None)

    def __getattr__(self, name: str):
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
    Final layer + slab read on last token. `current_slot` is set by the agent loop
    before each `generate_step` burst. `last_steered_h` holds post-blend last token.
    """

    def __init__(self, layer, slab_client, d: int, alpha: float = ALPHA_DEFAULT):
        super().__init__()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "slab_client", slab_client)
        object.__setattr__(self, "D", d)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "current_slot", 0)
        object.__setattr__(self, "last_steered_h", None)

    def __getattr__(self, name: str):
        if name in ("layer", "slab_client", "D", "alpha", "current_slot", "last_steered_h"):
            return super().__getattr__(name)
        try:
            return getattr(self.layer, name)
        except AttributeError:
            return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        h_step = h[:, -1:, :]

        h_modified = self.slab_client.fused_steer(
            int(self.current_slot),
            h_step,
            self.alpha,
            depends=h_step,
        )
        object.__setattr__(self, "last_steered_h", h_modified)

        if h.shape[1] > 1:
            h_final = mx.concatenate([h[:, :-1, :], h_modified], axis=1)
        else:
            h_final = h_modified

        return (h_final,) + out[1:] if isinstance(out, tuple) else h_final


def _encode_prompt(tokenizer, text: str) -> mx.array:
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return mx.array(tokenizer.encode(prompt))


def _goal_vector(model, tokenizer, prompt_text: str, d: int, original_layer) -> mx.array:
    wrapper = CaptureWrapper(original_layer)
    model.model.layers[-1] = wrapper
    toks = _encode_prompt(tokenizer, prompt_text)
    _ = model(toks[None])
    mx.eval(wrapper.captured_h)
    model.model.layers[-1] = original_layer
    h = wrapper.captured_h[:, -1:, :].astype(mx.float32)
    norm = mx.linalg.norm(h, ord=2, axis=-1, keepdims=True)
    out = (h / (norm + 1e-7)).reshape(d)
    mx.eval(out)
    return out


def _config_hidden_size(config: dict) -> int:
    """Architectural hidden size from Hugging Face config.json (not embedding weight shape)."""
    if "hidden_size" in config:
        return int(config["hidden_size"])
    tc = config.get("text_config")
    if isinstance(tc, dict) and "hidden_size" in tc:
        return int(tc["hidden_size"])
    raise ValueError(
        "Could not read hidden_size from model config (needed to match slab get_scent_dim)."
    )


def _cosine_np(vec_bf16_1d: mx.array, goal_f32_1d: np.ndarray) -> float:
    v = np.array(vec_bf16_1d.astype(mx.float32), dtype=np.float64)
    g = goal_f32_1d.astype(np.float64)
    nv = np.linalg.norm(v)
    ng = np.linalg.norm(g)
    if nv < 1e-12 or ng < 1e-12:
        return 0.0
    return float(np.dot(v, g) / (nv * ng))


def main() -> None:
    p = argparse.ArgumentParser(description="HiveClaw LLM swarm (Phase C)")
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

    d = slab_client.get_scent_dim()
    model, tokenizer, hf_config = load(MODEL_ID, return_config=True)
    try:
        hidden_size = _config_hidden_size(hf_config)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    if hidden_size != d:
        print(
            f"[ERROR] Model config hidden_size={hidden_size} does not match "
            f"HiveClaw slab get_scent_dim()={d}. Use a model with hidden_size {d}, "
            "or change SCENT_ELEMS in crates/hiveclaw-core/src/math.rs and rebuild.",
            file=sys.stderr,
        )
        sys.exit(1)

    original_layer = model.model.layers[-1]
    current_prompt_text = args.prompt.strip() or random.choice(PROMPT_POOL)
    goal_mx = _goal_vector(model, tokenizer, current_prompt_text, d, original_layer)
    goal_np = np.array(goal_mx, dtype=np.float32)

    steering = ActiveSteeringWrapper(original_layer, slab_client, d, alpha=alpha)
    model.model.layers[-1] = steering

    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1

    print(
        f"[llm_swarm] pid={os.getpid()} scent_dim={d} alpha={alpha} model={MODEL_ID}",
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
                scent = slab_client.read_scent_if_consistent(
                    slot, [d], context="llm_swarm_sense"
                )
                if scent is None:
                    continue
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

            h_f32 = last_h.astype(mx.float32)
            norm = mx.linalg.norm(h_f32, ord=2, axis=-1, keepdims=True)
            h_norm_bf16 = (h_f32 / (norm + 1e-7)).astype(mx.bfloat16).reshape(d)
            write_res = slab_client.write_scent(slot, h_norm_bf16)
            mx.eval(write_res)
            slab_client.release_task(slot)

            if eos_hit:
                current_prompt_text = random.choice(PROMPT_POOL)

    except KeyboardInterrupt:
        print("\n[llm_swarm] stopped.", flush=True)
    finally:
        model.model.layers[-1] = original_layer


if __name__ == "__main__":
    main()
