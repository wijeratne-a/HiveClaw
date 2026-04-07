#!/usr/bin/env python3
"""
HiveClaw latent committee: same task as string_swarm_baseline.py but coordination
via slab slots + ActiveSteeringWrapper (no growing text between agents).

Requires: pheromoned, hiveclaw_python, SAE weights, mlx_lm.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mlx.core as mx
import numpy as np
from hiveclaw_python.steering import ActiveSteeringWrapper, check_latent_dim, load_sae
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from string_swarm_baseline import (
    AGENT_ROLES,
    BenchmarkResult,
    MODEL_ID_DEFAULT,
    N_AGENTS_DEFAULT,
    N_ROUNDS_DEFAULT,
    MAX_TOKENS_PER_TURN_DEFAULT,
    TASK_CODE,
)

SAE_PATH = Path(__file__).resolve().parent.parent / "models/hiveclaw_sae_v1.safetensors"


def _config_hidden_size(config: dict) -> int:
    if "hidden_size" in config:
        return int(config["hidden_size"])
    tc = config.get("text_config")
    if isinstance(tc, dict) and "hidden_size" in tc:
        return int(tc["hidden_size"])
    raise ValueError("Could not read hidden_size from model config.")


def _encode_agent_prompt(tokenizer: Any, agent_idx: int, code: str) -> mx.array:
    role = AGENT_ROLES[agent_idx]
    messages = [
        {
            "role": "system",
            "content": (
                f"You are the {role} on a code review committee. "
                "Prior discussion is implicit via shared memory; respond concisely."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this Python code in one brief sentence.\n\n"
                f"```python\n{code.strip()}\n```"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return mx.array(tokenizer.encode(prompt))


def _claim_one(slab: Any, slot: int) -> bool:
    c = mx.array([slot], dtype=mx.int32)
    res = slab.claim_task(c)
    mx.eval(res)
    got = int(np.asarray(res).reshape(-1)[0])
    return got == slot


def run_hiveclaw_consensus(
    *,
    model_id: str = MODEL_ID_DEFAULT,
    n_rounds: int = N_ROUNDS_DEFAULT,
    n_agents: int = N_AGENTS_DEFAULT,
    max_tokens_per_turn: int = MAX_TOKENS_PER_TURN_DEFAULT,
    temperature: float = 0.8,
    sae_path: Path | None = None,
) -> BenchmarkResult:
    if n_agents > len(AGENT_ROLES):
        raise ValueError(f"n_agents must be <= {len(AGENT_ROLES)}")

    try:
        import hiveclaw_python
    except Exception as e:
        return BenchmarkResult(
            phase="hiveclaw_latent",
            rounds=n_rounds,
            agents=n_agents,
            total_coord_tokens=0,
            total_content_tokens=0,
            total_wall_ms=0.0,
            ok=False,
            error=f"hiveclaw_python import failed: {e}",
        )

    slab = hiveclaw_python.SlabClient()
    check_latent_dim(slab)
    spath = sae_path or SAE_PATH
    sae = load_sae(spath)
    W_enc = sae["encoder.weight"]
    b_enc = sae["encoder.bias"]
    b_dec = sae["decoder.bias"]

    model, tokenizer, hf_config = load(model_id, return_config=True)
    hidden_size = _config_hidden_size(hf_config)
    if hidden_size != 2048:
        return BenchmarkResult(
            phase="hiveclaw_latent",
            rounds=n_rounds,
            agents=n_agents,
            total_coord_tokens=0,
            total_content_tokens=0,
            total_wall_ms=0.0,
            ok=False,
            error=f"hidden_size={hidden_size}, expected 2048 for default SAE",
        )

    original_layer = model.model.layers[-1]
    d_latent = slab.get_latent_dim()
    steering = ActiveSteeringWrapper(
        original_layer,
        slab,
        W_enc,
        b_dec,
        alpha=0.1,
        slot_index=0,
    )
    model.model.layers[-1] = steering

    sampler = make_sampler(temp=temperature)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1

    total_content = 0
    t0_all = time.perf_counter()
    per_round_ctx: list[int] = []
    per_round_wall: list[float] = []

    try:
        for r in range(n_rounds):
            t_round = time.perf_counter()
            max_ctx_round = 0
            for a in range(n_agents):
                deadline = time.time() + 30.0
                while time.time() < deadline:
                    if _claim_one(slab, a):
                        break
                    time.sleep(random.uniform(0.002, 0.015))
                else:
                    raise RuntimeError(f"could not claim slot {a}")

                object.__setattr__(steering, "current_slot", a)
                _ = slab.read_slot_v5(a)
                mx.eval(_)

                encoded = _encode_agent_prompt(tokenizer, a, TASK_CODE)
                ctx_tokens = int(encoded.size)
                max_ctx_round = max(max_ctx_round, ctx_tokens)
                pieces: list[str] = []
                gen_t0 = time.perf_counter()
                for token, _ in generate_step(
                    encoded,
                    model,
                    sampler=sampler,
                    max_tokens=max_tokens_per_turn,
                ):
                    tok_id = int(token.item()) if hasattr(token, "item") else int(token)
                    pieces.append(tokenizer.decode([tok_id]))
                    total_content += 1
                    if eos_id >= 0 and tok_id == eos_id:
                        break
                gen_ms = (time.perf_counter() - gen_t0) * 1000.0

                last_h = steering.last_steered_h
                if last_h is not None:
                    h_f = last_h.astype(mx.float32)
                    latent = mx.maximum(
                        mx.matmul(h_f, mx.transpose(W_enc)) + b_enc, 0.0
                    ).astype(mx.bfloat16)
                    latent = latent.reshape(1, 1, d_latent)
                    w = slab.write_slot_v5(a, latent)
                    mx.eval(w)
                slab.release_task(a)

                sys.stderr.write(
                    json.dumps(
                        {
                            "event": "hiveclaw_round",
                            "round": r + 1,
                            "agent": a,
                            "role": AGENT_ROLES[a],
                            "latency_ms": round(gen_ms, 3),
                            "coord_tokens": 0,
                            "ctx_tokens": ctx_tokens,
                            "content_tokens": len(pieces),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            per_round_ctx.append(max_ctx_round)
            per_round_wall.append((time.perf_counter() - t_round) * 1000.0)

    except Exception as e:
        return BenchmarkResult(
            phase="hiveclaw_latent",
            rounds=n_rounds,
            agents=n_agents,
            total_coord_tokens=0,
            total_content_tokens=total_content,
            total_wall_ms=(time.perf_counter() - t0_all) * 1000.0,
            per_round_ctx_tokens=per_round_ctx,
            per_round_wall_ms=per_round_wall,
            ok=False,
            error=str(e),
        )
    finally:
        model.model.layers[-1] = original_layer

    total_wall_ms = (time.perf_counter() - t0_all) * 1000.0
    return BenchmarkResult(
        phase="hiveclaw_latent",
        rounds=n_rounds,
        agents=n_agents,
        total_coord_tokens=0,
        total_content_tokens=total_content,
        total_wall_ms=total_wall_ms,
        per_round_ctx_tokens=per_round_ctx,
        per_round_wall_ms=per_round_wall,
        ok=True,
    )


def main() -> int:
    r = run_hiveclaw_consensus()
    print(json.dumps(r.to_dict(), indent=2))
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
