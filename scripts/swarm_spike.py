#!/usr/bin/env python3
"""
Phase C multi-agent VRAM / slab contention spike (no LLM).
Sense unclaimed slots via CPU snapshot, pressure-sort by cosine vs a random goal,
claim → synthetic matmul hold → blend → write → release.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import mlx.core as mx
import numpy as np

FATAL_LINE = (
    "pheromoned is not running under launchd. From repo root: "
    "`cargo build --release -p hiveclaw-daemon` then `make daemon-load`. "
    "See scripts/README.md."
)

_N_SLOTS = 4096


def _chaos_seed() -> int:
    return (os.getpid() * 1_000_003) ^ (time.time_ns() & 0xFFFFFFFFFFFFFFFF)


def _cosine_read_np(vec_bf16_1d: mx.array, goal_f32_1d: np.ndarray) -> float:
    """Cosine similarity; flattens any leading dims (e.g. [1,1,D])."""
    v = np.array(vec_bf16_1d.astype(mx.float32), dtype=np.float64).reshape(-1)
    g = goal_f32_1d.astype(np.float64)
    nv = np.linalg.norm(v)
    ng = np.linalg.norm(g)
    if nv < 1e-12 or ng < 1e-12:
        return 0.0
    return float(np.dot(v, g) / (nv * ng))


def main() -> None:
    p = argparse.ArgumentParser(description="HiveClaw swarm contention spike")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Blend weight on current slot scent (default 0.1)",
    )
    p.add_argument(
        "--hold-steps",
        type=int,
        default=10,
        help="Synthetic matmul iterations while holding claim (default 10)",
    )
    p.add_argument(
        "--heartbeat",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Print a line every N successful claim→write→release cycles "
            "(default 10; use 0 to disable)"
        ),
    )
    args = p.parse_args()
    alpha = float(args.alpha)
    hold_steps = int(args.hold_steps)
    if not (0.0 <= alpha <= 1.0):
        print("--alpha must be in [0, 1]", file=sys.stderr)
        sys.exit(2)
    if hold_steps < 1:
        print("--hold-steps must be >= 1", file=sys.stderr)
        sys.exit(2)
    heartbeat = int(args.heartbeat)
    if heartbeat < 0:
        print("--heartbeat must be >= 0", file=sys.stderr)
        sys.exit(2)

    seed = _chaos_seed()
    random.seed(seed)
    rng = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)
    mx.random.seed(int(seed & 0xFFFFFFFF))

    try:
        import hiveclaw_python

        client = hiveclaw_python.SlabClient()
    except Exception:
        print(FATAL_LINE, file=sys.stderr)
        sys.exit(1)

    d = client.get_latent_dim()
    inner = max(1, d // 4)

    # Random normalized goal bf16 [1, 1, D]
    g = rng.standard_normal(d, dtype=np.float32)
    g /= float(np.linalg.norm(g) + 1e-7)
    goal = mx.array(g.reshape(1, 1, d), dtype=mx.bfloat16)
    goal_f32_1d = g.copy()

    # Fixed MLP-ish geometry: W [D, D//4], V [D//4, D] float32
    W = mx.array(
        rng.standard_normal((d, inner), dtype=np.float32),
        dtype=mx.float32,
    )
    V = mx.array(
        rng.standard_normal((inner, d), dtype=np.float32),
        dtype=mx.float32,
    )
    mx.eval(goal, W, V)

    print(
        f"[swarm_spike] pid={os.getpid()} latent_dim={d} alpha={alpha} "
        f"hold_steps={hold_steps} seed={seed & 0xFFFFFFFFFFFFFFFF:x}",
        flush=True,
    )

    try:
        cycles = 0
        while True:
            states = client.get_slot_states()
            unclaimed = [i for i, s in enumerate(states) if not s["claimed"]]
            if not unclaimed:
                time.sleep(random.uniform(0.001, 0.010))
                continue

            scored: list[tuple[float, int]] = []
            for slot in unclaimed:
                read_n = client.read_slot_v5(slot)
                mx.eval(read_n)
                cos = _cosine_read_np(read_n, goal_f32_1d)
                scored.append((cos, slot))

            if not scored:
                time.sleep(random.uniform(0.001, 0.010))
                continue

            scored.sort(key=lambda t: t[0], reverse=True)
            order = [s for _, s in scored]
            candidates = mx.array(order, dtype=mx.int32)
            claim_res = client.claim_task(candidates)
            mx.eval(claim_res)
            slot = int(np.asarray(claim_res).reshape(-1)[0])
            if slot < 0:
                time.sleep(random.uniform(0.001, 0.010))
                continue

            slot_scent = client.read_slot_v5(slot)
            mx.eval(slot_scent)
            slot_bf16_3d = slot_scent.reshape(1, 1, d)
            h = slot_bf16_3d.astype(mx.float32)
            for _ in range(hold_steps):
                h = (h @ W) @ V
            h_bf16 = h.astype(mx.bfloat16)
            mx.eval(h_bf16)

            blend = alpha * slot_bf16_3d + (1.0 - alpha) * goal
            to_write = blend.reshape(d).astype(mx.bfloat16)
            write_res = client.write_scent(slot, to_write)
            mx.eval(write_res)
            client.release_task(slot)
            cycles += 1
            if heartbeat and cycles % heartbeat == 0:
                print(
                    f"[swarm_spike] cycles={cycles} last_slot={slot}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[swarm_spike] stopped.", flush=True)


if __name__ == "__main__":
    main()
