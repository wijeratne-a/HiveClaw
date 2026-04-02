#!/usr/bin/env python3
"""
Phase C entropy watchdog: sample claimed slots, detect frozen geometry (low variance),
then inhibit via GPU kernel and clear local history.
"""

from __future__ import annotations

import argparse
import collections
import os
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
_K = 5


def main() -> None:
    p = argparse.ArgumentParser(description="HiveClaw slab entropy overseer")
    p.add_argument(
        "--tick-ms",
        type=int,
        default=500,
        help="Sleep between ticks in milliseconds (default 500)",
    )
    p.add_argument(
        "--var-threshold",
        type=float,
        default=1e-5,
        help="Inhibit if mean per-dim time variance falls below this (default 1e-5)",
    )
    args = p.parse_args()
    tick_ms = max(1, int(args.tick_ms))
    var_threshold = float(args.var_threshold)

    try:
        import hiveclaw_python

        client = hiveclaw_python.SlabClient()
    except Exception:
        print(FATAL_LINE, file=sys.stderr)
        sys.exit(1)

    d = client.get_latent_dim()
    history: list[collections.deque[np.ndarray]] = [
        collections.deque(maxlen=_K) for _ in range(_N_SLOTS)
    ]

    print(
        f"[overseer] pid={os.getpid()} latent_dim={d} tick_ms={tick_ms} "
        f"var_threshold={var_threshold}",
        flush=True,
    )

    try:
        while True:
            time.sleep(tick_ms / 1000.0)
            states = client.get_slot_states()
            for slot_idx, st in enumerate(states):
                if not st["claimed"]:
                    history[slot_idx].clear()
                    continue

                scent_bf16 = client.read_slot_v5(slot_idx)
                scent_f32 = scent_bf16.astype(mx.float32)
                mx.eval(scent_f32)
                sample = np.array(scent_f32, dtype=np.float32).reshape(-1)
                history[slot_idx].append(sample)

                if len(history[slot_idx]) < _K:
                    continue

                stack = np.stack(list(history[slot_idx]), axis=0)
                mean_var = float(np.mean(np.var(stack, axis=0)))

                if mean_var < var_threshold:
                    owner_id = int(st["owner_id"])
                    inhibit_res = client.inhibit(slot_idx, owner_id)
                    mx.eval(inhibit_res)
                    print(
                        f"[overseer] INHIBIT slot={slot_idx} owner_id={owner_id} "
                        f"mean_var={mean_var:.3e}",
                        flush=True,
                    )
                    history[slot_idx].clear()
    except KeyboardInterrupt:
        print("\n[overseer] stopped.", flush=True)


if __name__ == "__main__":
    main()
