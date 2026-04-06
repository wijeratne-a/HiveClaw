#!/usr/bin/env python3
"""
Minimal HiveClaw slab handshake: claim a slot, write/read v5 latent, release.

Requires: ``pheromoned`` under launchd and ``make python`` (see ``scripts/README.md``).
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

from hiveclaw_python import Swarm

_FATAL = (
    "pheromoned is not running under launchd, or XPC handshake failed.",
    "From repo root:  cargo build --release -p hiveclaw-daemon && make daemon-load",
    "Then:  make python",
    "See scripts/README.md.",
)


def main() -> int:
    try:
        swarm = Swarm()
    except Exception:
        print("\n".join(_FATAL), file=sys.stderr)
        return 1

    slot = swarm.try_claim_any()
    if slot < 0:
        print("[hello_swarm] no slot claimed (all busy?)", file=sys.stderr)
        return 1

    d = swarm.latent_dim()
    latent = mx.ones((1, 1, d), dtype=mx.bfloat16)
    w = swarm.client.write_slot_v5(slot, latent)
    mx.eval(w)
    r = swarm.client.read_slot_v5(slot)
    mx.eval(r)
    swarm.release(slot)

    err = float(
        np.max(
            np.abs(
                np.array(r.astype(mx.float32), dtype=np.float64)
                - np.array(latent.astype(mx.float32), dtype=np.float64)
            )
        )
    )
    print(f"hello_swarm: OK slot={slot} latent_dim={d} max_abs_err={err:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
