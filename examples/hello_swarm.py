#!/usr/bin/env python3
"""
First run (multi-agent via LocalSwarm)::

    python examples/hello_swarm.py

Low-level slab only (no HTTP server)::

    python examples/hello_swarm.py --slab-only

## Prerequisites

Requires **macOS + Apple Silicon**, ``make python``, models + SAE as in ``scripts/README.md``.
For the default path: ``pip install -r requirements/requirements-server.txt`` (httpx, mlx-lm, FastAPI stack).

**Primary:** :class:`LocalSwarm` bootstraps ``pheromoned`` and spawns ``hiveclaw_server``.

**Low-level:** raw :class:`Swarm` / ``SlabClient`` round-trip (no HTTP).
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx
import numpy as np


def _run_local_swarm() -> int:
    try:
        import hiveclaw_python as hc
    except ImportError as e:
        print("hiveclaw_python not importable:", e, file=sys.stderr)
        print("Run from repo venv after: make python", file=sys.stderr)
        return 1

    try:
        swarm = hc.LocalSwarm(
            model="mlx-community/Llama-3.2-1B-Instruct-4bit",
            port=8765,
            build_if_missing=False,
        )
        swarm.add_agent(
            slot=1,
            goal="Say hello in one short sentence.",
            max_tokens=32,
        )
        swarm.run(stream_output=True)
        swarm.stop()
    except Exception as e:
        print(f"[hello_swarm] LocalSwarm failed: {e}", file=sys.stderr)
        print(
            "[hello_swarm] If the server or daemon did not start, try:\n"
            '  python -c "import hiveclaw_python as hc; hc.init()"',
            file=sys.stderr,
        )
        return 1
    print("\n[hello_swarm] LocalSwarm: OK")
    return 0


def _run_slab_only() -> int:
    try:
        from hiveclaw_python import Swarm
    except ImportError as e:
        print("hiveclaw_python not importable:", e, file=sys.stderr)
        return 1

    _fatal = (
        "pheromoned is not running under launchd, or XPC handshake failed.",
        "From repo root:  cargo build --release -p hiveclaw-daemon && make daemon-load",
        "Then:  make python",
        "See scripts/README.md.",
    )
    try:
        swarm = Swarm()
    except Exception:
        print("\n".join(_fatal), file=sys.stderr)
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
    print(f"[hello_swarm] Slab-only: OK slot={slot} latent_dim={d} max_abs_err={err:.6g}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="HiveClaw hello examples")
    p.add_argument(
        "--slab-only",
        action="store_true",
        help="Only test SlabClient claim/write/read (no HTTP server)",
    )
    args = p.parse_args()
    if args.slab_only:
        return _run_slab_only()
    return _run_local_swarm()


if __name__ == "__main__":
    sys.exit(main())
