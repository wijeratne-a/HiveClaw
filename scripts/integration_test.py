#!/usr/bin/env python3
"""
Subprocess integration checks for HiveClaw v6 (macOS + pheromoned required).

Exit 0: SlabClient XPC handshake, global header, read_slot_v5 shape (latent dim from slab).
Review stderr for JSON telemetry (torn_epoch_skip, etc.) manually for PR merges.

Usage:
  source .venv/bin/activate
  # pheromoned must be running: make daemon-load
  python scripts/integration_test.py
  python scripts/integration_test.py --quick   # XPC + import only
  python scripts/integration_test.py --stress  # + claim/release loop + swarm_spike child
  python scripts/integration_test.py --stress --stress-max-slots 4096  # full slab gauntlet
  python scripts/integration_test.py --batched   # read_slots / write_slots (daemon required)
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

EXPECT_MAGIC = 0x48434C5700000006


def _stress_claim_release(n_slots: int) -> None:
    import mlx.core as mx

    import hiveclaw_python as h

    c = h.SlabClient()
    for i in range(n_slots):
        cand = mx.array([i], dtype=mx.int32)
        r = c.claim_task(cand)
        mx.eval(r)
        got = int(mx.array(r).item())
        assert got == i, (got, i)
        c.release_task(i)
    print(f"[integration_test] stress ok: {n_slots} claim/release cycles", flush=True)


def _batched_slab_roundtrip() -> None:
    import mlx.core as mx
    import numpy as np

    import hiveclaw_python as h

    c = h.SlabClient()
    d = int(c.get_latent_dim())
    slots = [200, 201, 202, 203]
    for s in slots:
        cand = mx.array([s], dtype=mx.int32)
        r = c.claim_task(cand)
        mx.eval(r)
        got = int(mx.array(r).item())
        assert got == s, (got, s)
    B = len(slots)
    si = mx.array(slots, dtype=mx.int32)
    data, st = c.read_slots(si)
    mx.eval(data, st)
    assert list(data.shape) == [B, 1, d], data.shape
    assert list(st.shape) == [B], st.shape
    latent = mx.ones((B, 1, d), dtype=mx.bfloat16)
    out, wst = c.write_slots(si, latent)
    mx.eval(out, wst)
    assert list(wst.shape) == [B]
    wnp = np.array(wst, dtype=np.uint8).reshape(-1)
    assert np.all(wnp == 0), wnp.tolist()
    for s in slots:
        c.release_task(s)
    print("[integration_test] batched read/write ok", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="HiveClaw v6 integration harness")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Only verify SlabClient / get_surface_v5 (no subprocess agents)",
    )
    p.add_argument(
        "--stress",
        action="store_true",
        help="Run claim/release stress + swarm_spike child SIGKILL",
    )
    p.add_argument(
        "--stress-max-slots",
        type=int,
        default=256,
        metavar="N",
        help="Sequential slots for --stress (default 256; use 4096 for full slab)",
    )
    p.add_argument(
        "--batched",
        action="store_true",
        help="Claim 4 slots, read_slots + write_slots, assert shapes and status",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    py = sys.executable
    env = os.environ.copy()

    smoke = subprocess.run(
        [
            py,
            "-c",
            "import hiveclaw_python as h; c=h.SlabClient(); "
            "d=int(c.get_latent_dim()); assert d>=1; "
            f"assert c.read_u64_at(0)=={EXPECT_MAGIC}; "
            "assert c.read_u32_at(8)==6; "
            "import mlx.core as mx; z=c.read_slot_v5(0); "
            "assert list(z.shape)==[1,1,d]; "
            "print('ok', d)",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )
    if smoke.returncode != 0:
        print(smoke.stdout, file=sys.stdout)
        print(smoke.stderr, file=sys.stderr)
        return smoke.returncode or 1
    if "nan" in (smoke.stdout + smoke.stderr).lower():
        print("FAIL: NaN in smoke output", file=sys.stderr)
        return 2

    print(smoke.stdout.strip(), flush=True)

    if args.quick:
        return 0

    if args.batched:
        try:
            _batched_slab_roundtrip()
        except Exception as e:
            print(f"batched slab test failed: {e}", file=sys.stderr)
            return 1
        if not args.stress:
            return 0

    if args.stress:
        n = int(args.stress_max_slots)
        if n < 1 or n > 4096:
            print("--stress-max-slots must be in [1, 4096]", file=sys.stderr)
            return 2
        try:
            _stress_claim_release(n)
        except Exception as e:
            print(f"stress failed: {e}", file=sys.stderr)
            return 1

        sp = subprocess.Popen(
            [py, str(repo / "scripts" / "swarm_spike.py")],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        time.sleep(8)
        sp.send_signal(signal.SIGKILL)
        try:
            sp.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sp.kill()

    return 0


if __name__ == "__main__":
    # Support `python -c` only for stress body — normal entry is main().
    raise SystemExit(main())
