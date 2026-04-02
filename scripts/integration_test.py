#!/usr/bin/env python3
"""
Subprocess integration checks for HiveClaw v4 (macOS + pheromoned required).

Exit 0: SlabClient XPC v4 handshake + optional short swarm stress.
Review stderr for JSON telemetry (torn_epoch_skip, etc.) manually for PR merges.

Usage:
  source .venv/bin/activate
  # pheromoned must be running: make daemon-load
  python scripts/integration_test.py
  python scripts/integration_test.py --quick   # XPC + import only
  python scripts/integration_test.py --stress  # + swarm_spike + SIGKILL child
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="HiveClaw v4 integration harness")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Only verify SlabClient / get_surface_v4 (no subprocess agents)",
    )
    p.add_argument(
        "--stress",
        action="store_true",
        help="Run swarm_spike child and SIGKILL it (requires daemon + MLX)",
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
            "assert c.get_scent_dim()>0; print('ok', c.get_scent_dim())",
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

    if args.stress:
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
    raise SystemExit(main())
