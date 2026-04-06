#!/usr/bin/env python3
"""
LocalSwarm with optional Catenar Proof-of-Task traces (one entry per agent SSE turn).

Prereqs:
  - HiveClaw: make python, daemon, models + SAE, pip install -r scripts/requirements-server.txt
  - Catenar: pip install per scripts/requirements-catenar.txt; verifier up (e.g. docker compose)

Traces append to ./catenar-trace-wal.jsonl; verifier POST /v1/verify returns receipts when healthy.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if os.environ.get("CATENAR_DEMO", "").strip() not in ("1", "true", "yes"):
        os.environ.setdefault("CATENAR_DEMO", "1")

    try:
        import hiveclaw_python as hc
    except ImportError as e:
        print("hiveclaw_python not importable:", e, file=sys.stderr)
        return 1

    swarm = hc.LocalSwarm(
        model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        port=8766,
        build_if_missing=False,
        catenar_enabled=True,
        catenar_url=os.environ.get("CATENAR_BASE_URL", "http://127.0.0.1:3000"),
        catenar_agent_id=os.environ.get("CATENAR_AGENT_ID", "hiveclaw-demo"),
    )
    swarm.add_agent(slot=1, goal="Say hello in one short sentence.", max_tokens=48)
    swarm.add_agent(slot=2, goal="Name one color in one word.", max_tokens=16)
    try:
        swarm.run(stream_output=True)
    finally:
        swarm.stop()
    print("\n[local_swarm_catenar] done. Check catenar-trace-wal.jsonl and Catenar dashboard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
