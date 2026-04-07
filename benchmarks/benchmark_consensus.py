#!/usr/bin/env python3
"""
Run string-passing baseline vs HiveClaw latent committee benchmark; print table + JSON.

Usage:
  python benchmarks/benchmark_consensus.py [--rounds 10] [--agents 5] [--tokens-per-turn 24]
      [--model MODEL_ID] [--no-hiveclaw] [--json-out path.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _print_table(baseline: Any, hive: Any | None) -> None:
    def row(label: str, coord: str, content: str, wall: str, growth: str) -> None:
        print(f"{label:24} | {coord:>12} | {content:>14} | {wall:>12} | {growth}")

    row("Phase", "coord_tokens", "content_tokens", "wall_ms", "ctx_growth")
    print("-" * 84)
    row(
        "String-passing baseline",
        str(baseline.total_coord_tokens),
        str(baseline.total_content_tokens),
        f"{baseline.total_wall_ms:.0f}",
        "grows per round",
    )
    if hive and hive.ok:
        avg_ctx = _avg([float(x) for x in hive.per_round_ctx_tokens])
        row(
            "HiveClaw latent path",
            str(hive.total_coord_tokens),
            str(hive.total_content_tokens),
            f"{hive.total_wall_ms:.0f}",
            f"~constant (~{avg_ctx:.0f} tok/rd)",
        )
        sp = baseline.total_wall_ms / max(hive.total_wall_ms, 1e-6)
        print("-" * 84)
        print(
            f"Wall-clock speedup (baseline / hiveclaw): {sp:.2f}x "
            f"(content_tokens baseline={baseline.total_content_tokens}, "
            f"hive={hive.total_content_tokens})"
        )
        print(
            "Coordination token tax (baseline only): "
            f"{baseline.total_coord_tokens} prompt tokens not attributable to raw code body."
        )
    elif hive:
        print("-" * 84)
        print(f"HiveClaw path failed: {hive.error}")
    else:
        print("-" * 84)
        print("HiveClaw path skipped (--no-hiveclaw).")


def main() -> int:
    p = argparse.ArgumentParser(description="Consensus benchmark: string vs latent")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--tokens-per-turn", type=int, default=24)
    p.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--no-hiveclaw", action="store_true")
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()

    from string_swarm_baseline import run_string_baseline

    common = dict(
        model_id=args.model,
        n_rounds=args.rounds,
        n_agents=args.agents,
        max_tokens_per_turn=args.tokens_per_turn,
    )

    baseline = run_string_baseline(**common)
    hive = None
    if not args.no_hiveclaw:
        from hiveclaw_consensus import run_hiveclaw_consensus

        hive = run_hiveclaw_consensus(**common)

    summary = {
        "event": "benchmark_consensus_summary",
        "baseline": baseline.to_dict(),
        "hiveclaw": hive.to_dict() if hive else None,
    }
    print(json.dumps(summary, indent=2))
    print()
    _print_table(baseline, hive)

    if args.json_out.strip():
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")

    if hive is not None and not hive.ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
