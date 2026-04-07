#!/usr/bin/env python3
"""
Run LangChain string committee vs HiveClaw latent committee; print table + JSON.

Usage:
  python benchmarks/benchmark_external.py [--rounds 10] [--agents 5] [--tokens-per-turn 24]
      [--model MODEL_ID] [--no-hiveclaw] [--no-langchain] [--json-out path.json]
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


def _token_tax(coord: int, content: int) -> float:
    t = coord + content
    return float(coord) / float(t) if t > 0 else 0.0


def _print_table(langchain: Any | None, hive: Any | None) -> None:
    def row(label: str, coord: str, content: str, wall: str, growth: str, tax: str) -> None:
        print(f"{label:28} | {coord:>12} | {content:>14} | {wall:>10} | {growth:>18} | {tax:>8}")

    row("Phase", "coord_tokens", "content_tokens", "wall_ms", "ctx_growth", "tax")
    print("-" * 100)
    if langchain and langchain.ok:
        tax = _token_tax(langchain.total_coord_tokens, langchain.total_content_tokens)
        row(
            "LangChain string baseline",
            str(langchain.total_coord_tokens),
            str(langchain.total_content_tokens),
            f"{langchain.total_wall_ms:.0f}",
            "grows per round",
            f"{tax:.3f}",
        )
    elif langchain:
        print(f"LangChain path failed: {langchain.error}")
    else:
        row("LangChain string baseline", "-", "-", "-", "skipped", "-")

    if hive and hive.ok:
        avg_ctx = _avg([float(x) for x in hive.per_round_ctx_tokens])
        tax = _token_tax(hive.total_coord_tokens, hive.total_content_tokens)
        row(
            "HiveClaw latent path",
            str(hive.total_coord_tokens),
            str(hive.total_content_tokens),
            f"{hive.total_wall_ms:.0f}",
            f"~constant (~{avg_ctx:.0f} tok/rd)",
            f"{tax:.3f}",
        )
        if langchain and langchain.ok:
            sp = langchain.total_wall_ms / max(hive.total_wall_ms, 1e-6)
            print("-" * 100)
            print(
                f"Wall-clock speedup (langchain / hiveclaw): {sp:.2f}x "
                f"(content_tokens langchain={langchain.total_content_tokens}, "
                f"hive={hive.total_content_tokens})"
            )
            print(
                "Coordination token tax (LangChain path): "
                f"{langchain.total_coord_tokens} prompt tokens not attributable to raw code body; "
                f"share={_token_tax(langchain.total_coord_tokens, langchain.total_content_tokens):.3f}"
            )
    elif hive:
        print("-" * 100)
        print(f"HiveClaw path failed: {hive.error}")
    else:
        print("-" * 100)
        print("HiveClaw path skipped (--no-hiveclaw).")


def main() -> int:
    p = argparse.ArgumentParser(description="External benchmark: LangChain vs HiveClaw consensus")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--tokens-per-turn", type=int, default=24)
    p.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--no-hiveclaw", action="store_true")
    p.add_argument("--no-langchain", action="store_true")
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()

    common = dict(
        model_id=args.model,
        n_rounds=args.rounds,
        n_agents=args.agents,
        max_tokens_per_turn=args.tokens_per_turn,
    )

    langchain_res = None
    if not args.no_langchain:
        from langchain_string_swarm import run_langchain_baseline

        langchain_res = run_langchain_baseline(**common)

    hive = None
    if not args.no_hiveclaw:
        from hiveclaw_consensus import run_hiveclaw_consensus

        hive = run_hiveclaw_consensus(**common)

    summary = {
        "event": "benchmark_external_summary",
        "langchain": langchain_res.to_dict() if langchain_res else None,
        "hiveclaw": hive.to_dict() if hive else None,
    }
    print(json.dumps(summary, indent=2))
    print()
    _print_table(langchain_res, hive)

    if args.json_out.strip():
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")

    if hive is not None and not hive.ok:
        return 1
    if langchain_res is not None and not langchain_res.ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
