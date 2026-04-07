#!/usr/bin/env python3
"""
Benchmark the quality-gated triple-threat pipeline over multiple runs.

Requires a running OpenAI-compatible server (e.g. HiveClaw) or cloud credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from demo_triple_threat import (  # noqa: E402
    _resolve_quality_profile,
    _run_openai_sdk_path,
)
from quality_controller import QualityController, QualityGateFailure  # noqa: E402


def _fixture_paths(root: Path) -> list[Path]:
    fx = root / "fixtures"
    if not fx.is_dir():
        return []
    return sorted(p for p in fx.glob("*.py") if p.is_file())


def main() -> int:
    p = argparse.ArgumentParser(description="Quality gate benchmark runner")
    p.add_argument(
        "--profile",
        default="python_refactor",
        help="Quality profile name or YAML path",
    )
    p.add_argument("--runs", type=int, default=10, help="Full pipeline runs per fixture")
    p.add_argument("--json-out", type=str, default="", help="Write summary JSON here")
    p.add_argument(
        "--base-url",
        default=os.environ.get("HIVECLAW_BASE_URL", "http://127.0.0.1:8080/v1"),
        help="OpenAI-compatible API base URL",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("HIVECLAW_MODEL", "hiveclaw-llama-1b"),
        help="Model id",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--rounds", type=int, default=1, help="3-agent pipeline repeats per run")
    p.add_argument(
        "--fixtures-dir",
        type=str,
        default="",
        help="Override fixtures directory (default: scripts/fixtures)",
    )
    p.add_argument(
        "--quality-report-only",
        action="store_true",
        help="Non-blocking quality mode",
    )
    args = p.parse_args()

    try:
        from openai import OpenAI
    except ImportError as e:
        print("pip install -r scripts/requirements-bench-openai.txt", file=sys.stderr)
        raise SystemExit(2) from e

    profile_path = _resolve_quality_profile(args.profile)
    qc = QualityController(
        profile_path,
        report_only=True if args.quality_report_only else None,
    )

    fixtures_root = Path(args.fixtures_dir) if args.fixtures_dir else _scripts
    paths = _fixture_paths(fixtures_root)
    if not paths:
        demo_target = _scripts / "demo_target.py"
        paths = [demo_target] if demo_target.is_file() else []

    if not paths:
        print("No fixtures found (scripts/fixtures/*.py or demo_target.py)", file=sys.stderr)
        return 2

    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key="sk-benchmark")

    passes = 0
    fails = 0
    blocker_freq: Counter[str] = Counter()
    total_retries = 0
    total_tokens = 0
    wall_times: list[float] = []

    for _ in range(args.runs):
        for fix in paths:
            src = fix.read_text(encoding="utf-8")
            t0 = time.perf_counter()
            try:
                m = _run_openai_sdk_path(
                    label=f"bench:{fix.name}",
                    client=client,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    rounds=args.rounds,
                    original_source=src,
                    compact_prompts=True,
                    estimate_cloud_cost=False,
                    cloud_provider=None,
                    quality=qc,
                )
            except QualityGateFailure as e:
                fails += 1
                wall_times.append(time.perf_counter() - t0)
                for rep in e.reports:
                    for v in rep.violations:
                        if v.rule_id in qc.profile.hard_blockers:
                            blocker_freq[v.rule_id] += 1
                continue
            wall_times.append(time.perf_counter() - t0)
            total_retries += m.total_retries
            total_tokens += m.total_tokens
            if m.gate_pass_rate >= 1.0 - 1e-9:
                passes += 1
            else:
                fails += 1
                for tr in m.transcripts:
                    q = tr.get("quality") or {}
                    for rep in q.get("reports") or []:
                        for vd in rep.get("violations") or []:
                            rid = vd.get("rule_id")
                            if rid in qc.profile.hard_blockers:
                                blocker_freq[rid] += 1

    n = passes + fails
    summary = {
        "profile": args.profile,
        "runs": args.runs,
        "fixtures": [str(x) for x in paths],
        "pass_rate": (passes / n) if n else 0.0,
        "passes": passes,
        "fails": fails,
        "blocker_frequencies": dict(blocker_freq),
        "mean_retries": (total_retries / n) if n else 0.0,
        "mean_wall_s": (sum(wall_times) / len(wall_times)) if wall_times else 0.0,
        "mean_total_tokens": (total_tokens / n) if n else 0.0,
    }

    print(json.dumps(summary, indent=2))
    if args.json_out.strip():
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
