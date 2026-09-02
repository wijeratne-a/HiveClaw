#!/usr/bin/env python3
"""Session 7 runner: Postgres over TCP, scaling + leases + connection drop.

Usage:
  HIVECLAW_PG_DSN=postgresql://hiveclaw:hiveclaw@127.0.0.1:55432/hiveclaw \\
    .venv/bin/python scripts/exp004_multi_host.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hiveclaw_causal.benchmark import measure_repair
from hiveclaw_causal.pg_store import PgStore, new_schema_name


def wait_pg(dsn: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            store = PgStore(dsn, schema="public")
            store.close()
            return
        except Exception as exc:
            last = str(exc)
            time.sleep(0.4)
    raise RuntimeError(f"Postgres not ready: {last}")


def one_pair(
    dsn: str,
    *,
    extra_unrelated: int = 0,
    extra_related_claims: int = 0,
    extra_unrelated_claims: int = 0,
) -> dict[str, object]:
    kwargs = {
        "seed": 42,
        "extra_unrelated": extra_unrelated,
        "extra_related_claims": extra_related_claims,
        "extra_unrelated_claims": extra_unrelated_claims,
    }
    t_store = PgStore(dsn, schema=new_schema_name("t"))
    n_store = PgStore(dsn, schema=new_schema_name("n"))
    try:
        _rt_t, targeted = measure_repair("pg:t", mode="targeted", store=t_store, **kwargs)
        _rt_n, naive = measure_repair("pg:n", mode="naive", store=n_store, **kwargs)
        return {
            "targeted": {
                "objects_before": targeted.objects_before,
                "objects_after": targeted.objects_after,
                "objects_touched": targeted.objects_touched,
                "eval_steps": targeted.eval_steps,
                "wall_s": targeted.wall_s,
                "support_pct": targeted.support_pct,
                "rollback_blocked": targeted.rollback_blocked,
                "followup_present": targeted.followup_present,
            },
            "naive": {
                "objects_before": naive.objects_before,
                "objects_after": naive.objects_after,
                "objects_touched": naive.objects_touched,
                "eval_steps": naive.eval_steps,
                "wall_s": naive.wall_s,
                "support_pct": naive.support_pct,
                "rollback_blocked": naive.rollback_blocked,
                "followup_present": naive.followup_present,
            },
        }
    finally:
        t_store.close()
        n_store.close()


def main() -> int:
    dsn = os.environ.get("HIVECLAW_PG_DSN", "").strip()
    if not dsn:
        print("HIVECLAW_PG_DSN is required", file=sys.stderr)
        return 2
    wait_pg(dsn)
    print("Postgres accepted a TCP connection.", flush=True)

    scales = [
        ("N~12", {"extra_unrelated": 0, "extra_related_claims": 0}),
        ("N~100", {"extra_unrelated": 29, "extra_related_claims": 1}),
        ("N~500", {"extra_unrelated": 162, "extra_related_claims": 2}),
        ("N~2000", {"extra_unrelated": 662, "extra_related_claims": 2}),
    ]
    claims = [
        ("C=0", {"extra_unrelated_claims": 0}),
        ("C=500", {"extra_unrelated_claims": 500}),
        ("C=2000", {"extra_unrelated_claims": 2000}),
    ]

    results: dict[str, object] = {"task_scales": {}, "claim_scales": {}}
    print("\n=== task scales (two runs) ===", flush=True)
    for label, kwargs in scales:
        runs = []
        for i in range(2):
            pair = one_pair(dsn, **kwargs)
            runs.append(pair)
            t = pair["targeted"]
            n = pair["naive"]
            print(
                f"{label} run{i+1}: eval {t['eval_steps']}/{n['eval_steps']} "
                f"touched {t['objects_touched']}/{n['objects_touched']} "
                f"wall {t['wall_s']:.6f}/{n['wall_s']:.6f} "
                f"support {t['support_pct']:.1f} rollback={t['rollback_blocked']}",
                flush=True,
            )
        results["task_scales"][label] = runs  # type: ignore[index]

    print("\n=== claim scales (two runs) ===", flush=True)
    for label, kwargs in claims:
        runs = []
        for i in range(2):
            pair = one_pair(dsn, **kwargs)
            runs.append(pair)
            t = pair["targeted"]
            n = pair["naive"]
            print(
                f"{label} run{i+1}: eval {t['eval_steps']}/{n['eval_steps']} "
                f"touched {t['objects_touched']}/{n['objects_touched']} "
                f"wall {t['wall_s']:.6f}/{n['wall_s']:.6f} "
                f"support {t['support_pct']:.1f} rollback={t['rollback_blocked']}",
                flush=True,
            )
        results["claim_scales"][label] = runs  # type: ignore[index]

    print("\n=== unittest (leases, drop, stall) ===", flush=True)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName("tests.test_hiveclaw_causal_pg")
    runner = unittest.TextTestRunner(verbosity=2)
    outcome = runner.run(suite)
    results["unittest"] = {
        "testsRun": outcome.testsRun,
        "failures": len(outcome.failures),
        "errors": len(outcome.errors),
        "skipped": len(outcome.skipped),
        "wasSuccessful": outcome.wasSuccessful(),
    }
    print(json.dumps(results, indent=2, default=str))
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
