"""Run The Rewind locally and print status explanations from SQLite."""

from __future__ import annotations

import argparse
from pathlib import Path

from .rewind import (
    ACTION_ROLLBACK,
    CLAIM_CACHE,
    CLAIM_OUTAGE,
    CLAIM_RESIDUAL,
    TASK_FOLLOWUP,
    run_rewind,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="HiveClaw Rewind demo: ingest, invalidate, block rollback, compute outage %."
    )
    p.add_argument(
        "--db",
        default="output/rewind.sqlite",
        help="SQLite path (created/overwritten by replacing the file)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    db = Path(args.db)
    if db.exists():
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)

    rt = run_rewind(db, seed=args.seed)
    pct = rt.computed_outage_support_pct()
    decision = rt.policy_authorize(ACTION_ROLLBACK)
    report = rt.last_revalidation()

    print(f"db={db}")
    print(f"outage_explains_pct={pct:.4f}  (from hiveclaw_causal.stats.outage_explains_pct)")
    print(f"policy_authorize({ACTION_ROLLBACK}): allowed={decision.allowed} reason={decision.reason}")
    print("scheduled:", ", ".join(report.scheduled_task_ids) or "(none)")
    print("not rerun:")
    for tid, why in report.skipped:
        print(f"  {tid}: {why}")
    print()
    for oid in (
        CLAIM_CACHE,
        ACTION_ROLLBACK,
        CLAIM_OUTAGE,
        CLAIM_RESIDUAL,
        TASK_FOLLOWUP,
    ):
        info = rt.why(oid)
        print(f"== {oid} ==")
        print(f"  status={info['status']} producer={info['producer']} trust={info['trust']}")
        print(f"  last_reason={info['last_reason']}")
        print(f"  last_edge_id={info['last_edge_id']} last_rule={info['last_rule']}")
        print(f"  evidence={list(info['evidence_ids'])}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
