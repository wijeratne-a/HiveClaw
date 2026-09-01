"""Measure targeted reverse-dep repair vs naive full re-evaluation on The Rewind."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from .fixture import build_rewind_fixture
from .rewind import ACTION_ROLLBACK, TASK_FOLLOWUP, RewindRuntime
from .types import ActionStatus, ArtifactKind, TrustClass
from .work import WorkMetrics

_KIND_URI_PROVIDER = "fixture://provider-incident/INC-8841"


def measure_repair(
    db_path: Path | str,
    *,
    mode: str,
    seed: int = 42,
) -> tuple[RewindRuntime, WorkMetrics]:
    if mode not in ("targeted", "naive"):
        raise ValueError(mode)
    path = Path(db_path)
    rt = RewindRuntime.create(path)
    fixture = build_rewind_fixture(seed=seed)
    rt.ingest_and_propose(fixture)
    before = {r.id: (r.status, r.updated_at) for r in rt.store.all_objects()}
    rt.work.reset()
    t0 = time.perf_counter()
    rt.ingest_artifact(
        kind=ArtifactKind.PROVIDER_INCIDENT,
        producer="ingestor",
        body=fixture.provider_report,
        trust=TrustClass.TRUSTED,
        source_uri=str(fixture.provider_report.get("uri", _KIND_URI_PROVIDER)),
        timestamp=str(fixture.provider_report.get("window_start")),
        repair=mode,
    )
    wall_s = time.perf_counter() - t0
    after = rt.store.all_objects()
    changed: set[str] = set()
    for rec in after:
        prev = before.get(rec.id)
        if prev is None or prev != (rec.status, rec.updated_at):
            changed.add(rec.id)
    touched = set(rt.work.touched) | changed
    untouched = tuple(sorted(oid for oid in before if oid not in touched))
    rollback = rt.get(ACTION_ROLLBACK)
    follow = rt.store.get_or_none(TASK_FOLLOWUP)
    return rt, WorkMetrics(
        mode=mode,
        objects_before=len(before),
        objects_after=len(after),
        objects_touched=len(touched),
        objects_untouched=len(untouched),
        eval_steps=rt.work.eval_steps,
        wall_s=wall_s,
        support_pct=rt.computed_outage_support_pct(),
        rollback_status=rollback.status,
        rollback_blocked=rollback.status == ActionStatus.BLOCKED.value,
        followup_present=follow is not None,
        touched_ids=tuple(sorted(touched)),
        untouched_ids=untouched,
    )


def format_table(targeted: WorkMetrics, naive: WorkMetrics) -> str:
    rows = [
        ("metric", "targeted", "naive"),
        ("objects_before", str(targeted.objects_before), str(naive.objects_before)),
        ("objects_after", str(targeted.objects_after), str(naive.objects_after)),
        ("objects_touched", str(targeted.objects_touched), str(naive.objects_touched)),
        ("objects_untouched", str(targeted.objects_untouched), str(naive.objects_untouched)),
        ("eval_steps", str(targeted.eval_steps), str(naive.eval_steps)),
        ("wall_s", f"{targeted.wall_s:.6f}", f"{naive.wall_s:.6f}"),
        ("support_pct", f"{targeted.support_pct:.1f}", f"{naive.support_pct:.1f}"),
        ("rollback_blocked", str(targeted.rollback_blocked), str(naive.rollback_blocked)),
        ("followup_present", str(targeted.followup_present), str(naive.followup_present)),
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    lines = []
    for r in rows:
        lines.append("  ".join(r[i].ljust(widths[i]) for i in range(3)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true")
    p.add_argument("--dir", type=Path, default=None)
    args = p.parse_args(argv)
    td: tempfile.TemporaryDirectory[str] | None = None
    base = args.dir
    if base is None:
        td = tempfile.TemporaryDirectory()
        base = Path(td.name)
    try:
        t_rt, t_m = measure_repair(base / "targeted.sqlite", mode="targeted", seed=args.seed)
        n_rt, n_m = measure_repair(base / "naive.sqlite", mode="naive", seed=args.seed)
        _ = (t_rt, n_rt)
        print(format_table(t_m, n_m))
        print()
        print("targeted untouched:", ", ".join(t_m.untouched_ids) or "(none)")
        print("naive untouched:   ", ", ".join(n_m.untouched_ids) or "(none)")
        if args.json:
            print(
                json.dumps(
                    {
                        "targeted": t_m.__dict__,
                        "naive": n_m.__dict__,
                    },
                    indent=2,
                )
            )
        return 0
    finally:
        if td is not None:
            td.cleanup()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
