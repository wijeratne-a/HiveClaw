#!/usr/bin/env python3
"""Concurrent task leases: real processes, shared SQLite, no worker-to-worker messages."""

from __future__ import annotations

import multiprocessing
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hiveclaw_causal.fixture import build_rewind_fixture  # noqa: E402
from hiveclaw_causal.lease import mp_drain  # noqa: E402
from hiveclaw_causal.rewind import RewindRuntime  # noqa: E402
from hiveclaw_causal.store import Store  # noqa: E402
from hiveclaw_causal.types import (  # noqa: E402
    ArtifactKind,
    ObjectKind,
    Provenance,
    Record,
    SourceRef,
    TaskStatus,
    TrustClass,
)
from hiveclaw_causal.util import content_hash  # noqa: E402


def _seed_pending_tasks(db_path: Path, n: int) -> list[str]:
    store = Store(db_path)
    ids: list[str] = []
    ts = "2026-08-31T00:00:00Z"
    for i in range(n):
        tid = f"task-lease-{i:04d}"
        payload = {"kind": "synthetic", "i": i}
        rec = Record(
            id=tid,
            kind=ObjectKind.TASK,
            status=TaskStatus.PENDING.value,
            provenance=Provenance(
                producer="test",
                sources=(
                    SourceRef(
                        uri=f"test://task/{i}",
                        version="1",
                        content_hash=content_hash(payload),
                    ),
                ),
                timestamp=ts,
                trust=TrustClass.DETERMINISTIC,
            ),
            payload=payload,
            evidence_ids=(),
            source_snapshot=content_hash(payload),
            created_at=ts,
            updated_at=ts,
        )
        store.put_object(rec, ts=ts, event_type="seed_task", reason="lease stress seed")
        ids.append(tid)
    store.close()
    return ids


def _run_workers(db_path: Path, n_workers: int, pause_s: float) -> list[tuple[str, list[str]]]:
    ctx = multiprocessing.get_context("spawn")
    jobs = [
        (str(db_path), f"worker-{i}", pause_s) for i in range(n_workers)
    ]
    with ctx.Pool(processes=n_workers) as pool:
        return list(pool.map(mp_drain, jobs))


class TestConcurrentLeases(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_two_workers_after_rewind_injection(self) -> None:
        db = self.dir / "rewind.sqlite"
        rt = RewindRuntime.create(db)
        fixture = build_rewind_fixture(seed=42)
        rt.ingest_and_propose(fixture)
        rt.ingest_artifact(
            kind=ArtifactKind.PROVIDER_INCIDENT,
            producer="ingestor",
            body=fixture.provider_report,
            trust=TrustClass.TRUSTED,
            source_uri=str(fixture.provider_report["uri"]),
            timestamp=str(fixture.provider_report["window_start"]),
        )
        pending = [
            t.id
            for t in rt.objects_of(ObjectKind.TASK)
            if t.status == TaskStatus.PENDING.value
        ]
        self.assertGreaterEqual(len(pending), 2)
        rt.store.close()

        results = _run_workers(db, n_workers=2, pause_s=0.003)
        leased_all = [tid for _w, ids in results for tid in ids]
        self.assertEqual(sorted(leased_all), sorted(pending))
        self.assertEqual(len(leased_all), len(set(leased_all)), Counter(leased_all))

        store = Store(db)
        try:
            tasks = store.objects_of(ObjectKind.TASK)
            self.assertTrue(all(t.status == TaskStatus.DONE.value for t in tasks))
            owners = [t.payload.get("lease_owner") for t in tasks]
            self.assertTrue(all(owners))
        finally:
            store.close()

    def test_more_workers_than_tasks_no_double_lease(self) -> None:
        """5 processes, 3 tasks, 8 repeats. Fail if any id is leased twice."""
        double = 0
        dropped = 0
        for trial in range(8):
            db = self.dir / f"stress-{trial}.sqlite"
            expected = _seed_pending_tasks(db, 3)
            results = _run_workers(db, n_workers=5, pause_s=0.004)
            leased_all = [tid for _w, ids in results for tid in ids]
            counts = Counter(leased_all)
            if any(c != 1 for c in counts.values()):
                double += 1
            if sorted(leased_all) != sorted(expected):
                dropped += 1
            store = Store(db)
            try:
                tasks = {t.id: t for t in store.objects_of(ObjectKind.TASK)}
                for tid in expected:
                    self.assertEqual(tasks[tid].status, TaskStatus.DONE.value)
            finally:
                store.close()
        self.assertEqual(double, 0, "double-lease trials")
        self.assertEqual(dropped, 0, "dropped-task trials")


if __name__ == "__main__":
    unittest.main()
