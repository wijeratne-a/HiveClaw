#!/usr/bin/env python3
"""Concurrent task leases: real processes, shared SQLite, no worker-to-worker messages."""

from __future__ import annotations

import multiprocessing
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hiveclaw_causal.fixture import build_rewind_fixture  # noqa: E402
from hiveclaw_causal.lease import (  # noqa: E402
    drain_pending_tasks,
    lease_one_and_die,
    mp_drain,
    mp_drain_until_idle,
    work_slow_with_renew,
)
from hiveclaw_causal.lease_policy import LEASE_TTL_CEILING_S  # noqa: E402
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


def _seed_pending_tasks(db_path: Path, n: int, start: int = 0) -> list[str]:
    store = Store(db_path)
    ids: list[str] = []
    ts = "2026-08-31T00:00:00Z"
    for i in range(start, start + n):
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

    def test_oversized_client_ttl_does_not_strand_after_silence(self) -> None:
        """A client-requested hour-long TTL must not make a silent owner unrecoverable.

        Session 7: TCP death does not release a lease. If the client can set
        lease_until arbitrarily far in the future, a dropped path strands the
        task. Desired: store clamps TTL; after the ceiling (1s here) a survivor
        reclaims. Client still asks for 3600s.
        """
        db = self.dir / "strand.sqlite"
        expected = _seed_pending_tasks(db, 1)
        tid = expected[0]
        store = Store(db, max_lease_ttl_s=1.0)
        rec = store.try_lease_one_task(
            "stranded", "2026-08-31T00:00:00Z", lease_ttl_s=3600.0
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        remaining = float(rec.payload["lease_until"]) - time.time()
        self.assertLessEqual(remaining, 1.5, f"client 3600s TTL was honored: {remaining:.1f}s left")
        store.close()

        time.sleep(1.25)
        recovered = drain_pending_tasks(
            str(db), "survivor", pause_s=0.0, lease_ttl_s=30.0
        )
        self.assertEqual(recovered, [tid])
        store = Store(db)
        try:
            rec = store.get(tid)
            self.assertEqual(rec.status, TaskStatus.DONE.value)
            self.assertEqual(rec.payload.get("completed_by"), "survivor")
            self.assertEqual(rec.payload.get("reclaimed_from"), "stranded")
        finally:
            store.close()

    def test_infinite_client_ttl_is_clamped_to_absolute_ceiling(self) -> None:
        db = self.dir / "inf.sqlite"
        _seed_pending_tasks(db, 1)
        store = Store(db)
        rec = store.try_lease_one_task(
            "w", "2026-08-31T00:00:00Z", lease_ttl_s=float("inf")
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        remaining = float(rec.payload["lease_until"]) - time.time()
        self.assertLessEqual(remaining, LEASE_TTL_CEILING_S + 0.5)
        self.assertGreater(remaining, LEASE_TTL_CEILING_S - 2.0)
        store.close()

    def test_killed_worker_lease_is_reclaimed(self) -> None:
        db = self.dir / "crash.sqlite"
        expected = _seed_pending_tasks(db, 1)
        tid = expected[0]
        ttl_s = 0.25
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=lease_one_and_die,
            args=((str(db), "crasher", ttl_s),),
        )
        p.start()
        deadline = time.time() + 8.0
        saw_crasher_lease = False
        while time.time() < deadline:
            store = Store(db)
            try:
                rec = store.get(tid)
                if (
                    rec.status == TaskStatus.LEASED.value
                    and rec.payload.get("lease_owner") == "crasher"
                ):
                    saw_crasher_lease = True
                    break
            finally:
                store.close()
            if not p.is_alive() and not saw_crasher_lease:
                time.sleep(0.01)
                continue
            time.sleep(0.01)
        p.join(timeout=3)
        self.assertTrue(saw_crasher_lease, "crasher never committed a lease")
        self.assertNotEqual(p.exitcode, 0)

        store = Store(db)
        try:
            rec = store.get(tid)
            self.assertEqual(rec.status, TaskStatus.LEASED.value)
            self.assertEqual(rec.payload.get("lease_owner"), "crasher")
        finally:
            store.close()

        time.sleep(ttl_s + 0.15)
        recovered = drain_pending_tasks(
            str(db), "survivor", pause_s=0.0, lease_ttl_s=30.0
        )
        self.assertEqual(recovered, [tid])

        store = Store(db)
        try:
            rec = store.get(tid)
            self.assertEqual(rec.status, TaskStatus.DONE.value)
            self.assertEqual(rec.payload.get("completed_by"), "survivor")
            self.assertEqual(rec.payload.get("reclaimed_from"), "crasher")
            reclaim_evs = [
                e
                for e in store.events_for(tid)
                if e.event_type == "lease_reclaim"
            ]
            self.assertTrue(reclaim_evs, "expected lease_reclaim event")
        finally:
            store.close()

    def test_slow_alive_worker_that_renews_is_not_reclaimed(self) -> None:
        """Wall-clock past TTL is not enough: a live worker that heartbeats keeps the lease."""
        db = self.dir / "slow.sqlite"
        expected = _seed_pending_tasks(db, 1)
        tid = expected[0]
        ttl_s = 0.2
        work_s = 0.7
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=work_slow_with_renew,
            args=((str(db), "slow", ttl_s, work_s, 0.05),),
        )
        p.start()
        deadline = time.time() + 8.0
        saw = False
        while time.time() < deadline:
            store = Store(db)
            try:
                rec = store.get(tid)
                if rec.status == TaskStatus.LEASED.value and rec.payload.get("lease_owner") == "slow":
                    saw = True
                    break
            finally:
                store.close()
            time.sleep(0.01)
        self.assertTrue(saw, "slow worker never leased")

        time.sleep(ttl_s + 0.2)
        store = Store(db)
        try:
            stolen = store.try_lease_one_task("poacher", "2026-08-31T00:00:01Z", lease_ttl_s=30.0)
            self.assertIsNone(stolen, "slow-but-alive lease was reclaimed")
            rec = store.get(tid)
            self.assertEqual(rec.status, TaskStatus.LEASED.value)
            self.assertEqual(rec.payload.get("lease_owner"), "slow")
        finally:
            store.close()

        p.join(timeout=5)
        self.assertEqual(p.exitcode, 0)

        store = Store(db)
        try:
            rec = store.get(tid)
            self.assertEqual(rec.status, TaskStatus.DONE.value)
            self.assertEqual(rec.payload.get("completed_by"), "slow")
            self.assertIsNone(rec.payload.get("reclaimed_from"))
            reclaim_evs = [
                e for e in store.events_for(tid) if e.event_type == "lease_reclaim"
            ]
            self.assertEqual(reclaim_evs, [])
        finally:
            store.close()

    def test_continuous_insert_while_workers_drain(self) -> None:
        """Producer inserts while workers drain; not a pre-seeded one-shot queue."""
        db = self.dir / "churn.sqlite"
        n_tasks = 24
        n_workers = 3
        ctx = multiprocessing.get_context("spawn")
        stop_path = self.dir / "churn.stop"
        jobs = [
            (str(db), f"worker-{i}", 0.002, str(stop_path)) for i in range(n_workers)
        ]
        with ctx.Pool(processes=n_workers) as pool:
            async_result = pool.map_async(mp_drain_until_idle, jobs)
            expected: list[str] = []
            for i in range(n_tasks):
                expected.extend(_seed_pending_tasks(db, 1, start=i))
                time.sleep(0.003)
            stop_path.write_text("stop\n", encoding="utf-8")
            results = async_result.get(timeout=30)

        leased_all = [tid for _w, ids in results for tid in ids]
        self.assertEqual(sorted(leased_all), sorted(expected), Counter(leased_all))
        self.assertEqual(len(leased_all), len(set(leased_all)), Counter(leased_all))

        store = Store(db)
        try:
            tasks = {t.id: t for t in store.objects_of(ObjectKind.TASK)}
            for tid in expected:
                self.assertEqual(tasks[tid].status, TaskStatus.DONE.value)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
