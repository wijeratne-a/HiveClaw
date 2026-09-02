#!/usr/bin/env python3
"""Session 9: verify-store, store-status, backup/restore, migrate, contention.

Does not loosen double-lease or Rewind conclusion assertions.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hiveclaw_causal.backup import backup_sqlite, restore_sqlite, verify_after_restore  # noqa: E402
from hiveclaw_causal.cli import main as cli_main  # noqa: E402
from hiveclaw_causal.fixture import build_rewind_fixture  # noqa: E402
from hiveclaw_causal.lease import mp_drain  # noqa: E402
from hiveclaw_causal.migrate import MigrationError, migrate_to_latest  # noqa: E402
from hiveclaw_causal.ops_status import store_status  # noqa: E402
from hiveclaw_causal.rewind import RewindRuntime  # noqa: E402
from hiveclaw_causal.schema import SCHEMA_VERSION, current_version  # noqa: E402
from hiveclaw_causal.store import Store  # noqa: E402
from hiveclaw_causal.types import (  # noqa: E402
    ArtifactKind,
    ObjectKind,
    ObjectStatus,
    Provenance,
    Record,
    SourceRef,
    TaskStatus,
    TrustClass,
)
from hiveclaw_causal.util import content_hash  # noqa: E402
from hiveclaw_causal.verify import verify_store  # noqa: E402

TS = "2026-08-31T00:00:00Z"


def _cli(argv: list[str]) -> int:
    buf = StringIO()
    err = StringIO()
    with redirect_stdout(buf), redirect_stderr(err):
        return cli_main(argv)


def _seed_pending_tasks(db_path: Path, n: int, start: int = 0) -> list[str]:
    store = Store(db_path)
    ids: list[str] = []
    for i in range(start, start + n):
        tid = f"task-ops-{i:04d}"
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
                timestamp=TS,
                trust=TrustClass.DETERMINISTIC,
            ),
            payload=payload,
            evidence_ids=(),
            source_snapshot=content_hash(payload),
            created_at=TS,
            updated_at=TS,
        )
        store.put_object(rec, ts=TS, event_type="seed_task", reason="ops seed")
        ids.append(tid)
    store.close()
    return ids


def _run_workers(db_path: Path, n_workers: int, pause_s: float) -> list[tuple[str, list[str]]]:
    ctx = multiprocessing.get_context("spawn")
    jobs = [(str(db_path), f"ops-w-{i}", pause_s) for i in range(n_workers)]
    with ctx.Pool(processes=n_workers) as pool:
        return list(pool.map(mp_drain, jobs))


class TestVerifyAndStatus(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _ingest(self, path: Path) -> RewindRuntime:
        rt = RewindRuntime.create(path)
        rt.ingest_and_propose(build_rewind_fixture(seed=42))
        return rt

    def test_verify_store_ok_on_rewind_fixture(self) -> None:
        db = self.dir / "ok.sqlite"
        rt = self._ingest(db)
        try:
            report = verify_store(rt.store)
        finally:
            rt.store.close()
        self.assertTrue(report["ok"], report)
        names = {c["name"]: c for c in report["checks"]}
        self.assertTrue(names["seq_monotonic"]["ok"])
        self.assertTrue(names["projection_replay"]["ok"])
        self.assertTrue(names["append_only_triggers"]["ok"])
        self.assertTrue(names["lease_ttl_ceiling_trigger"]["ok"])
        self.assertFalse(names["seq_gap_free"]["invariant"])
        self.assertFalse(names["event_checksums"]["recorded"])
        self.assertEqual(_cli(["verify-store", "--db", str(db), "--json"]), 0)

    def test_verify_detects_projection_mismatch(self) -> None:
        db = self.dir / "bad.sqlite"
        rt = self._ingest(db)
        oid = rt.objects_of(ObjectKind.CLAIM)[0].id
        rt.store._conn.execute(
            "UPDATE objects SET status = 'stale' WHERE id = ?", (oid,)
        )
        rt.store._conn.commit()
        report = verify_store(rt.store)
        rt.store.close()
        self.assertFalse(report["ok"])
        names = {c["name"]: c for c in report["checks"]}
        self.assertFalse(names["projection_replay"]["ok"])
        self.assertEqual(_cli(["verify-store", "--db", str(db), "--json"]), 1)

    def test_verify_detects_missing_reverse_dep(self) -> None:
        db = self.dir / "idx.sqlite"
        rt = self._ingest(db)
        edge = rt.store.all_edges()[0]
        rt.store._conn.execute("DELETE FROM reverse_deps WHERE edge_id = ?", (edge.id,))
        rt.store._conn.commit()
        report = verify_store(rt.store)
        rt.store.close()
        self.assertFalse(report["ok"])
        names = {c["name"]: c for c in report["checks"]}
        self.assertFalse(names["reverse_deps_match_edges"]["ok"])

    def test_store_status_clamped_lease_and_counts(self) -> None:
        db = self.dir / "status.sqlite"
        _seed_pending_tasks(db, 2)
        store = Store(db, max_lease_ttl_s=1.0)
        rec = store.try_lease_one_task("clamp-w", TS, lease_ttl_s=3600.0)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertTrue(rec.payload.get("ttl_clamped"))
        store.close()
        ro = Store(db, read_only=True)
        try:
            report = store_status(ro)
        finally:
            ro.close()
        self.assertEqual(report["backend"], "sqlite")
        self.assertGreaterEqual(report["task_counts"].get(TaskStatus.LEASED.value, 0), 1)
        self.assertGreaterEqual(report["clamped_lease_events"], 1)
        self.assertFalse(report["failed_lease_attempts"]["recorded"])
        self.assertFalse(report["reclaim_latency_s"]["recorded"])
        self.assertFalse(report["owner_process_health"]["recorded"])
        self.assertTrue(report["ttl_policy"]["within_hard_maximum"])
        self.assertEqual(report["ttl_policy"]["configured_max_ttl_s"], 1.0)
        owners = {row["owner"] for row in report["active_leases"]}
        self.assertIn("clamp-w", owners)
        self.assertEqual(_cli(["store-status", "--db", str(db), "--json"]), 0)

    def test_read_only_open_does_not_write(self) -> None:
        db = self.dir / "ro.sqlite"
        store = Store(db)
        n_events = len(store.all_events())
        store.close()
        ro = Store(db, read_only=True)
        try:
            verify_store(ro)
            store_status(ro)
            with self.assertRaises(sqlite3.OperationalError):
                ro._conn.execute("CREATE TABLE hiveclaw_ro_probe (id INTEGER)")
        finally:
            ro.close()
        again = Store(db, read_only=True)
        try:
            self.assertEqual(len(again.all_events()), n_events)
        finally:
            again.close()


class TestBackupRestoreMigrate(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_sqlite_backup_restore_drill(self) -> None:
        src = self.dir / "live.sqlite"
        bak = self.dir / "backup.sqlite"
        dest = self.dir / "restored.sqlite"
        rt = RewindRuntime.create(src)
        fixture = build_rewind_fixture(seed=42)
        rt.ingest_and_propose(fixture)
        before = verify_store(rt.store)
        self.assertTrue(before["ok"], before)
        n_events = before["events"]
        n_objects = before["objects"]
        # Backup while the writer connection is still open.
        report = backup_sqlite(src, bak)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["integrity_dest"], "ok")
        rt.store.close()

        restored = restore_sqlite(bak, dest)
        self.assertTrue(restored["ok"], restored)
        after = verify_after_restore(str(dest))
        self.assertTrue(after["ok"], after)
        self.assertEqual(after["events"], n_events)
        self.assertEqual(after["objects"], n_objects)

        store = Store(dest)
        try:
            leased = store.try_lease_one_task("post-restore", TS, lease_ttl_s=5.0)
            self.assertIsNotNone(leased)
            assert leased is not None
            store.complete_task(leased.id, "post-restore", TS)
            store.complete_task(leased.id, "post-restore", TS)
            completes = [
                e for e in store.events_for(leased.id) if e.event_type == "complete_task"
            ]
            self.assertEqual(len(completes), 1)
        finally:
            store.close()

        rt2 = RewindRuntime.create(dest)
        rt2.fixture = fixture
        try:
            rt2.ingest_artifact(
                kind=ArtifactKind.PROVIDER_INCIDENT,
                producer="ingestor",
                body=fixture.provider_report,
                trust=TrustClass.TRUSTED,
                source_uri=str(fixture.provider_report["uri"]),
                timestamp=str(fixture.provider_report["window_start"]),
            )
            cache = rt2.get("claim-cache-regression")
            self.assertIn(
                cache.status,
                (ObjectStatus.CHALLENGED.value, ObjectStatus.STALE.value),
            )
            self.assertTrue(verify_store(rt2.store)["ok"])
        finally:
            rt2.store.close()

        self.assertEqual(
            _cli(
                [
                    "restore",
                    "--backup",
                    str(bak),
                    "--db",
                    str(self.dir / "cli-restored.sqlite"),
                    "--confirm",
                ]
            ),
            0,
        )
        self.assertEqual(
            _cli(
                ["restore", "--backup", str(bak), "--db", str(self.dir / "nope.sqlite")]
            ),
            2,
        )

    def test_migrate_requires_confirm_and_upgrades_v1(self) -> None:
        db = self.dir / "v1.sqlite"
        store = Store(db)
        store._conn.execute("DROP TABLE IF EXISTS schema_migrations")
        store._conn.execute("DROP TABLE IF EXISTS lease_config")
        store._conn.execute("DROP TRIGGER IF EXISTS lease_until_absolute_ceiling_update")
        store._conn.execute("DROP TRIGGER IF EXISTS lease_until_absolute_ceiling_insert")
        store._conn.commit()
        self.assertEqual(current_version(store), 1)
        with self.assertRaises(MigrationError):
            migrate_to_latest(store, confirm=False)
        report = migrate_to_latest(store, confirm=True)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["from_version"], 1)
        self.assertEqual(report["to_version"], SCHEMA_VERSION)
        self.assertIn("1->2", " ".join(report["steps"]))
        self.assertTrue(report["downgrade"].startswith("not supported"))
        self.assertTrue(verify_store(store)["ok"])
        store.close()
        self.assertEqual(_cli(["migrate", "--db", str(db), "--to-latest"]), 2)
        self.assertEqual(
            _cli(["migrate", "--db", str(db), "--to-latest", "--confirm"]),
            0,
        )

    def test_migrate_already_latest_is_idempotent(self) -> None:
        db = self.dir / "v2.sqlite"
        store = Store(db)
        try:
            report = migrate_to_latest(store, confirm=True)
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["from_version"], SCHEMA_VERSION)
            self.assertEqual(report["to_version"], SCHEMA_VERSION)
        finally:
            store.close()


class TestContentionAndRetry(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_oversubscribed_workers_zero_double_lease(self) -> None:
        """More workers than tasks. Duplicate execution must stay zero."""
        double = 0
        dropped = 0
        for trial in range(3):
            db = self.dir / f"cont-{trial}.sqlite"
            expected = _seed_pending_tasks(db, 5)
            results = _run_workers(db, n_workers=8, pause_s=0.003)
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

    def test_duplicate_complete_retry_does_not_append_second_event(self) -> None:
        db = self.dir / "retry.sqlite"
        expected = _seed_pending_tasks(db, 1)
        store = Store(db)
        try:
            rec = store.try_lease_one_task("w", TS)
            self.assertIsNotNone(rec)
            assert rec is not None
            self.assertIsNone(store.try_lease_one_task("w", TS))
            store.complete_task(rec.id, "w", TS)
            store.complete_task(rec.id, "w", TS)
            completes = [
                e for e in store.events_for(rec.id) if e.event_type == "complete_task"
            ]
            self.assertEqual(len(completes), 1)
            self.assertEqual(store.get(expected[0]).status, TaskStatus.DONE.value)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
