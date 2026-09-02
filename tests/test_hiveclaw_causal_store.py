#!/usr/bin/env python3
"""Store invariants: events table is append-only at the SQLite layer."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hiveclaw_causal.store import Store, _json  # noqa: E402
from hiveclaw_causal.types import (  # noqa: E402
    ObjectKind,
    ObjectStatus,
    Provenance,
    Record,
    SourceRef,
    TaskStatus,
    TrustClass,
)
from hiveclaw_causal.util import content_hash  # noqa: E402

TS = "2026-08-31T00:00:00Z"


def _seed(store: Store) -> None:
    rec = Record(
        id="obj-seed",
        kind=ObjectKind.OBSERVATION,
        status=ObjectStatus.ACTIVE.value,
        provenance=Provenance(
            producer="test",
            sources=(
                SourceRef(
                    uri="test://seed",
                    version="1",
                    content_hash=content_hash({"k": "v"}),
                ),
            ),
            timestamp=TS,
            trust=TrustClass.DETERMINISTIC,
        ),
        payload={"k": "v"},
        evidence_ids=(),
        source_snapshot=content_hash("seed"),
        created_at=TS,
        updated_at=TS,
    )
    store.put_object(rec, ts=TS, event_type="seed", reason="seed")


class TestEventsAppendOnly(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db_path = Path(self._td.name) / "append.sqlite"
        self.store = Store(self.db_path)
        _seed(self.store)
        n = self.store._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        self.assertGreaterEqual(int(n), 1)

    def tearDown(self) -> None:
        self.store.close()
        self._td.cleanup()

    def test_raw_update_events_is_aborted(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError) as ctx:
            self.store._conn.execute("UPDATE events SET reason = 'mutated' WHERE seq = 1")
            self.store._conn.commit()
        self.assertIn("append-only", str(ctx.exception).lower())
        reason = self.store._conn.execute(
            "SELECT reason FROM events WHERE seq = 1"
        ).fetchone()[0]
        self.assertEqual(reason, "seed")

    def test_raw_delete_events_is_aborted(self) -> None:
        before = int(
            self.store._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        )
        with self.assertRaises(sqlite3.IntegrityError) as ctx:
            self.store._conn.execute("DELETE FROM events WHERE seq = 1")
            self.store._conn.commit()
        self.assertIn("append-only", str(ctx.exception).lower())
        after = int(
            self.store._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        )
        self.assertEqual(after, before)

    def test_insert_events_still_succeeds(self) -> None:
        ev = self.store.append_event(
            ts=TS,
            event_type="note",
            object_id="obj-seed",
            reason="legitimate append",
        )
        self.assertGreaterEqual(ev.seq, 2)


class TestLeaseTtlCeiling(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.db_path = Path(self._td.name) / "lease.sqlite"
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self._td.cleanup()

    def test_raw_sql_cannot_set_lease_until_beyond_absolute_ceiling(self) -> None:
        rec = Record(
            id="task-ceiling",
            kind=ObjectKind.TASK,
            status=TaskStatus.LEASED.value,
            provenance=Provenance(
                producer="test",
                sources=(
                    SourceRef(
                        uri="test://task",
                        version="1",
                        content_hash=content_hash({"k": 1}),
                    ),
                ),
                timestamp=TS,
                trust=TrustClass.DETERMINISTIC,
            ),
            payload={"lease_owner": "w", "lease_until": 1.0},
            evidence_ids=(),
            source_snapshot=content_hash("t"),
            created_at=TS,
            updated_at=TS,
        )
        self.store.put_object(rec, ts=TS, event_type="seed_task", reason="ceiling")
        far = {"lease_owner": "w", "lease_until": 1.0 + 10**12}
        with self.assertRaises(sqlite3.IntegrityError) as ctx:
            self.store._conn.execute(
                "UPDATE objects SET payload = ? WHERE id = ?",
                (_json(far), "task-ceiling"),
            )
            self.store._conn.commit()
        self.assertIn("ceiling", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
