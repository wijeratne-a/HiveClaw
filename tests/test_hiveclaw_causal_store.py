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

from hiveclaw_causal.store import Store  # noqa: E402
from hiveclaw_causal.types import (  # noqa: E402
    ObjectKind,
    ObjectStatus,
    Provenance,
    Record,
    SourceRef,
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


if __name__ == "__main__":
    unittest.main()
