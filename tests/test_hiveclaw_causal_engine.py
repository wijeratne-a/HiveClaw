#!/usr/bin/env python3
"""Unit tests for the causal invalidation engine (no Rewind fixture, no LLM)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hiveclaw_causal.engine import InvalidationEngine  # noqa: E402
from hiveclaw_causal.store import Store  # noqa: E402
from hiveclaw_causal.types import (  # noqa: E402
    ActionStatus,
    Edge,
    EdgeMode,
    InvalidationRule,
    ObjectKind,
    ObjectStatus,
    Provenance,
    Record,
    SourceRef,
    TrustClass,
)
from hiveclaw_causal.util import content_hash  # noqa: E402

TS = "2026-08-30T14:00:00Z"


def _prov(name: str) -> Provenance:
    body = {"n": name}
    return Provenance(
        producer="test",
        sources=(SourceRef(uri=f"test://{name}", version="1", content_hash=content_hash(body)),),
        timestamp=TS,
        trust=TrustClass.DETERMINISTIC,
    )


def _rec(oid: str, kind: ObjectKind, status: str) -> Record:
    return Record(
        id=oid,
        kind=kind,
        status=status,
        provenance=_prov(oid),
        payload={"name": oid},
        evidence_ids=(),
        source_snapshot=content_hash(oid),
        created_at=TS,
        updated_at=TS,
    )


class TestInvalidationEngine(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._td.name) / "e.sqlite")
        self.clock_n = 0

        def clock() -> str:
            self.clock_n += 1
            return f"2026-08-30T14:40:{self.clock_n:02d}Z"

        self.engine = InvalidationEngine(self.store, clock=clock)

    def tearDown(self) -> None:
        self.store.close()
        self._td.cleanup()

    def _put(self, rec: Record) -> None:
        self.store.put_object(rec, ts=TS, event_type="seed", reason="seed")

    def test_direct_hard_dependency(self) -> None:
        ev = _rec("e1", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value)
        claim = _rec("c1", ObjectKind.CLAIM, ObjectStatus.ACTIVE.value)
        self._put(ev)
        self._put(claim)
        self.store.add_edge(
            Edge(
                id="edge-depends-c1-e1",
                src="c1",
                dst="e1",
                mode=EdgeMode.DEPENDS_ON,
                rule=InvalidationRule.HARD_CHALLENGE,
                declared_effect="challenge claim",
            )
        )
        src = _rec("src-new", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value)
        self._put(src)
        self.engine.fire(
            src_id="src-new",
            dst_id="e1",
            mode=EdgeMode.SUPERSEDES,
            rule=InvalidationRule.HARD_STALE,
            reason="evidence superseded",
        )
        self.assertEqual(self.store.get("e1").status, ObjectStatus.STALE.value)
        self.assertEqual(self.store.get("c1").status, ObjectStatus.CHALLENGED.value)
        evs = [
            e
            for e in self.store.events_for("c1")
            if e.event_type == "status_transition"
        ]
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].old_status, ObjectStatus.ACTIVE.value)
        self.assertEqual(evs[0].rule, InvalidationRule.HARD_CHALLENGE.value)
        self.assertTrue(evs[0].edge_id)

    def test_multi_hop_propagation(self) -> None:
        self._put(_rec("e1", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value))
        self._put(_rec("c1", ObjectKind.CLAIM, ObjectStatus.ACTIVE.value))
        self._put(_rec("a1", ObjectKind.ACTION, ActionStatus.PROPOSED.value))
        self.store.add_edge(
            Edge(
                id="edge-depends-c1-e1",
                src="c1",
                dst="e1",
                mode=EdgeMode.DEPENDS_ON,
                rule=InvalidationRule.HARD_CHALLENGE,
                declared_effect="challenge",
            )
        )
        self.store.add_edge(
            Edge(
                id="edge-justifies-c1-a1",
                src="c1",
                dst="a1",
                mode=EdgeMode.JUSTIFIES,
                rule=InvalidationRule.BLOCK_ACTION,
                declared_effect="block",
            )
        )
        self._put(_rec("src-new", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value))
        self.engine.fire(
            src_id="src-new",
            dst_id="e1",
            mode=EdgeMode.SUPERSEDES,
            rule=InvalidationRule.HARD_STALE,
            reason="multi-hop",
        )
        self.assertEqual(self.store.get("e1").status, ObjectStatus.STALE.value)
        self.assertEqual(self.store.get("c1").status, ObjectStatus.CHALLENGED.value)
        self.assertEqual(self.store.get("a1").status, ActionStatus.BLOCKED.value)

    def test_duplicate_event_idempotent(self) -> None:
        self._put(_rec("c1", ObjectKind.CLAIM, ObjectStatus.ACTIVE.value))
        self._put(_rec("src", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value))
        self.engine.fire(
            src_id="src",
            dst_id="c1",
            mode=EdgeMode.CONTRADICTS,
            rule=InvalidationRule.HARD_CHALLENGE,
            reason="first",
        )
        before = len(self.store.events_for("c1"))
        self.engine.fire(
            src_id="src",
            dst_id="c1",
            mode=EdgeMode.CONTRADICTS,
            rule=InvalidationRule.HARD_CHALLENGE,
            reason="second",
        )
        after = len(self.store.events_for("c1"))
        self.assertEqual(before, after)
        trans = [
            e
            for e in self.store.events_for("c1")
            if e.event_type == "status_transition"
        ]
        self.assertEqual(len(trans), 1)

    def test_cycle_safety(self) -> None:
        self._put(_rec("a", ObjectKind.CLAIM, ObjectStatus.ACTIVE.value))
        self._put(_rec("b", ObjectKind.CLAIM, ObjectStatus.ACTIVE.value))
        self.store.add_edge(
            Edge(
                id="edge-depends-a-b",
                src="a",
                dst="b",
                mode=EdgeMode.DEPENDS_ON,
                rule=InvalidationRule.HARD_STALE,
                declared_effect="stale",
            )
        )
        self.store.add_edge(
            Edge(
                id="edge-depends-b-a",
                src="b",
                dst="a",
                mode=EdgeMode.DEPENDS_ON,
                rule=InvalidationRule.HARD_STALE,
                declared_effect="stale",
            )
        )
        self._put(_rec("src", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value))
        self.engine.fire(
            src_id="src",
            dst_id="a",
            mode=EdgeMode.CONTRADICTS,
            rule=InvalidationRule.HARD_CHALLENGE,
            reason="cycle",
        )
        self.assertEqual(self.store.get("a").status, ObjectStatus.CHALLENGED.value)
        self.assertEqual(self.store.get("b").status, ObjectStatus.STALE.value)

    def test_unrelated_subtree_isolated(self) -> None:
        self._put(_rec("e1", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value))
        self._put(_rec("c1", ObjectKind.CLAIM, ObjectStatus.ACTIVE.value))
        unrelated = _rec("u1", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value)
        self._put(unrelated)
        self.store.add_edge(
            Edge(
                id="edge-depends-c1-e1",
                src="c1",
                dst="e1",
                mode=EdgeMode.DEPENDS_ON,
                rule=InvalidationRule.HARD_CHALLENGE,
                declared_effect="challenge",
            )
        )
        self._put(_rec("src", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value))
        self.engine.fire(
            src_id="src",
            dst_id="e1",
            mode=EdgeMode.SUPERSEDES,
            rule=InvalidationRule.HARD_STALE,
            reason="isolate",
        )
        u = self.store.get("u1")
        self.assertEqual(u.status, ObjectStatus.ACTIVE.value)
        self.assertEqual(u.updated_at, TS)
        trans_u = [
            e for e in self.store.events_for("u1") if e.event_type == "status_transition"
        ]
        self.assertEqual(trans_u, [])


if __name__ == "__main__":
    unittest.main()
