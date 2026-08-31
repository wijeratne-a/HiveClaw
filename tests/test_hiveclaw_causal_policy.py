#!/usr/bin/env python3
"""Policy gate: authorize rollback only while justifying claims stay fresh."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hiveclaw_causal.engine import InvalidationEngine  # noqa: E402
from hiveclaw_causal.policy import authorize  # noqa: E402
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
        trust=TrustClass.INFERRED,
    )


def _rec(oid: str, kind: ObjectKind, status: str, **payload: object) -> Record:
    return Record(
        id=oid,
        kind=kind,
        status=status,
        provenance=_prov(oid),
        payload=dict(payload) if payload else {"name": oid},
        evidence_ids=("c1",) if kind == ObjectKind.ACTION else (),
        source_snapshot=content_hash(oid),
        created_at=TS,
        updated_at=TS,
    )


class TestPolicyGate(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._td.name) / "p.sqlite")
        self.engine = InvalidationEngine(self.store, clock=lambda: "2026-08-30T14:41:00Z")
        claim = _rec("c1", ObjectKind.CLAIM, ObjectStatus.ACTIVE.value)
        action = _rec(
            "a1",
            ObjectKind.ACTION,
            ActionStatus.PROPOSED.value,
            approved=True,
            executed=False,
            irreversible=True,
            action_kind="rollback",
        )
        self.store.put_object(claim, ts=TS, event_type="seed", reason="seed")
        self.store.put_object(action, ts=TS, event_type="seed", reason="seed")
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

    def tearDown(self) -> None:
        self.store.close()
        self._td.cleanup()

    def test_valid_rollback_allowed(self) -> None:
        d = authorize(self.store, "a1")
        self.assertTrue(d.allowed)
        self.assertEqual(d.reason, "prerequisites-hold")
        self.assertEqual(d.failed_preconditions, ())

    def test_blocked_after_justification_invalidated(self) -> None:
        self.assertTrue(authorize(self.store, "a1").allowed)
        self.store.put_object(
            _rec("src", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value),
            ts=TS,
            event_type="seed",
            reason="seed",
        )
        self.engine.fire(
            src_id="src",
            dst_id="c1",
            mode=EdgeMode.CONTRADICTS,
            rule=InvalidationRule.HARD_CHALLENGE,
            reason="provider evidence",
        )
        self.assertEqual(self.store.get("c1").status, ObjectStatus.CHALLENGED.value)
        self.assertEqual(self.store.get("a1").status, ActionStatus.BLOCKED.value)
        d = authorize(self.store, "a1")
        self.assertFalse(d.allowed)
        self.assertIn("edge=", d.reason)
        self.assertIn(InvalidationRule.BLOCK_ACTION.value, d.reason)
        self.assertIn("action-blocked", d.failed_preconditions)

    def test_reattempt_blocked_action_fails_deterministically(self) -> None:
        self.store.put_object(
            _rec("src", ObjectKind.OBSERVATION, ObjectStatus.ACTIVE.value),
            ts=TS,
            event_type="seed",
            reason="seed",
        )
        self.engine.fire(
            src_id="src",
            dst_id="c1",
            mode=EdgeMode.CONTRADICTS,
            rule=InvalidationRule.HARD_CHALLENGE,
            reason="provider evidence",
        )
        first = authorize(self.store, "a1")
        second = authorize(self.store, "a1")
        self.assertFalse(first.allowed)
        self.assertEqual(first.reason, second.reason)
        self.assertEqual(first.failed_preconditions, second.failed_preconditions)


if __name__ == "__main__":
    unittest.main()
