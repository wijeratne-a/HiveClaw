#!/usr/bin/env python3
"""Rewind e2e: typed ingest, causal invalidation, policy block. CPU only, no LLM.

Asserts real object state (status, blocking reasons, task creation, computed
percentage). Written before the engine/policy/fixture implementations.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hiveclaw_causal.fixture import build_rewind_fixture  # noqa: E402
from hiveclaw_causal.rewind import RewindRuntime  # noqa: E402
from hiveclaw_causal.types import (  # noqa: E402
    ActionStatus,
    ArtifactKind,
    EdgeMode,
    InvalidationRule,
    ObjectKind,
    ObjectStatus,
    Record,
    TrustClass,
)

CACHE_GAP = "still missing: provider-status cross-check"
GOAL = "find the cause and safely fix it."


def _payload_text(rec: Record) -> str:
    return " ".join(str(v) for v in rec.payload.values()).lower()


class TestRewindE2E(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _assert_guarantee_a(self, rec: Record) -> None:
        self.assertTrue(rec.provenance.producer, rec.id)
        self.assertTrue(rec.provenance.sources, rec.id)
        for src in rec.provenance.sources:
            self.assertTrue(src.uri, rec.id)
            self.assertTrue(src.content_hash or src.version, rec.id)
        self.assertTrue(rec.provenance.timestamp, rec.id)
        self.assertIsInstance(rec.provenance.trust, TrustClass)

    def _assert_guarantee_b_claim(self, rec: Record) -> None:
        self.assertTrue(rec.evidence_ids, rec.id)
        self.assertTrue(rec.source_snapshot, rec.id)
        self.assertTrue(rec.invalidation_conditions, rec.id)

    def _find(self, rows: list[Record], needle: str) -> Record:
        hits = [
            r
            for r in rows
            if needle.lower() in r.id.lower() or needle.lower() in _payload_text(r)
        ]
        self.assertEqual(len(hits), 1, f"need one match for {needle!r}, got { [r.id for r in rows] }")
        return hits[0]

    def test_rewind_scenario_state_transitions(self) -> None:
        rt = RewindRuntime.create(self.dir / "rewind.sqlite")
        fixture = build_rewind_fixture(seed=42)
        self.assertIn("cause", fixture.goal.lower())
        self.assertEqual(fixture.goal, GOAL)

        # --- Step 1: ingest typed records; cache-regression claim; locked rollback ---
        rt.ingest_and_propose(fixture)

        artifacts = rt.objects_of(ObjectKind.ARTIFACT)
        observations = rt.objects_of(ObjectKind.OBSERVATION)
        claims = rt.objects_of(ObjectKind.CLAIM)
        actions = rt.objects_of(ObjectKind.ACTION)
        tasks = rt.objects_of(ObjectKind.TASK)
        self.assertGreaterEqual(len(artifacts), 3, "repo, incident log, deploy (and goal)")
        self.assertGreaterEqual(len(tasks), 2, "gap cross-check + unrelated task")
        cache_claim = self._find(claims, "cache")
        self.assertEqual(cache_claim.status, ObjectStatus.ACTIVE.value)
        conf = float(cache_claim.payload["confidence"])
        self.assertGreaterEqual(conf, 0.4)
        self.assertLessEqual(conf, 0.75)
        self.assertIn(CACHE_GAP, cache_claim.payload.get("gaps", []))
        self._assert_guarantee_a(cache_claim)
        self._assert_guarantee_b_claim(cache_claim)
        self.assertEqual(cache_claim.provenance.trust, TrustClass.INFERRED)

        for rec in observations + actions:
            self._assert_guarantee_a(rec)

        rollback = self._find(actions, "rollback")
        self.assertIn(
            rollback.status,
            (ActionStatus.PROPOSED.value, ActionStatus.BLOCKED.value),
        )
        self.assertNotEqual(rollback.status, ActionStatus.EXECUTED.value)
        self.assertFalse(bool(rollback.payload.get("executed")))
        phase1_auth = rt.policy_authorize(rollback.id)
        self.assertFalse(phase1_auth.allowed)
        self.assertTrue(phase1_auth.reason)

        unrelated = self._find(observations, "unrelated")
        self.assertEqual(unrelated.status, ObjectStatus.ACTIVE.value)
        unrelated_updated = unrelated.updated_at
        unrelated_status = unrelated.status

        # --- Step 2: same ingest path, provider outage overlapping the window ---
        provider_id = rt.ingest_artifact(
            kind=ArtifactKind.PROVIDER_INCIDENT,
            producer="ingestor",
            body=fixture.provider_report,
            trust=TrustClass.TRUSTED,
            source_uri=str(fixture.provider_report["uri"]),
        )
        self.assertTrue(provider_id)
        provider_art = rt.get(provider_id)
        self.assertEqual(provider_art.kind, ObjectKind.ARTIFACT)
        self._assert_guarantee_a(provider_art)

        # --- Step 3: runtime (not LLM) challenges claim / blocks rollback with edge+rule ---
        cache_claim = rt.get(cache_claim.id)
        self.assertIn(
            cache_claim.status,
            (ObjectStatus.CHALLENGED.value, ObjectStatus.STALE.value),
        )
        evs = rt.events_for(cache_claim.id)
        status_evs = [
            e
            for e in evs
            if e.old_status == ObjectStatus.ACTIVE.value
            and e.new_status == cache_claim.status
        ]
        self.assertTrue(status_evs, [e for e in evs])
        fired = status_evs[-1]
        self.assertTrue(fired.edge_id)
        self.assertTrue(fired.rule)
        self.assertIn(fired.rule, {r.value for r in InvalidationRule})
        self.assertTrue(fired.reason)

        rollback = rt.get(rollback.id)
        self.assertEqual(rollback.status, ActionStatus.BLOCKED.value)
        rb_evs = [
            e
            for e in rt.events_for(rollback.id)
            if e.new_status == ActionStatus.BLOCKED.value
        ]
        self.assertTrue(rb_evs)
        self.assertTrue(rb_evs[-1].edge_id)
        self.assertEqual(rb_evs[-1].rule, InvalidationRule.BLOCK_ACTION.value)

        # --- Step 4: only affected work scheduled; unrelated subtree untouched ---
        report = rt.last_revalidation()
        self.assertTrue(report.scheduled_task_ids)
        self.assertTrue(report.skipped)
        skipped_ids = {tid for tid, _why in report.skipped}
        self.assertTrue(all(why for _tid, why in report.skipped))
        self.assertTrue(skipped_ids.isdisjoint(set(report.scheduled_task_ids)))

        unrelated2 = rt.get(unrelated.id)
        self.assertEqual(unrelated2.status, unrelated_status)
        self.assertEqual(unrelated2.updated_at, unrelated_updated)

        # --- Step 5: computed outage %, residual cache claim, rollback still denied ---
        from hiveclaw_causal.stats import outage_explains_pct

        independent_pct = outage_explains_pct(fixture.failures, fixture.outage_window)
        runtime_pct = rt.computed_outage_support_pct()
        self.assertAlmostEqual(independent_pct, runtime_pct, places=6)
        self.assertGreaterEqual(runtime_pct, 70.0)
        self.assertLess(runtime_pct, 100.0)

        claims_after = rt.objects_of(ObjectKind.CLAIM)
        outage_claim = self._find(claims_after, "outage")
        self.assertIn(
            outage_claim.status,
            (
                ObjectStatus.ACTIVE.value,
                ObjectStatus.CORROBORATED.value,
                ObjectStatus.VERIFIED.value,
            ),
        )
        self.assertAlmostEqual(
            float(outage_claim.payload["support_pct"]), runtime_pct, places=6
        )
        self.assertNotIsInstance(outage_claim.payload["support_pct"], str)
        self._assert_guarantee_a(outage_claim)
        self._assert_guarantee_b_claim(outage_claim)
        self.assertEqual(outage_claim.provenance.trust, TrustClass.DETERMINISTIC)

        residual = self._find(claims_after, "residual")
        self.assertIn(
            residual.status,
            (ObjectStatus.PROPOSED.value, ObjectStatus.ACTIVE.value),
        )
        self._assert_guarantee_b_claim(residual)

        final_auth = rt.policy_authorize(rollback.id)
        self.assertFalse(final_auth.allowed)
        self.assertTrue(final_auth.reason)
        retry = rt.policy_authorize(rollback.id)
        self.assertFalse(retry.allowed)
        self.assertEqual(retry.reason, final_auth.reason)

        followups = [
            t
            for t in rt.objects_of(ObjectKind.TASK)
            if "follow" in t.id.lower()
            or "investig" in _payload_text(t)
            or t.payload.get("kind") == "followup_investigation"
        ]
        self.assertTrue(followups, [t.id for t in rt.objects_of(ObjectKind.TASK)])
        self.assertIn(
            followups[0].status,
            ("pending", "proposed", ObjectStatus.PROPOSED.value, ObjectStatus.ACTIVE.value),
        )

        why = rt.why(rollback.id)
        self.assertEqual(why["status"], ActionStatus.BLOCKED.value)
        self.assertTrue(why["events"])

        _ = EdgeMode  # imported for contract completeness; edges live on events

    def test_rewind_deterministic_twice(self) -> None:
        """Same seed must yield the same computed percentage and terminal statuses."""
        from hiveclaw_causal.rewind import run_rewind

        a = run_rewind(self.dir / "a.sqlite", seed=42)
        b = run_rewind(self.dir / "b.sqlite", seed=42)
        self.assertAlmostEqual(
            a.computed_outage_support_pct(), b.computed_outage_support_pct(), places=6
        )
        a_claims = {c.id: c.status for c in a.objects_of(ObjectKind.CLAIM)}
        b_claims = {c.id: c.status for c in b.objects_of(ObjectKind.CLAIM)}
        self.assertTrue(a_claims, "scenario must produce claims; empty state is not success")
        self.assertEqual(a_claims, b_claims)
        self.assertGreaterEqual(a.computed_outage_support_pct(), 70.0)
        a_actions = {x.id: x.status for x in a.objects_of(ObjectKind.ACTION)}
        b_actions = {x.id: x.status for x in b.objects_of(ObjectKind.ACTION)}
        self.assertEqual(a_actions, b_actions)


if __name__ == "__main__":
    unittest.main()
