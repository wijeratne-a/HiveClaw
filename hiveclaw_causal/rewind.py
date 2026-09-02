"""Rewind orchestrator: ingest → extract → investigate → invalidate → verify → policy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .engine import InvalidationEngine
from .fixture import RewindFixture, build_rewind_fixture
from .inspect import explain
from .policy import authorize
from .stats import outage_explains_pct
from .store import Store
from .types import (
    ActionStatus,
    ArtifactKind,
    Edge,
    EdgeMode,
    InvalidationCondition,
    InvalidationRule,
    ObjectKind,
    ObjectStatus,
    PolicyDecision,
    Provenance,
    Record,
    RevalidationReport,
    SourceRef,
    TaskStatus,
    TrustClass,
)
from .util import content_hash
from .work import WorkCounter

CACHE_GAP = "still missing: provider-status cross-check"

ART_REPO = "art-repo-checkout"
ART_LOGS = "art-incident-log"
ART_DEPLOY = "art-deploy"
ART_GOAL = "art-goal"
ART_HEALTH = "art-health-note"
ART_PROVIDER = "art-provider-incident"
OBS_TIMEOUTS = "obs-timeouts"
OBS_DEPLOY = "obs-deploy-before-incident"
OBS_UNRELATED = "obs-unrelated-health"
OBS_PROVIDER = "obs-provider-outage"
CLAIM_CACHE = "claim-cache-regression"
CLAIM_OUTAGE = "claim-provider-outage"
CLAIM_RESIDUAL = "claim-residual-cache"
ACTION_ROLLBACK = "action-rollback-release"
TASK_GAP = "task-provider-crosscheck"
TASK_DOCS = "task-unrelated-docs"
TASK_VERIFY = "task-verify-outage-pct"
TASK_FOLLOWUP = "task-followup-cache"
VER_OUTAGE = "ver-outage-support"

_KIND_TO_ID = {
    ArtifactKind.REPO_SNAPSHOT: ART_REPO,
    ArtifactKind.INCIDENT_LOG: ART_LOGS,
    ArtifactKind.DEPLOY_RECORD: ART_DEPLOY,
    ArtifactKind.GOAL: ART_GOAL,
    ArtifactKind.HEALTH_NOTE: ART_HEALTH,
    ArtifactKind.PROVIDER_INCIDENT: ART_PROVIDER,
}

_KIND_URI = {
    ArtifactKind.REPO_SNAPSHOT: "fixture://repo/checkout-payment",
    ArtifactKind.INCIDENT_LOG: "fixture://logs/incident",
    ArtifactKind.DEPLOY_RECORD: "fixture://deploy/rel-2026.08.30.1",
    ArtifactKind.GOAL: "fixture://goal",
    ArtifactKind.HEALTH_NOTE: "fixture://health/nightly",
    ArtifactKind.PROVIDER_INCIDENT: "fixture://provider-incident/INC-8841",
}


class RewindRuntime:
    def __init__(self, db_path: Path, store: Store) -> None:
        self.db_path = Path(db_path)
        self.store = store
        self.fixture: RewindFixture | None = None
        self._clock_s = 0
        self._revalidation = RevalidationReport(scheduled_task_ids=(), skipped=())
        self.work = WorkCounter()
        self.engine = InvalidationEngine(store, clock=self._now, counter=self.work)

    @classmethod
    def create(cls, db_path: Path | str) -> RewindRuntime:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path, Store(path))

    def _now(self) -> str:
        self._clock_s += 1
        dt = datetime(2026, 8, 30, 14, 35, 0, tzinfo=timezone.utc) + timedelta(
            seconds=self._clock_s
        )
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def ingest_and_propose(self, fixture: RewindFixture) -> None:
        self.fixture = fixture
        self.ingest_artifact(
            kind=ArtifactKind.REPO_SNAPSHOT,
            producer="ingestor",
            body={"files": fixture.repo_files},
            trust=TrustClass.TRUSTED,
            source_uri=_KIND_URI[ArtifactKind.REPO_SNAPSHOT],
            timestamp=fixture.deploy["timestamp"],
        )
        self.ingest_artifact(
            kind=ArtifactKind.INCIDENT_LOG,
            producer="ingestor",
            body={
                "failures": [
                    {"ts": f.ts, "kind": f.kind, "request_id": f.request_id}
                    for f in fixture.failures
                ],
                "window": {
                    "start": fixture.incident_window.start,
                    "end": fixture.incident_window.end,
                },
            },
            trust=TrustClass.TRUSTED,
            source_uri=_KIND_URI[ArtifactKind.INCIDENT_LOG],
            timestamp=fixture.incident_window.start,
        )
        self.ingest_artifact(
            kind=ArtifactKind.DEPLOY_RECORD,
            producer="ingestor",
            body=fixture.deploy,
            trust=TrustClass.TRUSTED,
            source_uri=_KIND_URI[ArtifactKind.DEPLOY_RECORD],
            timestamp=str(fixture.deploy["timestamp"]),
        )
        self.ingest_artifact(
            kind=ArtifactKind.GOAL,
            producer="ingestor",
            body={"goal": fixture.goal},
            trust=TrustClass.HUMAN_AUTHORED,
            source_uri=_KIND_URI[ArtifactKind.GOAL],
            timestamp=fixture.incident_window.start,
        )
        self.ingest_artifact(
            kind=ArtifactKind.HEALTH_NOTE,
            producer="ingestor",
            body=fixture.unrelated_note,
            trust=TrustClass.TRUSTED,
            source_uri=str(fixture.unrelated_note["uri"]),
            timestamp=str(fixture.unrelated_note["timestamp"]),
        )
        self._investigate()
        self._ingest_scale_extras(fixture)

    def ingest_artifact(
        self,
        *,
        kind: ArtifactKind,
        producer: str,
        body: dict[str, Any],
        trust: TrustClass,
        source_uri: str,
        timestamp: str | None = None,
        repair: str = "targeted",
    ) -> str:
        if repair not in ("targeted", "naive", "none"):
            raise ValueError(f"repair must be targeted|naive|none, got {repair!r}")
        ts = timestamp or self._now()
        aid = _KIND_TO_ID[kind]
        rec = Record(
            id=aid,
            kind=ObjectKind.ARTIFACT,
            status=ObjectStatus.ACTIVE.value,
            provenance=_prov(producer, source_uri, body, ts, trust),
            payload={"kind": kind.value, "body": body},
            evidence_ids=(),
            source_snapshot=content_hash(body),
            created_at=ts,
            updated_at=ts,
        )
        self.store.put_object(
            rec, ts=ts, event_type="ingest", reason=f"ingest {kind.value} via {producer}"
        )
        self.work.touch(aid)
        new_obs = self._extract(rec)
        if new_obs is not None:
            if new_obs.payload.get("kind") == "provider_outage":
                if repair == "targeted":
                    self._repair_targeted(new_obs)
                elif repair == "naive":
                    self._repair_naive(new_obs)
            else:
                self.engine.apply_provider_overlap_rule(new_obs)
        return aid

    def objects_of(self, kind: ObjectKind) -> list[Record]:
        return self.store.objects_of(kind)

    def get(self, object_id: str) -> Record:
        return self.store.get(object_id)

    def events_for(self, object_id: str) -> list[Any]:
        return self.store.events_for(object_id)

    def all_events(self) -> list[Any]:
        return self.store.all_events()

    def policy_authorize(self, action_id: str) -> PolicyDecision:
        self.work.inspect(action_id)
        return authorize(self.store, action_id)

    def last_revalidation(self) -> RevalidationReport:
        return self._revalidation

    def why(self, object_id: str) -> dict[str, Any]:
        return explain(self.store, object_id)

    def computed_outage_support_pct(self) -> float:
        claim = self.store.get_or_none(CLAIM_OUTAGE)
        if claim is not None and "support_pct" in claim.payload:
            return float(claim.payload["support_pct"])
        if self.fixture is None:
            return 0.0
        return outage_explains_pct(self.fixture.failures, self.fixture.outage_window)

    def _extract(self, artifact: Record) -> Record | None:
        kind = ArtifactKind(artifact.payload["kind"])
        body = artifact.payload["body"]
        ts = artifact.created_at
        if kind == ArtifactKind.INCIDENT_LOG:
            return self._put_obs(
                OBS_TIMEOUTS,
                ts=ts,
                uri=artifact.provenance.sources[0].uri,
                body={
                    "kind": "timeout_burst",
                    "failure_count": len(body.get("failures") or []),
                    "window": body.get("window"),
                },
                artifact_id=artifact.id,
            )
        if kind == ArtifactKind.DEPLOY_RECORD:
            return self._put_obs(
                OBS_DEPLOY,
                ts=ts,
                uri=artifact.provenance.sources[0].uri,
                body={
                    "kind": "deploy_before_incident",
                    "release": body.get("release"),
                    "timestamp": body.get("timestamp"),
                    "notes": body.get("notes"),
                },
                artifact_id=artifact.id,
            )
        if kind == ArtifactKind.HEALTH_NOTE:
            return self._put_obs(
                OBS_UNRELATED,
                ts=ts,
                uri=str(body.get("uri") or artifact.provenance.sources[0].uri),
                body={
                    "kind": "unrelated_health",
                    "summary": body.get("summary"),
                },
                artifact_id=artifact.id,
            )
        if kind == ArtifactKind.PROVIDER_INCIDENT:
            return self._put_obs(
                OBS_PROVIDER,
                ts=ts,
                uri=str(body.get("uri") or artifact.provenance.sources[0].uri),
                body={
                    "kind": "provider_outage",
                    "window_start": body.get("window_start"),
                    "window_end": body.get("window_end"),
                    "incident_id": body.get("incident_id"),
                    "summary": body.get("summary"),
                },
                artifact_id=artifact.id,
            )
        return None

    def _put_obs(
        self, oid: str, *, ts: str, uri: str, body: dict[str, Any], artifact_id: str
    ) -> Record:
        rec = Record(
            id=oid,
            kind=ObjectKind.OBSERVATION,
            status=ObjectStatus.ACTIVE.value,
            provenance=_prov("extractor", uri, body, ts, TrustClass.DETERMINISTIC),
            payload=body,
            evidence_ids=(artifact_id,),
            source_snapshot=content_hash(body),
            created_at=ts,
            updated_at=ts,
        )
        self.store.put_object(
            rec, ts=ts, event_type="extract", reason=f"extract {oid} from {artifact_id}"
        )
        self.work.touch(oid)
        self.store.add_edge(
            Edge(
                id=f"edge-produced-{artifact_id}-{oid}",
                src=artifact_id,
                dst=oid,
                mode=EdgeMode.PRODUCED_FROM,
                rule=InvalidationRule.HARD_STALE,
                declared_effect="observation stale if artifact superseded",
            )
        )
        return rec

    def _investigate(self) -> None:
        assert self.fixture is not None
        ts = self.fixture.incident_window.start
        snapshot = content_hash(
            {
                "logs": self.store.get(ART_LOGS).source_snapshot,
                "deploy": self.store.get(ART_DEPLOY).source_snapshot,
            }
        )
        gaps = [CACHE_GAP]
        claim = Record(
            id=CLAIM_CACHE,
            kind=ObjectKind.CLAIM,
            status=ObjectStatus.ACTIVE.value,
            provenance=_prov(
                "investigator",
                "fixture://claim/cache-regression",
                {"hypothesis": "cache invalidation regression"},
                ts,
                TrustClass.INFERRED,
            ),
            payload={
                "hypothesis": "probable cause is a cache invalidation regression",
                "confidence": 0.55,
                "gaps": gaps,
                "incident_window": {
                    "start": self.fixture.incident_window.start,
                    "end": self.fixture.incident_window.end,
                },
            },
            evidence_ids=(OBS_TIMEOUTS, OBS_DEPLOY),
            source_snapshot=snapshot,
            invalidation_conditions=(
                InvalidationCondition(
                    description="provider-status cross-check contradicts cache-primary cause",
                    evidence_ids=(OBS_TIMEOUTS, OBS_DEPLOY),
                    rule=InvalidationRule.HARD_CHALLENGE,
                ),
            ),
            created_at=ts,
            updated_at=ts,
        )
        self.store.put_object(
            claim,
            ts=ts,
            event_type="propose_claim",
            reason="investigator: deploy just before timeouts; cache path changed",
        )
        for ev in (OBS_TIMEOUTS, OBS_DEPLOY):
            self.store.add_edge(
                Edge(
                    id=f"edge-depends-{CLAIM_CACHE}-{ev}",
                    src=CLAIM_CACHE,
                    dst=ev,
                    mode=EdgeMode.DEPENDS_ON,
                    rule=InvalidationRule.HARD_CHALLENGE,
                    declared_effect="challenge claim if evidence is contradicted",
                )
            )
        rollback = Record(
            id=ACTION_ROLLBACK,
            kind=ObjectKind.ACTION,
            status=ActionStatus.PROPOSED.value,
            provenance=_prov(
                "investigator",
                "fixture://action/rollback",
                {"release": self.fixture.deploy["release"]},
                ts,
                TrustClass.INFERRED,
            ),
            payload={
                "kind": "rollback",
                "release": self.fixture.deploy["release"],
                "executed": False,
                "approved": False,
                "irreversible": True,
            },
            evidence_ids=(CLAIM_CACHE,),
            source_snapshot=snapshot,
            created_at=ts,
            updated_at=ts,
        )
        self.store.put_object(
            rollback,
            ts=ts,
            event_type="propose_action",
            reason="investigator proposed rollback; locked pending authorization",
        )
        self.store.add_edge(
            Edge(
                id=f"edge-justifies-{CLAIM_CACHE}-{ACTION_ROLLBACK}",
                src=CLAIM_CACHE,
                dst=ACTION_ROLLBACK,
                mode=EdgeMode.JUSTIFIES,
                rule=InvalidationRule.BLOCK_ACTION,
                declared_effect="block rollback if justifying claim is not fresh",
            )
        )
        self._put_task(
            TASK_GAP,
            ts=ts,
            payload={
                "kind": "provider_crosscheck",
                "target_id": CLAIM_CACHE,
                "note": CACHE_GAP,
            },
        )
        self._put_task(
            TASK_DOCS,
            ts=ts,
            payload={
                "kind": "docs_typo",
                "target_id": OBS_UNRELATED,
                "note": "unrelated README lint",
            },
        )

    def _ingest_scale_extras(self, fixture: RewindFixture) -> None:
        """Unrelated triples (not in the cache cone) plus optional claims on CLAIM_CACHE."""
        for i in range(fixture.extra_unrelated):
            ts = self._now()
            aid = f"art-unrel-{i:04d}"
            oid = f"obs-unrel-{i:04d}"
            tid = f"task-unrel-{i:04d}"
            body = {"summary": f"unrelated note {i}", "i": i}
            art = Record(
                id=aid,
                kind=ObjectKind.ARTIFACT,
                status=ObjectStatus.ACTIVE.value,
                provenance=_prov(
                    "ingestor",
                    f"fixture://health/unrel/{i}",
                    body,
                    ts,
                    TrustClass.TRUSTED,
                ),
                payload={"kind": ArtifactKind.HEALTH_NOTE.value, "body": body},
                evidence_ids=(),
                source_snapshot=content_hash(body),
                created_at=ts,
                updated_at=ts,
            )
            self.store.put_object(
                art, ts=ts, event_type="ingest", reason=f"scale extra artifact {i}"
            )
            obs = Record(
                id=oid,
                kind=ObjectKind.OBSERVATION,
                status=ObjectStatus.ACTIVE.value,
                provenance=_prov(
                    "extractor",
                    f"fixture://health/unrel/{i}",
                    body,
                    ts,
                    TrustClass.DETERMINISTIC,
                ),
                payload={"kind": "unrelated_health", "summary": body["summary"]},
                evidence_ids=(aid,),
                source_snapshot=content_hash(body),
                created_at=ts,
                updated_at=ts,
            )
            self.store.put_object(
                obs, ts=ts, event_type="extract", reason=f"scale extra observation {i}"
            )
            self.store.add_edge(
                Edge(
                    id=f"edge-produced-{aid}-{oid}",
                    src=aid,
                    dst=oid,
                    mode=EdgeMode.PRODUCED_FROM,
                    rule=InvalidationRule.HARD_STALE,
                    declared_effect="observation stale if artifact superseded",
                )
            )
            self._put_task(
                tid,
                ts=ts,
                payload={
                    "kind": "docs_typo",
                    "target_id": oid,
                    "note": f"unrelated lint {i}",
                },
            )
        for i in range(fixture.extra_related_claims):
            ts = self._now()
            cid = f"claim-related-{i:04d}"
            snapshot = content_hash({"parent": CLAIM_CACHE, "i": i})
            rec = Record(
                id=cid,
                kind=ObjectKind.CLAIM,
                status=ObjectStatus.ACTIVE.value,
                provenance=_prov(
                    "investigator",
                    f"fixture://claim/related/{i}",
                    {"i": i},
                    ts,
                    TrustClass.INFERRED,
                ),
                payload={
                    "hypothesis": f"follow-on interpretation {i} of the primary cause claim",
                    "confidence": 0.4,
                },
                evidence_ids=(CLAIM_CACHE,),
                source_snapshot=snapshot,
                created_at=ts,
                updated_at=ts,
            )
            self.store.put_object(
                rec, ts=ts, event_type="propose_claim", reason=f"scale related claim {i}"
            )
            self.store.add_edge(
                Edge(
                    id=f"edge-depends-{cid}-{CLAIM_CACHE}",
                    src=cid,
                    dst=CLAIM_CACHE,
                    mode=EdgeMode.DEPENDS_ON,
                    rule=InvalidationRule.HARD_CHALLENGE,
                    declared_effect="challenge follow-on if cache claim is challenged",
                )
            )

    def _put_task(self, tid: str, *, ts: str, payload: dict[str, Any]) -> Record:
        rec = Record(
            id=tid,
            kind=ObjectKind.TASK,
            status=TaskStatus.PENDING.value,
            provenance=_prov(
                "investigator",
                f"fixture://task/{tid}",
                payload,
                ts,
                TrustClass.INFERRED,
            ),
            payload=payload,
            evidence_ids=tuple(
                x for x in (payload.get("target_id"),) if x
            ),
            source_snapshot=content_hash(payload),
            created_at=ts,
            updated_at=ts,
        )
        self.store.put_object(
            rec, ts=ts, event_type="schedule_task", reason=f"schedule {tid}"
        )
        self.work.touch(tid)
        target = payload.get("target_id")
        if target:
            self.store.add_edge(
                Edge(
                    id=f"edge-depends-{tid}-{target}",
                    src=tid,
                    dst=str(target),
                    mode=EdgeMode.DEPENDS_ON,
                    rule=InvalidationRule.HARD_STALE,
                    declared_effect="task is in the reverse-dep cone of its target",
                )
            )
        return rec

    def _repair_targeted(self, obs: Record) -> None:
        self.engine.apply_provider_overlap_rule(obs)
        self._after_provider(obs)

    def _repair_naive(self, obs: Record) -> None:
        """Re-evaluate every object in the store, then apply the same conclusion steps.

        Does not use reverse-dependency traversal to skip subtrees. Same semantic
        rules still fire (overlap → challenge cache claim → block rollback).
        """
        snapshot = list(self.store.all_objects())
        for rec in snapshot:
            self.work.inspect(rec.id)
            if rec.kind == ObjectKind.CLAIM:
                self.work.touch(rec.id)
            elif rec.kind == ObjectKind.ACTION:
                self.work.inspect(rec.id)
                authorize(self.store, rec.id)
                self.work.touch(rec.id)
            elif rec.kind == ObjectKind.TASK:
                self.work.touch(rec.id)
            elif rec.kind == ObjectKind.OBSERVATION:
                self.work.touch(rec.id)
            elif rec.kind == ObjectKind.ARTIFACT:
                self.work.touch(rec.id)
            elif rec.kind == ObjectKind.VERIFICATION:
                self.work.touch(rec.id)
        self.engine.apply_provider_overlap_rule(obs)
        self._put_task(
            TASK_VERIFY,
            ts=self._now(),
            payload={
                "kind": "verify_outage_pct",
                "target_id": obs.id,
            },
        )
        self._verify_outage(obs)
        follow = self._put_task(
            TASK_FOLLOWUP,
            ts=self._now(),
            payload={
                "kind": "followup_investigation",
                "target_id": CLAIM_RESIDUAL,
                "note": "bounded look at residual timeouts outside the outage window",
            },
        )
        scheduled = tuple(
            t.id for t in self.store.objects_of(ObjectKind.TASK)
        )
        self._revalidation = RevalidationReport(
            scheduled_task_ids=scheduled,
            skipped=(),
        )
        _ = follow

    def _after_provider(self, obs: Record) -> None:
        assert self.fixture is not None
        cone = _transitive_dependents(self.store, obs.id)
        cone.add(CLAIM_CACHE)
        cone.add(ACTION_ROLLBACK)
        cone.add(TASK_GAP)
        scheduled: list[str] = []
        skipped: list[tuple[str, str]] = []

        self._put_task(
            TASK_VERIFY,
            ts=self._now(),
            payload={
                "kind": "verify_outage_pct",
                "target_id": obs.id,
            },
        )
        scheduled.append(TASK_VERIFY)

        # Walk reverse_deps from the cone only. Do not objects_of(TASK): that
        # inspects every unrelated queue item and is why eval_steps tracked N.
        for task in _tasks_in_cone(self.store, cone):
            self.work.inspect(task.id)
            if task.id in scheduled:
                continue
            target = str(task.payload.get("target_id") or "")
            if task.id == TASK_GAP or target == CLAIM_CACHE:
                skipped.append(
                    (
                        task.id,
                        "provider-status gap filled by ingested provider incident; not rerun",
                    )
                )
                continue
            if task.id in cone or target in cone:
                scheduled.append(task.id)
            else:
                skipped.append(
                    (
                        task.id,
                        f"in reverse-dep cone of {obs.id} but not selected; target={target or 'none'}",
                    )
                )

        self._verify_outage(obs)
        follow = self._put_task(
            TASK_FOLLOWUP,
            ts=self._now(),
            payload={
                "kind": "followup_investigation",
                "target_id": CLAIM_RESIDUAL,
                "note": "bounded look at residual timeouts outside the outage window",
            },
        )
        scheduled.append(follow.id)
        self._revalidation = RevalidationReport(
            scheduled_task_ids=tuple(scheduled),
            skipped=tuple(skipped),
        )

    def _verify_outage(self, obs: Record) -> None:
        assert self.fixture is not None
        ts = self._now()
        pct = outage_explains_pct(self.fixture.failures, self.fixture.outage_window)
        snapshot = content_hash(
            {
                "obs": obs.source_snapshot,
                "logs": self.store.get(ART_LOGS).source_snapshot,
            }
        )
        ver = Record(
            id=VER_OUTAGE,
            kind=ObjectKind.VERIFICATION,
            status=ObjectStatus.ACTIVE.value,
            provenance=_prov(
                "verifier",
                "fixture://verify/outage-pct",
                {"support_pct": pct},
                ts,
                TrustClass.DETERMINISTIC,
            ),
            payload={"support_pct": pct, "method": "outage_explains_pct"},
            evidence_ids=(obs.id, OBS_TIMEOUTS, ART_LOGS),
            source_snapshot=snapshot,
            created_at=ts,
            updated_at=ts,
        )
        self.store.put_object(
            ver, ts=ts, event_type="verify", reason="deterministic outage_explains_pct"
        )
        self.work.touch(ver.id)
        self.work.inspect(ver.id)
        claim = Record(
            id=CLAIM_OUTAGE,
            kind=ObjectKind.CLAIM,
            status=ObjectStatus.ACTIVE.value,
            provenance=_prov(
                "verifier",
                "fixture://claim/provider-outage",
                {"support_pct": pct},
                ts,
                TrustClass.DETERMINISTIC,
            ),
            payload={
                "hypothesis": "external provider outage is the primary cause",
                "support_pct": pct,
                "confidence": min(0.99, pct / 100.0),
            },
            evidence_ids=(obs.id, OBS_TIMEOUTS, VER_OUTAGE),
            source_snapshot=snapshot,
            invalidation_conditions=(
                InvalidationCondition(
                    description="outage window or failure timestamps change",
                    evidence_ids=(obs.id, ART_LOGS),
                    rule=InvalidationRule.HARD_STALE,
                ),
            ),
            created_at=ts,
            updated_at=ts,
        )
        self.store.put_object(
            claim,
            ts=ts,
            event_type="propose_claim",
            reason="verifier: outage window covers computed share of failures",
        )
        self.work.touch(claim.id)
        residual_n = sum(
            1
            for f in self.fixture.failures
            if not (
                self.fixture.outage_window.start
                <= f.ts
                <= self.fixture.outage_window.end
            )
        )
        residual = Record(
            id=CLAIM_RESIDUAL,
            kind=ObjectKind.CLAIM,
            status=ObjectStatus.ACTIVE.value,
            provenance=_prov(
                "investigator",
                "fixture://claim/residual-cache",
                {"residual_failures": residual_n},
                ts,
                TrustClass.INFERRED,
            ),
            payload={
                "hypothesis": "possible minor cache issue remains for residual timeouts",
                "residual_failures": residual_n,
                "open": True,
            },
            evidence_ids=(OBS_TIMEOUTS, OBS_DEPLOY),
            source_snapshot=snapshot,
            invalidation_conditions=(
                InvalidationCondition(
                    description="residual timestamps explained by another cause",
                    evidence_ids=(OBS_TIMEOUTS,),
                    rule=InvalidationRule.HARD_STALE,
                ),
            ),
            created_at=ts,
            updated_at=ts,
        )
        self.store.put_object(
            residual,
            ts=ts,
            event_type="propose_claim",
            reason="investigator: residual failures sit outside the outage window",
        )
        self.work.touch(residual.id)


def run_rewind(db_path: Path | str, seed: int = 42) -> RewindRuntime:
    rt = RewindRuntime.create(db_path)
    fixture = build_rewind_fixture(seed=seed)
    rt.ingest_and_propose(fixture)
    rt.ingest_artifact(
        kind=ArtifactKind.PROVIDER_INCIDENT,
        producer="ingestor",
        body=fixture.provider_report,
        trust=TrustClass.TRUSTED,
        source_uri=str(fixture.provider_report.get("uri", _KIND_URI[ArtifactKind.PROVIDER_INCIDENT])),
        timestamp=str(fixture.provider_report.get("window_start")),
    )
    return rt


def _prov(
    producer: str,
    uri: str,
    body: Any,
    ts: str,
    trust: TrustClass,
    version: str | None = "1",
) -> Provenance:
    return Provenance(
        producer=producer,
        sources=(
            SourceRef(uri=uri, version=version, content_hash=content_hash(body)),
        ),
        timestamp=ts,
        trust=trust,
    )


def _transitive_dependents(store: Store, origin_id: str) -> set[str]:
    cone: set[str] = set()
    queue = [origin_id]
    while queue:
        n = queue.pop()
        if n in cone:
            continue
        cone.add(n)
        for dep, _, _ in store.reverse_lookup(n):
            queue.append(dep)
    return cone


def _tasks_in_cone(store: Store, cone: set[str]) -> list[Record]:
    """Tasks reachable from the invalidated cone via reverse_deps, plus tasks in the cone."""
    found: dict[str, Record] = {}
    for oid in cone:
        rec = store.get_or_none(oid)
        if rec is not None and rec.kind == ObjectKind.TASK:
            found[rec.id] = rec
        for task in store.dependent_tasks(oid):
            found[task.id] = task
    return [found[k] for k in sorted(found)]
