"""Rewind orchestrator stub — ingest/invalidation/policy are not implemented yet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import (
    ArtifactKind,
    ObjectKind,
    PolicyDecision,
    Record,
    RevalidationReport,
    TrustClass,
)


class RewindRuntime:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._objects: dict[str, Record] = {}
        self._events: list[Any] = []
        self._revalidation = RevalidationReport(scheduled_task_ids=(), skipped=())

    @classmethod
    def create(cls, db_path: Path | str) -> RewindRuntime:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path)

    def ingest_and_propose(self, fixture: Any) -> None:
        _ = fixture

    def ingest_artifact(
        self,
        *,
        kind: ArtifactKind,
        producer: str,
        body: dict[str, Any],
        trust: TrustClass,
        source_uri: str,
        timestamp: str | None = None,
    ) -> str:
        _ = (kind, producer, body, trust, source_uri, timestamp)
        return "art-stub"

    def objects_of(self, kind: ObjectKind) -> list[Record]:
        return [r for r in self._objects.values() if r.kind == kind]

    def get(self, object_id: str) -> Record:
        return self._objects[object_id]

    def events_for(self, object_id: str) -> list[Any]:
        return [e for e in self._events if getattr(e, "object_id", None) == object_id]

    def all_events(self) -> list[Any]:
        return list(self._events)

    def policy_authorize(self, action_id: str) -> PolicyDecision:
        return PolicyDecision(
            allowed=False,
            action_id=action_id,
            reason="stub-unauthorized",
            failed_preconditions=("not-implemented",),
        )

    def last_revalidation(self) -> RevalidationReport:
        return self._revalidation

    def why(self, object_id: str) -> dict[str, Any]:
        rec = self._objects.get(object_id)
        return {
            "object_id": object_id,
            "status": None if rec is None else rec.status,
            "events": self.events_for(object_id),
        }

    def computed_outage_support_pct(self) -> float:
        return 0.0


def run_rewind(db_path: Path | str, seed: int = 42) -> RewindRuntime:
    """Full scenario helper used by the demo. E2E test drives steps itself."""
    from .fixture import build_rewind_fixture

    rt = RewindRuntime.create(db_path)
    fixture = build_rewind_fixture(seed=seed)
    rt.ingest_and_propose(fixture)
    rt.ingest_artifact(
        kind=ArtifactKind.PROVIDER_INCIDENT,
        producer="ingestor",
        body=fixture.provider_report,
        trust=TrustClass.TRUSTED,
        source_uri=str(fixture.provider_report.get("uri", "fixture://provider")),
    )
    return rt
