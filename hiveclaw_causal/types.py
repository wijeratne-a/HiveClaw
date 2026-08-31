"""Minimum causal-runtime types for The Rewind (four guarantees).

These are data records only. Invalidation, policy, and persistence live in
sibling modules. Do not import hiveclaw_python from this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrustClass(str, Enum):
    TRUSTED = "trusted"
    INFERRED = "inferred"
    HUMAN_AUTHORED = "human-authored"
    DETERMINISTIC = "deterministic"


class ObjectKind(str, Enum):
    ARTIFACT = "artifact"
    OBSERVATION = "observation"
    CLAIM = "claim"
    VERIFICATION = "verification"
    CONFLICT = "conflict"
    TASK = "task"
    ACTION = "action"


class ObjectStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    CHALLENGED = "challenged"
    CORROBORATED = "corroborated"
    VERIFIED = "verified"
    SUPERSEDED = "superseded"
    STALE = "stale"
    INVALIDATED = "invalidated"
    RETRACTED = "retracted"


class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    BLOCKED = "blocked"
    APPROVED = "approved"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    DONE = "done"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class EdgeMode(str, Enum):
    SUPPORTS = "supports"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    JUSTIFIES = "justifies"
    PRODUCED_FROM = "produced_from"


class InvalidationRule(str, Enum):
    """Declared per-edge effect when the source is contradicted, superseded, or itself invalidated."""

    HARD_CHALLENGE = "hard_challenge"
    HARD_STALE = "hard_stale"
    HARD_INVALIDATE = "hard_invalidate"
    BLOCK_ACTION = "block_action"
    DOWNGRADE_CONFIDENCE = "downgrade_confidence"
    SUPERSEDE_CLAIM = "supersede_claim"


class ArtifactKind(str, Enum):
    REPO_SNAPSHOT = "repo_snapshot"
    INCIDENT_LOG = "incident_log"
    DEPLOY_RECORD = "deploy_record"
    PROVIDER_INCIDENT = "provider_incident"
    GOAL = "goal"
    HEALTH_NOTE = "health_note"


@dataclass(frozen=True)
class SourceRef:
    """Exact source pointer with version and/or content hash (Guarantee A)."""

    uri: str
    version: str | None = None
    content_hash: str = ""


@dataclass(frozen=True)
class Provenance:
    producer: str
    sources: tuple[SourceRef, ...]
    timestamp: str
    trust: TrustClass


@dataclass(frozen=True)
class InvalidationCondition:
    """What would invalidate or downgrade a claim (Guarantee B)."""

    description: str
    evidence_ids: tuple[str, ...]
    rule: InvalidationRule


@dataclass
class Record:
    """Queryable stigmergy object. Claims must carry evidence_ids and invalidation_conditions."""

    id: str
    kind: ObjectKind
    status: str
    provenance: Provenance
    payload: dict[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    source_snapshot: str = ""
    invalidation_conditions: tuple[InvalidationCondition, ...] = ()
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class Edge:
    """Typed edge with semantic mode and declared invalidation effect (Guarantee C)."""

    id: str
    src: str
    dst: str
    mode: EdgeMode
    rule: InvalidationRule
    declared_effect: str


@dataclass(frozen=True)
class CausalEvent:
    """Append-only history row. Never overwritten (Guarantee C)."""

    seq: int
    ts: str
    event_type: str
    object_id: str
    old_status: str | None
    new_status: str | None
    reason: str
    edge_id: str | None
    rule: str | None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    """Deterministic authorize/deny (Guarantee D). LLM must not produce this."""

    allowed: bool
    action_id: str
    reason: str
    failed_preconditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RevalidationReport:
    scheduled_task_ids: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]  # (task_id, why_not_rerun)
