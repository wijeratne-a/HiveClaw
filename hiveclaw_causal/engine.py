"""Deterministic causal invalidation. No LLM. Cycle-safe and idempotent."""

from __future__ import annotations

from collections.abc import Callable

from .store import Store
from .types import (
    ActionStatus,
    Edge,
    EdgeMode,
    InvalidationRule,
    ObjectKind,
    ObjectStatus,
    Record,
)
from .work import WorkCounter


def next_status(kind: ObjectKind, rule: InvalidationRule) -> str:
    if kind == ObjectKind.ACTION:
        return ActionStatus.BLOCKED.value
    mapping = {
        InvalidationRule.HARD_CHALLENGE: ObjectStatus.CHALLENGED.value,
        InvalidationRule.HARD_STALE: ObjectStatus.STALE.value,
        InvalidationRule.HARD_INVALIDATE: ObjectStatus.INVALIDATED.value,
        InvalidationRule.SUPERSEDE_CLAIM: ObjectStatus.SUPERSEDED.value,
        InvalidationRule.BLOCK_ACTION: ActionStatus.BLOCKED.value,
        InvalidationRule.DOWNGRADE_CONFIDENCE: ObjectStatus.CHALLENGED.value,
    }
    return mapping[rule]


TOPIC_PROVIDER_STATUS = "topic-provider-status"


def index_provider_interest(store: Store, rec: Record) -> None:
    """Index a claim that declared provider-status interest.

    A new provider-outage observation is not already in reverse_deps of existing
    claims (the contradict edge is created by this rule). Claims cannot be found
    via the observation cone the way tasks are found via their target. They are
    indexed on a stable topic key instead of scanning objects_of(CLAIM).
    """
    if rec.kind != ObjectKind.CLAIM:
        return
    if not _mentions_provider(rec):
        return
    store.add_edge(
        Edge(
            id=f"edge-depends-{rec.id}-{TOPIC_PROVIDER_STATUS}",
            src=rec.id,
            dst=TOPIC_PROVIDER_STATUS,
            mode=EdgeMode.DEPENDS_ON,
            rule=InvalidationRule.HARD_CHALLENGE,
            declared_effect="claim is eligible for provider-outage window overlap",
        )
    )


class InvalidationEngine:
    def __init__(
        self,
        store: Store,
        clock: Callable[[], str],
        counter: WorkCounter | None = None,
    ) -> None:
        self.store = store
        self.clock = clock
        self.counter = counter if counter is not None else WorkCounter()

    def fire(
        self,
        *,
        src_id: str,
        dst_id: str,
        mode: EdgeMode,
        rule: InvalidationRule,
        reason: str,
    ) -> str:
        edge_id = f"edge-{mode.value}-{src_id}-{dst_id}"
        edge = Edge(
            id=edge_id,
            src=src_id,
            dst=dst_id,
            mode=mode,
            rule=rule,
            declared_effect=f"{rule.value} via {mode.value}",
        )
        self.store.add_edge(edge)
        self._propagate(
            dst_id,
            rule=rule,
            edge_id=edge_id,
            reason=reason,
            stack=set(),
        )
        return edge_id

    def apply_provider_overlap_rule(self, new_obs: Record) -> list[str]:
        """If a provider-outage observation overlaps a claim's incident window and the
        claim declared a provider-status invalidation condition, contradict it.
        """
        fired: list[str] = []
        if new_obs.payload.get("kind") != "provider_outage":
            return fired
        obs_start = str(new_obs.payload.get("window_start", ""))
        obs_end = str(new_obs.payload.get("window_end", ""))
        for claim in self.store.dependent_claims(TOPIC_PROVIDER_STATUS):
            self.counter.inspect(claim.id)
            if claim.status not in (
                ObjectStatus.PROPOSED.value,
                ObjectStatus.ACTIVE.value,
                ObjectStatus.CORROBORATED.value,
            ):
                continue
            if not _mentions_provider(claim):
                continue
            win = claim.payload.get("incident_window") or {}
            if _overlaps(obs_start, obs_end, str(win.get("start", "")), str(win.get("end", ""))):
                reason = (
                    f"new observation {new_obs.id} overlaps claim {claim.id} incident window; "
                    f"rule={InvalidationRule.HARD_CHALLENGE.value} mode={EdgeMode.CONTRADICTS.value}"
                )
                eid = self.fire(
                    src_id=new_obs.id,
                    dst_id=claim.id,
                    mode=EdgeMode.CONTRADICTS,
                    rule=InvalidationRule.HARD_CHALLENGE,
                    reason=reason,
                )
                fired.append(eid)
        return fired

    def _propagate(
        self,
        object_id: str,
        *,
        rule: InvalidationRule,
        edge_id: str,
        reason: str,
        stack: set[str],
    ) -> None:
        if object_id in stack:
            return
        rec = self.store.get_or_none(object_id)
        if rec is None:
            return
        # Tasks are indexed in reverse_deps so repair can find the cone without
        # scanning the queue. They are not status-propagation nodes: a challenged
        # claim must not flip every depending task to stale (that would change
        # lease/skip semantics). Scheduling decides skip vs run.
        if rec.kind == ObjectKind.TASK:
            return
        self.counter.inspect(object_id)
        new_status = next_status(rec.kind, rule)
        if self.store.has_applied(object_id, edge_id, new_status):
            return
        stack.add(object_id)
        self.store.transition(
            object_id,
            new_status,
            ts=self.clock(),
            reason=reason,
            edge_id=edge_id,
            rule=rule.value,
        )
        self.counter.touch(object_id)
        for dep_id, dep_edge, dep_rule in self.store.reverse_lookup(object_id):
            self._propagate(
                dep_id,
                rule=InvalidationRule(dep_rule),
                edge_id=dep_edge,
                reason=reason,
                stack=stack,
            )
        stack.remove(object_id)


def _mentions_provider(claim: Record) -> bool:
    for cond in claim.invalidation_conditions:
        blob = (cond.description + " " + " ".join(cond.evidence_ids)).lower()
        if "provider" in blob:
            return True
    gaps = claim.payload.get("gaps") or []
    return any("provider" in str(g).lower() for g in gaps)


def _overlaps(a0: str, a1: str, b0: str, b1: str) -> bool:
    if not (a0 and a1 and b0 and b1):
        return False
    return a0 <= b1 and b0 <= a1
