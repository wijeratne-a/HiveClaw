"""Deterministic policy gate for irreversible actions. No LLM."""

from __future__ import annotations

from .store import Store
from .types import (
    ActionStatus,
    EdgeMode,
    ObjectKind,
    ObjectStatus,
    PolicyDecision,
)

_FRESH = {
    ObjectStatus.ACTIVE.value,
    ObjectStatus.CORROBORATED.value,
    ObjectStatus.VERIFIED.value,
}


def authorize(store: Store, action_id: str) -> PolicyDecision:
    rec = store.get_or_none(action_id)
    if rec is None:
        return PolicyDecision(
            allowed=False,
            action_id=action_id,
            reason="unknown-action",
            failed_preconditions=("missing-object",),
        )
    if rec.kind != ObjectKind.ACTION:
        return PolicyDecision(
            allowed=False,
            action_id=action_id,
            reason="not-an-action",
            failed_preconditions=(f"kind={rec.kind.value}",),
        )
    failed: list[str] = []
    if rec.status == ActionStatus.EXECUTED.value:
        failed.append("already-executed")
    if rec.status == ActionStatus.BLOCKED.value:
        failed.append("action-blocked")
    if rec.payload.get("executed"):
        failed.append("payload-executed")
    if not rec.payload.get("approved"):
        failed.append("locked-unauthorized")
    justifying = store.edges_to(action_id, EdgeMode.JUSTIFIES)
    if not justifying:
        failed.append("no-justifying-claim")
    for edge in justifying:
        claim = store.get_or_none(edge.src)
        if claim is None:
            failed.append(f"missing-claim:{edge.src}")
            continue
        if claim.status not in _FRESH:
            failed.append(f"justifying-claim-not-fresh:{claim.id}:{claim.status}")
    if failed:
        reason = failed[0]
        if rec.status == ActionStatus.BLOCKED.value:
            evs = [e for e in store.events_for(action_id) if e.new_status == ActionStatus.BLOCKED.value]
            if evs:
                last = evs[-1]
                reason = (
                    f"action-blocked edge={last.edge_id} rule={last.rule} "
                    f"({last.reason})"
                )
        elif any(f.startswith("justifying-claim-not-fresh") for f in failed):
            stale = [f for f in failed if f.startswith("justifying-claim-not-fresh")][0]
            reason = stale
        return PolicyDecision(
            allowed=False,
            action_id=action_id,
            reason=reason,
            failed_preconditions=tuple(failed),
        )
    return PolicyDecision(
        allowed=True,
        action_id=action_id,
        reason="prerequisites-hold",
        failed_preconditions=(),
    )
