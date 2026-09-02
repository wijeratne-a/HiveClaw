"""Read-only lease / store operational view."""

from __future__ import annotations

import time
from typing import Any

from .lease_policy import LEASE_TTL_CEILING_S
from .schema import SCHEMA_VERSION, current_version
from .types import ObjectKind, TaskStatus


NEAR_EXPIRY_S = 5.0


def store_status(store: Any) -> dict[str, Any]:
    now = time.time()
    tasks = store.objects_of(ObjectKind.TASK)
    by_state: dict[str, int] = {}
    for t in tasks:
        by_state[t.status] = by_state.get(t.status, 0) + 1
    expired = 0
    for t in tasks:
        if t.status != TaskStatus.LEASED.value:
            continue
        until = t.payload.get("lease_until")
        if until is not None and float(until) < now:
            expired += 1
    by_state["expired_still_leased"] = expired

    active: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    longest_age = 0.0
    for t in tasks:
        if t.status != TaskStatus.LEASED.value:
            continue
        until = float(t.payload.get("lease_until") or 0.0)
        acquired = float(t.payload.get("lease_acquired_at") or 0.0)
        remaining = until - now
        age = (now - acquired) if acquired else None
        if age is not None:
            longest_age = max(longest_age, age)
        row = {
            "task_id": t.id,
            "owner": t.payload.get("lease_owner"),
            "acquired_at": acquired or None,
            "expiry_at": until or None,
            "ttl_remaining_s": remaining,
            "renewal_count": int(t.payload.get("renewal_count") or 0),
            "reclaim_count": int(t.payload.get("reclaim_count") or 0),
            "ttl_granted_s": t.payload.get("ttl_granted_s"),
            "ttl_clamped": bool(t.payload.get("ttl_clamped")),
        }
        active.append(row)
        if 0 < remaining <= NEAR_EXPIRY_S:
            near.append(row)

    events = store.all_events()
    reclaim_evs = [e for e in events if e.event_type == "lease_reclaim"]
    clamped_evs = [
        e
        for e in events
        if e.event_type in ("lease_task", "lease_reclaim") and e.payload.get("ttl_clamped")
    ]

    cfg_max = _config_max(store)
    return {
        "backend": getattr(store, "backend", "unknown"),
        "schema_version": current_version(store),
        "code_schema_version": SCHEMA_VERSION,
        "task_counts": by_state,
        "active_leases": sorted(active, key=lambda r: r["task_id"]),
        "near_expiry": near,
        "clamped_lease_events": len(clamped_evs),
        "failed_lease_attempts": {
            "recorded": False,
            "reason": "try_lease returning None is not persisted; CAS miss is not an audit row",
        },
        "reclaim_count": len(reclaim_evs),
        "reclaim_latency_s": {
            "recorded": False,
            "reason": "event ts is 1s ISO; reclaim latency is not separately timed",
        },
        "longest_observed_lease_age_s": longest_age if longest_age else None,
        "owner_process_health": {
            "recorded": False,
            "reason": "workers share only the store; no process heartbeat table",
        },
        "ttl_policy": {
            "configured_max_ttl_s": cfg_max,
            "hard_ceiling_s": LEASE_TTL_CEILING_S,
            "within_hard_maximum": cfg_max is not None and cfg_max <= LEASE_TTL_CEILING_S + 1e-9,
        },
    }


def _config_max(store: Any) -> float | None:
    try:
        row = store._conn.execute(
            "SELECT value FROM lease_config WHERE key = 'max_ttl_s'"
        ).fetchone()
    except Exception:
        return getattr(store, "max_lease_ttl_s", None)
    if row is None:
        return getattr(store, "max_lease_ttl_s", None)
    if isinstance(row, dict):
        return float(row["value"])
    return float(row[0] if not hasattr(row, "keys") else row["value"])


def format_status(report: dict[str, Any]) -> str:
    counts = report.get("task_counts") or {}
    lines = [
        f"backend={report.get('backend')} schema={report.get('schema_version')}/{report.get('code_schema_version')}",
        "tasks: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        or "(none)",
        f"active_leases={len(report.get('active_leases') or [])} near_expiry={len(report.get('near_expiry') or [])}",
        f"reclaims={report.get('reclaim_count')} clamped_events={report.get('clamped_lease_events')}",
        (
            f"ttl configured_max={report['ttl_policy'].get('configured_max_ttl_s')} "
            f"ceiling={report['ttl_policy'].get('hard_ceiling_s')} "
            f"within_ceiling={report['ttl_policy'].get('within_hard_maximum')}"
        ),
        "failed_attempts: not recorded (see JSON)",
        "reclaim_latency: not recorded (see JSON)",
        "owner_process_health: not recorded (see JSON)",
    ]
    return "\n".join(lines)
