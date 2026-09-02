"""Lease TTL bounds. Enforced in Store/PgStore, not a client convention.

TCP death does not release a lease. The only reclaim path for a silent owner
is `lease_until` expiry. An unbounded client-requested TTL would strand work
after a dropped connection. These constants are the hard ceiling.
"""

from __future__ import annotations

import math
from typing import Any

LEASE_TTL_DEFAULT_S = 30.0
LEASE_TTL_CEILING_S = 30.0
# Trigger slack so Python float now + ceiling is not rejected vs integer unixepoch.
LEASE_TTL_TRIGGER_SLACK_S = 2.0


def configured_max_ttl_s(requested: float | None) -> float:
    """Store-level max, itself capped by the absolute ceiling."""
    if requested is None:
        return LEASE_TTL_CEILING_S
    if not math.isfinite(requested) or requested <= 0:
        return LEASE_TTL_CEILING_S
    return min(float(requested), LEASE_TTL_CEILING_S)


def clamp_lease_ttl_s(requested: float, *, max_ttl_s: float) -> float:
    """TTL actually written to lease_until. Clients cannot exceed max_ttl_s or the ceiling."""
    cap = configured_max_ttl_s(max_ttl_s)
    if not math.isfinite(requested) or requested <= 0:
        return min(LEASE_TTL_DEFAULT_S, cap)
    return min(float(requested), cap)


def requested_ttl_for_log(requested: float) -> float | None:
    if not math.isfinite(requested):
        return None
    return float(requested)


def apply_lease_observe(
    payload: dict[str, Any],
    *,
    worker_id: str,
    lease_until: float,
    ttl_requested: float,
    ttl_granted: float,
    now: float,
    reclaim: bool,
    previous_owner: Any,
) -> dict[str, Any]:
    """Fields operators can read; heartbeats still do not append events."""
    req = requested_ttl_for_log(ttl_requested)
    out = {
        **payload,
        "lease_owner": worker_id,
        "lease_until": lease_until,
        "lease_acquired_at": now,
        "ttl_requested_s": req,
        "ttl_granted_s": ttl_granted,
        "ttl_clamped": req is None or ttl_granted + 1e-9 < req,
        "renewal_count": 0,
    }
    if reclaim:
        out["reclaimed_from"] = previous_owner
        out["reclaim_count"] = int(payload.get("reclaim_count") or 0) + 1
    else:
        out["reclaim_count"] = int(payload.get("reclaim_count") or 0)
    return out


def apply_renew_observe(
    payload: dict[str, Any],
    *,
    lease_until: float,
    ttl_requested: float,
    ttl_granted: float,
) -> dict[str, Any]:
    req = requested_ttl_for_log(ttl_requested)
    return {
        **payload,
        "lease_until": lease_until,
        "ttl_requested_s": req,
        "ttl_granted_s": ttl_granted,
        "ttl_clamped": req is None or ttl_granted + 1e-9 < req,
        "renewal_count": int(payload.get("renewal_count") or 0) + 1,
    }
