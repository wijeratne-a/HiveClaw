"""Lease TTL bounds. Enforced in Store/PgStore, not a client convention.

TCP death does not release a lease. The only reclaim path for a silent owner
is `lease_until` expiry. An unbounded client-requested TTL would strand work
after a dropped connection. These constants are the hard ceiling.
"""

from __future__ import annotations

import math

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
