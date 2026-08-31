"""Deterministic stats over fixture timestamps. No hardcoded claim strings."""

from __future__ import annotations

from .fixture import FailureEvent, TimeWindow


def in_window(ts: str, window: TimeWindow) -> bool:
    return window.start <= ts <= window.end


def outage_explains_pct(
    failures: tuple[FailureEvent, ...] | list[FailureEvent],
    outage_window: TimeWindow,
) -> float:
    """Percentage of observed failures whose timestamps fall inside the provider outage window."""
    if not failures:
        return 0.0
    n = sum(1 for f in failures if in_window(f.ts, outage_window))
    return 100.0 * n / len(failures)
