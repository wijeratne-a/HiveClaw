"""Generative Rewind fixture: outage window covers a large majority of failure timestamps."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class TimeWindow:
    start: str
    end: str


@dataclass(frozen=True)
class FailureEvent:
    ts: str
    kind: str
    request_id: str


@dataclass(frozen=True)
class RewindFixture:
    repo_files: dict[str, str]
    failures: tuple[FailureEvent, ...]
    incident_window: TimeWindow
    outage_window: TimeWindow
    deploy: dict[str, Any]
    goal: str
    provider_report: dict[str, Any]
    unrelated_note: dict[str, Any]


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, TS_FMT).replace(tzinfo=timezone.utc)


def format_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).strftime(TS_FMT)


def _spread(
    rng: random.Random,
    start: str,
    end: str,
    n: int,
    *,
    prefix: str,
    kind: str,
) -> tuple[FailureEvent, ...]:
    """Place n timestamps strictly inside (start, end) so inclusive-window stats stay stable."""
    t0 = parse_ts(start)
    t1 = parse_ts(end)
    span = (t1 - t0).total_seconds()
    if span <= 2:
        raise ValueError("window too small to spread failures")
    out: list[FailureEvent] = []
    for i in range(n):
        frac = (i + 1) / (n + 1)
        frac = min(0.98, max(0.02, frac + (rng.random() - 0.5) * 0.01))
        ts = t0 + timedelta(seconds=span * frac)
        out.append(
            FailureEvent(
                ts=format_ts(ts),
                kind=kind,
                request_id=f"{prefix}-{i:03d}",
            )
        )
    return tuple(out)


def build_rewind_fixture(seed: int = 42) -> RewindFixture:
    rng = random.Random(seed)
    incident = TimeWindow(start="2026-08-30T14:00:00Z", end="2026-08-30T14:30:00Z")
    outage = TimeWindow(start="2026-08-30T14:02:00Z", end="2026-08-30T14:28:00Z")
    deploy_ts = "2026-08-30T13:55:00Z"

    n_total = 100
    n_residual = 8
    n_outage = n_total - n_residual
    n_before = n_residual // 2
    n_after = n_residual - n_before

    in_outage = _spread(
        rng, outage.start, outage.end, n_outage, prefix="to", kind="timeout"
    )
    before = _spread(
        rng,
        incident.start,
        outage.start,
        n_before,
        prefix="tb",
        kind="timeout",
    )
    after = _spread(
        rng,
        outage.end,
        incident.end,
        n_after,
        prefix="ta",
        kind="timeout",
    )
    failures = tuple(sorted(in_outage + before + after, key=lambda f: f.ts))

    repo_files = {
        "checkout/payment.py": (
            "CACHE_TTL = 5\n"
            "def capture(order_id):\n"
            "    invalidate_cart_cache(order_id)\n"
            "    return provider.charge(order_id)\n"
        ),
        "checkout/provider.py": (
            "def charge(order_id):\n"
            "    return http.post('/v1/charge', json={'order': order_id})\n"
        ),
    }

    return RewindFixture(
        repo_files=repo_files,
        failures=failures,
        incident_window=incident,
        outage_window=outage,
        deploy={
            "release": "rel-2026.08.30.1",
            "sha": "c0ffee1234ab",
            "timestamp": deploy_ts,
            "notes": "cache invalidation refactor on capture path",
        },
        goal="find the cause and safely fix it.",
        provider_report={
            "uri": "fixture://provider-incident/INC-8841",
            "provider": "PayNorth",
            "incident_id": "INC-8841",
            "window_start": outage.start,
            "window_end": outage.end,
            "summary": "external payment-provider outage overlapping checkout timeouts",
        },
        unrelated_note={
            "uri": "fixture://health/nightly",
            "summary": "unrelated nightly health check passed",
            "timestamp": "2026-08-30T12:00:00Z",
        },
    )
