"""Rewind fixture shape. Generative timestamps land in a later increment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def build_rewind_fixture(seed: int = 42) -> RewindFixture:
    """Stub: empty failures. The generative builder is implemented after this test exists."""
    _ = seed
    return RewindFixture(
        repo_files={},
        failures=(),
        incident_window=TimeWindow(start="", end=""),
        outage_window=TimeWindow(start="", end=""),
        deploy={},
        goal="find the cause and safely fix it.",
        provider_report={"uri": "fixture://provider-incident", "body": {}},
        unrelated_note={},
    )
