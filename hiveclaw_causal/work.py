"""Work accounting for targeted vs naive repair (deterministic evals, no LLM)."""

from __future__ import annotations

from dataclasses import dataclass, field


class WorkCounter:
    """Counts inspection steps and explicitly touched object ids during a repair."""

    def __init__(self) -> None:
        self.eval_steps = 0
        self.touched: set[str] = set()

    def reset(self) -> None:
        self.eval_steps = 0
        self.touched = set()

    def inspect(self, object_id: str) -> None:
        self.eval_steps += 1
        _ = object_id

    def touch(self, object_id: str) -> None:
        self.touched.add(object_id)


@dataclass(frozen=True)
class WorkMetrics:
    mode: str
    objects_before: int
    objects_after: int
    objects_touched: int
    objects_untouched: int
    eval_steps: int
    wall_s: float
    support_pct: float
    rollback_status: str
    rollback_blocked: bool
    followup_present: bool
    touched_ids: tuple[str, ...] = field(default_factory=tuple)
    untouched_ids: tuple[str, ...] = field(default_factory=tuple)
