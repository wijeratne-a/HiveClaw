"""Language-agnostic checks: fenced code extraction and violation types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ViolationSeverity(str, Enum):
    critical = "critical"
    warning = "warning"


@dataclass(frozen=True)
class Violation:
    rule_id: str
    severity: ViolationSeverity
    message: str
    line: int | None = None

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "line": self.line,
        }


_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_single_fence(text: str) -> tuple[str | None, list[Violation]]:
    """
    Require exactly one non-empty ```python (or ```) fenced block.
    Returns (code_body, violations). code_body is None if extraction failed.
    """
    violations: list[Violation] = []
    matches = list(_FENCE_RE.finditer(text))
    if not matches:
        violations.append(
            Violation(
                "FMT_NO_FENCE",
                ViolationSeverity.critical,
                "No ```python fenced block found.",
            )
        )
        return None, violations
    if len(matches) > 1:
        violations.append(
            Violation(
                "FMT_MULTIPLE_FENCES",
                ViolationSeverity.critical,
                "Output must contain exactly one ```python fenced block.",
            )
        )
        return None, violations
    body = matches[0].group(1).strip()
    if not body:
        violations.append(
            Violation(
                "FMT_EMPTY_FENCE",
                ViolationSeverity.critical,
                "Fenced Python block is empty.",
            )
        )
        return None, violations
    return body, []


def check_fence_extraction(text: str, fence_required: bool) -> tuple[str | None, list[Violation]]:
    """If fence_required is False, skip fence rules and return (None, []) — caller uses raw text."""
    if not fence_required:
        return None, []
    return extract_single_fence(text)
