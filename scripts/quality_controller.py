#!/usr/bin/env python3
"""
Generic Generate → Verify → Repair quality controller.
Task rules live in YAML profiles; this module runs the loop and telemetry.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    import yaml
except ImportError as e:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = e
else:
    _YAML_IMPORT_ERROR = None

from quality_checks import (
    Violation,
    ViolationSeverity,
    check_fence_extraction,
    check_docstrings,
    check_python_parse,
    check_py_compile,
    check_ruff,
    check_security,
    check_type_hints,
)


class QualityGateFailure(Exception):
    """Raised when max retries exhausted and blocking mode is on."""

    def __init__(self, message: str, reports: list["GateReport"]) -> None:
        super().__init__(message)
        self.reports = reports


@dataclass(frozen=True)
class QualityProfile:
    artifact_type: str
    hard_blockers: frozenset[str]
    warn_checks: frozenset[str]
    max_retries: int
    report_only: bool
    fence_required: bool
    ruff: bool
    py_compile: bool
    require_type_hints: bool
    require_docstrings: bool
    retry_on_warn: bool


def load_profile(path: Path) -> QualityProfile:
    if not path.is_file():
        raise FileNotFoundError(f"Quality profile not found: {path}")
    if yaml is None:
        raise ImportError(
            "PyYAML is required for quality profiles. Install with: pip install pyyaml"
        ) from _YAML_IMPORT_ERROR
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Profile must be a mapping: {path}")

    def _list(key: str, default: list[str] | None = None) -> list[str]:
        v = raw.get(key, default or [])
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError(f"Profile {key} must be a list")
        return [str(x) for x in v]

    return QualityProfile(
        artifact_type=str(raw.get("artifact_type", "python")),
        hard_blockers=frozenset(_list("hard_blockers")),
        warn_checks=frozenset(_list("warn_checks")),
        max_retries=int(raw.get("max_retries", 2)),
        report_only=bool(raw.get("report_only", False)),
        fence_required=bool(raw.get("fence_required", True)),
        ruff=bool(raw.get("ruff", True)),
        py_compile=bool(raw.get("py_compile", True)),
        require_type_hints=bool(raw.get("require_type_hints", False)),
        require_docstrings=bool(raw.get("require_docstrings", False)),
        retry_on_warn=bool(raw.get("retry_on_warn", False)),
    )


Decision = Literal["ACCEPT", "RETRY", "FAIL"]


@dataclass
class GateReport:
    decision: Decision
    violations: list[Violation]
    score: float
    extracted_code: str | None
    raw_assistant: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "violations": [v.to_dict() for v in self.violations],
            "score": self.score,
            "has_extracted_code": self.extracted_code is not None,
        }


@dataclass
class QualityTurnResult:
    accepted_code: str | None
    final_decision: Decision
    attempts_used: int
    reports: list[GateReport]
    last_assistant_text: str
    usage_snapshots: list[dict[str, Any]] = field(default_factory=list)


def _score_violations(
    violations: list[Violation], profile: QualityProfile
) -> float:
    if not violations:
        return 1.0
    hard_hits = sum(
        1 for v in violations if v.rule_id in profile.hard_blockers
    )
    warn_hits = sum(1 for v in violations if v.rule_id in profile.warn_checks)
    # Simple bounded score in [0, 1]
    penalty = min(1.0, 0.15 * hard_hits + 0.05 * warn_hits)
    return max(0.0, 1.0 - penalty)


def verify_assistant_output(
    raw_assistant: str, profile: QualityProfile, *, report_only: bool
) -> GateReport:
    """Run all checks for profile.artifact_type; return GateReport."""
    violations: list[Violation] = []
    code: str | None = None

    if profile.artifact_type == "python":
        extracted, fence_v = check_fence_extraction(
            raw_assistant, profile.fence_required
        )
        violations.extend(fence_v)
        if profile.fence_required and extracted is not None:
            code = extracted
        elif not profile.fence_required:
            code = raw_assistant.strip() or None
            if not code:
                violations.append(
                    Violation(
                        "FMT_EMPTY_BODY",
                        ViolationSeverity.critical,
                        "Empty assistant output.",
                    )
                )
        else:
            code = None

        if code is not None:
            tree, parse_v = check_python_parse(code)
            violations.extend(parse_v)
            if tree is not None:
                violations.extend(check_security(tree))
                if profile.require_docstrings:
                    violations.extend(check_docstrings(tree))
                if profile.require_type_hints:
                    violations.extend(check_type_hints(tree))
            if profile.py_compile and code:
                violations.extend(check_py_compile(code))
            if profile.ruff and code:
                violations.extend(check_ruff(code))
    else:
        violations.append(
            Violation(
                "CFG_UNSUPPORTED_ARTIFACT",
                ViolationSeverity.critical,
                f"artifact_type {profile.artifact_type!r} not supported",
            )
        )

    score = _score_violations(violations, profile)

    blocking_ids = {v.rule_id for v in violations if v.rule_id in profile.hard_blockers}
    warn_blocking = profile.retry_on_warn and any(
        v.rule_id in profile.warn_checks for v in violations
    )

    if report_only or profile.report_only:
        decision: Decision = "ACCEPT"
    elif blocking_ids:
        decision = "RETRY"
    elif warn_blocking:
        decision = "RETRY"
    else:
        decision = "ACCEPT"

    return GateReport(
        decision=decision,
        violations=violations,
        score=score,
        extracted_code=code,
        raw_assistant=raw_assistant,
    )


def format_repair_prompt(violations: list[Violation]) -> str:
    """Repair instructions listing only violated rules (deduped)."""
    seen: set[str] = set()
    lines: list[str] = [
        "Your previous output failed automated verification. Fix ONLY these issues "
        "and reply again with EXACTLY ONE ```python fenced block containing the full file:"
    ]
    for v in violations:
        if v.rule_id in seen:
            continue
        seen.add(v.rule_id)
        lines.append(f"- [{v.rule_id}] {v.message}")
    return "\n".join(lines)


def _emit_event(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, separators=(",", ":")), file=sys.stderr)


class QualityController:
    def __init__(
        self,
        profile_path: Path,
        *,
        report_only: bool | None = None,
    ) -> None:
        self.profile_path = profile_path
        self.profile = load_profile(profile_path)
        self.cli_report_only = report_only

    def effective_report_only(self) -> bool:
        if self.cli_report_only is not None:
            return bool(self.cli_report_only)
        return self.profile.report_only

    def run_turn(
        self,
        call_fn: Callable[[str], tuple[str, Any]],
        user_content: str,
        *,
        label: str,
        role_name: str,
    ) -> QualityTurnResult:
        """
        call_fn(user_message) -> (assistant_text, usage_object_or_none)
        usage_object may have prompt_tokens, completion_tokens, total_tokens attrs.
        """
        reports: list[GateReport] = []
        usage_snapshots: list[dict[str, Any]] = []
        ro = self.effective_report_only()
        max_r = max(0, self.profile.max_retries)
        attempts = 0
        current_user = user_content
        last_text = ""

        while True:
            attempts += 1
            last_text, usage = call_fn(current_user)
            usage_snapshots.append(_usage_to_dict(usage))

            report = verify_assistant_output(last_text, self.profile, report_only=ro)
            reports.append(report)

            vdicts = [v.to_dict() for v in report.violations]
            if report.decision == "ACCEPT":
                _emit_event(
                    {
                        "event": "demo_triple_threat_validation",
                        "path": label,
                        "role": role_name,
                        "decision": report.decision,
                        "score": round(report.score, 4),
                        "attempts": attempts,
                        "violations": vdicts,
                    }
                )
                return QualityTurnResult(
                    accepted_code=report.extracted_code or last_text.strip(),
                    final_decision="ACCEPT",
                    attempts_used=attempts,
                    reports=reports,
                    last_assistant_text=last_text,
                    usage_snapshots=usage_snapshots,
                )

            can_retry = attempts <= max_r
            if can_retry:
                _emit_event(
                    {
                        "event": "demo_triple_threat_retry",
                        "path": label,
                        "role": role_name,
                        "attempt": attempts,
                        "violations": vdicts,
                    }
                )
                current_user = user_content + "\n\n" + format_repair_prompt(
                    report.violations
                )
                continue

            _emit_event(
                {
                    "event": "demo_triple_threat_rejected",
                    "path": label,
                    "role": role_name,
                    "violations": vdicts,
                }
            )
            if ro:
                return QualityTurnResult(
                    accepted_code=report.extracted_code,
                    final_decision="FAIL",
                    attempts_used=attempts,
                    reports=reports,
                    last_assistant_text=last_text,
                    usage_snapshots=usage_snapshots,
                )
            raise QualityGateFailure(
                f"Quality gate rejected output for {label}/{role_name}",
                reports,
            )


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    out: dict[str, Any] = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if hasattr(usage, k):
            out[k] = getattr(usage, k)
    # google-genai usage_metadata
    if hasattr(usage, "prompt_token_count"):
        out["prompt_tokens"] = int(getattr(usage, "prompt_token_count") or 0)
    if hasattr(usage, "candidates_token_count"):
        out["completion_tokens"] = int(getattr(usage, "candidates_token_count") or 0)
    tot = getattr(usage, "total_token_count", None)
    if tot is not None:
        out["total_tokens"] = int(tot)
    elif "prompt_tokens" in out or "completion_tokens" in out:
        out["total_tokens"] = out.get("prompt_tokens", 0) + out.get(
            "completion_tokens", 0
        )
    return out
