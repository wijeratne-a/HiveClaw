#!/usr/bin/env python3
"""
HiveClaw multi-agent repo audit path for Repo Pulse demo.

Pipeline:
  scanner output -> Agent A (Rust JSON findings)
                 -> Agent B (Python JSON findings)
                 -> Agent C (Architect Markdown report)
                 -> HEALTH_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from demos.corpus_scanner import CorpusReport, build_corpus_report
from demos.health_report_validate import format_allowed_paths_instructions, validate_health_report_markdown
from demos.llm_ab import complete_issue_agent_round
from demos.llm_client import httpx_client, stream_chat_completion


@dataclass
class AuditMetrics:
    wall_s: float
    ttft_s: float
    coord_context_chars: int
    report_items: int
    slab_rw_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    report_path: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _template_path() -> Path:
    return _repo_root() / "demos" / "data" / "health_report_template.md"


def _default_report_path() -> Path:
    return _repo_root() / "HEALTH_REPORT.md"


def _slab_claimed_count() -> int:
    try:
        import hiveclaw_python as hc
    except Exception:
        return 0
    try:
        c = hc.SlabClient()
        states = c.get_slot_states()
        return sum(1 for s in states if bool(s["claimed"]))
    except Exception:
        return 0


def _prompt_agent_a(rep: CorpusReport) -> tuple[str, str]:
    system = (
        "You are a Senior Rust Safety Engineer. "
        "Output ONLY a JSON array. Each item must be an object with keys: "
        "file, line, category, risk (high|med|low), suggestion. No prose."
    )
    compact = [asdict(f) for f in rep.rust_findings[:120]]
    user = (
        "Analyze these Rust findings and produce triaged issues.\n\n"
        f"{json.dumps(compact, ensure_ascii=False)}"
    )
    return system, user


def _prompt_agent_b(rep: CorpusReport) -> tuple[str, str]:
    system = (
        "You are a Python Type-Safety Expert. "
        "Output ONLY a JSON array. Each item must be an object with keys: "
        "file, line, category, risk (high|med|low), suggestion. No prose."
    )
    compact = [asdict(f) for f in rep.py_findings[:120]]
    user = (
        "Analyze these Python findings and produce triaged issues.\n\n"
        f"{json.dumps(compact, ensure_ascii=False)}"
    )
    return system, user


def _prompt_agent_c(
    rep: CorpusReport,
    rust_issues: list[dict[str, Any]],
    py_issues: list[dict[str, Any]],
    template: str,
) -> tuple[str, str]:
    allow = format_allowed_paths_instructions(rep)
    system = (
        "You are the Hive Architect. Output ONLY Markdown matching the template sections: "
        "Rust Safety Issues, Python Type Safety Issues, Summary."
    )
    user = (
        f"{allow}\n"
        "Use this template:\n"
        f"{template}\n\n"
        "Input findings JSON:\n"
        f"{json.dumps({'rust': rust_issues[:40], 'python': py_issues[:40]}, ensure_ascii=False)}\n\n"
        "Populate concrete table rows with concise suggestions. Keep the report actionable."
    )
    return system, user


def run_audit_swarm(
    *,
    base_url: str,
    model: str,
    report_path: Path,
    max_files: int,
) -> AuditMetrics:
    rep = build_corpus_report(max_files=max_files)
    template = _template_path().read_text(encoding="utf-8")
    slab_before = _slab_claimed_count()
    t0 = time.perf_counter()

    sum_prompt = 0
    sum_completion = 0
    sum_total = 0
    total_coord_chars = 0
    first_ttft = 0.0

    with httpx_client() as client:
        s_a, u_a = _prompt_agent_a(rep)
        _txt_a, ttft_a, usage_a, cc_a, rust_issues = complete_issue_agent_round(
            client, base_url=base_url, model=model, system=s_a, user=u_a
        )
        first_ttft = ttft_a
        total_coord_chars += cc_a
        sum_prompt += usage_a["prompt_tokens"]
        sum_completion += usage_a["completion_tokens"]
        sum_total += usage_a["total_tokens"]

        s_b, u_b = _prompt_agent_b(rep)
        _txt_b, _ttft_b, usage_b, cc_b, py_issues = complete_issue_agent_round(
            client, base_url=base_url, model=model, system=s_b, user=u_b
        )
        total_coord_chars += cc_b
        sum_prompt += usage_b["prompt_tokens"]
        sum_completion += usage_b["completion_tokens"]
        sum_total += usage_b["total_tokens"]

        s_c, u_c = _prompt_agent_c(rep, rust_issues, py_issues, template)
        text_c = ""
        for attempt in range(3):
            text_c, _ttft_c, usage_c, cc_c = stream_chat_completion(
                client,
                base_url=base_url,
                model=model,
                system=s_c,
                user=u_c,
                temperature=0.1,
            )
            total_coord_chars += cc_c
            sum_prompt += usage_c["prompt_tokens"]
            sum_completion += usage_c["completion_tokens"]
            sum_total += usage_c["total_tokens"]
            if validate_health_report_markdown(text_c):
                break
            u_c = (
                u_c
                + "\n\nRepair instruction: output markdown with all required headings exactly. "
                "No ellipsis placeholder rows (no |...|). Use only allowed file paths listed above."
            )

    report_path.write_text(text_c.strip() + "\n", encoding="utf-8")
    slab_after = _slab_claimed_count()
    wall = time.perf_counter() - t0
    return AuditMetrics(
        wall_s=wall,
        ttft_s=first_ttft,
        coord_context_chars=total_coord_chars,
        report_items=len(rust_issues) + len(py_issues),
        slab_rw_count=abs(slab_after - slab_before),
        prompt_tokens=sum_prompt,
        completion_tokens=sum_completion,
        total_tokens=sum_total,
        report_path=str(report_path),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Run HiveClaw audit swarm demo path")
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8080")
    p.add_argument("--model", type=str, default="hiveclaw-swarm-8b")
    p.add_argument("--max-files", type=int, default=60)
    p.add_argument("--report-path", type=str, default=str(_default_report_path()))
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()

    metrics = run_audit_swarm(
        base_url=args.base_url,
        model=args.model,
        report_path=Path(args.report_path),
        max_files=max(1, int(args.max_files)),
    )
    payload = json.dumps(asdict(metrics), indent=2)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
