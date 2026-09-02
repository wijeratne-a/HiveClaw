#!/usr/bin/env python3
"""
Transcript-heavy baseline audit path for Repo Pulse demo.

Same corpus/task as audit_swarm, but agents coordinate by repeatedly passing
the full prior transcript (JSON anti-pattern) and we track serialization growth.
Uses the same httpx streaming client as audit_swarm for comparable wall times.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from demos.corpus_scanner import CorpusReport, build_corpus_report
from demos.health_report_validate import format_allowed_paths_instructions, validate_health_report_markdown
from demos.llm_ab import complete_issue_agent_round
from demos.llm_client import httpx_client, stream_chat_completion


@dataclass
class BaselineMetrics:
    wall_s: float
    serialized_chars: int
    coord_context_chars: int
    report_items: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    report_path: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _template_path() -> Path:
    return _repo_root() / "demos" / "data" / "health_report_template.md"


def _default_report_path() -> Path:
    return _repo_root() / "HEALTH_REPORT_BASELINE.md"


def run_baseline(
    *,
    base_url: str,
    model: str,
    report_path: Path,
    max_files: int,
) -> BaselineMetrics:
    rep: CorpusReport = build_corpus_report(max_files=max_files)
    template = _template_path().read_text(encoding="utf-8")
    allow = format_allowed_paths_instructions(rep)

    t0 = time.perf_counter()
    transcript = ""
    serialized_chars = 0
    coord_chars = 0
    sum_prompt = 0
    sum_completion = 0
    sum_total = 0

    rust_payload = json.dumps([asdict(f) for f in rep.rust_findings[:120]], ensure_ascii=False)
    py_payload = json.dumps([asdict(f) for f in rep.py_findings[:120]], ensure_ascii=False)

    with httpx_client() as client:
        s_a = (
            "You are a Senior Rust Safety Engineer. Output ONLY JSON array with keys: "
            "file,line,category,risk,suggestion."
        )
        u_a = (
            "Analyze Rust findings and produce issues.\n"
            f"Current shared transcript:\n{transcript or '(empty)'}\n\n"
            f"Rust findings:\n{rust_payload}"
        )
        t_a, _, u_a_usage, c_a, rust_issues = complete_issue_agent_round(
            client, base_url=base_url, model=model, system=s_a, user=u_a
        )
        transcript += f"\n[AgentA]\n{t_a}\n"
        serialized_chars += len(json.dumps({"shared": transcript}))
        coord_chars += c_a
        sum_prompt += u_a_usage["prompt_tokens"]
        sum_completion += u_a_usage["completion_tokens"]
        sum_total += u_a_usage["total_tokens"]

        s_b = (
            "You are a Python Type-Safety Expert. Output ONLY JSON array with keys: "
            "file,line,category,risk,suggestion."
        )
        u_b = (
            "Analyze Python findings and produce issues.\n"
            f"Current shared transcript:\n{transcript}\n\n"
            f"Python findings:\n{py_payload}"
        )
        t_b, _, u_b_usage, c_b, py_issues = complete_issue_agent_round(
            client, base_url=base_url, model=model, system=s_b, user=u_b
        )
        transcript += f"\n[AgentB]\n{t_b}\n"
        serialized_chars += len(json.dumps({"shared": transcript}))
        coord_chars += c_b
        sum_prompt += u_b_usage["prompt_tokens"]
        sum_completion += u_b_usage["completion_tokens"]
        sum_total += u_b_usage["total_tokens"]

        s_c = "You are the Architect. Output only Markdown matching the provided template."
        u_c = (
            f"{allow}\n"
            f"Template:\n{template}\n\n"
            "Generate report from these findings and shared transcript.\n"
            f"Shared transcript:\n{transcript}\n\n"
            f"Rust: {json.dumps(rust_issues[:40], ensure_ascii=False)}\n"
            f"Python: {json.dumps(py_issues[:40], ensure_ascii=False)}"
        )
        t_c = ""
        u_arch = u_c
        for attempt in range(2):
            t_c, _, u_c_usage, c_c = stream_chat_completion(
                client,
                base_url=base_url,
                model=model,
                system=s_c,
                user=u_arch,
                temperature=0.1,
            )
            coord_chars += c_c
            sum_prompt += u_c_usage["prompt_tokens"]
            sum_completion += u_c_usage["completion_tokens"]
            sum_total += u_c_usage["total_tokens"]
            if validate_health_report_markdown(t_c):
                break
            u_arch = (
                u_c
                + "\n\nRepair: required headings exactly. No ellipsis rows (no |...|). "
                "Use only allowed file paths listed above."
            )

        transcript += f"\n[AgentC]\n{t_c}\n"
        serialized_chars += len(json.dumps({"shared": transcript}))

    report_path.write_text(t_c.strip() + "\n", encoding="utf-8")
    wall = time.perf_counter() - t0
    return BaselineMetrics(
        wall_s=wall,
        serialized_chars=serialized_chars,
        coord_context_chars=coord_chars,
        report_items=len(rust_issues) + len(py_issues),
        prompt_tokens=sum_prompt,
        completion_tokens=sum_completion,
        total_tokens=sum_total,
        report_path=str(report_path),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Run transcript-heavy baseline audit demo")
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8080")
    p.add_argument("--model", type=str, default="hiveclaw-swarm-8b")
    p.add_argument("--max-files", type=int, default=60)
    p.add_argument("--report-path", type=str, default=str(_default_report_path()))
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()

    met = run_baseline(
        base_url=args.base_url,
        model=args.model,
        report_path=Path(args.report_path),
        max_files=max(1, int(args.max_files)),
    )
    payload = json.dumps(asdict(met), indent=2)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
