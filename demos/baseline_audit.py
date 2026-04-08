#!/usr/bin/env python3
"""
Transcript-heavy baseline audit path for Repo Pulse demo.

Same corpus/task as audit_swarm, but agents coordinate by repeatedly passing
the full prior transcript (JSON anti-pattern) and we track serialization growth.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from demos.corpus_scanner import CorpusReport, build_corpus_report

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


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


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _safe_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception:
        return []


def _call_nonstream(
    client: Any,
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> tuple[str, dict[str, int], int]:
    resp = client.chat.completions.create(
        model=model,
        messages=_messages(system, user),
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    text = resp.choices[0].message.content or ""
    usage_obj = getattr(resp, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }
    return text, usage, len(system) + len(user)


def run_baseline(
    *,
    base_url: str,
    model: str,
    report_path: Path,
    max_files: int,
) -> BaselineMetrics:
    if OpenAI is None:
        raise SystemExit("openai package required: pip install openai")
    rep: CorpusReport = build_corpus_report(max_files=max_files)
    template = _template_path().read_text(encoding="utf-8")
    client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="local")

    t0 = time.perf_counter()
    transcript = ""
    serialized_chars = 0
    coord_chars = 0
    sum_prompt = 0
    sum_completion = 0
    sum_total = 0

    rust_payload = json.dumps([asdict(f) for f in rep.rust_findings[:120]], ensure_ascii=False)
    py_payload = json.dumps([asdict(f) for f in rep.py_findings[:120]], ensure_ascii=False)

    s_a = (
        "You are a Senior Rust Safety Engineer. Output ONLY JSON array with keys: "
        "file,line,category,risk,suggestion."
    )
    u_a = (
        "Analyze Rust findings and produce issues.\n"
        f"Current shared transcript:\n{transcript or '(empty)'}\n\n"
        f"Rust findings:\n{rust_payload}"
    )
    t_a, u_a_usage, c_a = _call_nonstream(client, base_url=base_url, model=model, system=s_a, user=u_a)
    transcript += f"\n[AgentA]\n{t_a}\n"
    serialized_chars += len(json.dumps({"shared": transcript}))
    coord_chars += c_a
    sum_prompt += u_a_usage["prompt_tokens"]
    sum_completion += u_a_usage["completion_tokens"]
    sum_total += u_a_usage["total_tokens"]
    rust_issues = _safe_json(t_a)

    s_b = (
        "You are a Python Type-Safety Expert. Output ONLY JSON array with keys: "
        "file,line,category,risk,suggestion."
    )
    u_b = (
        "Analyze Python findings and produce issues.\n"
        f"Current shared transcript:\n{transcript}\n\n"
        f"Python findings:\n{py_payload}"
    )
    t_b, u_b_usage, c_b = _call_nonstream(client, base_url=base_url, model=model, system=s_b, user=u_b)
    transcript += f"\n[AgentB]\n{t_b}\n"
    serialized_chars += len(json.dumps({"shared": transcript}))
    coord_chars += c_b
    sum_prompt += u_b_usage["prompt_tokens"]
    sum_completion += u_b_usage["completion_tokens"]
    sum_total += u_b_usage["total_tokens"]
    py_issues = _safe_json(t_b)

    s_c = "You are the Architect. Output only Markdown matching the provided template."
    u_c = (
        f"Template:\n{template}\n\n"
        "Generate report from these findings and shared transcript.\n"
        f"Shared transcript:\n{transcript}\n\n"
        f"Rust: {json.dumps(rust_issues[:40], ensure_ascii=False)}\n"
        f"Python: {json.dumps(py_issues[:40], ensure_ascii=False)}"
    )
    t_c, u_c_usage, c_c = _call_nonstream(client, base_url=base_url, model=model, system=s_c, user=u_c, temperature=0.1)
    transcript += f"\n[AgentC]\n{t_c}\n"
    serialized_chars += len(json.dumps({"shared": transcript}))
    coord_chars += c_c
    sum_prompt += u_c_usage["prompt_tokens"]
    sum_completion += u_c_usage["completion_tokens"]
    sum_total += u_c_usage["total_tokens"]

    report_path.write_text(t_c.strip() + "\n", encoding="utf-8")
    wall = time.perf_counter() - t0
    return BaselineMetrics(
        wall_s=wall,
        serialized_chars=serialized_chars,
        coord_context_chars=coord_chars,
        report_items=(len(rust_issues) if isinstance(rust_issues, list) else 0)
        + (len(py_issues) if isinstance(py_issues, list) else 0),
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
