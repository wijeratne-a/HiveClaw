#!/usr/bin/env python3
"""
Triple-Threat Refactor demo: same 3-agent pipeline on ``demo_target.py`` via
HiveClaw (local OpenAI-compatible) vs a **cloud** leg — **OpenAI** or **Gemini**.

Tracks wall-clock and token usage. HiveClaw uses ``stream=False`` (usage in JSON).
OpenAI: same. Gemini: ``usage_metadata`` from the ``google-genai`` client.

HiveClaw path: compact prompts (latest code only).
Cloud path: growing prior-transcript (coordination token growth).

API keys (cloud)::

  export OPENAI_API_KEY=sk-...     # OpenAI
  export GEMINI_API_KEY=...        # or GOOGLE_API_KEY for Gemini

Default ``--cloud-provider auto``: use OpenAI if ``OPENAI_API_KEY`` is set,
otherwise Gemini if ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` is set.

Usage::

  pip install -r requirements/requirements-bench-openai.txt
  python scripts/hiveclaw_server.py --host 127.0.0.1 --port 8080   # terminal A
  export GEMINI_API_KEY=...
  python examples/demo_triple_threat.py --cloud-provider gemini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EXAMPLES_DIR = Path(__file__).resolve().parent
_DEMO_TARGET = _EXAMPLES_DIR / "demo_target.py"

# Approximate list prices (USD per token); update when vendors change pricing.
_GPT4O_MINI_INPUT_PER_TOKEN = 0.15 / 1_000_000
_GPT4O_MINI_OUTPUT_PER_TOKEN = 0.60 / 1_000_000
# Gemini 2.0 Flash–class (approximate; verify on Google AI pricing).
_GEMINI_FLASH_INPUT_PER_TOKEN = 0.10 / 1_000_000
_GEMINI_FLASH_OUTPUT_PER_TOKEN = 0.40 / 1_000_000

_OUTPUT_CONTRACT = (
    "Output contract: EXACTLY ONE ```python fenced block containing the full file. "
    "No text before or after the fence. "
    "Forbidden: eval(), exec(), subprocess calls with shell=True, open() without a "
    "context manager (use ``with open(...)``). "
    "Preserve all function names and public interfaces."
)

AGENTS: list[tuple[str, str]] = [
    (
        "Architect",
        "You are the Architect. Improve module structure, naming, and readability. "
        "Do not remove security-critical rewrites from later agents; preserve intent. "
        f"{_OUTPUT_CONTRACT}",
    ),
    (
        "Security Sentinel",
        "You are the Security Sentinel. Remove dangerous patterns: eval on user input, "
        "subprocess with shell=True on interpolated strings, unsafe file IO. "
        "Replace with safe alternatives. "
        f"{_OUTPUT_CONTRACT}",
    ),
    (
        "Performance Optimizer",
        "You are the Performance Optimizer. Fix obvious O(n^2) patterns where a set or "
        "dict suffices. Add type hints and concise docstrings for public functions. "
        f"{_OUTPUT_CONTRACT}",
    ),
]


@dataclass
class PathMetrics:
    label: str
    wall_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    rounds: int
    transcripts: list[dict[str, Any]]
    cloud_provider: Literal["openai", "gemini"] | None = None
    # Quality gate (0.0–1.0 pass rate over gated turns; 1.0 when gate disabled).
    gate_pass_rate: float = 1.0
    total_retries: int = 0


def _resolve_quality_profile(spec: str) -> Path:
    """Profile name (e.g. python_refactor) or path to a YAML file."""
    p = Path(spec)
    if p.is_file():
        return p.resolve()
    name = spec.removesuffix(".yaml")
    cand = _REPO_ROOT / "quality_gate" / "quality_profiles" / f"{name}.yaml"
    if cand.is_file():
        return cand.resolve()
    raise SystemExit(f"Quality profile not found: {spec!r} (tried {cand})")


def _load_source() -> str:
    if not _DEMO_TARGET.is_file():
        raise SystemExit(f"Missing {_DEMO_TARGET}")
    return _DEMO_TARGET.read_text(encoding="utf-8")


def _extract_python_fenced(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _estimate_openai_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        float(prompt_tokens) * _GPT4O_MINI_INPUT_PER_TOKEN
        + float(completion_tokens) * _GPT4O_MINI_OUTPUT_PER_TOKEN
    )


def _estimate_gemini_flash_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        float(prompt_tokens) * _GEMINI_FLASH_INPUT_PER_TOKEN
        + float(completion_tokens) * _GEMINI_FLASH_OUTPUT_PER_TOKEN
    )


def _run_openai_sdk_path(
    *,
    label: str,
    client: Any,
    model: str,
    max_tokens: int,
    rounds: int,
    original_source: str,
    compact_prompts: bool,
    estimate_cloud_cost: bool,
    cloud_provider: Literal["openai", "gemini"] | None,
    quality: Any | None,
) -> PathMetrics:
    from quality_gate.quality_controller import QualityController

    t_wall0 = time.perf_counter()
    sum_prompt = 0
    sum_completion = 0
    sum_total = 0
    transcripts: list[dict[str, Any]] = []

    code_state = original_source
    prior_transcript = ""

    gate_passes = 0
    gate_turns = 0
    total_retries = 0

    for _r in range(rounds):
        for role_name, system_content in AGENTS:
            if compact_prompts:
                user_content = (
                    "Refactor pipeline step. Output ONLY a ```python block with the full file.\n\n"
                    "Current file:\n```python\n"
                    f"{code_state.strip()}\n```"
                )
            else:
                user_content = (
                    "Refactor pipeline step. Output ONLY a ```python block with the full file.\n\n"
                    "Original file (reference):\n```python\n"
                    f"{original_source.strip()}\n```\n\n"
                    "--- Prior agent outputs (full transcript; coordination context) ---\n"
                    f"{prior_transcript if prior_transcript.strip() else '(none yet)'}\n"
                )

            t0 = time.perf_counter()
            if quality is not None:
                assert isinstance(quality, QualityController)

                def call_fn(user_msg: str) -> tuple[str, Any]:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_msg},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.3,
                        stream=False,
                    )
                    choice = resp.choices[0]
                    t = (choice.message.content or "").strip()
                    return t, resp.usage

                q_result = quality.run_turn(
                    call_fn,
                    user_content,
                    label=label,
                    role_name=role_name,
                )
                text = q_result.last_assistant_text
                for snap in q_result.usage_snapshots:
                    pt = int(snap.get("prompt_tokens") or 0)
                    ct = int(snap.get("completion_tokens") or 0)
                    sum_prompt += pt
                    sum_completion += ct
                    tt = snap.get("total_tokens")
                    if tt is not None:
                        sum_total += int(tt)
                    else:
                        sum_total += pt + ct
                gate_turns += 1
                total_retries += max(0, q_result.attempts_used - 1)
                if q_result.final_decision == "ACCEPT":
                    gate_passes += 1
                q_meta = {
                    "attempts_used": q_result.attempts_used,
                    "final_decision": q_result.final_decision,
                    "reports": [r.to_dict() for r in q_result.reports],
                }
            else:
                q_meta = None
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,
                    stream=False,
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                usage = resp.usage
                if usage is not None:
                    sum_prompt += int(usage.prompt_tokens or 0)
                    sum_completion += int(usage.completion_tokens or 0)
                    sum_total += int(usage.total_tokens or 0)
            dt = time.perf_counter() - t0

            print(
                json.dumps(
                    {
                        "event": "demo_triple_threat_turn",
                        "path": label,
                        "role": role_name,
                        "turn_wall_s": round(dt, 3),
                    },
                    separators=(",", ":"),
                ),
                file=sys.stderr,
            )
            print(f"\n===== [{label}] {role_name} =====\n{text}\n", file=sys.stderr)

            entry: dict[str, Any] = {
                "role": role_name,
                "assistant": text,
                "turn_wall_s": dt,
            }
            if q_meta is not None:
                entry["quality"] = q_meta
            transcripts.append(entry)

            if quality is not None:
                extracted = (
                    q_result.accepted_code
                    if q_result.final_decision == "ACCEPT"
                    else None
                )
                if extracted:
                    code_state = extracted
                elif q_result.accepted_code and quality.effective_report_only():
                    code_state = q_result.accepted_code
            else:
                extracted = _extract_python_fenced(text)
                if extracted:
                    code_state = extracted
            if not compact_prompts:
                prior_transcript += f"\n## {role_name}\n{text}\n"

    wall = time.perf_counter() - t_wall0
    cost = 0.0
    if estimate_cloud_cost and cloud_provider == "openai":
        cost = _estimate_openai_cost(sum_prompt, sum_completion)

    gate_pass_rate = (gate_passes / gate_turns) if gate_turns else 1.0

    return PathMetrics(
        label=label,
        wall_s=wall,
        prompt_tokens=sum_prompt,
        completion_tokens=sum_completion,
        total_tokens=sum_total,
        estimated_cost_usd=cost,
        rounds=rounds,
        transcripts=transcripts,
        cloud_provider=cloud_provider if not compact_prompts else None,
        gate_pass_rate=gate_pass_rate,
        total_retries=total_retries,
    )


def _run_gemini_path(
    *,
    label: str,
    api_key: str,
    model: str,
    max_tokens: int,
    rounds: int,
    original_source: str,
    compact_prompts: bool,
    estimate_cloud_cost: bool,
    quality: Any | None,
) -> PathMetrics:
    from google import genai
    from google.genai import types

    from quality_gate.quality_controller import QualityController

    client = genai.Client(api_key=api_key)
    t_wall0 = time.perf_counter()
    sum_prompt = 0
    sum_completion = 0
    sum_total = 0
    transcripts: list[dict[str, Any]] = []

    code_state = original_source
    prior_transcript = ""

    gate_passes = 0
    gate_turns = 0
    total_retries = 0

    for _r in range(rounds):
        for role_name, system_content in AGENTS:
            if compact_prompts:
                user_content = (
                    "Refactor pipeline step. Output ONLY a ```python block with the full file.\n\n"
                    "Current file:\n```python\n"
                    f"{code_state.strip()}\n```"
                )
            else:
                user_content = (
                    "Refactor pipeline step. Output ONLY a ```python block with the full file.\n\n"
                    "Original file (reference):\n```python\n"
                    f"{original_source.strip()}\n```\n\n"
                    "--- Prior agent outputs (full transcript; coordination context) ---\n"
                    f"{prior_transcript if prior_transcript.strip() else '(none yet)'}\n"
                )

            t0 = time.perf_counter()
            if quality is not None:
                assert isinstance(quality, QualityController)

                def call_fn(user_msg: str) -> tuple[str, Any]:
                    r = client.models.generate_content(
                        model=model,
                        contents=user_msg,
                        config=types.GenerateContentConfig(
                            system_instruction=system_content,
                            max_output_tokens=max_tokens,
                            temperature=0.3,
                        ),
                    )
                    t = (r.text or "").strip()
                    return t, r.usage_metadata

                q_result = quality.run_turn(
                    call_fn,
                    user_content,
                    label=label,
                    role_name=role_name,
                )
                text = q_result.last_assistant_text
                for snap in q_result.usage_snapshots:
                    pt = int(snap.get("prompt_tokens") or 0)
                    ct = int(snap.get("completion_tokens") or 0)
                    sum_prompt += pt
                    sum_completion += ct
                    tt = snap.get("total_tokens")
                    if tt is not None:
                        sum_total += int(tt)
                    else:
                        sum_total += pt + ct
                gate_turns += 1
                total_retries += max(0, q_result.attempts_used - 1)
                if q_result.final_decision == "ACCEPT":
                    gate_passes += 1
                q_meta = {
                    "attempts_used": q_result.attempts_used,
                    "final_decision": q_result.final_decision,
                    "reports": [r.to_dict() for r in q_result.reports],
                }
            else:
                q_meta = None
                resp = client.models.generate_content(
                    model=model,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_content,
                        max_output_tokens=max_tokens,
                        temperature=0.3,
                    ),
                )
                text = (resp.text or "").strip()
                um = resp.usage_metadata
                if um is not None:
                    sum_prompt += int(um.prompt_token_count or 0)
                    sum_completion += int(um.candidates_token_count or 0)
                    tot = um.total_token_count
                    if tot is not None:
                        sum_total += int(tot)
                    else:
                        sum_total += int(um.prompt_token_count or 0) + int(
                            um.candidates_token_count or 0
                        )
            dt = time.perf_counter() - t0

            print(
                json.dumps(
                    {
                        "event": "demo_triple_threat_turn",
                        "path": label,
                        "role": role_name,
                        "turn_wall_s": round(dt, 3),
                    },
                    separators=(",", ":"),
                ),
                file=sys.stderr,
            )
            print(f"\n===== [{label}] {role_name} =====\n{text}\n", file=sys.stderr)

            entry: dict[str, Any] = {
                "role": role_name,
                "assistant": text,
                "turn_wall_s": dt,
            }
            if q_meta is not None:
                entry["quality"] = q_meta
            transcripts.append(entry)

            if quality is not None:
                extracted = (
                    q_result.accepted_code
                    if q_result.final_decision == "ACCEPT"
                    else None
                )
                if extracted:
                    code_state = extracted
                elif q_result.accepted_code and quality.effective_report_only():
                    code_state = q_result.accepted_code
            else:
                extracted = _extract_python_fenced(text)
                if extracted:
                    code_state = extracted
            if not compact_prompts:
                prior_transcript += f"\n## {role_name}\n{text}\n"

    wall = time.perf_counter() - t_wall0
    cost = (
        _estimate_gemini_flash_cost(sum_prompt, sum_completion)
        if estimate_cloud_cost
        else 0.0
    )

    gate_pass_rate = (gate_passes / gate_turns) if gate_turns else 1.0

    return PathMetrics(
        label=label,
        wall_s=wall,
        prompt_tokens=sum_prompt,
        completion_tokens=sum_completion,
        total_tokens=sum_total,
        estimated_cost_usd=cost,
        rounds=rounds,
        transcripts=transcripts,
        cloud_provider="gemini" if not compact_prompts else None,
        gate_pass_rate=gate_pass_rate,
        total_retries=total_retries,
    )


def _print_table(hive: PathMetrics | None, cloud: PathMetrics | None) -> None:
    def row(lab: str, w: str, p: str, c: str, t: str, money: str) -> None:
        print(f"{lab:32} | {w:>8} | {p:>12} | {c:>16} | {t:>11} | {money}")

    print()
    row("Phase", "wall_s", "prompt_tok", "completion_tok", "total_tok", "cost_usd")
    print("-" * 99)
    if hive:
        row(
            "HiveClaw (local)",
            f"{hive.wall_s:.3f}",
            str(hive.prompt_tokens),
            str(hive.completion_tokens),
            str(hive.total_tokens),
            f"${hive.estimated_cost_usd:.6f}",
        )
    else:
        print(f"{'HiveClaw (local)':32} | {'skipped':>8} |")
    if cloud:
        prov = (cloud.cloud_provider or "cloud").upper()
        row(
            f"Cloud ({prov} {cloud.label})",
            f"{cloud.wall_s:.3f}",
            str(cloud.prompt_tokens),
            str(cloud.completion_tokens),
            str(cloud.total_tokens),
            f"${cloud.estimated_cost_usd:.6f}",
        )
    else:
        print(f"{'Cloud (OpenAI/Gemini)':32} | {'skipped':>8} |")
    print("-" * 99)
    if hive and cloud and hive.wall_s > 0 and cloud.wall_s > 0:
        sp = cloud.wall_s / hive.wall_s
        print(f"Wall-clock speedup (cloud_wall / hiveclaw_wall): {sp:.2f}x")
        tok_delta = cloud.total_tokens - hive.total_tokens
        print(f"Total token delta (cloud - hiveclaw): {tok_delta:+d}")
    if hive:
        print(
            f"HiveClaw quality: gate_pass_rate={hive.gate_pass_rate:.3f} "
            f"total_retries={hive.total_retries}",
            flush=True,
        )
    if cloud:
        print(
            f"Cloud quality: gate_pass_rate={cloud.gate_pass_rate:.3f} "
            f"total_retries={cloud.total_retries}",
            flush=True,
        )
    print()
    print(
        "Note: Cloud leg uses a growing prior-transcript prompt (coordination tax). "
        "HiveClaw leg uses compact rolling code state only.",
        flush=True,
    )


def _resolve_cloud_provider(
    args: argparse.Namespace,
) -> tuple[Literal["openai", "gemini"] | None, str]:
    """Returns (provider or None if skip cloud, error_message)."""
    if args.hiveclaw_only:
        return None, ""
    if args.openai_only:
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            return None, "OPENAI_API_KEY required for --openai-only"
        return "openai", ""
    if args.gemini_only:
        if not (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        ):
            return None, "GEMINI_API_KEY or GOOGLE_API_KEY required for --gemini-only"
        return "gemini", ""

    mode = args.cloud_provider
    oa = os.environ.get("OPENAI_API_KEY", "").strip()
    gm = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get(
        "GOOGLE_API_KEY", ""
    ).strip()

    if mode == "openai":
        if not oa:
            return None, "OPENAI_API_KEY required for --cloud-provider openai"
        return "openai", ""
    if mode == "gemini":
        if not gm:
            return None, "GEMINI_API_KEY or GOOGLE_API_KEY required for --cloud-provider gemini"
        return "gemini", ""

    # auto
    if oa:
        return "openai", ""
    if gm:
        return "gemini", ""
    return None, ""


def main() -> int:
    p = argparse.ArgumentParser(
        description="Triple-Threat Refactor: HiveClaw vs OpenAI or Gemini metrics"
    )
    p.add_argument(
        "--hiveclaw-base-url",
        default="http://127.0.0.1:8080/v1",
        help="OpenAI-compatible base URL",
    )
    p.add_argument("--hiveclaw-model", default="hiveclaw-llama-1b")
    p.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    p.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="Or OPENAI_CHEAP_MODEL",
    )
    p.add_argument(
        "--gemini-model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        help="Gemini model id (default: gemini-2.0-flash or env GEMINI_MODEL)",
    )
    p.add_argument(
        "--cloud-provider",
        choices=("auto", "openai", "gemini"),
        default="auto",
        help="Cloud backend: auto prefers OpenAI if OPENAI_API_KEY is set, else Gemini",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--rounds", type=int, default=1, help="Repeat full 3-agent pipeline")
    p.add_argument("--hiveclaw-only", action="store_true")
    p.add_argument(
        "--openai-only",
        action="store_true",
        help="Run only OpenAI cloud leg (no HiveClaw)",
    )
    p.add_argument(
        "--gemini-only",
        action="store_true",
        help="Run only Gemini cloud leg (no HiveClaw)",
    )
    p.add_argument("--json-out", type=str, default="")
    p.add_argument(
        "--quality-profile",
        default="python_refactor",
        help="Quality profile name (e.g. python_refactor) or path to a YAML file",
    )
    p.add_argument(
        "--quality-report-only",
        action="store_true",
        help="Log violations without blocking (overrides profile report_only)",
    )
    args = p.parse_args()

    if args.hiveclaw_only and (args.openai_only or args.gemini_only):
        print(
            "Cannot combine --hiveclaw-only with --openai-only / --gemini-only",
            file=sys.stderr,
        )
        return 2
    if args.openai_only and args.gemini_only:
        print("Cannot use both --openai-only and --gemini-only", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit(
            "pip install -r requirements/requirements-bench-openai.txt"
        ) from e

    from quality_gate.quality_controller import QualityController, QualityGateFailure

    try:
        profile_path = _resolve_quality_profile(args.quality_profile)
        quality = QualityController(
            profile_path,
            report_only=True if args.quality_report_only else None,
        )
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 2

    openai_model = os.environ.get("OPENAI_CHEAP_MODEL", "").strip() or args.openai_model
    gemini_model = args.gemini_model.strip()

    src = _load_source()
    hive_metrics: PathMetrics | None = None
    cloud_metrics: PathMetrics | None = None

    prov, prov_err = _resolve_cloud_provider(args)
    if prov_err:
        print(f"[cloud] {prov_err}", file=sys.stderr)
        return 2

    try:
        if not args.openai_only and not args.gemini_only:
            hc = OpenAI(base_url=args.hiveclaw_base_url.rstrip("/"), api_key="sk-demo")
            hive_metrics = _run_openai_sdk_path(
                label="HiveClaw",
                client=hc,
                model=args.hiveclaw_model,
                max_tokens=args.max_tokens,
                rounds=args.rounds,
                original_source=src,
                compact_prompts=True,
                estimate_cloud_cost=False,
                cloud_provider=None,
                quality=quality,
            )

        if prov == "openai":
            oa_client = OpenAI(
                base_url=args.openai_base_url.rstrip("/"),
                api_key=os.environ["OPENAI_API_KEY"].strip(),
            )
            cloud_metrics = _run_openai_sdk_path(
                label=openai_model,
                client=oa_client,
                model=openai_model,
                max_tokens=args.max_tokens,
                rounds=args.rounds,
                original_source=src,
                compact_prompts=False,
                estimate_cloud_cost=True,
                cloud_provider="openai",
                quality=quality,
            )
        elif prov == "gemini":
            try:
                from google import genai  # noqa: F401
            except ImportError as e:
                raise SystemExit(
                    "Gemini requires: pip install -r requirements/requirements-bench-openai.txt"
                ) from e
            gkey = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get(
                "GOOGLE_API_KEY", ""
            ).strip()
            cloud_metrics = _run_gemini_path(
                label=gemini_model,
                api_key=gkey,
                model=gemini_model,
                max_tokens=args.max_tokens,
                rounds=args.rounds,
                original_source=src,
                compact_prompts=False,
                estimate_cloud_cost=True,
                quality=quality,
            )

    except QualityGateFailure as e:
        print(
            json.dumps(
                {
                    "event": "demo_triple_threat_quality_failure",
                    "message": str(e),
                    "last_reports": [r.to_dict() for r in e.reports[-3:]],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3

    if (
        not args.hiveclaw_only
        and not args.openai_only
        and not args.gemini_only
        and prov is None
    ):
        print(
            "[cloud] skip: set OPENAI_API_KEY or GEMINI_API_KEY/GOOGLE_API_KEY, "
            "or use --hiveclaw-only / --cloud-provider …",
            file=sys.stderr,
        )

    def _metrics_dict(m: PathMetrics | None) -> dict[str, Any] | None:
        if m is None:
            return None
        d = asdict(m)
        d.pop("transcripts", None)
        return d

    summary = {
        "event": "demo_triple_threat_summary",
        "hiveclaw": _metrics_dict(hive_metrics),
        "cloud": _metrics_dict(cloud_metrics),
        "cloud_provider": prov,
    }

    print(json.dumps(summary, indent=2))
    _print_table(hive_metrics, cloud_metrics)

    if args.json_out.strip():
        full = {
            "summary": summary,
            "hiveclaw_transcripts": [
                t for t in (hive_metrics.transcripts if hive_metrics else [])
            ],
            "cloud_transcripts": [
                t for t in (cloud_metrics.transcripts if cloud_metrics else [])
            ],
        }
        Path(args.json_out).write_text(json.dumps(full, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
