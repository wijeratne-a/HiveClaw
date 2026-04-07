#!/usr/bin/env python3
"""
Compare wall-clock + TTFT: LangChain-orchestrated Ollama (string handoff) vs HiveClaw HTTP (stream).

Requires:
  - HiveClaw: `hiveclaw-server` on --hiveclaw-url (default http://127.0.0.1:8080/v1)
  - LangChain path: Ollama on --ollama-url (default http://127.0.0.1:11434) and:
        pip install langchain-core langchain-community openai

Raw results append to benchmarks/results/cursor_simulation_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _results_path() -> Path:
    d = _repo_root() / "benchmarks" / "results"
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return d / f"cursor_simulation_{ts}.csv"


def run_ollama_langchain(
    *,
    user_prompt: str,
    ollama_model: str,
    ollama_host: str,
) -> tuple[float, int, int, str]:
    """Two-step chain (coder -> reviewer) via LangChain + ChatOllama. Returns wall_s, p_tok, c_tok, text."""
    try:
        from langchain_community.chat_models import ChatOllama
        from langchain_core.messages import HumanMessage
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "LangChain + Ollama path needs: pip install langchain-core langchain-community"
        ) from e

    base = ollama_host.rstrip("/")
    llm = ChatOllama(model=ollama_model, base_url=base, temperature=0.8)

    t0 = time.perf_counter()
    m1 = HumanMessage(
        content=(
            "You are the coder. Write a short answer (under 120 words) to:\n\n"
            f"{user_prompt}"
        )
    )
    r1 = llm.invoke([m1])
    coder_text = str(getattr(r1, "content", r1))

    m2 = HumanMessage(
        content=(
            "You are the reviewer. Improve concision and fix any issues in this draft:\n\n"
            f"{coder_text}\n\nReply with the final version only."
        )
    )
    r2 = llm.invoke([m2])
    final = str(getattr(r2, "content", r2))
    wall = time.perf_counter() - t0

    meta = getattr(r2, "response_metadata", None) or {}
    pt = int(meta.get("prompt_eval_count", 0) or 0)
    ct = int(meta.get("eval_count", 0) or 0)
    if pt == 0 and ct == 0:
        pt = max(1, len(coder_text) // 4 + len(final) // 4)
        ct = max(1, len(final) // 4)
    return wall, pt, ct, final


def run_hiveclaw_stream(
    *,
    user_prompt: str,
    base_url: str,
    model: str,
    max_tokens: int,
) -> tuple[float, float, int, int, str]:
    """OpenAI-compatible streaming chat. Returns wall_s, ttft_s, p_tok, c_tok, text."""
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise SystemExit("HiveClaw path needs: pip install openai") from e

    client = OpenAI(base_url=base_url.rstrip("/"), api_key="local")
    t0 = time.perf_counter()
    ttft: float | None = None
    parts: list[str] = []
    usage_pt = 0
    usage_ct = 0

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    for ch in stream:
        if ttft is None:
            ttft = time.perf_counter() - t0
        choice = ch.choices[0]
        delta = choice.delta
        if delta and delta.content:
            parts.append(delta.content)
        u = getattr(ch, "usage", None)
        if u is not None:
            usage_pt = int(getattr(u, "prompt_tokens", 0) or 0)
            usage_ct = int(getattr(u, "completion_tokens", 0) or 0)

    wall = time.perf_counter() - t0
    text = "".join(parts)
    if usage_ct == 0:
        usage_ct = max(1, len(text) // 4)
    if usage_pt == 0:
        usage_pt = max(1, len(user_prompt) // 4)
    return wall, float(ttft or wall), usage_pt, usage_ct, text


def run_hiveclaw_models_smoke(base_url: str) -> bool:
    """GET /v1/models without extra deps."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    url = f"{root}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5.0) as r:
            json.loads(r.read().decode())
        return True
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompt",
        type=str,
        default="Explain stigmergy in multi-agent systems in 3 bullet points.",
    )
    p.add_argument(
        "--hiveclaw-url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        help="OpenAI base URL for HiveClaw",
    )
    p.add_argument(
        "--hiveclaw-model",
        type=str,
        default="hiveclaw-swarm-8b",
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server (LangChain path)",
    )
    p.add_argument("--ollama-model", type=str, default="llama3.2")
    p.add_argument(
        "--skip-langchain",
        action="store_true",
        help="Only run HiveClaw (no Ollama/LangChain)",
    )
    p.add_argument(
        "--skip-hiveclaw",
        action="store_true",
        help="Only run LangChain/Ollama",
    )
    args = p.parse_args()

    out_csv = _results_path()
    rows: list[dict[str, object]] = []

    if not args.skip_langchain:
        try:
            lc_wall, lc_pt, lc_ct, _ = run_ollama_langchain(
                user_prompt=args.prompt,
                ollama_model=args.ollama_model,
                ollama_host=args.ollama_url,
            )
            print(
                f"LangChain:  {lc_wall:5.1f}s  |  prompt_tokens: {lc_pt}  "
                f"completion_tokens: {lc_ct}"
            )
            rows.append(
                {
                    "path": "langchain_ollama",
                    "wall_s": lc_wall,
                    "ttft_s": "",
                    "prompt_tokens": lc_pt,
                    "completion_tokens": lc_ct,
                }
            )
        except SystemExit as e:
            print(f"LangChain:  skipped ({e})", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"LangChain:  failed ({type(e).__name__}: {e})", file=sys.stderr)

    if not args.skip_hiveclaw:
        if not run_hiveclaw_models_smoke(args.hiveclaw_url):
            print(
                "HiveClaw:  /v1/models unreachable — start hiveclaw-server?",
                file=sys.stderr,
            )
        try:
            hc_wall, hc_ttft, hc_pt, hc_ct, _ = run_hiveclaw_stream(
                user_prompt=args.prompt,
                base_url=args.hiveclaw_url,
                model=args.hiveclaw_model,
                max_tokens=args.max_tokens,
            )
            print(
                f"HiveClaw:   {hc_wall:5.1f}s  |  prompt_tokens: {hc_pt}  "
                f"completion_tokens: {hc_ct}  TTFT: {hc_ttft:.2f}s"
            )
            rows.append(
                {
                    "path": "hiveclaw_http_stream",
                    "wall_s": hc_wall,
                    "ttft_s": hc_ttft,
                    "prompt_tokens": hc_pt,
                    "completion_tokens": hc_ct,
                }
            )
        except SystemExit as e:
            print(f"HiveClaw:   skipped ({e})", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"HiveClaw:   failed ({type(e).__name__}: {e})", file=sys.stderr)

    lc_wall: float | None = None
    hc_wall: float | None = None
    for row in rows:
        if row["path"] == "langchain_ollama":
            lc_wall = float(row["wall_s"])
        elif row["path"] == "hiveclaw_http_stream":
            hc_wall = float(row["wall_s"])
    if lc_wall is not None and hc_wall is not None and hc_wall > 0:
        sp = lc_wall / hc_wall
        print(f"Speedup:    {sp:5.1f}x  (LangChain wall / HiveClaw wall)")

    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
