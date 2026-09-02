"""
OpenAI-compatible streaming chat via httpx (shared by Repo Pulse demo paths).
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

CHAT_TIMEOUT_S = 180.0
DEFAULT_MAX_TOKENS = 512


def stream_chat_completion(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
) -> tuple[str, float, dict[str, int], int]:
    """Return (full_text, ttft_s, usage, coord_context_chars)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    coord_chars = len(system) + len(user)
    t0 = time.perf_counter()
    ttft = 0.0
    got_first = False
    parts: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with client.stream(
        "POST",
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers={"Accept": "text/event-stream"},
        json=body,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or line.startswith(":"):
                continue
            if line == "data: [DONE]":
                break
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "usage" in obj and isinstance(obj["usage"], dict):
                for k in usage:
                    usage[k] = int(obj["usage"].get(k, usage[k]))
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            content = delta.get("content")
            if content:
                if not got_first:
                    ttft = time.perf_counter() - t0
                    got_first = True
                parts.append(str(content))
    return "".join(parts), ttft, usage, coord_chars


def httpx_client() -> httpx.Client:
    return httpx.Client(timeout=CHAT_TIMEOUT_S)
