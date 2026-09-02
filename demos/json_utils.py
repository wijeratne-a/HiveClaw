"""
Extract JSON issue arrays from LLM output for Repo Pulse agents A/B.
"""

from __future__ import annotations

import json
from typing import Any


def _strip_markdown_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if len(lines) < 2:
        return t.strip("`").strip()
    rest = "\n".join(lines[1:])
    fence = rest.rfind("```")
    if fence >= 0:
        rest = rest[:fence]
    return rest.strip()


def extract_json_array(text: str) -> list[Any]:
    """Parse the first JSON array from model output (handles fences and leading prose)."""
    raw = _strip_markdown_fence(text)
    if not raw:
        return []
    start = raw.find("[")
    if start < 0:
        return []
    try:
        val, _ = json.JSONDecoder().raw_decode(raw, start)
    except json.JSONDecodeError:
        return []
    return val if isinstance(val, list) else []


def parse_issue_dicts(text: str) -> list[dict[str, Any]]:
    """Return list of objects from agent A/B JSON array output."""
    arr = extract_json_array(text)
    return [x for x in arr if isinstance(x, dict)]
