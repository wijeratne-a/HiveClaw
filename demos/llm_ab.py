"""Shared Agents A/B completion + JSON issue extraction for Repo Pulse."""

from __future__ import annotations

from typing import Any

import httpx

from demos.json_utils import parse_issue_dicts
from demos.llm_client import stream_chat_completion

AB_TEMPERATURE = 0.1
AB_EMPTY_RETRY_MIN_CHARS = 40


def complete_issue_agent_round(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
) -> tuple[str, float, dict[str, int], int, list[dict[str, Any]]]:
    """
    Stream one A/B completion; if output is non-empty but JSON parses to no dicts,
    retry once with a repair suffix (bounded cost).
    """
    text, ttft, usage, cc = stream_chat_completion(
        client,
        base_url=base_url,
        model=model,
        system=system,
        user=user,
        temperature=AB_TEMPERATURE,
    )
    issues = parse_issue_dicts(text)
    if not issues and len(text.strip()) >= AB_EMPTY_RETRY_MIN_CHARS:
        user_r = user + "\n\nRepair: output a single valid JSON array only. No markdown fences or prose."
        text2, ttft2, usage2, cc2 = stream_chat_completion(
            client,
            base_url=base_url,
            model=model,
            system=system,
            user=user_r,
            temperature=AB_TEMPERATURE,
        )
        for k in usage:
            usage[k] = usage[k] + usage2[k]
        issues = parse_issue_dicts(text2)
        text = text2
        ttft = ttft2
        cc = cc + cc2
    return text, ttft, usage, cc, issues
