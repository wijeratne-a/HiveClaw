#!/usr/bin/env python3
"""
LangChain-orchestrated string committee: same task and token metrics as string_swarm_baseline.py.

Uses LangChain (ChatPromptTemplate + Runnable) only for message assembly; generation is mlx_lm
so results stay comparable to the plain baseline and HiveClaw path.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import mlx.core as mx

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "LangChain required: pip install -r scripts/requirements-bench-langchain.txt"
    ) from e

try:
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "mlx_lm required: pip install -r scripts/requirements-server.txt"
    ) from e

from string_swarm_baseline import (
    AGENT_ROLES,
    BenchmarkResult,
    MODEL_ID_DEFAULT,
    MAX_TOKENS_PER_TURN_DEFAULT,
    N_AGENTS_DEFAULT,
    N_ROUNDS_DEFAULT,
    TASK_CODE,
    _build_user_message,
    _coord_tokens_for_turn,
    _encode_messages,
)


def _lc_messages_to_openai(msgs: list[Any]) -> list[dict[str, str]]:
    """Convert LangChain message list to OpenAI-style dicts for the tokenizer."""
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

    if not isinstance(msgs, list) or not msgs or not isinstance(msgs[0], BaseMessage):
        raise TypeError(f"expected list[BaseMessage], got {type(msgs)!r}")

    out: list[dict[str, str]] = []
    for m in msgs:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": str(m.content)})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": str(m.content)})
        else:
            out.append({"role": "user", "content": str(m.content)})
    return out


def run_langchain_baseline(
    *,
    model_id: str = MODEL_ID_DEFAULT,
    n_rounds: int = N_ROUNDS_DEFAULT,
    n_agents: int = N_AGENTS_DEFAULT,
    max_tokens_per_turn: int = MAX_TOKENS_PER_TURN_DEFAULT,
    temperature: float = 0.8,
) -> BenchmarkResult:
    """Same loop as string baseline; prompts flow through LangChain templates."""
    if n_agents > len(AGENT_ROLES):
        raise ValueError(f"n_agents must be <= {len(AGENT_ROLES)}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_content}"),
            ("human", "{user_content}"),
        ]
    )

    model, tokenizer, _hf = load(model_id, return_config=True)
    sampler = make_sampler(temp=temperature)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1

    prior = ""
    total_coord = 0
    total_content = 0
    t0_all = time.perf_counter()
    per_round_ctx: list[int] = []
    per_round_wall: list[float] = []

    for r in range(n_rounds):
        t_round = time.perf_counter()
        max_ctx_this_round = 0
        round_outputs: list[str] = []

        for a in range(n_agents):
            role = AGENT_ROLES[a]
            system_content = (
                f"You are the {role} on a code review committee. "
                "Respond concisely."
            )
            user_content = _build_user_message(TASK_CODE, prior)

            coord_turn, ctx_turn = _coord_tokens_for_turn(
                tokenizer, TASK_CODE, prior, system_content
            )
            total_coord += coord_turn
            max_ctx_this_round = max(max_ctx_this_round, ctx_turn)

            lc_msgs = prompt.format_messages(
                system_content=system_content,
                user_content=user_content,
            )
            messages = _lc_messages_to_openai(lc_msgs)
            encoded = _encode_messages(tokenizer, messages)
            pieces: list[str] = []
            gen_t0 = time.perf_counter()
            for token, _ in generate_step(
                encoded,
                model,
                sampler=sampler,
                max_tokens=max_tokens_per_turn,
            ):
                tok_id = int(token.item()) if hasattr(token, "item") else int(token)
                pieces.append(tokenizer.decode([tok_id]))
                total_content += 1
                if eos_id >= 0 and tok_id == eos_id:
                    break
            gen_ms = (time.perf_counter() - gen_t0) * 1000.0
            text = "".join(pieces).strip()
            round_outputs.append(f"[Round {r + 1} {role}]: {text}")

            sys.stderr.write(
                json.dumps(
                    {
                        "event": "langchain_baseline_round",
                        "round": r + 1,
                        "agent": a,
                        "role": role,
                        "latency_ms": round(gen_ms, 3),
                        "coord_tokens": coord_turn,
                        "ctx_tokens": ctx_turn,
                        "content_tokens": len(pieces),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

        prior = prior + "\n".join(round_outputs) + "\n"
        per_round_ctx.append(max_ctx_this_round)
        per_round_wall.append((time.perf_counter() - t_round) * 1000.0)

    total_wall_ms = (time.perf_counter() - t0_all) * 1000.0
    return BenchmarkResult(
        phase="langchain_string_baseline",
        rounds=n_rounds,
        agents=n_agents,
        total_coord_tokens=total_coord,
        total_content_tokens=total_content,
        total_wall_ms=total_wall_ms,
        per_round_ctx_tokens=per_round_ctx,
        per_round_wall_ms=per_round_wall,
        ok=True,
    )


def main() -> int:
    r = run_langchain_baseline()
    print(json.dumps(r.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
