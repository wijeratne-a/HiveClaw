#!/usr/bin/env python3
"""
String-passing multi-agent baseline: coordination via growing text context (no slab).
Used by benchmarks/benchmark_consensus.py — does NOT import hiveclaw_python.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import mlx.core as mx

try:
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "mlx_lm required: pip install -r requirements/requirements-server.txt"
    ) from e

MODEL_ID_DEFAULT = "mlx-community/Llama-3.2-1B-Instruct-4bit"

TASK_CODE = """\
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
"""

AGENT_ROLES = [
    "correctness reviewer",
    "performance reviewer",
    "readability reviewer",
    "security reviewer",
    "documentation reviewer",
]

N_ROUNDS_DEFAULT = 10
N_AGENTS_DEFAULT = 5
MAX_TOKENS_PER_TURN_DEFAULT = 24


@dataclass
class BenchmarkResult:
    """Aggregated metrics for one benchmark path (baseline or HiveClaw)."""

    phase: str
    rounds: int
    agents: int
    total_coord_tokens: int
    total_content_tokens: int
    total_wall_ms: float
    per_round_ctx_tokens: list[int] = field(default_factory=list)
    per_round_wall_ms: list[float] = field(default_factory=list)
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _encode_messages(tokenizer: Any, messages: list[dict[str, str]]) -> mx.array:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return mx.array(tokenizer.encode(prompt))


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def _build_user_message(code: str, prior: str) -> str:
    return (
        "The following Python code is under review by a committee.\n\n"
        f"```python\n{code.strip()}\n```\n\n"
        "Prior discussion (verbatim from earlier rounds and agents):\n"
        f"{prior if prior.strip() else '(none yet)'}\n\n"
        "Give one brief sentence of analysis from your perspective."
    )


def _coord_tokens_for_turn(
    tokenizer: Any,
    code: str,
    prior: str,
    system_content: str,
) -> tuple[int, int]:
    """
    Returns (coord_tokens, ctx_tokens) for the prompt only.
    Coordination = everything in the rendered chat prompt except the raw code body tokens.
    """
    user_msg = _build_user_message(code, prior)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    ctx_tokens = _token_len(tokenizer, prompt)
    code_tokens = _token_len(tokenizer, code.strip())
    coord_tokens = max(0, ctx_tokens - code_tokens)
    return coord_tokens, ctx_tokens


def run_string_baseline(
    *,
    model_id: str = MODEL_ID_DEFAULT,
    n_rounds: int = N_ROUNDS_DEFAULT,
    n_agents: int = N_AGENTS_DEFAULT,
    max_tokens_per_turn: int = MAX_TOKENS_PER_TURN_DEFAULT,
    temperature: float = 0.8,
) -> BenchmarkResult:
    """Run the string-passing committee benchmark."""
    if n_agents > len(AGENT_ROLES):
        raise ValueError(f"n_agents must be <= {len(AGENT_ROLES)}")

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
            coord_turn, ctx_turn = _coord_tokens_for_turn(
                tokenizer, TASK_CODE, prior, system_content
            )
            total_coord += coord_turn
            max_ctx_this_round = max(max_ctx_this_round, ctx_turn)

            messages = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": _build_user_message(TASK_CODE, prior),
                },
            ]
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
                        "event": "baseline_round",
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
        phase="string_passing_baseline",
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
    r = run_string_baseline()
    print(json.dumps(r.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
