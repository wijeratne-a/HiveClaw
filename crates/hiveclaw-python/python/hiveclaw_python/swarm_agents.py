"""
Two-agent Coder + Reviewer pipeline (HIVECLAW_TWO_AGENT=1).

Uses two fixed slab slots (configurable via env) and sequential generation:
coder steers on the first slot, then the reviewer continues with an extended
chat transcript on the second slot.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from .steering import ActiveSteeringWrapper


def coder_reviewer_slots() -> tuple[int, int]:
    c = int(os.environ.get("HIVECLAW_TWO_AGENT_CODER_SLOT", "0"))
    r = int(os.environ.get("HIVECLAW_TWO_AGENT_REVIEWER_SLOT", "1"))
    return c, r


def _claim_slot_only(ctx: Any, slot: int) -> bool:
    claim_res = ctx.slab_client.claim_task(mx.array([int(slot)], dtype=mx.int32))
    mx.eval(claim_res)
    got = int(np.asarray(claim_res).reshape(-1)[0])
    return got == int(slot)


def _finalize_steering_write(
    ctx: Any, steering: ActiveSteeringWrapper | None, slot: int
) -> None:
    if steering is None or slot < 0:
        return
    last_h = steering.last_steered_h
    if last_h is not None:
        h_f = last_h.astype(mx.float32)
        latent = mx.maximum(
            mx.matmul(h_f, mx.transpose(ctx.W_enc)) + ctx.b_enc, 0.0
        ).astype(mx.bfloat16)
        latent = latent.reshape(1, 1, ctx.d_latent)
        write_res = ctx.slab_client.write_slot_v5(slot, latent)
        mx.eval(write_res)


def _split_two_agent_budget(max_total: int) -> tuple[int, int]:
    mc = int(
        os.environ.get(
            "HIVECLAW_TWO_AGENT_CODER_TOKENS", str(max(1, max_total // 2))
        )
    )
    mr_env = os.environ.get("HIVECLAW_TWO_AGENT_REVIEWER_TOKENS")
    if mr_env is not None:
        mr = max(1, int(mr_env))
        mc = max(1, max_total - mr)
    else:
        mr = max(1, max_total - mc)
    if mc + mr > max_total:
        mr = max(1, max_total - mc)
    return mc, mr


def two_agent_pipeline(ctx: Any, body: Any, msgs: list[dict[str, str]]) -> tuple[str, int]:
    """Run coder then reviewer; return (final reviewer text, total new tokens across both phases)."""
    from .openai_server import _encode_messages, _goal_latent, _goal_text_from_messages

    tokenizer = ctx.tokenizer
    model = ctx.model
    goal_text = _goal_text_from_messages(msgs).strip()
    if not goal_text:
        raise ValueError("no user/content text for goal latent")

    _ = _goal_latent(
        model, tokenizer, goal_text, ctx.original_layer, ctx.W_enc, ctx.b_enc
    )

    coder_slot, reviewer_slot = coder_reviewer_slots()
    if coder_slot == reviewer_slot:
        raise RuntimeError(
            "HIVECLAW_TWO_AGENT_CODER_SLOT and HIVECLAW_TWO_AGENT_REVIEWER_SLOT must differ"
        )

    alpha = float(body.alpha if body.alpha is not None else 0.1)
    temp = float(body.temperature if body.temperature is not None else 0.8)
    sampler = make_sampler(temp=temp)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1
    max_total = int(body.max_tokens)
    max_coder, max_reviewer = _split_two_agent_budget(max_total)

    steering_c: ActiveSteeringWrapper | None = None
    steering_r: ActiveSteeringWrapper | None = None
    claimed_coder = False
    claimed_rev = False
    n_completion = 0

    try:
        if not _claim_slot_only(ctx, coder_slot):
            raise RuntimeError(f"could not claim coder slab slot {coder_slot}")
        claimed_coder = True
        if not _claim_slot_only(ctx, reviewer_slot):
            raise RuntimeError(f"could not claim reviewer slab slot {reviewer_slot}")
        claimed_rev = True

        encoded = _encode_messages(tokenizer, msgs)
        steering_c = ActiveSteeringWrapper(
            ctx.original_layer,
            ctx.slab_client,
            ctx.W_enc,
            ctx.b_dec,
            alpha=alpha,
            slot_index=coder_slot,
        )
        model.model.layers[-1] = steering_c
        object.__setattr__(steering_c, "current_slot", coder_slot)
        pieces_c: list[str] = []
        for token, _ in generate_step(
            encoded, model, sampler=sampler, max_tokens=max_coder
        ):
            tok_id = int(token.item()) if hasattr(token, "item") else int(token)
            pieces_c.append(tokenizer.decode([tok_id]))
            n_completion += 1
            if eos_id >= 0 and tok_id == eos_id:
                break
        _finalize_steering_write(ctx, steering_c, coder_slot)

        text_c = "".join(pieces_c)
        follow_user = os.environ.get(
            "HIVECLAW_TWO_AGENT_REVIEWER_PROMPT",
            "Review and refine the above. Produce the final answer only.",
        )
        msgs2 = list(msgs) + [
            {"role": "assistant", "content": text_c},
            {"role": "user", "content": follow_user},
        ]
        encoded2 = _encode_messages(tokenizer, msgs2)
        steering_r = ActiveSteeringWrapper(
            ctx.original_layer,
            ctx.slab_client,
            ctx.W_enc,
            ctx.b_dec,
            alpha=alpha,
            slot_index=reviewer_slot,
        )
        model.model.layers[-1] = steering_r
        object.__setattr__(steering_r, "current_slot", reviewer_slot)
        pieces_r: list[str] = []
        for token, _ in generate_step(
            encoded2, model, sampler=sampler, max_tokens=max_reviewer
        ):
            tok_id = int(token.item()) if hasattr(token, "item") else int(token)
            pieces_r.append(tokenizer.decode([tok_id]))
            n_completion += 1
            if eos_id >= 0 and tok_id == eos_id:
                break
        _finalize_steering_write(ctx, steering_r, reviewer_slot)
        return "".join(pieces_r), n_completion
    finally:
        model.model.layers[-1] = ctx.original_layer
        if claimed_rev:
            try:
                ctx.slab_client.release_task(reviewer_slot)
            except Exception:
                pass
        if claimed_coder:
            try:
                ctx.slab_client.release_task(coder_slot)
            except Exception:
                pass


def iter_two_agent_stream_chunks(
    ctx: Any,
    body: Any,
    msgs: list[dict[str, str]],
    chunk: Callable[..., dict[str, Any]],
    completion_count: list[int] | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Same as two_agent_pipeline but yields OpenAI-style chunk dicts for reviewer tokens only
    (coder phase is silent on the wire).
    """
    from .openai_server import _encode_messages, _goal_latent, _goal_text_from_messages

    tokenizer = ctx.tokenizer
    model = ctx.model
    goal_text = _goal_text_from_messages(msgs).strip()
    if not goal_text:
        raise ValueError("no user/content text for goal latent")

    _ = _goal_latent(
        model, tokenizer, goal_text, ctx.original_layer, ctx.W_enc, ctx.b_enc
    )

    coder_slot, reviewer_slot = coder_reviewer_slots()
    if coder_slot == reviewer_slot:
        raise RuntimeError(
            "HIVECLAW_TWO_AGENT_CODER_SLOT and HIVECLAW_TWO_AGENT_REVIEWER_SLOT must differ"
        )

    alpha = float(body.alpha if body.alpha is not None else 0.1)
    temp = float(body.temperature if body.temperature is not None else 0.8)
    sampler = make_sampler(temp=temp)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1
    max_total = int(body.max_tokens)
    max_coder, max_reviewer = _split_two_agent_budget(max_total)

    steering_c: ActiveSteeringWrapper | None = None
    steering_r: ActiveSteeringWrapper | None = None
    claimed_coder = False
    claimed_rev = False

    try:
        if not _claim_slot_only(ctx, coder_slot):
            raise RuntimeError(f"could not claim coder slab slot {coder_slot}")
        claimed_coder = True
        if not _claim_slot_only(ctx, reviewer_slot):
            raise RuntimeError(f"could not claim reviewer slab slot {reviewer_slot}")
        claimed_rev = True

        encoded = _encode_messages(tokenizer, msgs)
        steering_c = ActiveSteeringWrapper(
            ctx.original_layer,
            ctx.slab_client,
            ctx.W_enc,
            ctx.b_dec,
            alpha=alpha,
            slot_index=coder_slot,
        )
        model.model.layers[-1] = steering_c
        object.__setattr__(steering_c, "current_slot", coder_slot)
        pieces_c: list[str] = []
        for token, _ in generate_step(
            encoded, model, sampler=sampler, max_tokens=max_coder
        ):
            tok_id = int(token.item()) if hasattr(token, "item") else int(token)
            pieces_c.append(tokenizer.decode([tok_id]))
            if completion_count is not None:
                completion_count[0] += 1
            if eos_id >= 0 and tok_id == eos_id:
                break
        _finalize_steering_write(ctx, steering_c, coder_slot)

        text_c = "".join(pieces_c)
        follow_user = os.environ.get(
            "HIVECLAW_TWO_AGENT_REVIEWER_PROMPT",
            "Review and refine the above. Produce the final answer only.",
        )
        msgs2 = list(msgs) + [
            {"role": "assistant", "content": text_c},
            {"role": "user", "content": follow_user},
        ]
        encoded2 = _encode_messages(tokenizer, msgs2)
        steering_r = ActiveSteeringWrapper(
            ctx.original_layer,
            ctx.slab_client,
            ctx.W_enc,
            ctx.b_dec,
            alpha=alpha,
            slot_index=reviewer_slot,
        )
        model.model.layers[-1] = steering_r
        object.__setattr__(steering_r, "current_slot", reviewer_slot)
        for token, _ in generate_step(
            encoded2, model, sampler=sampler, max_tokens=max_reviewer
        ):
            tok_id = int(token.item()) if hasattr(token, "item") else int(token)
            text = tokenizer.decode([tok_id])
            if completion_count is not None:
                completion_count[0] += 1
            yield chunk({"content": text})
            if eos_id >= 0 and tok_id == eos_id:
                break
        _finalize_steering_write(ctx, steering_r, reviewer_slot)
    finally:
        model.model.layers[-1] = ctx.original_layer
        if claimed_rev:
            try:
                ctx.slab_client.release_task(reviewer_slot)
            except Exception:
                pass
        if claimed_coder:
            try:
                ctx.slab_client.release_task(coder_slot)
            except Exception:
                pass