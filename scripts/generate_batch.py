#!/usr/bin/env python3
"""
Phase 7: continuous batching for HiveClaw + mlx-lm.

Single ``swarm_batch_worker`` thread owns ``model()`` and ``mx.eval``.
See ``docs/adr/BATCHED_STEERING_CONTRACT.md`` Phase 7.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from hiveclaw_kv_mask import (
    install_hiveclaw_kv_cache,
    rebuild_hive_kv_metadata,
    sync_hive_metadata_to_fa_cache,
)
from hiveclaw_steering import BatchedSteeringWrapper

try:
    from mlx_lm.models.cache import make_prompt_cache
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mlx_lm is required for generate_batch. pip install -r scripts/requirements-server.txt"
    ) from e


def probe_max_batch(model_id: str, env_max: int, probe_ctx_len: int) -> int:
    """Exponential batch-size probe using a temporary model; returns last successful B (≥1)."""
    from mlx_lm import load as mlx_load
    from mlx_lm.models.cache import make_prompt_cache

    last_ok = 0
    tmp_model = None
    try:
        tmp_model, _ = mlx_load(model_id)
        B = 1
        while B <= env_max:
            try:
                cache = make_prompt_cache(tmp_model)
                dummy_prefill = mx.zeros((B, probe_ctx_len), dtype=mx.int32)
                logits = tmp_model(dummy_prefill, cache=cache)
                mx.eval(logits)
                dummy_decode = mx.zeros((B, 1), dtype=mx.int32)
                logits = tmp_model(dummy_decode, cache=cache)
                mx.eval(logits)
                del cache, logits
                last_ok = B
                B *= 2
            except RuntimeError:
                break
    finally:
        if tmp_model is not None:
            del tmp_model
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()
    return last_ok if last_ok > 0 else 1


def next_pow2_bucket(n: int, cap: int = 32) -> int:
    """Smallest power of two >= n, capped at ``cap``."""
    if n <= 0:
        return 1
    p = 1
    while p < n and p < cap:
        p <<= 1
    return min(p, cap)


def slice_kv_cache_batch_dim(kv_cache: list[Any], m_keep: list[int]) -> None:
    """Slice batch axis 0 of each layer cache (mlx-lm ``KVCache`` / ``RotatingKVCache``)."""
    if not m_keep:
        return
    idx = mx.array(m_keep, dtype=mx.int32)
    for c in kv_cache:
        keys = getattr(c, "keys", None)
        if keys is None:
            continue
        c.keys = mx.take(c.keys, idx, axis=0)
        c.values = mx.take(c.values, idx, axis=0)


def pad_kv_cache_batch_dim(kv_cache: list[Any], B_bucket: int, B_current: int) -> None:
    """Append zero KV rows so batch dim matches ``B_bucket``."""
    if B_current >= B_bucket:
        return
    pad_n = B_bucket - B_current
    for c in kv_cache:
        keys = getattr(c, "keys", None)
        if keys is None:
            continue
        zk = mx.zeros((pad_n,) + tuple(c.keys.shape[1:]), dtype=c.keys.dtype)
        zv = mx.zeros((pad_n,) + tuple(c.values.shape[1:]), dtype=c.values.dtype)
        c.keys = mx.concatenate([c.keys, zk], axis=0)
        c.values = mx.concatenate([c.values, zv], axis=0)


def build_left_padded_batch(
    prompts: list[mx.array],
    pad_id: int,
    B_bucket: int,
) -> mx.array:
    """Stack 1-D token arrays left-padded to max length; pad batch dim to ``B_bucket``.

    When :func:`install_hiveclaw_kv_cache` succeeds, :class:`HiveClawKVCache` supplies
    composite additive masks so padded columns and dummy rows are blinded in attention.
    """
    lens = [int(p.size) for p in prompts]
    max_len = max(lens)
    rows: list[mx.array] = []
    for p in prompts:
        L = int(p.size)
        flat = mx.reshape(p, (L,))
        pl = max_len - L
        if pl > 0:
            pad = mx.full((pl,), pad_id, dtype=flat.dtype)
            row = mx.concatenate([pad, flat], axis=0)
        else:
            row = flat
        rows.append(row)
    x = mx.stack(rows, axis=0)
    B_active = x.shape[0]
    if B_bucket > B_active:
        dummy = mx.full((B_bucket - B_active, max_len), pad_id, dtype=x.dtype)
        x = mx.concatenate([x, dummy], axis=0)
    return x


def _eval_cache_states(kv_cache: list[Any]) -> None:
    states = []
    for c in kv_cache:
        if getattr(c, "keys", None) is not None and c.keys is not None:
            states.append(c.state)
    if states:
        mx.eval(*states)


def _kv_compile_output_arrays(kv_cache: list[Any]) -> list[mx.array]:
    """Arrays mlx-lm mutates in-place during ``model(..., cache=)`` (for ``mx.compile``)."""
    out: list[mx.array] = []
    for c in kv_cache:
        keys = getattr(c, "keys", None)
        if keys is not None:
            out.append(keys)
            out.append(c.values)
    return out


def _build_bucket_slots(entries: list[ChatBatchEntry], B_bucket: int) -> mx.array:
    """``[B_bucket]`` int32: real slot ids then ``-1`` sentinel dummies."""
    slots = [e.slot_id for e in entries] + [-1] * (B_bucket - len(entries))
    return mx.array(slots, dtype=mx.int32)


def _pad_token_row_to_bucket(
    t: mx.array, B_active: int, B_bucket: int, pad_id: int
) -> mx.array:
    """``t`` is ``[B_active]`` last-token ids; return ``[B_bucket]`` with ``pad_id`` tail."""
    if B_active >= B_bucket:
        return t
    tail = mx.full((B_bucket - B_active,), pad_id, dtype=t.dtype)
    return mx.concatenate([t, tail], axis=0)


def _try_compile_inner_step(
    model: Any,
    kv_cache: list[Any],
    steering_wrapper: Any,
) -> Callable[[mx.array], mx.array] | None:
    """Compile ``model.model`` only (transformer inner); LM head stays eager."""
    if os.environ.get("HIVECLAW_COMPILE_DECODE", "0") != "1":
        return None
    outputs = _kv_compile_output_arrays(kv_cache)
    if len(outputs) < 2:
        return None
    outputs.append(steering_wrapper.last_steered_norm)

    def inner_step(ny: mx.array) -> mx.array:
        return model.model(ny, cache=kv_cache)

    try:
        return mx.compile(inner_step, outputs=outputs, shapeless=True)
    except Exception:
        try:
            return mx.compile(inner_step, outputs=outputs, shapeless=False)
        except Exception:
            return None


def warmup_compiled_decode(
    compiled_fn: Callable[[mx.array], mx.array] | None,
    *,
    B_bucket: int,
    pad_id: int,
) -> None:
    """Run one ``mx.eval`` on a compiled decode fn (set ``HIVECLAW_COMPILE_WARMUP=1`` at startup)."""
    if (
        compiled_fn is None
        or os.environ.get("HIVECLAW_COMPILE_WARMUP", "0") != "1"
    ):
        return
    try:
        dummy = mx.full((B_bucket, 1), pad_id, dtype=mx.int32)
        mx.eval(compiled_fn(dummy))
    except Exception:
        pass


@dataclass
class ChatBatchEntry:
    """One client request in a batch session (worker thread reads this)."""

    entry_id: str
    loop: asyncio.AbstractEventLoop
    client_queue: asyncio.Queue
    slot_id: int
    prompt_tokens: mx.array
    max_tokens: int
    temperature: float
    alpha: float
    eos_id: int
    cancelled: threading.Event
    completion_id: str
    model_name: str
    created: int
    tokens_emitted: int = 0
    released: bool = False
    done_enqueued: bool = False


@dataclass
class BatchedStreamJob:
    """Queued by FastAPI; worker converts to ``ChatBatchEntry`` (uses ``model()`` for claim)."""

    body: Any
    messages: list[dict[str, str]]
    client_queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    cancelled: threading.Event


def _openai_chunk(
    e: ChatBatchEntry, delta: dict[str, Any], finish: str | None = None
) -> dict[str, Any]:
    return {
        "id": e.completion_id,
        "object": "chat.completion.chunk",
        "created": e.created,
        "model": e.model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def release_entry_slab(slab_client: Any, e: ChatBatchEntry) -> None:
    if e.released:
        return
    e.released = True
    try:
        slab_client.release_task(e.slot_id)
    except Exception:
        pass


def log_slow_consumer_event(slot_id: int, depth: int) -> None:
    sys.stderr.write(
        json.dumps(
            {
                "event": "client_evicted_slow_consumer",
                "slot": int(slot_id),
                "queue_depth": int(depth),
                "ts_ns": time.time_ns(),
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def push_openai_chunk(
    loop: asyncio.AbstractEventLoop, q: asyncio.Queue, openai_chunk: dict
) -> bool:
    """Queue format matches ``hiveclaw_server`` stream worker: ``(\"chunk\", payload)``."""

    async def _put() -> bool:
        try:
            q.put_nowait(("chunk", openai_chunk))
            return True
        except asyncio.QueueFull:
            return False

    fut = asyncio.run_coroutine_threadsafe(_put(), loop)
    return bool(fut.result(timeout=120))


def push_done(loop: asyncio.AbstractEventLoop, q: asyncio.Queue) -> None:
    async def _put() -> None:
        await q.put(("done", None))

    fut = asyncio.run_coroutine_threadsafe(_put(), loop)
    fut.result(timeout=120)


def _evict_indices(
    entries: list[ChatBatchEntry],
    kv_cache: list[Any],
    slab_client: Any,
    B_bucket: int,
    pad_id: int,
    next_y: mx.array,
    to_drop: set[int],
) -> tuple[mx.array, bool]:
    B_active = len(entries)
    m_keep = [i for i in range(B_active) if i not in to_drop]
    victims = [entries[i] for i in sorted(to_drop)]
    for e in victims:
        if not e.done_enqueued:
            push_openai_chunk(
                e.loop,
                e.client_queue,
                _openai_chunk(e, {}, finish="stop"),
            )
            push_done(e.loop, e.client_queue)
            e.done_enqueued = True
        release_entry_slab(slab_client, e)
    if not m_keep:
        return next_y, True
    slice_kv_cache_batch_dim(kv_cache, m_keep)
    entries[:] = [entries[i] for i in m_keep]
    pad_kv_cache_batch_dim(kv_cache, B_bucket, len(entries))
    _eval_cache_states(kv_cache)
    idx_mx = mx.array(m_keep, dtype=mx.int32)
    ny = mx.take(next_y, idx_mx, axis=0)
    if ny.shape[0] < B_bucket:
        tail = mx.full((B_bucket - ny.shape[0], 1), pad_id, dtype=ny.dtype)
        ny = mx.concatenate([ny, tail], axis=0)
    return ny, False


def _emit_row_tokens(
    tokenizer: Any,
    entries: list[ChatBatchEntry],
    tok_np: np.ndarray,
    max_client_queue_depth: int,
    eos_pending: set[int],
) -> None:
    B_active = len(entries)
    for i in range(B_active):
        e = entries[i]
        if e.tokens_emitted >= e.max_tokens:
            eos_pending.add(i)
            continue
        tid = int(tok_np[i])
        text = tokenizer.decode([tid])
        ok = push_openai_chunk(
            e.loop,
            e.client_queue,
            _openai_chunk(e, {"content": text}),
        )
        if not ok:
            sys.stderr.write(
                json.dumps(
                    {"error": "Slow consumer", "entry_id": e.entry_id},
                    separators=(",", ":"),
                )
                + "\n"
            )
            log_slow_consumer_event(e.slot_id, max_client_queue_depth)
            e.cancelled.set()
            continue
        e.tokens_emitted += 1
        if e.eos_id >= 0 and tid == e.eos_id:
            eos_pending.add(i)


def generate_batch_session(
    *,
    slab_client: Any,
    model: Any,
    tokenizer: Any,
    original_layer: Any,
    W_enc: mx.array,
    b_dec: mx.array,
    b_enc: mx.array,
    d_latent: int,
    entries: list[ChatBatchEntry],
    max_client_queue_depth: int = 50,
) -> None:
    """Run prefill + decode for a batch; mutates ``entries`` (evictions)."""
    if not entries:
        return

    started = list(entries)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    pad_id = int(pad_id)

    B_active0 = len(entries)
    B_bucket = next_pow2_bucket(B_active0)
    prompts = [e.prompt_tokens for e in entries]
    x = build_left_padded_batch(prompts, pad_id, B_bucket)
    cache = make_prompt_cache(model)
    lp, act = rebuild_hive_kv_metadata(entries, B_bucket)
    install_hiveclaw_kv_cache(model, cache, lp, act)
    batch_slots = _build_bucket_slots(entries, B_bucket)
    alpha = float(entries[0].alpha)

    steering = BatchedSteeringWrapper(
        original_layer,
        slab_client,
        W_enc,
        b_dec,
        alpha=alpha,
        batch_slots=batch_slots,
        b_enc=b_enc,
        d_latent=int(d_latent),
    )
    model.model.layers[-1] = steering

    eos_pending: set[int] = set()
    compiled_inner: Callable[[mx.array], mx.array] | None = None

    try:
        logits = model(x, cache=cache)
        _eval_cache_states(cache)
        mx.eval(logits)
        compiled_inner = _try_compile_inner_step(model, cache, steering)

        logits_active = logits[:B_active0]
        t = mx.argmax(logits_active[:, -1, :], axis=-1)
        mx.eval(t)
        full_t = np.asarray(t).reshape(-1)
        tok_line = full_t[:B_active0].copy()

        to_drop0 = {i for i in range(len(entries)) if entries[i].cancelled.is_set()}
        t_b0 = _pad_token_row_to_bucket(t, B_active0, B_bucket, pad_id)
        next_y = mx.reshape(t_b0, (B_bucket, 1))
        if to_drop0:
            next_y, emptied = _evict_indices(
                entries, cache, slab_client, B_bucket, pad_id, next_y, to_drop0
            )
            if emptied:
                return
            m_keep0 = [i for i in range(B_active0) if i not in to_drop0]
            tok_line = tok_line[m_keep0]
            lp0, act0 = rebuild_hive_kv_metadata(entries, B_bucket)
            sync_hive_metadata_to_fa_cache(cache, model, lp0, act0)
            compiled_inner = _try_compile_inner_step(model, cache, steering)

        B_active = len(entries)
        if B_active == 0:
            return
        batch_slots = _build_bucket_slots(entries, B_bucket)
        object.__setattr__(steering, "current_batch_slots", batch_slots)
        _emit_row_tokens(
            tokenizer, entries, tok_line, max_client_queue_depth, eos_pending
        )

        step = 0
        while step < 8192:
            B_active = len(entries)
            if B_active == 0:
                break

            to_drop: set[int] = set()
            for i in range(B_active):
                if entries[i].cancelled.is_set():
                    to_drop.add(i)
            to_drop |= eos_pending
            eos_pending.clear()

            if to_drop:
                next_y, emptied = _evict_indices(
                    entries, cache, slab_client, B_bucket, pad_id, next_y, to_drop
                )
                if emptied:
                    break
                lp1, act1 = rebuild_hive_kv_metadata(entries, B_bucket)
                sync_hive_metadata_to_fa_cache(cache, model, lp1, act1)
                compiled_inner = _try_compile_inner_step(model, cache, steering)

            B_active = len(entries)
            if B_active == 0:
                break

            batch_slots = _build_bucket_slots(entries, B_bucket)
            object.__setattr__(steering, "current_batch_slots", batch_slots)

            if compiled_inner is not None:
                try:
                    hidden = compiled_inner(next_y)
                    mx.eval(hidden)
                    tie = getattr(getattr(model, "args", None), "tie_word_embeddings", False)
                    if tie:
                        logits = model.model.embed_tokens.as_linear(hidden)
                    else:
                        logits = model.lm_head(hidden)
                    mx.eval(logits)
                    if os.environ.get("HIVECLAW_TELEMETRY", "1") != "0":
                        norm_np = np.array(steering.last_steered_norm, dtype=np.float32)
                        real_slots_np = np.array(batch_slots, dtype=np.int32)
                        Bb = int(batch_slots.shape[0])
                        clamped = [
                            int(real_slots_np[i])
                            for i in range(Bb)
                            if int(real_slots_np[i]) != -1
                            and float(norm_np.reshape(-1)[i]) > 2.0
                        ]
                        if clamped:
                            sys.stderr.write(
                                json.dumps(
                                    {
                                        "event": "poison_clamp_batch",
                                        "slots": clamped,
                                        "ts_ns": time.time_ns(),
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                except RuntimeError:
                    compiled_inner = None
                    logits = model(next_y, cache=cache)
                    _eval_cache_states(cache)
                    mx.eval(logits)
            else:
                logits = model(next_y, cache=cache)
                _eval_cache_states(cache)
                mx.eval(logits)

            logits_active = logits[:B_active]
            t = mx.argmax(logits_active[:, -1, :], axis=-1)
            mx.eval(t)
            tok_np = np.asarray(t).reshape(-1)

            _emit_row_tokens(
                tokenizer, entries, tok_np, max_client_queue_depth, eos_pending
            )
            t_b = _pad_token_row_to_bucket(t, B_active, B_bucket, pad_id)
            next_y = mx.reshape(t_b, (B_bucket, 1))
            step += 1

    finally:
        model.model.layers[-1] = original_layer
        for e in started:
            if not e.done_enqueued:
                try:
                    push_done(e.loop, e.client_queue)
                except Exception:
                    pass
                e.done_enqueued = True
            release_entry_slab(slab_client, e)
        entries.clear()


def _server_module() -> Any:
    """Resolve server module whether started as ``uvicorn hiveclaw_server:app`` or ``python hiveclaw_server.py``."""
    import sys

    m = sys.modules.get("hiveclaw_server")
    if m is not None:
        return m
    return sys.modules["__main__"]


def batch_jobs_to_entries(ctx: Any, jobs: list[BatchedStreamJob]) -> list[ChatBatchEntry]:
    """Runs on worker thread; uses the loaded server module for claim / goal helpers."""
    srv = _server_module()

    entries: list[ChatBatchEntry] = []
    for job in jobs:
        if job.cancelled.is_set():
            continue
        msgs = job.messages
        goal_text = srv._goal_text_from_messages(msgs)
        if not goal_text.strip():
            push_openai_chunk(
                job.loop,
                job.client_queue,
                {
                    "error": {
                        "message": "no user/content text for goal latent",
                        "type": "ValueError",
                    }
                },
            )
            push_done(job.loop, job.client_queue)
            continue
        goal_np = srv._goal_latent(
            ctx.model,
            ctx.tokenizer,
            goal_text.strip(),
            ctx.original_layer,
            ctx.W_enc,
            ctx.b_enc,
        )
        slot = srv._claim_slot(ctx, goal_np)
        if slot < 0:
            push_openai_chunk(
                job.loop,
                job.client_queue,
                {
                    "error": {
                        "message": "no slab slot available",
                        "type": "RuntimeError",
                    }
                },
            )
            push_done(job.loop, job.client_queue)
            continue
        prompt_tokens = srv._encode_messages(ctx.tokenizer, msgs)
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model_name = getattr(job.body, "model", None) or srv.DEFAULT_MODEL_NAME
        eos_id = (
            int(ctx.tokenizer.eos_token_id)
            if ctx.tokenizer.eos_token_id is not None
            else -1
        )
        entries.append(
            ChatBatchEntry(
                entry_id=cid,
                loop=job.loop,
                client_queue=job.client_queue,
                slot_id=slot,
                prompt_tokens=prompt_tokens,
                max_tokens=int(job.body.max_tokens),
                temperature=float(
                    job.body.temperature if job.body.temperature is not None else 0.8
                ),
                alpha=float(job.body.alpha if job.body.alpha is not None else 0.1),
                eos_id=eos_id,
                cancelled=job.cancelled,
                completion_id=cid,
                model_name=str(model_name),
                created=created,
            )
        )
    return entries


def swarm_batch_worker(
    *,
    ctx: Any,
    master_queue: queue.Queue,
    stop_event: threading.Event,
    max_batch: int = 8,
    max_client_queue_depth: int = 50,
) -> None:
    """Blocking worker loop (OS thread). ``ctx`` must expose server fields (model, slab, SAE, etc.)."""
    while not stop_event.is_set():
        try:
            first = master_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        batch_jobs: list[BatchedStreamJob] = [first]
        while len(batch_jobs) < max_batch:
            try:
                batch_jobs.append(master_queue.get_nowait())
            except queue.Empty:
                break
        entries: list[ChatBatchEntry] = []
        try:
            entries = batch_jobs_to_entries(ctx, batch_jobs)
            if not entries:
                continue
            generate_batch_session(
                slab_client=ctx.slab_client,
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                original_layer=ctx.original_layer,
                W_enc=ctx.W_enc,
                b_dec=ctx.b_dec,
                b_enc=ctx.b_enc,
                d_latent=int(ctx.d_latent),
                entries=entries,
                max_client_queue_depth=max_client_queue_depth,
            )
        except Exception as e:
            for ent in entries:
                try:
                    push_openai_chunk(
                        ent.loop,
                        ent.client_queue,
                        {
                            "error": {
                                "message": str(e),
                                "type": type(e).__name__,
                            }
                        },
                    )
                except Exception:
                    pass
                try:
                    push_done(ent.loop, ent.client_queue)
                except Exception:
                    pass
                release_entry_slab(ctx.slab_client, ent)
            print(f"[swarm_batch_worker] batch failed: {e}", file=sys.stderr, flush=True)


def start_swarm_batch_worker(
    ctx: Any,
    *,
    max_batch: int = 8,
    max_client_queue_depth: int = 50,
) -> tuple[queue.Queue, threading.Thread, threading.Event]:
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    t = threading.Thread(
        target=swarm_batch_worker,
        kwargs={
            "ctx": ctx,
            "master_queue": q,
            "stop_event": stop,
            "max_batch": max_batch,
            "max_client_queue_depth": max_client_queue_depth,
        },
        name="swarm_batch_worker",
        daemon=True,
    )
    t.start()
    return q, t, stop
