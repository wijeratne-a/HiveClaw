#!/usr/bin/env python3
"""
Phase 5: OpenAI-compatible FastAPI server backed by HiveClaw slab + SAE-steered MLX generation.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from hiveclaw_steering import ActiveSteeringWrapper, CaptureWrapper, check_latent_dim, load_sae
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
SAE_PATH = Path(__file__).resolve().parent.parent / "models/hiveclaw_sae_v1.safetensors"
MAX_CONCURRENT = 1
MAX_CANDIDATE_SLOTS = 512
DEFAULT_MODEL_NAME = "hiveclaw-llama-1b"

# Serialize all MLX + layer-swap work (executor thread vs stream worker thread).
_MLX_LOCK = threading.Lock()

FATAL_LINE = (
    "pheromoned is not running under launchd. From repo root: "
    "`cargo build --release -p hiveclaw-daemon` then `make daemon-load`. "
    "See scripts/README.md."
)


@dataclass
class ServerContext:
    slab_client: Any
    model: Any
    tokenizer: Any
    original_layer: Any
    W_enc: mx.array
    b_enc: mx.array
    b_dec: mx.array
    d_latent: int
    hf_config: dict
    # Phase 7 continuous batching (optional)
    master_queue: Any = None
    batch_worker_thread: Any = None
    batch_worker_stop: Any = None
    max_client_queue_depth: int = 50


def _config_hidden_size(config: dict) -> int:
    if "hidden_size" in config:
        return int(config["hidden_size"])
    tc = config.get("text_config")
    if isinstance(tc, dict) and "hidden_size" in tc:
        return int(tc["hidden_size"])
    raise ValueError("Could not read hidden_size from model config.")


def _encode_messages(tokenizer, messages: list[dict[str, str]]) -> mx.array:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return mx.array(tokenizer.encode(prompt))


def _goal_latent(
    model,
    tokenizer,
    prompt_text: str,
    original_layer,
    W_enc: mx.array,
    b_enc: mx.array,
) -> np.ndarray:
    wrapper = CaptureWrapper(original_layer)
    model.model.layers[-1] = wrapper
    toks = _encode_messages(tokenizer, [{"role": "user", "content": prompt_text}])
    _ = model(toks[None])
    mx.eval(wrapper.captured_h)
    model.model.layers[-1] = original_layer
    h = wrapper.captured_h[:, -1:, :].astype(mx.float32)
    norm = mx.linalg.norm(h, ord=2, axis=-1, keepdims=True)
    h_n = h / (norm + 1e-7)
    z = mx.maximum(mx.matmul(h_n, mx.transpose(W_enc)) + b_enc, 0.0).reshape(-1)
    mx.eval(z)
    return np.array(z, dtype=np.float32)


def _goal_text_from_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        if m.get("role") == "user" and (c := m.get("content")):
            parts.append(str(c))
    if parts:
        return parts[-1]
    for m in messages:
        if c := m.get("content"):
            return str(c)
    return ""


def _cosine_np(vec_bf16: mx.array, goal_f32_1d: np.ndarray) -> float:
    v = np.array(vec_bf16.astype(mx.float32), dtype=np.float64).reshape(-1)
    g = goal_f32_1d.astype(np.float64).reshape(-1)
    nv = np.linalg.norm(v)
    ng = np.linalg.norm(g)
    if nv < 1e-12 or ng < 1e-12:
        return 0.0
    return float(np.dot(v, g) / (nv * ng))


def _claim_slot(ctx: ServerContext, goal_np: np.ndarray) -> int:
    slab = ctx.slab_client
    states = slab.get_slot_states()
    unclaimed = [i for i, s in enumerate(states) if not s["claimed"]]
    if not unclaimed:
        return -1
    if len(unclaimed) > MAX_CANDIDATE_SLOTS:
        random.shuffle(unclaimed)
        unclaimed = unclaimed[:MAX_CANDIDATE_SLOTS]

    scored: list[tuple[float, int]] = []
    for slot in unclaimed:
        scent = slab.read_slot_v5(slot)
        mx.eval(scent)
        scored.append((_cosine_np(scent, goal_np), slot))
    scored.sort(key=lambda t: t[0], reverse=True)
    order = [s for _, s in scored]
    candidates = mx.array(order, dtype=mx.int32)
    claim_res = slab.claim_task(candidates)
    mx.eval(claim_res)
    return int(np.asarray(claim_res).reshape(-1)[0])


def _sync_startup() -> ServerContext:
    try:
        import hiveclaw_python

        slab_client = hiveclaw_python.SlabClient()
    except Exception:
        print(FATAL_LINE, file=sys.stderr)
        raise SystemExit(1) from None

    check_latent_dim(slab_client)
    sae = load_sae(SAE_PATH)
    W_enc = sae["encoder.weight"]
    b_enc = sae["encoder.bias"]
    b_dec = sae["decoder.bias"]

    model, tokenizer, hf_config = load(MODEL_ID, return_config=True)
    try:
        hidden_size = _config_hidden_size(hf_config)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1) from e
    if hidden_size != 2048:
        print(
            f"[ERROR] Model hidden_size={hidden_size}; SAE expects 2048 (Llama 3.2 1B).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    original_layer = model.model.layers[-1]
    d_latent = slab_client.get_latent_dim()
    print(
        f"[hiveclaw_server] MLX + slab ready latent_dim={d_latent} model={MODEL_ID}",
        flush=True,
    )
    return ServerContext(
        slab_client=slab_client,
        model=model,
        tokenizer=tokenizer,
        original_layer=original_layer,
        W_enc=W_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        d_latent=d_latent,
        hf_config=hf_config,
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL_NAME
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=8192)
    stream: bool = False
    temperature: float | None = Field(default=0.8, ge=0.0, le=2.0)
    alpha: float | None = Field(default=0.1, ge=0.0)


def _messages_as_dicts(msgs: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in msgs]


def _sync_chat_completion(
    ctx: ServerContext,
    body: ChatCompletionRequest,
) -> dict[str, Any]:
    with _MLX_LOCK:
        return _sync_chat_completion_unlocked(ctx, body)


def _sync_chat_completion_unlocked(
    ctx: ServerContext,
    body: ChatCompletionRequest,
) -> dict[str, Any]:
    msgs = _messages_as_dicts(body.messages)
    if not msgs:
        raise ValueError("messages must be non-empty")

    tokenizer = ctx.tokenizer
    model = ctx.model
    goal_text = _goal_text_from_messages(msgs)
    if not goal_text.strip():
        raise ValueError("no user/content text for goal latent")

    goal_np = _goal_latent(
        model, tokenizer, goal_text.strip(), ctx.original_layer, ctx.W_enc, ctx.b_enc
    )
    slot = _claim_slot(ctx, goal_np)
    if slot < 0:
        raise RuntimeError("no slab slot available")

    alpha = float(body.alpha if body.alpha is not None else 0.1)
    steering = ActiveSteeringWrapper(
        ctx.original_layer,
        ctx.slab_client,
        ctx.W_enc,
        ctx.b_dec,
        alpha=alpha,
        slot_index=slot,
    )
    model.model.layers[-1] = steering

    encoded = _encode_messages(tokenizer, msgs)
    temp = float(body.temperature if body.temperature is not None else 0.8)
    sampler = make_sampler(temp=temp)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1

    pieces: list[str] = []
    try:
        object.__setattr__(steering, "current_slot", slot)
        for token, _ in generate_step(
            encoded,
            model,
            sampler=sampler,
            max_tokens=int(body.max_tokens),
        ):
            tok_id = int(token.item()) if hasattr(token, "item") else int(token)
            pieces.append(tokenizer.decode([tok_id]))
            if eos_id >= 0 and tok_id == eos_id:
                break
    finally:
        model.model.layers[-1] = ctx.original_layer

    last_h = steering.last_steered_h
    if last_h is not None:
        h_f = last_h.astype(mx.float32)
        latent = mx.maximum(
            mx.matmul(h_f, mx.transpose(ctx.W_enc)) + ctx.b_enc, 0.0
        ).astype(mx.bfloat16)
        latent = latent.reshape(1, 1, ctx.d_latent)
        write_res = ctx.slab_client.write_slot_v5(slot, latent)
        mx.eval(write_res)
    ctx.slab_client.release_task(slot)

    content = "".join(pieces)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    prompt_ids = tokenizer.encode(
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    )
    usage = {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(tokenizer.encode(content)),
        "total_tokens": len(prompt_ids) + len(tokenizer.encode(content)),
    }
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": body.model or DEFAULT_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _sync_stream_chunks(
    ctx: ServerContext,
    body: ChatCompletionRequest,
) -> Iterator[dict[str, Any]]:
    with _MLX_LOCK:
        yield from _sync_stream_chunks_unlocked(ctx, body)


def _sync_stream_chunks_unlocked(
    ctx: ServerContext,
    body: ChatCompletionRequest,
) -> Iterator[dict[str, Any]]:
    msgs = _messages_as_dicts(body.messages)
    if not msgs:
        raise ValueError("messages must be non-empty")

    tokenizer = ctx.tokenizer
    model = ctx.model
    goal_text = _goal_text_from_messages(msgs)
    if not goal_text.strip():
        raise ValueError("no user/content text for goal latent")

    goal_np = _goal_latent(
        model, tokenizer, goal_text.strip(), ctx.original_layer, ctx.W_enc, ctx.b_enc
    )
    slot = _claim_slot(ctx, goal_np)
    if slot < 0:
        raise RuntimeError("no slab slot available")

    alpha = float(body.alpha if body.alpha is not None else 0.1)
    steering = ActiveSteeringWrapper(
        ctx.original_layer,
        ctx.slab_client,
        ctx.W_enc,
        ctx.b_dec,
        alpha=alpha,
        slot_index=slot,
    )
    model.model.layers[-1] = steering

    encoded = _encode_messages(tokenizer, msgs)
    temp = float(body.temperature if body.temperature is not None else 0.8)
    sampler = make_sampler(temp=temp)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = body.model or DEFAULT_MODEL_NAME

    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    try:
        object.__setattr__(steering, "current_slot", slot)
        yield chunk({"role": "assistant", "content": ""})
        for token, _ in generate_step(
            encoded,
            model,
            sampler=sampler,
            max_tokens=int(body.max_tokens),
        ):
            tok_id = int(token.item()) if hasattr(token, "item") else int(token)
            text = tokenizer.decode([tok_id])
            yield chunk({"content": text})
            if eos_id >= 0 and tok_id == eos_id:
                break
        yield chunk({}, finish="stop")
    finally:
        model.model.layers[-1] = ctx.original_layer
        last_h = steering.last_steered_h
        if last_h is not None:
            h_f = last_h.astype(mx.float32)
            latent = mx.maximum(
                mx.matmul(h_f, mx.transpose(ctx.W_enc)) + ctx.b_enc, 0.0
            ).astype(mx.bfloat16)
            latent = latent.reshape(1, 1, ctx.d_latent)
            write_res = ctx.slab_client.write_slot_v5(slot, latent)
            mx.eval(write_res)
        ctx.slab_client.release_task(slot)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    ctx = await loop.run_in_executor(None, _sync_startup)
    app.state.ctx = ctx
    app.state.sem = asyncio.Semaphore(MAX_CONCURRENT)
    app.state.executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="hiveclaw_mlx"
    )
    continuous = os.environ.get("HIVECLAW_CONTINUOUS_BATCH", "0") == "1"
    app.state.continuous_batch = continuous
    if continuous:
        from generate_batch import start_swarm_batch_worker

        ctx.max_client_queue_depth = int(
            os.environ.get("HIVECLAW_MAX_QUEUE_DEPTH", "50")
        )
        mq, th, st = start_swarm_batch_worker(
            ctx,
            max_batch=int(os.environ.get("HIVECLAW_MAX_BATCH", "8")),
            max_client_queue_depth=ctx.max_client_queue_depth,
        )
        ctx.master_queue = mq
        ctx.batch_worker_thread = th
        ctx.batch_worker_stop = st
        print(
            "[hiveclaw_server] Phase 7 continuous batching enabled "
            f"(max_queue_depth={ctx.max_client_queue_depth})",
            flush=True,
        )
    yield
    if continuous and ctx.batch_worker_stop is not None:
        ctx.batch_worker_stop.set()
        if ctx.batch_worker_thread is not None:
            ctx.batch_worker_thread.join(timeout=5.0)
    app.state.executor.shutdown(wait=True)


app = FastAPI(title="HiveClaw Chat API", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    ctx: ServerContext = request.app.state.ctx
    return {"status": "ok", "latent_dim": int(ctx.d_latent)}


@app.get("/v1/slots")
async def v1_slots(request: Request) -> dict[str, Any]:
    ctx: ServerContext = request.app.state.ctx

    def _snapshot() -> list[dict[str, Any]]:
        raw = ctx.slab_client.get_slot_states()
        out: list[dict[str, Any]] = []
        for i, s in enumerate(raw):
            claimed = bool(s["claimed"])
            oid = int(s["owner_id"])
            out.append(
                {
                    "slot": i,
                    "claimed": claimed,
                    "owner_id": oid,
                    "state": "CLAIMED" if claimed else "FREE",
                }
            )
        return out

    loop = asyncio.get_running_loop()
    slots = await loop.run_in_executor(request.app.state.executor, _snapshot)
    return {"slots": slots, "count": len(slots)}


async def _chat_completions_batched_stream(
    request: Request, body: ChatCompletionRequest
) -> EventSourceResponse:
    """Phase 7: enqueue to ``swarm_batch_worker``; stream from per-client asyncio.Queue."""
    from generate_batch import BatchedStreamJob

    ctx: ServerContext = request.app.state.ctx
    if ctx.master_queue is None:
        raise HTTPException(
            status_code=503, detail="continuous batch worker not initialized"
        )

    msgs = _messages_as_dicts(body.messages)
    if not msgs:
        raise HTTPException(status_code=400, detail="messages must be non-empty")

    loop = asyncio.get_running_loop()
    max_q = int(ctx.max_client_queue_depth)
    client_q: asyncio.Queue = asyncio.Queue(maxsize=max_q)
    cancelled = threading.Event()

    job = BatchedStreamJob(
        body=body,
        messages=msgs,
        client_queue=client_q,
        loop=loop,
        cancelled=cancelled,
    )

    await loop.run_in_executor(None, ctx.master_queue.put, job)

    model_name = body.model or DEFAULT_MODEL_NAME
    created = int(time.time())
    cid0 = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    async def event_gen() -> AsyncIterator[dict[str, str]]:
        yield {
            "data": json.dumps(
                {
                    "id": cid0,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        }
        while True:
            if await request.is_disconnected():
                cancelled.set()
            try:
                kind, payload = await asyncio.wait_for(client_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if kind == "done":
                yield {"data": "[DONE]"}
                break
            if kind == "chunk":
                if "error" in payload:
                    yield {"data": json.dumps(payload)}
                    break
                yield {"data": json.dumps(payload)}
                continue
            break

    return EventSourceResponse(event_gen())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    ctx: ServerContext = request.app.state.ctx
    sem: asyncio.Semaphore = request.app.state.sem
    executor: concurrent.futures.ThreadPoolExecutor = request.app.state.executor
    loop = asyncio.get_running_loop()

    if getattr(request.app.state, "continuous_batch", False):
        if not body.stream:
            raise HTTPException(
                status_code=400,
                detail="HIVECLAW_CONTINUOUS_BATCH=1 requires stream=true",
            )
        return await _chat_completions_batched_stream(request, body)

    if body.stream:

        async def event_gen() -> AsyncIterator[dict[str, str]]:
            q: asyncio.Queue = asyncio.Queue()
            loop_ref = asyncio.get_running_loop()

            def worker() -> None:
                try:
                    for item in _sync_stream_chunks(ctx, body):
                        fut = asyncio.run_coroutine_threadsafe(q.put(("chunk", item)), loop_ref)
                        fut.result(timeout=600)
                    fut = asyncio.run_coroutine_threadsafe(q.put(("done", None)), loop_ref)
                    fut.result(timeout=30)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(q.put(("err", e)), loop_ref)

            async with sem:
                t = threading.Thread(target=worker, daemon=True)
                t.start()
                while True:
                    kind, payload = await q.get()
                    if kind == "done":
                        yield {"data": "[DONE]"}
                        break
                    if kind == "err":
                        err = payload
                        err_body = {
                            "error": {
                                "message": str(err),
                                "type": type(err).__name__,
                            }
                        }
                        yield {"data": json.dumps(err_body)}
                        break
                    assert kind == "chunk"
                    yield {"data": json.dumps(payload)}

            t.join(timeout=1.0)

        return EventSourceResponse(event_gen())

    async with sem:
        try:
            result = await loop.run_in_executor(
                executor,
                lambda: _sync_chat_completion(ctx, body),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    return JSONResponse(result)


def main() -> None:
    p = argparse.ArgumentParser(description="HiveClaw OpenAI-compatible chat server (Phase 5)")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
