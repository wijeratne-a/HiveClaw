# ADR: Batched IOSurface v5 Read/Write + Steering (Phase 6)

## Status

Accepted — implementation reference for engineers.

## Context

Single-slot `read_slot_v5` / `write_slot_v5` limit throughput when many agents need slab I/O in one MLX step. This ADR locks contracts for batched Metal kernels, nanobind, Python `SlabClient`, batched steering, tests, and future server work.

## Normative math

**Source of truth:** [`hiveclaw_python/steering.py`](../../crates/hiveclaw-python/python/hiveclaw_python/steering.py) (repo shims under `scripts/hiveclaw_steering.py` re-export this module). Markdown narrative (e.g. `PR2_NEXT.md`) is non-normative if it conflicts with that file.

## Phase 1.1–1.2 — Metal, bindings, `depends`

### File layout

Batched primitives live in `crates/hiveclaw-mlx/src/slab_primitives.cpp` and headers/bindings beside existing v5 IOSurface code.

### `slot_indices`

- Passed as **`mx.array` int32, shape `[B]`** from Python → C++ `std::vector<uint32_t>` via **value cast** (`int32` **`-1` → `0xFFFFFFFF`** sentinel: no IOSurface access, read row zeros / write no-op / `status=0`).
- **Duplicates rejected in C++** among **real** slot indices only; sentinels skip range and uniqueness checks.
- **Row `i` aligns with batch row `i`** — callers must **not** reorder slots vs latents; optional coalescing is the caller’s responsibility (sorting in `SlabClient` would misalign `depends`).

### Grid / kernels

- **3D grid:** `dispatch_threadgroups((1,1,B), (32,1,1))` — Z = batch index; each threadgroup mirrors single-slot `32×1` threads (8× uint16 per thread = 256 elems).

### Torn epoch (read)

- Per-row: torn → zero that row’s 256-D output, `status[i]=1`.
- Telemetry: one JSON line per batch:  
  `{"event":"torn_epoch_skip_batch","slots":[...],"ts_ns":<int>}`  
  when `HIVECLAW_TELEMETRY != "0"`.

### Write protocol

- Match single-slot v5: bump `front_epoch`, copy 512-byte payload, set `back_epoch` at slot-relative +576.
- **Per-row status** required; never fail the whole batch for one bad row.

### `depends` (read/write)

- **Single optional dependency** tensor; **rank 3 only**; shapes **`[B,1,2048]`** (read path) or **`[B,1,256]`** (write path when used).
- **Strict:** `dep.shape[0] == B`; reject 2D `[B,2048]` etc.

### C++ return ABI

- `std::pair<mlx::core::array, mlx::core::array>` → Python **`tuple[mx.array, mx.array]`** via nanobind.
- **Order:** **`(data, status)`** always.
- `status`: **`uint8`**, shape **`[B]`**:
  - `0` = success  
  - `1` = torn read  
  - `2` = invalid / not claimed (write path)

### Write `depends` standardization

- Optional `depends` on **encoded latent** `[B,1,256]` bf16 so the write op is ordered after encoding.

## Phase 1.3 — Batched steering

- **`steer_hidden_batched`:** `read_slots` → decode `matmul(scents_f32, W_enc) + b_dec` → **L2 clamp** on `alpha * decoded` (per row, same as `steer_hidden`) → add to `h_batch`.
- Telemetry: **`{"event":"poison_clamp_batch","slots":[...],"ts_ns":...}`** (one line per batch when any row clamps).
- Injection: **last token** only: `h[:, -1:, :]`.
- **`BatchedSteeringWrapper`:** holds **`current_batch_slots`** (`mx.array` int32 **`[B_bucket]`** with **`-1`** dummies), updated each tick when batch membership changes.
- **`last_steered_h`:** **`[B_bucket,1,2048]`** (full bucket; dummy rows are zeros). **`last_steered_norm`:** **`[B_bucket,1,1]`** for poison-clamp telemetry (host read after `mx.eval`).

### Encode write-back

- Same as `llm_swarm.py`:  
  `mx.maximum(mx.matmul(h_f, W_enc.T) + b_enc, 0.0)` → bf16 → `write_slots`.

## Phase 1.4–1.5 — Server / OOM / telemetry (future)

- **Milestone 1:** HTTP **`MAX_CONCURRENT=1`**; batched Metal validated in **synthetic Python** only.
- **Future:** hand-rolled MLX KV + continuous batching; `mlx_lm.generate_step` insufficient for dynamic B.
- **SSE disconnect:** `await request.is_disconnected()` in yield loop → signal worker → evict row, release slab slot (bridge async → worker).
- **Re-claim:** max **3** retries or **50 ms** wall time; else requeue / reject.
- **Queue:** max wait (e.g. **5000 ms**), **queue depth > N** → **503**.
- **OOM / MAX_BATCH:** empirical probe at startup (prefill + one decode, grow B); **fixed for process lifetime**.
- **Parity tests:** **`temp=0.0`**, fixed **`mx.random` / `np.random`** seeds; compare within **epsilon** at logits / sequence level.

## Last-mile bindings (nanobind / tests)

1. **`std::pair<array,array>`** → Python tuple **`(data, status)`**.
2. **Greedy + seeds** for parity harness.
3. **`depends` rank 3** only (`[B,1,2048]` or `[B,1,256]`).
4. **Re-claim:** 3 retries or 50 ms (server phase; not required for Phase 6 kernel PR).
5. **Disconnect:** `is_disconnected()` + worker eviction signal (server phase).

## Dashboard

- Parse **`torn_epoch_skip_batch`** and **`poison_clamp_batch`**: count `len(slots)` per line when using `--telemetry-log`.

## Tests

- **Python first**, **daemon required** (real IOSurface).
- **`integration_test.py --batched`:** smoke read/write shapes + status.
- **`tests/test_batched_steering.py`:** B=1 parity vs `steer_hidden`, torn row, B=2 shapes, clamp batch telemetry.

## Backward compatibility

- **`read_slot_v5` / `write_slot_v5`** remain unchanged for `llm_swarm.py`, `intelligence_spike.py`, `integration_test.py`.

---

## Phase 7 — Continuous batching engine (accepted)

**Code:** [`hiveclaw_python/batching/generate_batch.py`](../../crates/hiveclaw-python/python/hiveclaw_python/batching/generate_batch.py), `hiveclaw-server` / [`server_main.py`](../../crates/hiveclaw-python/python/hiveclaw_python/server_main.py) (`HIVECLAW_CONTINUOUS_BATCH=1`), [`hiveclaw_python/steering.py`](../../crates/hiveclaw-python/python/hiveclaw_python/steering.py) (Steering Sandwich).

### Per-step invariant

1. **Evaluation and eviction (top of step, before `model()`):** Check disconnect flags and queue-full conditions. Eviction is **atomic**: stop sending to client queue → slice Python trackers (`B_active`, `M_keep`) → `release_task`. MLX and the slab **never** touch a released slot in the same step.
2. **EOS:** EOS is observed **after** logits; eviction for that row runs at **next** step top (client still receives the EOS token).
3. **Bucketing (no shrinking):** On first forward of a session, `B_bucket = min(32, next_pow2(B_active_initial))`. If `B_active` shrinks, **`B_bucket` never shrinks** until the batch drains to zero. Prefer masked dummy FLOPs over recompile.
4. **Dummy rows:** `pad_token_id` (or EOS if undefined). **Attention:** [`HiveClawKVCache`](../../crates/hiveclaw-python/python/hiveclaw_python/batching/kv_mask.py) (installed on the full-attention cache slot) applies an **additive float16 mask** (`-1e4` on masked positions) so dummy rows are **fully blinded** in SDPA on prefill and decode; real rows also mask **left-padded** key columns. Shared KV length across the batch tensor is unchanged. **Steering sandwich** still pads dummies with zeros before the LM head.
5. **Steering Sandwich (before LM head):** Full-bucket static shape — no `[:B_active]` slice on the batch axis. `batch_slots` is **`[B_bucket]` int32** with **`-1`** sentinel for dummy rows (C++ `0xFFFFFFFF`: read zeros / write no-op). On `H_orig [B_bucket,1,2048]`: **(a)** `read_slots(batch_slots, depends=H_orig)` (full bucket); **torn** rows → zero scents; decode, L2 clamp; **(b)** **active mask** `(batch_slots != -1)` zeroes delta on dummies; **(c)** **Sandwich Gate:** `H_steered` is the steered hidden for real rows and **exact zeros** for dummies (before LM head); **(d)** optional `write_slots` over full bucket (sentinel rows no-op in C++). **(e)** LM head / logits: compile boundary ends at transformer inner; **`logits[:B_active]`** is taken in Python on the eager logits tensor. **`last_steered_norm`:** sidecar **`[B_bucket,1,1]`** float32 on `BatchedSteeringWrapper` for host telemetry (`poison_clamp_batch`) after compiled decode steps.
6. **P95 latency:** Server-side only (time between successive generated tokens for a row); excludes SSE network.

### KV cache (mlx-lm `KVCache`)

- Layout per layer: **`keys` / `values`** shape **`[B, n_kv_heads, S_buf, head_dim]`** (batch axis **0**). **`slice_kv_cache`** uses `mx.take(..., axis=0)` with `M_keep` as **`mx.int32`** on GPU (constructed from a Python index list first).
7. **`M_keep`:** Indices into the **pre-slice** batch; `active_requests = [active[i] for i in M_keep]`. `batch_slots_new[j]` = slot of surviving agent **`j`** for `j in 0..B_active-1`.

### Threading and queues

- **Single MLX thread:** `swarm_batch_worker` is the **only** thread that calls `model()` and `mx.eval`. **No** `mx.eval` on asyncio threads.
- **Master queue:** `threading.Queue` hands work to the worker; FastAPI uses `run_in_executor` to **put** non-blocking from async.
- **`HIVECLAW_MAX_QUEUE_DEPTH`** (Pydantic / env, default **50**): if `client_queue` is full or `put_nowait` raises, run the same eviction path as disconnect; emit `{"error":"Slow consumer"}` (best effort); telemetry `client_evicted_slow_consumer` on stderr.
- **`_released`:** Set-once; first of disconnect / slow-consumer wins; second path no-ops (no double `release_task`).

### Golden / regression tests

- **Cross-bucket logits (masked batch vs B=1):** compare **fp32** logits after `mx.eval` with **`atol=5e-2`, `rtol=1e-2`** and require **greedy argmax parity** (`temperature=0`). Stricter tolerances are flaky under bf16/float16 attention.
- **mlx-lm bump:** Re-audit this section when the pin in **`requirements/requirements-spike.txt`** / server requirements changes (KVCache API).

### Phase 7+ — Compiled decode (default on)

- **`HIVECLAW_COMPILE_DECODE`:** Defaults to **`1`**. After prefill, `generate_batch` attempts **`mx.compile(inner_step, outputs=<KV keys/values per layer + steering.last_steered_norm>, shapeless=True)`** where **`inner_step(ny) = model.model(ny, cache=kv_cache)`** — LM head (`embed_tokens.as_linear` / `lm_head`) stays **outside** the compiled region. Set **`HIVECLAW_COMPILE_DECODE=0`** to skip compilation entirely. Falls back to `shapeless=False`, then **eager** `model(...)` if **`RuntimeError`** during compiled eval (stderr JSON **`eager_fallback`**). Emits **`compile_status`** after each successful (re)compile. **Recompile** after **every** eviction: `mx.take` replaces `keys`/`values` array objects, so `outputs=` must be rebuilt.
- **`HIVECLAW_COMPILE_WARMUP=1`:** Optional one-shot `mx.eval(compiled(dummy_tokens))` via `warmup_compiled_decode` (caller supplies compiled inner fn); may advance cache — intended for dedicated warmup harnesses, not mid-session.
- **Sliding-window / SWA models:** `install_hiveclaw_kv_cache` **always succeeds** on supported mlx-lm caches: replaces **`cache[fa_idx]`** with `HiveClawKVCache` or **`HiveClawRotatingKVCache`** (when the slot holds `RotatingKVCache`), and **`cache[swa_idx]`** the same way when **`swa_idx`** is present. Composite masks intersect base causal / ring masks with HiveClaw left-pad and dummy-row blinds.

### Feature flag

- **`HIVECLAW_CONTINUOUS_BATCH`:** default **`0`** (Phase 5 single-stream + `_MLX_LOCK`). Set **`1`** to enable the batch worker path.

### Operations

- Do **not** run Phase 7 MLX tests or batched server load tests while **`harvester.py`** (or another heavy Metal workload) owns the GPU.
- **`make python`** does not run these tests.
