# ADR: Batched IOSurface v5 Read/Write + Steering (Phase 6)

## Status

Accepted — implementation reference for engineers.

## Context

Single-slot `read_slot_v5` / `write_slot_v5` limit throughput when many agents need slab I/O in one MLX step. This ADR locks contracts for batched Metal kernels, nanobind, Python `SlabClient`, batched steering, tests, and future server work.

## Normative math

**Source of truth:** [`scripts/hiveclaw_steering.py`](../../scripts/hiveclaw_steering.py). Markdown narrative (e.g. `PR2_NEXT.md`) is non-normative if it conflicts with that file.

## Phase 1.1–1.2 — Metal, bindings, `depends`

### File layout

Batched primitives live in `crates/hiveclaw-mlx/src/slab_primitives.cpp` and headers/bindings beside existing v5 IOSurface code.

### `slot_indices`

- Passed as **`mx.array` int32, shape `[B]`** from Python → C++ `std::vector<uint32_t>`.
- **Duplicates rejected in C++** (`std::set` / uniqueness check).
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
- **`BatchedSteeringWrapper`:** holds **`current_batch_slots`** (`mx.array` int32 `[B]`), updated each tick when batch membership changes.
- **`last_steered_h`:** **`[B,1,2048]`** for batched encode + `write_slots`.

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
- **`scripts/test_batched_steering.py`:** B=1 parity vs `steer_hidden`, torn row, B=2 shapes, clamp batch telemetry.

## Backward compatibility

- **`read_slot_v5` / `write_slot_v5`** remain unchanged for `llm_swarm.py`, `intelligence_spike.py`, `integration_test.py`.

---

## Phase 7 — Continuous batching engine (accepted)

**Code:** [`scripts/generate_batch.py`](../../scripts/generate_batch.py), [`scripts/hiveclaw_server.py`](../../scripts/hiveclaw_server.py) (`HIVECLAW_CONTINUOUS_BATCH=1`), [`scripts/hiveclaw_steering.py`](../../scripts/hiveclaw_steering.py) (Steering Sandwich).

### Per-step invariant

1. **Evaluation and eviction (top of step, before `model()`):** Check disconnect flags and queue-full conditions. Eviction is **atomic**: stop sending to client queue → slice Python trackers (`B_active`, `M_keep`) → `release_task`. MLX and the slab **never** touch a released slot in the same step.
2. **EOS:** EOS is observed **after** logits; eviction for that row runs at **next** step top (client still receives the EOS token).
3. **Bucketing (no shrinking):** On first forward of a session, `B_bucket = min(32, next_pow2(B_active_initial))`. If `B_active` shrinks, **`B_bucket` never shrinks** until the batch drains to zero. Prefer masked dummy FLOPs over recompile.
4. **Dummy rows:** `pad_token_id` (or EOS if undefined); **frozen** `seq_length`; position IDs do not advance for dummies. During decode, rows are independent; dummy KV is zero-initialized and updated each step so attention over zero values yields zero contribution to logits (discarded in Python).
5. **Steering Sandwich (before LM head):** Steering does **not** use dummy rows. On `H_orig [B_bucket,1,2048]`: **(a)** slice to `H_active [B_active,1,2048]`; **(b)** `read_slots` with `depends=H_active`; **torn** rows → zero scents before matmul; decode, L2 clamp, add → `H_steered`; **(c)** `write_slots` encode (`max(matmul(h,W_enc.T)+b_enc,0)` → bf16) with `depends=H_steered`; **(d)** pad with zeros to `[B_bucket,1,2048]`; pass to LM head. Dummy logits discarded.
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

- **Cross-bucket logits match (B=1 minimal bucket vs B=4 evicted-to-1 in 8-bucket)** is a **regression sentinel**, not a formal proof; MLX reduction order may drift — use fp32 logits after `mx.eval`, `atol=1e-5` where stable.
- **mlx-lm bump:** Re-audit this section when the pin in **`scripts/requirements-spike.txt`** / server requirements changes (KVCache API).

### Feature flag

- **`HIVECLAW_CONTINUOUS_BATCH`:** default **`0`** (Phase 5 single-stream + `_MLX_LOCK`). Set **`1`** to enable the batch worker path.

### Operations

- Do **not** run Phase 7 MLX tests or batched server load tests while **`harvester.py`** (or another heavy Metal workload) owns the GPU.
- **`make python`** does not run these tests.
