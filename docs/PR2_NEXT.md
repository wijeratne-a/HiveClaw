# PR2: zero-copy fused steering (implemented)

- `FusedSteerScent` CustomOp in `crates/hiveclaw-mlx/src/slab_primitives.cpp`: epoch check + L2 ceiling (2.0) on `||alpha * scent||` + `h_step` blend; embedded MSL `fused_steer_bf16` runs on **Metal** by default.
- **GPU path (restored):** `eval_gpu` dispatches **one threadgroup of 256 threads** (`dispatch_threadgroups((1,1,1), (256,1,1))`) to match the kernel’s `thread_index_in_threadgroup` + `tid * 8` coverage over 2048 dims. The auxiliary **status** buffer is kept alive across async completion via an extra `retain()` before `addCompletedHandler` and matching `release()` in the handler (fixes prior `kIOGPUCommandBufferCallbackErrorInvalidResource` from destructor vs in-flight command buffers).
- **Rollback:** set **`HIVECLAW_FUSED_GPU=0`** to force the CPU implementation (same numerics/telemetry, no fused Metal kernel).
- **Telemetry:** after the GPU kernel finishes, the completion handler reads shared `status` and emits stderr JSON for **`torn_epoch_skip`** (status `1`) and **`poison_clamp`** (status `2`); status `0` is silent. **`HIVECLAW_TELEMETRY=0`** suppresses stderr (handler still runs for buffer lifetime).
- Python: `SlabClient.fused_steer` (casts **float16/float32** `h_step` to bf16 for the C++ op, then casts back); `intelligence_spike.py` / `llm_swarm.py` use fused path (no `read_scent_for_steering` on the hot path). Rebuild with **`make python`** after changing the MLX extension.

**Phase 4 / SAE:** runtime `D` from tensor shape (not fixed 2048) remains future work.

See internal roadmap for ordering vs Mach dead-name eviction.
