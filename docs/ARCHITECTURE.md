# HiveClaw — implementation architecture

This document is the **technical deep dive** for contributors and operators. The main [README](../README.md) stays jargon-light for a quick product read; the details below cover **IOSurface / XPC**, **slab layout**, **SAE latent geometry**, **continuous batching (Phase 7)**, **burn-in / ironclad** gates, and related env vars.

---

## Terminology (internal names vs industry language)

| Internal / code name | Meaning |
|---------------------|---------|
| **`pheromoned`** | **IPC broker daemon** — the macOS background binary that owns the shared memory surface and brokers access (XPC + IOSurface handoff) for client processes. LaunchAgent service id: `com.hiveclaw.pheromoned`. |
| **Stigmergy** (e.g. `benchmark_stigmergy`, server env `HIVECLAW_STIGMERGY`) | **Latent-cache synchronization** — agents read/write coordination state in shared slab slots instead of sending full text transcripts through a message bus each round. |
| **IOSurface slab** | **Shared memory pool** — a GPU-accessible buffer split into fixed-size **slots**; agents claim, write, and release slots under daemon supervision. |
| **SAE latent (256-D bf16)** | **Compressed coordination vector** — a small bfloat16 tensor per slot representing agent/hidden state for steering, avoiding re-tokenizing large chat history for peer sync. |
| **`eager_fallback`** (server telemetry) | **Compile fallback** — when the optional compiled decode path (`mx.compile`) cannot be used, the server falls back to eager execution and may log this event (burn-in gates watch for unexpected fallback under load). |

User-facing Python APIs should prefer neutral names (**`HiveClawManager`**, **`SlabClient`**, **`LocalSwarm`**) while this document and code retain historical identifiers where renaming would break Mach services, plist paths, or release artifacts.

---

## High-level stack

- **`pheromoned`** (Rust, macOS LaunchAgent) owns Mach service **`com.hiveclaw.pheromoned`** and brokers **XPC** plus **IOSurface** handoff to clients.
- Python **`hiveclaw_python.SlabClient`** talks to the daemon; **`hiveclaw_mlx`** can accelerate batched slab traffic on **Metal** (CPU paths remain the correctness baseline).
- The OpenAI-compatible **API gateway** is FastAPI-based: [`hiveclaw_python.openai_server`](../crates/hiveclaw-python/python/hiveclaw_python/openai_server.py), launched via `hiveclaw-server` / `python -m hiveclaw_python.server_main` (see [`server_main.py`](../crates/hiveclaw-python/python/hiveclaw_python/server_main.py)).

Peer coordination does not require a central JSON message bus: the **shared surface** plus **SAE** geometry is the coordination substrate (see steering and slab sections below).

---

## Slab layout (v5)

The PyO3 client handshakes with **`cmd=get_surface_v5`**. The daemon returns **`surface_id`**, **`magic_version`** (`0x48434C5700000005`), and optional identity strings **`daemon_exe`**, **`daemon_crate_version`**. Legacy commands receive **`error`** = **`INVALID_COMMAND_OR_UNSUPPORTED_VERSION`**.

**Global header:** 4096 bytes (magic `u64` @ 0, version `u32` @ 8, `n_slots` @ 12, `stride` @ 16).

**Each slot:** **640 bytes** total:

- 64-byte header (`slot_state`, Mach `last_claim`, `front_epoch`)
- **256×bf16** payload (512 B) — the SAE latent written/read by agents
- 64-byte footer with **`back_epoch`** at +576 from slot base

Writers bump **`front_epoch`**, copy 512 B, set **`back_epoch`**. **`read_slot_v5` / write paths** enforce **torn-read detection** (zeros or skip on mismatch); C++ may emit **`torn_epoch_skip`** to stderr unless **`HIVECLAW_TELEMETRY=0`**.

Python APIs: `read_slot_v5`, `write_slot_v5`, batched variants, `claim_task` / `release_task`. See [`scripts/README.md`](../scripts/README.md) for integration test commands.

---

## SAE latent geometry (2048 ↔ 256)

Slot payloads are **256×bf16** SAE latents (**`SCENT_ELEMS`** in [`crates/hiveclaw-core/src/math.rs`](../crates/hiveclaw-core/src/math.rs)). **`SlabClient.get_latent_dim()`** (and **`hiveclaw_mlx_ext.get_latent_dim()`**) expose the dimension.

**Llama 3.2 1B** hidden size is **2048**; the trained SAE maps **2048 → 256** for the slab. Default artifact: **`models/hiveclaw_sae_v1.safetensors`** (see `training/harvester.py`, `training/train_sae.py`).

**Breaking change:** any edit to `math.rs` (especially `SCENT_ELEMS`) changes the IOSurface layout — rebuild native code, rebuild **`pheromoned`**, and reload the daemon (see canonical reset steps in [`scripts/README.md`](../scripts/README.md)).

**Steered generation:** Llama-class runs with an SAE tying **2048-D hidden ↔ 256-D slab** so the last layer can inject peer state without re-tokenizing large chat histories.

---

## Continuous batching (Phase 7)

Set **`HIVECLAW_CONTINUOUS_BATCH=1`** with **`requirements/requirements-server.txt`** (includes **mlx-lm** + **httpx**). The server uses a **`swarm_batch_worker`** thread; **`stream=true`** paths are primary for this mode.

- Helpers: [`hiveclaw_python.batching.generate_batch`](../crates/hiveclaw-python/python/hiveclaw_python/batching/generate_batch.py), [`hiveclaw_python.batching.kv_mask`](../crates/hiveclaw-python/python/hiveclaw_python/batching/kv_mask.py) (`HiveClawKVCache` masks).
- Tests: **`tests/test_continuous_batching.py`** (KV slice/pad, mask shapes; optional golden via **`HIVECLAW_PHASE7_GOLDEN=1`**).
- **`HIVECLAW_COMPILE_DECODE`** defaults to **`1`** (try **`mx.compile`** on the inner decode; emits **`compile_status`** / **`eager_fallback`** JSON on stderr; set **`0`** for eager-only).
- **`HIVECLAW_COMPILE_WARMUP=1`** is **required** with **`HIVECLAW_CONTINUOUS_BATCH=1`** and default **`HIVECLAW_COMPILE_DECODE=1`** (server and batch worker raise **`ValueError`** otherwise). See **[ADR: batched steering contract](adr/BATCHED_STEERING_CONTRACT.md)**.
- Opt-in GPU batched slab: **`HIVECLAW_GPU_BATCH_READ=1`**, **`HIVECLAW_GPU_BATCH_WRITE=1`** (Metal fast path in **`hiveclaw_mlx`**; default remains CPU batched eval).

Load testing: **`python scripts/burn_in.py`** (`--spawn-server`, `--concurrency`, `--swapin-delta-max`, etc.).

---

## Ironclad engine proof (burn-in)

The repo ships an **exit-0** gate for a **local Apple Silicon Mac** with GUI **`launchctl`**, MLX weights, and the default SAE:

```bash
bash .github/scripts/ci_ironclad_verify.sh
```

This runs **`make doctor`** (daemon path + **`SlabClient`** handshake) then **[`scripts/verify_burn_in.sh`](../scripts/verify_burn_in.sh)**:

- SSE load under configured concurrency
- **`burn_in.py`** success criteria (including HTTP 503 behavior under overload where applicable)
- **Zero** **`eager_fallback`** JSON events on the server log

Phase 7 defaults compile the inner decode step when **`HIVECLAW_COMPILE_DECODE=1`** and **`HIVECLAW_COMPILE_WARMUP=1`**.

**Overrides** (env on `verify_burn_in.sh`): **`HEALTH_TIMEOUT_S`** (default **900** — wait for **`/health`** while Phase 7 probe + compile warmup run), **`HIVECLAW_MAX_QUEUE_DEPTH`**, **`CONCURRENCY`**, **`SWAPIN_DELTA_MAX`**, **`PORT`**, **`VERIFY_LOG`**.

**CI:** [`.github/workflows/ironclad-burn-in.yml`](../.github/workflows/ironclad-burn-in.yml) (typically needs **self-hosted macOS**). Lighter smoke: **`bash .github/scripts/ci_mac_smoke.sh`**.

---

## Mach-era semantics and IOSurface (summary)

Agents **claim**, **read/write** v5 latents, and **release** slots using the daemon-brokered surface. Epoch words provide **torn-read protection**; the design intentionally mirrors low-level shared-memory discipline while exposing a safe Python API.

For day-to-day setup, **`make daemon-load`**, **`make doctor`**, and troubleshooting (e.g. **`launchctl`** EIO from IDE terminals), see **[`scripts/README.md`](../scripts/README.md)**.

---

## Related documentation

- **[`scripts/README.md`](../scripts/README.md)** — canonical workspace, venv, Phase 6/7 env matrix, integration tests.
- **[`docs/adr/BATCHED_STEERING_CONTRACT.md`](adr/BATCHED_STEERING_CONTRACT.md)** — per-step invariants and Phase 7 compile contract.
