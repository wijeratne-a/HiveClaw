# HiveClaw — living agent context

**Purpose:** Orientation for humans and coding agents working in this checkout. Update this file whenever architecture, layout, APIs, env vars, or working-tree intent change. Do **not** treat stale README/ARCHITECTURE copy as source of truth if it conflicts with this file or with code.

**Canonical checkout:** `~/dev/HiveClaw` (`/Users/wijeratne/dev/HiveClaw`). Single venv: `.venv` (Python ≥ 3.11). Do not keep a second clone for day-to-day builds (`CANONICAL.md`).

**Last full review:** 2026-08-30. **HEAD:** `d577ed0` on `main` (tracks `origin/main`). **This session:** Rewind causal-runtime slice — discovery complete; application code not started.

---

## How to maintain this file

After any meaningful session:

1. Update **Snapshot** and **Working tree**.
2. If layout, handshake, env vars, or public APIs changed, update those sections and bump **Last full review**.
3. Append a short **Session log** entry (date, intent, what changed, what is still open).
4. Prefer recording *code truth* here even when `docs/` still uses older names.

---

## Snapshot (2026-08-30)

| Item | State |
|------|--------|
| Product phase (README) | **Phase 1 shipped** — Apple Silicon / Metal / macOS IPC proof of concept |
| Implementation depth | Slab **v6** (runtime latent width), Phase 6 batched steering, Phase 7 continuous batching |
| Default model | `mlx-community/Llama-3.2-1B-Instruct-4bit` (hidden 2048) |
| Default SAE | `models/hiveclaw_sae_v1.safetensors` (2048 → 256, gitignored; present locally) |
| License | AGPL-3.0; commercial dual-license intended |
| Platform guard | `hiveclaw_python` raises `NotImplementedError` unless Darwin + arm64 |
| Remote | `origin/main` at `d577ed0` |
| Uncommitted | Demo WIP (see Working tree) + Rewind discovery docs in progress |
| Rewind slice | Types in `hiveclaw_causal/`; ADR `docs/adr/CAUSAL_RUNTIME_H5.md`. Engine/policy/e2e not implemented yet. |

---

## What it is

Local, hardware-native Llama-class inference on Apple Silicon with an **OpenAI-compatible** HTTP API. Multi-agent coordination does **not** ship growing JSON transcripts between peers. Agents share a **VRAM-backed IOSurface slab** of compressed SAE latents (historically called “pheromone / stigmergy”). The pitch: **zero coordination-token tax**, data stays on device.

**Not claimed:** “zero tokens for the whole model.” Each agent still has a task prompt. What disappears is the *growing committee transcript*. Do not quote a fixed speedup without measuring on this machine.

---

## Two phase numbering systems (do not mix)

**Product roadmap** (`README.md`):

| Phase | Target | Status |
|-------|--------|--------|
| 1 | Apple Silicon, Metal, macOS IPC | Shipped |
| 2 | Linux x86, POSIX shm, Docker | Planned |
| 3 | Multi-node CUDA / NVLink | Planned |
| 4 | vLLM backend | Planned |
| 5 | Enterprise auth / audit / SOC 2 hooks | Planned |

**Implementation phases** (code + `scripts/README.md` + ADR):

| Phase | Meaning |
|-------|---------|
| 4 / 4A | XPC Mach service + IOSurface + PyO3 + SAE steering spike |
| 5 | FastAPI OpenAI gateway + TUI dashboard |
| 6 | Batched slab read/write + batched steering |
| 7 | Continuous batching (`HIVECLAW_CONTINUOUS_BATCH=1`), compiled decode, KV masks |

When someone says “Phase 4,” ask which system they mean.

---

## Architecture (code truth)

```
Client (OpenAI SDK / Cursor / LocalSwarm)
    → hiveclaw-server (FastAPI, openai_server.py)
        → mlx-lm generation + SAE steering (steering.py)
        → SlabClient (PyO3 _core)  --XPC-->  pheromoned (LaunchAgent)
        → hiveclaw_mlx_ext (nanobind) maps IOSurface, Metal kernels
    → quality_gate (optional generate → verify → repair)
```

### Processes and names

| Name | Role |
|------|------|
| `pheromoned` | IPC broker daemon. Mach service `com.hiveclaw.pheromoned`. Owns the IOSurface. |
| `SlabClient` | Python client: XPC handshake + high-level slot APIs. |
| `hiveclaw_mlx_ext` | C++/MSL extension: map surface, v5-named kernels, batched I/O. |
| Stigmergy | Latent-cache sync on the slab instead of a JSON bus (`HIVECLAW_STIGMERGY`). |
| Overseer | Entropy watchdog: inhibit slots whose geometry freezes (`hiveclaw-overseer`). |

User-facing names: `HiveClawManager`, `SlabClient`, `LocalSwarm`. Historical names (`pheromoned`, scent, stigmergy) stay in code because they are Mach/plist/ABI.

### Slab layout — **v6 in code**, v5 names in APIs/docs

**Docs lag:** `docs/ARCHITECTURE.md`, `docs/PR2_NEXT.md`, and much of `scripts/README.md` still describe **v5** magic `0x48434C5700000005` and a fixed 640-byte stride. **Code is v6.**

Source of truth for layout: `crates/hiveclaw-core/src/math.rs` and `crates/hiveclaw-core/include/hiveclaw_layout_v5.h` (header filename is historical; contents are v6).

| Field | Value |
|-------|--------|
| Magic+version | `0x48434C5700000006` (`SLAB_MAGIC << 32 \| 6`) |
| XPC command | still **`get_surface_v5`** (name frozen; layout is v6) |
| Global header | 4096 B |
| Slots | 4096 |
| Slot geometry | 64 B header + `D * 2` payload + 64 B footer; stride = `128 + 2*D` |
| Default `D` | 256 bf16 (`DEFAULT_LATENT_ELEMS`); **runtime** from header offset 20 |
| Extra header fields | `zeta_t` @ 32, `decay_rate` @ 36 (daemon decay) |
| Epoch protocol | bump `front_epoch`, copy payload, set `back_epoch`; torn read → zeros |
| Slot states | FREE / CLAIMED / INHIBITED / FAULT |

Python methods stay named `read_slot_v5` / `write_slot_v5` / `read_slots` / `write_slots`. Payload shape is `[1,1,D]` or `[B,1,D]` bf16, not hardcoded 256.

**Breaking change:** editing `math.rs` latent/layout constants requires `make python-clean && make python`, `cargo build --release -p hiveclaw-daemon`, then `make daemon-uninstall && make daemon-load`. Mismatched daemon ↔ Python shows `magic_version 0x0` or `Connection invalid`.

### SAE steering

- Artifact: `models/hiveclaw_sae_v1.safetensors`.
- Tied weights: decode `matmul(scent_D, W_enc) + b_dec`; encode `relu(h @ W_enc.T + b_enc)`.
- Inject on **last token** only; L2 poison clamp on `alpha * decoded` (radius 2.0).
- Normative math: `hiveclaw_python/steering.py` — **not** `PR2_NEXT.md` if they disagree.
- Sentinel dummy slot: Python `-1` → C++ `0xFFFFFFFF` (no IOSurface access).

### Continuous batching (Phase 7)

- Flag: `HIVECLAW_CONTINUOUS_BATCH=1` (default off; single-stream + `_MLX_LOCK`).
- Single MLX thread: `swarm_batch_worker`. No `mx.eval` on asyncio threads.
- `stream=true` required in this mode.
- `HIVECLAW_COMPILE_DECODE` default `1`; with continuous batch, **`HIVECLAW_COMPILE_WARMUP=1` is required** or the server raises `ValueError`.
- `B_bucket` never shrinks until the batch drains (masked dummy FLOPs over recompile).
- Ironclad gate: zero `eager_fallback` JSON events under load (`scripts/verify_burn_in.sh`).

ADR: `docs/adr/BATCHED_STEERING_CONTRACT.md`.

---

## Crate / package split (respect this)

Rust workspace members (`Cargo.toml`): `hiveclaw-core`, `hiveclaw-backend-metal`, `hiveclaw-daemon`, `hiveclaw-python`.

| Path | Role |
|------|------|
| `crates/hiveclaw-core` | Shared layout math + placeholder traits. **No backend / no daemon.** |
| `crates/hiveclaw-backend-metal` | IOSurface / Metal buffer (`MetalPheromoneBuffer`). |
| `crates/hiveclaw-daemon` | `pheromoned` binary + XPC + CPU decay loop. |
| `crates/hiveclaw-python` | PyO3 `hiveclaw_python._core` + Python package. |
| `crates/hiveclaw-mlx` | **Not a workspace member.** CMake + nanobind MLX extension (`hiveclaw_mlx_ext`). |
| `crates/hiveclaw-worker` | Empty / unused in this tree. |

`.cursorrules`: prefer workspace path deps; keep backend code out of core.

Python package: `crates/hiveclaw-python/python/hiveclaw_python/`. Console scripts: `hiveclaw-server`, `hiveclaw-dashboard`, `hiveclaw-overseer`.

---

## Directory map

| Path | Role |
|------|------|
| `examples/` | Supported user demos (`hello_swarm.py`, `hiveclaw_top.py`, …). |
| `demos/` | Repo Pulse + Consensus Showdown showcase (Rich TUIs, health reports). |
| `benchmarks/` | Committee / LangChain / stigmergy / Cursor simulation harnesses. |
| `tests/` | Integration + unit (`integration_test.py`, `test_batched_steering.py`, …). |
| `scripts/` | Doctor, burn-in, verify, **thin shims** to the Python package. |
| `quality_gate/` | YAML-profile generate → verify → repair. |
| `training/` | SAE harvest + train (optional; not needed if SAE artifact exists). |
| `internal/spikes/` | Unsupported research (`intelligence_spike`, `llm_swarm`, `swarm_spike`). |
| `docs/` | Product-light README companion: `ARCHITECTURE.md` (lags v6), ADR. Research: `docs/research/repository-baseline.md`. |
| `hiveclaw_causal/` | CPU-only causal runtime for The Rewind. Types landed; engine/policy pending. Must not import `hiveclaw_python`. |
| `requirements/` | Optional dep sets (`requirements-server.txt`, spike, bench, catenar). |
| `models/` | SAE + latent traces (safetensors/npz gitignored). |
| `.github/` | Wheel CI (macos-arm64) + ironclad burn-in (needs self-hosted Mac). |

**Implementation lives in the package, not `scripts/`.** Shims: `scripts/hiveclaw_server.py`, `hiveclaw_steering.py`, `hiveclaw_dashboard.py`, `overseer.py`, `hiveclaw_kv_mask.py`, `generate_batch.py`. Prefer `hiveclaw-server` / `python -m hiveclaw_python.server_main`.

---

## Public Python surface

```python
import hiveclaw_python as hc
hc.init()                          # find repo root, optional daemon bootstrap
client = hc.SlabClient()           # XPC + IOSurface
swarm = hc.LocalSwarm(...)         # spawn server subprocess + SSE agents
mgr = hc.HiveClawManager(...)      # make/launchctl helpers
```

| API | Notes |
|-----|--------|
| `SlabClient.read_slot_v5` / `write_slot_v5` | Single slot `[1,1,D]` bf16 |
| `read_slots` / `write_slots` | Batched; status uint8 `[B]` (0 ok, 1 torn, 2 invalid write) |
| `claim_task` / `release_task` / `inhibit` | Slot lifecycle; claim uses pid as owner |
| `LocalSwarm` | Auto-starts daemon + server with continuous batch + compile warmup |
| `Swarm` | Thin slab helper (cosine ranking, claim) — not the LLM orchestrator |
| `ActiveSteeringWrapper` / `steer_hidden_batched` | Last-layer injection |
| `two_agent_pipeline` | Coder+Reviewer when `HIVECLAW_TWO_AGENT=1` |

OpenAI subset: `POST /v1/chat/completions` (string content, max_tokens, temperature, stream), `GET /v1/models`, `GET /health`, `GET /v1/slots`. No tools, no JSON mode, no embeddings.

Cursor integration: `HIVECLAW_TWO_AGENT=1 hiveclaw-server`; base URL `http://127.0.0.1:8080/v1`; model alias default `hiveclaw-swarm-8b`.

---

## Environment variables

| Var | Default | Meaning |
|-----|---------|---------|
| `HIVECLAW_MODEL_ID` | Llama-3.2-1B-Instruct-4bit | mlx-lm model id |
| `HIVECLAW_SAE_PATH` | `models/hiveclaw_sae_v1.safetensors` | SAE weights |
| `HIVECLAW_STIGMERGY` | `1` | Slab claim/read/write on generate path |
| `HIVECLAW_CONTINUOUS_BATCH` | `0` | Phase 7 worker |
| `HIVECLAW_COMPILE_DECODE` | `1` | `mx.compile` inner decode |
| `HIVECLAW_COMPILE_WARMUP` | `0` | Required `1` if continuous + compile |
| `HIVECLAW_MAX_QUEUE_DEPTH` | `50` | 503 / evict slow consumers |
| `HIVECLAW_MAX_BATCH` | `8` | OOM probe cap |
| `HIVECLAW_GPU_BATCH_READ/WRITE` | off | Metal batched slab (CPU is default) |
| `HIVECLAW_TELEMETRY` | on | stderr JSON (`torn_epoch_skip*`, `poison_clamp*`, `eager_fallback`) |
| `HIVECLAW_TWO_AGENT` | `0` | Coder+Reviewer pipeline |
| `HIVECLAW_TWO_AGENT_CODER_SLOT` / `_REVIEWER_SLOT` | `0` / `1` | Must differ |
| `HIVECLAW_CURSOR_MODEL_ALIAS` | `hiveclaw-swarm-8b` | `/v1/models` display name |
| `HIVECLAW_SHOW_THINKING` | `0` | Extra SSE status chunk |
| `HIVECLAW_REPO_ROOT` | auto | Checkout for models/make when not cwd |
| `HIVECLAW_SKIP_LAUNCHD_TEST` | unset | Skip daemon launchd cargo test |

---

## Day-to-day commands

Always from repo root, venv active. If `.venv` exists: `make python PYTHON="$(pwd)/.venv/bin/python3"`.

```bash
source .venv/bin/activate
make python PYTHON="$(pwd)/.venv/bin/python3"   # PyO3 + mlx ext only; does not run tests
make daemon-load                                 # build release pheromoned + launchctl
make doctor
make daemon-status                               # program must be this checkout's target/release/pheromoned
hiveclaw-server --host 127.0.0.1 --port 8080
```

Daemon bootstrap from **Cursor/VS Code terminals often fails with launchctl EIO 5**. Use **Terminal.app**. `make daemon-load` treats EIO as OK if the job is already running with this binary.

**Never** run MLX slab/LLM tests while `training/harvester.py` owns the GPU.

Integration:

```bash
python tests/integration_test.py --quick
python tests/integration_test.py --batched
python tests/test_batched_steering.py          # needs daemon + SAE
bash .github/scripts/ci_mac_smoke.sh
bash .github/scripts/ci_ironclad_verify.sh     # heavy; self-hosted Mac
```

---

## Conventions and gotchas

1. **One interpreter.** Conda + venv → nanobind `std::bad_cast`. `conda deactivate` until `(base)` is gone. Pin MLX via `requirements/requirements-spike.txt`; after `pip install -U mlx`, rebuild extensions.
2. **`make python` vs tests.** Build only. Do not assume it ran integration tests.
3. **Plist drift.** Template is `crates/hiveclaw-daemon/data/com.hiveclaw.pheromoned.plist.in`. Wheel copy under `hiveclaw_python/data/` must match (`make check-plist`). Installed copy lives in `~/Library/LaunchAgents/`.
4. **Do not pass `like=` into C++ read** if bindings only take `(slot, shape[, dep])`.
5. **Batched `depends`:** rank 3 only (`[B,1,2048]` read, `[B,1,256]` write). Row `i` aligns with batch row `i` — do not reorder slots vs latents in `SlabClient`.
6. **Duplicate real slot indices** rejected in C++; sentinels skipped.
7. **HEALTH_REPORT.md** files in the repo root are **demo outputs**, not a trusted static analyzer. The current generated reports suggest Rust types for Python files — ignore as engineering signal.
8. Safe demo claims: coordination context stays near-zero; SAE dashboard is a *proxy*; local-only path. Avoid “Agent C decodes findings from epochs alone” and “feature labels are semantic truth.”

---

## Working tree (as of 2026-08-30)

Clean **committed** main is Repo Pulse (`d577ed0`). **Uncommitted / untracked** (do not discard blindly):

- Modified: `demos/README.md`, `demos/audit_swarm.py`, `demos/baseline_audit.py`, `demos/run_repo_pulse.py`
- Untracked: `demos/consensus_showdown.py`, `demos/health_report_validate.py`, `demos/json_utils.py`, `demos/llm_ab.py`, `demos/llm_client.py`
- Generated: `HEALTH_REPORT.md`, `HEALTH_REPORT_BASELINE.md`, `.DS_Store`

Other branch (not checked out): `feat/llm-swarm-integration`.

---

## Docs vs code (fix later; do not “fix” by editing code in a docs-only pass)

- Architecture / scripts README still say v5 magic and fixed 640 B slots.
- `hiveclaw-python/src/lib.rs` comments still say “Phase 4 v5”; handshake function is `connect_and_fetch_surface_v5`.
- C header file is named `hiveclaw_layout_v5.h` but defines `HCLW_MAGIC_V6`.
- Product README “Phase 1 current” vs implementation Phase 7 — both true in different numbering systems.

---

## Session log

### 2026-08-30 — Rewind domain types + H5 ADR

- Added `hiveclaw_causal/types.py` (Artifact/Observation/Claim/…, provenance, edges, events, policy decision).
- ADR: `docs/adr/CAUSAL_RUNTIME_H5.md` — H5 default; H1–H4 have no existing primitive in-tree.

### 2026-08-30 — Rewind discovery (no application code)

- Ran existing test baseline on this machine (venv 3.11.1, daemon running).
- **PASS:** quality_controller 19/19; integration `--quick` and `--batched`; continuous_batching (golden skipped); SAE tied weights; batched_steering; `cargo test -p hiveclaw-core` (1 test).
- **Not run:** `--stress`, daemon crate IPC tests (Mach label collision with live LaunchAgent), ironclad burn-in.
- Confirmed: no typed claim/event/invalidation graph exists. Slab `claim_task` ≠ causal claim. `quality_gate` is code verify/repair only.
- Decision recorded for next step: new root package `hiveclaw_causal/` (CPU, SQLite H5). See `docs/research/repository-baseline.md`.

### 2026-08-30 — context bootstrap

- Explored checkout, architecture, crates, Python package, demos, Makefile, git history.
- Created this file. **No application code changes.**
- Noted v6 layout vs v5 documentation/API names; recorded uncommitted demo WIP.

---

## Pointers (read when the task needs them)

- Product: `README.md`
- Operator/setup: `scripts/README.md`
- Deep dive (verify against code): `docs/ARCHITECTURE.md`
- Batched steering + Phase 7 invariants: `docs/adr/BATCHED_STEERING_CONTRACT.md`
- Python package: `crates/hiveclaw-python/README-PYTHON.md`
- Demos: `demos/README.md`
- Spikes: `internal/spikes/README.md`
