# Intelligence spike (Phase 4A)

**Layout note:** benchmarks and tests moved to [`benchmarks/`](../benchmarks/) and [`tests/`](../tests/); quality gate code lives in [`quality_gate/`](../quality_gate/); pip requirement files in [`requirements/`](../requirements/); CI helpers in [`.github/scripts/`](../.github/scripts/); experimental demos in [`internal/spikes/`](../internal/spikes/). This file still documents daemon, server, and Metal workflows.

**Deep-dive architecture** (IOSurface / XPC, slab v5 layout, SAE geometry, Phase 7 continuous batching, burn-in / ironclad): **[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)**. The repo **[`README.md`](../README.md)** is the product-oriented landing page. Terminal demo / hero GIF workflow: **[`examples/hiveclaw_top.py`](../examples/hiveclaw_top.py)** and **[`docs/assets/README.md`](../docs/assets/README.md)**.

## Canonical workspace (macOS Metal)

Use **one** checkout path for all commands and for the **virtualenv** so PyO3, maturin, and MLX never mix binaries across folders.

**Recommended:** work only under **`~/dev/HiveClaw`** (e.g. `/Users/you/dev/HiveClaw`). Keep the venv at **`~/dev/HiveClaw/.venv`**. Avoid maintaining a second checkout (e.g. `~/Desktop/HiveClaw`) with a different `.venv` unless it is a throwaway clone.

All command blocks below assume:

```bash
cd ~/dev/HiveClaw
```

---

### Slab v5 (XPC `get_surface_v5`, 640-byte slots, 256-D SAE latent)

The PyO3 client handshakes with **`cmd=get_surface_v5`**; the daemon replies with **`surface_id`**, **`magic_version`** (`0x48434C5700000005`), and optional identity strings **`daemon_exe`** (absolute path of `pheromoned`) and **`daemon_crate_version`** (hiveclaw-daemon crate version). Any other command (including legacy surface handshakes) receives **`error`** = **`INVALID_COMMAND_OR_UNSUPPORTED_VERSION`**.

**Layout:** global header **4096** bytes (magic `u64` @ 0, version `u32` @ 8, `n_slots` @ 12, `stride` @ 16). Each slot is **640** bytes: 64-byte header (`slot_state`, Mach `last_claim`, `front_epoch`), **256×bf16** payload (512 B), 64-byte footer with **`back_epoch` at +576** from slot base. Writers bump **`front_epoch`**, copy **512** B, set **`back_epoch`**. **`read_slot_v5` / `WriteSlab`** enforce torn detection (zeros on mismatch); C++ may emit **`torn_epoch_skip`** to stderr unless **`HIVECLAW_TELEMETRY=0`**. LLM scripts use a **trained SAE** in **`models/hiveclaw_sae_v1.safetensors`** (see `training/harvester.py`, `training/train_sae.py`).

**Integration harness (subprocess-safe):** with `pheromoned` loaded and venv active:

```bash
python tests/integration_test.py          # XPC v5 + header + read_slot_v5 smoke
python tests/integration_test.py --quick  # same as default smoke path
python tests/integration_test.py --stress # claim/release (256 slots) + swarm_spike + SIGKILL
python tests/integration_test.py --stress --stress-max-slots 4096   # full slab gauntlet
python tests/integration_test.py --batched   # read_slots / write_slots (daemon + venv)
```

**Phase 6 batched steering:** `tests/test_batched_steering.py` exercises `read_slots`, `write_slots`, and `steer_hidden_batched` (requires daemon + `models/hiveclaw_sae_v1.safetensors`). Do **not** run MLX integration or batched tests while **`training/harvester.py`** (or another Metal-heavy workload) is using the same GPU — contention can flake reads/writes. **`make python`** only builds native extensions; it never runs those tests.

**Phase 7 continuous batching:** Set **`HIVECLAW_CONTINUOUS_BATCH=1`** and install **`requirements/requirements-server.txt`** (includes **`mlx-lm`** + **`httpx`**). **`hiveclaw-server`** then uses a **`swarm_batch_worker`** thread and **`stream=true`** only. Helpers: **`crates/hiveclaw-python/python/hiveclaw_python/batching/generate_batch.py`**, **`crates/hiveclaw-python/python/hiveclaw_python/batching/kv_mask.py`** (`HiveClawKVCache` masks). Tests: **`tests/test_continuous_batching.py`** (KV slice/pad, mask shapes; optional golden via **`HIVECLAW_PHASE7_GOLDEN=1`**). **`HIVECLAW_COMPILE_DECODE`** defaults to **`1`** (try **`mx.compile`** inner decode; emits **`compile_status`** / **`eager_fallback`** JSON on stderr; set **`0`** for eager-only). **`HIVECLAW_COMPILE_WARMUP=1`** is **required** with **`HIVECLAW_CONTINUOUS_BATCH=1`** and default **`HIVECLAW_COMPILE_DECODE=1`** (server and batch worker raise **`ValueError`** otherwise); see ADR. Opt-in GPU batched slab: **`HIVECLAW_GPU_BATCH_READ=1`**, **`HIVECLAW_GPU_BATCH_WRITE=1`** (Metal fast path in **`hiveclaw_mlx`**; default remains CPU batched eval). Load test: **`python scripts/burn_in.py`** (see **`--spawn-server`** / **`--concurrency`** / **`--swapin-delta-max`**). One-shot full gate: **`./scripts/verify_burn_in.sh`** (starts server with **`HIVECLAW_MAX_QUEUE_DEPTH=10`**, **`CONCURRENCY=50`**, relaxed swap budget; requires zero **`eager_fallback`** events and a successful **`burn_in`** run). A/B stigmergy vs plain final layer: **`python scripts/benchmark_stigmergy.py`** (spawns two **`hiveclaw_server`** runs; **`--no-stigmergy`** or **`HIVECLAW_STIGMERGY=0`** disables slab claim/read/write on the hot path while keeping **`SlabClient`**). **Consensus + token tax:** **`python scripts/benchmark_consensus.py`** (string-passing multi-agent baseline vs slab latent committee; **`--json-out`** for tables). **External baseline (LangChain):** install **`scripts/requirements-bench-langchain.txt`**, then **`python scripts/benchmark_external.py`** (LangChain-orchestrated string committee vs HiveClaw; **`--no-langchain`** / **`--no-hiveclaw`** to skip either side). See **`docs/adr/BATCHED_STEERING_CONTRACT.md`** Phase 7.

**Ironclad verification (exit-0 burn-in):** Run **`bash .github/scripts/ci_ironclad_verify.sh`** after **`make python`**, **`make daemon-load`**, and **`pip install -r requirements/requirements-server.txt`** (same models / SAE as normal server). The script runs **`make doctor`** then **`verify_burn_in.sh`**. Override load/swap tuning via env on **`verify_burn_in.sh`**: **`HEALTH_TIMEOUT_S`** (default **900** — wait for **`/health`** while Phase 7 probe + compile warmup run), **`HIVECLAW_MAX_QUEUE_DEPTH`**, **`CONCURRENCY`**, **`SWAPIN_DELTA_MAX`**, **`PORT`**, **`VERIFY_LOG`**. GitHub-hosted macOS runners often lack a healthy GUI **`launchctl`** domain or cached MLX weights; use a **self-hosted Mac** (or local machine) for a reliable green. Optional manual workflow: **`.github/workflows/ironclad-burn-in.yml`**.

**Doctor (macOS):** After **`make python`** and **`make daemon-load`**, run **`make doctor`** from the same repo root. It checks **`launchctl print gui/$UID/com.hiveclaw.pheromoned`** (running state, **`program`** path vs **`target/release/pheromoned`**) and **`SlabClient()`**. If integration tests show **`magic_version 0x0`** or **`Connection invalid`**, the loaded daemon usually does not match this checkout or is not running—fix with **`make daemon-uninstall && make daemon-load`** (one canonical tree; rebuild release before reload).

**CI / smoke (macOS):** On a self-hosted or interactive Mac with GPU and venv ready, **`bash .github/scripts/ci_mac_smoke.sh`** runs **`make doctor`** then **`tests/integration_test.py --quick`**. For the full SSE + **`burn_in`** Ironclad gate, use **`bash .github/scripts/ci_ironclad_verify.sh`** (heavier; see **Ironclad verification** above).

---

### Phase C latent dimension (`get_latent_dim`)

IOSurface slot payloads are **256×bf16** SAE latents (**`SCENT_ELEMS`** in [`crates/hiveclaw-core/src/math.rs`](../crates/hiveclaw-core/src/math.rs)). Python uses **`SlabClient.get_latent_dim()`** (and **`hiveclaw_mlx_ext.get_latent_dim()`**). **Llama 3.2 1B** hidden size remains **2048**; the SAE maps **2048 → 256** for the slab.

**Any edit to `math.rs` (especially `SCENT_ELEMS`) is a breaking layout change.** You **must** rebuild everything and restart the daemon or Python and `pheromoned` will disagree on IOSurface layout:

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
make python-clean
make python PYTHON="$(pwd)/.venv/bin/python3"
cargo build --release -p hiveclaw-daemon
make daemon-unload   # if loaded
make daemon-load
make daemon-status
```

---

The script runs **Agent A** (prefill → SAE encode → **`write_slot_v5`**) on slot 0, then **Agent B** generates with **`read_slot_v5` → decode → L2 clamp → inject** on the last token. Requires **`models/hiveclaw_sae_v1.safetensors`**. See `internal/spikes/intelligence_spike.py`.

### 1. Environment reset (Conda off)

Conda and nanobind fight over libraries on macOS. Enforce a **single** Python:

1. Run **`conda deactivate`** until **`(base)`** is gone from your prompt.
2. **`cd ~/dev/HiveClaw`** (only this tree).

### 2. Pristine venv + pinned deps

Python **≥ 3.11** (project requirement):

```bash
cd ~/dev/HiveClaw
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/requirements-spike.txt
```

Build the PyO3 + MLX extensions (from repo root; also runs `pip install -r requirements/requirements-spike.txt`):

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
make python PYTHON="$(pwd)/.venv/bin/python3"
```

Always pass **`PYTHON="$(pwd)/.venv/bin/python3"`** when a **`.venv`** exists so maturin does not build for a different interpreter (**`RuntimeError: std::bad_cast`**).

**Conda + venv:** If maturin errors with “Both VIRTUAL_ENV and CONDA_PREFIX are set”, either run `conda deactivate` or rely on the Makefile, which runs maturin with `CONDA_PREFIX` / `CONDA_DEFAULT_ENV` unset.

**Wrong Python / “cross-compilation” / old path in errors:** If `make python` fails mentioning another directory’s `.venv` or “Unsupported Python interpreter for cross-compilation”, your shell may have **`VIRTUAL_ENV`** set to another checkout (left over from `activate` elsewhere) or **`CARGO_TARGET_DIR`** pointing at another checkout’s `target/` (PyO3 then reuses a stale interpreter path). The Makefile sets **`VIRTUAL_ENV`** from the chosen interpreter when it lives in a venv, and sets **`CARGO_TARGET_DIR`** to this repo’s `target/`. If problems persist, run `unset CARGO_TARGET_DIR CARGO_BUILD_TARGET`, then `make python-clean`, `cargo clean -p hiveclaw-python` (or `cargo clean`), and `make python PYTHON="$(pwd)/.venv/bin/python3"`. If **`.venv/pyvenv.cfg`**’s `command = ... -m venv <path>` points at the wrong folder, recreate `.venv` in this repo.

**`make python` vs repo `.venv`:** If **`$(pwd)/.venv`** exists, you must run **`make python PYTHON=$(pwd)/.venv/bin/python3`** (or activate that venv and use `python3` so it resolves to the same binary). Otherwise maturin can still **discover** `.venv` while building for a **different** interpreter → wrong arch / wrong CPython → **`RuntimeError: std::bad_cast`** at `write_scent` / `read_scent`. The Makefile **`python-check-maturin`** target enforces this match.

Alternative without `make python`: `cd crates/hiveclaw-mlx && python setup.py build_ext --inplace`, copy `hiveclaw_mlx_ext*.so` into `crates/hiveclaw-python/python/hiveclaw_python/`, then `cd ../hiveclaw-python && maturin develop --release` with the **same** `python` and an active **`VIRTUAL_ENV`** (still run `pip install -r requirements/requirements-spike.txt` once).

### Python ↔ C++ `SlabHandle` API

`read_scent` / `read_scent_at_offset` must **not** pass **`like`** into **`read_slot` / `read`** if C++ (`crates/hiveclaw-mlx/src/bindings.cpp`) only exposes **`read_slot(slot_index, shape)`** and **`read_slot(slot_index, shape, dep)`**. After any change, audit **`crates/hiveclaw-python/python/hiveclaw_python/__init__.py`** against **`bindings.cpp`**.

### macOS Metal — clean rebuild

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
make python-clean
make python PYTHON="$(pwd)/.venv/bin/python3"
cargo build --release -p hiveclaw-daemon
```

### macOS Metal — system acceptance

Treat the MLX path as stable only after:

1. **Daemon (prefer Terminal.app if the IDE terminal fails):** from repo root, `make daemon-load`
2. **Verify:** `make daemon-status` — `program` must be **`~/dev/HiveClaw/target/release/pheromoned`** (path to **`pheromoned`** in your active checkout).
3. **Smoke:** `source .venv/bin/activate` → `python internal/spikes/swarm_spike.py` — must get past the **first `read_scent`** without **`RuntimeError: std::bad_cast`**.

### 3. `RuntimeError: std::bad_cast` (MLX / nanobind ABI)

This happens when **`hiveclaw_mlx_ext`** was built against a **different MLX / nanobind / Python** than the one loaded at runtime: nanobind cannot cast Python **`mlx.core.array`** to C++ **`mlx::core::array`**.

1. **One interpreter:** Use a **single** venv (Python **≥ 3.11**). Avoid **`(conda base)` + `.venv`** — run `conda deactivate` until base is gone, then `source .venv/bin/activate`.
2. **Pinned MLX:** `requirements/requirements-spike.txt` pins **`mlx`**, **`mlx-metal`**, and **`mlx-lm`**. After **`pip install -U mlx`**, rebuild: **`make python-clean`** then **`make python PYTHON="$(pwd)/.venv/bin/python3"`**.
3. **Deep clean (optional):** `make python-clean`, then optionally `rm -rf target/` and `cargo clean`, then **`make python`** again from repo root.
4. **Verify linkage (optional):** `otool -L crates/hiveclaw-mlx/hiveclaw_mlx_ext*.so` — extension should match your machine (**arm64** vs **x86_64**) and typical MLX install under site-packages.
5. **Unset stray loader paths:** If you use `DYLD_LIBRARY_PATH`, try unsetting it for a test run (SIP may strip it for some binaries; conda can still inject conflicting `libmlx`).

When fixed, **`write_scent`** and **`read_scent`** run without crossing a broken nanobind cast, and **`SlabHandle(surface_id)`** still maps the same IOSurface as **`pheromoned`**.

## Two-terminal run

**Terminal A** — register the XPC Mach service with launchd (starts `pheromoned`):

```bash
cd ~/dev/HiveClaw
cargo build --release -p hiveclaw-daemon
make daemon-load
```

To stop the service:

```bash
make daemon-unload
```

To unload and remove the LaunchAgent plist from `~/Library/LaunchAgents/` (e.g. before switching to another checkout): `make daemon-uninstall`.

**Terminal B** — run the spike (do not run a bare `make` line that starts with `#`; that is interpreted as the target name `#`):

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
make python PYTHON="$(pwd)/.venv/bin/python3"
python internal/spikes/intelligence_spike.py
```

If `SlabClient` cannot connect, the script prints to stderr:

`pheromoned is not running...` — ensure `make daemon-load` succeeded. The active plist is **`~/Library/LaunchAgents/com.hiveclaw.pheromoned.plist`** (a copy also appears as `com.hiveclaw.pheromoned.gen.plist` in the repo). The template is **`crates/hiveclaw-daemon/data/com.hiveclaw.pheromoned.plist.in`** (wheel copy under `hiveclaw_python/data/` must match — run **`make check-plist`**). `ProgramArguments` must point at **`…/target/release/pheromoned`** for this checkout.

**`make daemon-load` → `Bootstrap failed: 5 / Input/output error`:** The GUI launchd domain (`gui/$(id -u)`) is not available from every shell. **Cursor, VS Code, and some SSH sessions** often return this error even though the plist path is correct.

1. Quit relying on the IDE terminal for this step.
2. Open **Terminal.app** (Applications → Utilities → Terminal) while logged in at the Mac console (not Screen Sharing-only quirks, if you can avoid them).
3. Run:
   ```bash
   cd ~/dev/HiveClaw
   make daemon-load
   ```
4. Verify: `make daemon-status` (should print launchd state for `com.hiveclaw.pheromoned`).

`make daemon-load` treats **`launchctl bootstrap` EIO 5** as success when **`com.hiveclaw.pheromoned` is already `state = running`** with the same **`pheromoned` binary** you just built (avoids a false failure after `bootout`/`bootstrap` races).

The Makefile already installs the plist under **`~/Library/LaunchAgents/`** (not the repo path). Ensure that file is **`chmod 644`**.

If **`make daemon-load` still returns EIO 5 from Terminal.app**, the GUI launchd domain may be unavailable (e.g. SSH-only session, some remote-desktop setups, or MDM restrictions). Check recent errors with:

```bash
log show --style syslog --predicate 'eventMessage CONTAINS "launchd" OR eventMessage CONTAINS "pheromoned"' --last 15m
```

## Integration test (`cargo test -p hiveclaw-daemon`)

The XPC test uses `launchctl bootstrap`. In restricted environments (some CI sandboxes), set:

`HIVECLAW_SKIP_LAUNCHD_TEST=1 cargo test -p hiveclaw-daemon`

to skip that test. On a normal macOS login session, leave this unset so the test runs end-to-end.

## VRAM contention test

To test VRAM contention and the entropy watchdog:

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
make daemon-load
make python PYTHON="$(pwd)/.venv/bin/python3"
```

In **4** separate terminals (each `cd ~/dev/HiveClaw`, `source .venv/bin/activate`):

```bash
python internal/spikes/swarm_spike.py
```

Minimal claim → write → read → release (one process): **`python examples/hello_swarm.py`**.

In a **5th** terminal:

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
hiveclaw-overseer
```

Each `swarm_spike.py` instance races to claim slab slots, runs synthetic matmul FLOPs while holding the lock, then blends slot scent toward a random goal. `overseer.py` inhibits any slot whose geometry freezes (mean per-dimension variance &lt; 1e-5 over the last 5 ticks).

**Single-terminal demo:** `python internal/spikes/overseer_demo.py` — two synthetic agent threads (fixed slots) plus an overseer thread; prints `INHIBIT` → `REROUTE` lifecycle to stdout (no LLM, no extra terminals).

## LLM swarm integration (`internal/spikes/llm_swarm.py`)

Multi-process test: **real** `mlx_lm` generation (default **`mlx-community/Llama-3.2-1B-Instruct-4bit`**) plus slab **sense → claim → generate (≤10 tokens) → write post-steer scent → release**. Each agent uses a **static goal vector** from a one-time prefill on its initial prompt; cosine pressure ranks unclaimed slots. **`--alpha`** tunes steering (default `0.1`). On EOS before 10 tokens, the agent picks a new prompt from a built-in pool.

**Requirements:** Mac + Apple Silicon + GPU + `pheromoned` under launchd (no headless mock). The model’s **`hidden_size` from `config.json`** must be **2048** for the default SAE (do not use `embed_tokens.weight` shape). **`get_latent_dim()`** is **256** (slab latent width).

### Four-terminal live integration test

1. **Terminal 1** — daemon (after full rebuild if you changed `math.rs`):

   ```bash
   cd ~/dev/HiveClaw
   cargo build --release -p hiveclaw-daemon
   make daemon-load
   make daemon-status
   ```

2. **Terminal 2** — entropy overseer:

   ```bash
   cd ~/dev/HiveClaw
   source .venv/bin/activate
   hiveclaw-overseer
   ```

3. **Terminals 3–5** — LLM agents (use different `--prompt` strings per terminal):

   ```bash
   cd ~/dev/HiveClaw
   source .venv/bin/activate
   python internal/spikes/llm_swarm.py --prompt "Tell a story about a dog."
   ```

   ```bash
   python internal/spikes/llm_swarm.py --prompt "Write a poem about space."
   ```

   ```bash
   python internal/spikes/llm_swarm.py --prompt "Explain quantum physics in simple terms."
   ```

### Action 5 observation checklist

- [ ] **Contention:** Agents sometimes sleep/retry when they cannot claim a slot.
- [ ] **Generation:** Each agent streams text to stdout in bursts (up to 10 tokens per hold).
- [ ] **Steering:** Narratives may drift as slots exchange blended hidden-state scents.
- [ ] **Immune system:** With default overseer timing (~500 ms × 5 samples), killing an agent mid-hold (**Ctrl+C**, no cleanup handler) should eventually lead to **`INHIBIT`** on that slot as variance collapses.

Do not merge the feature branch to `main` until this checklist is green in your environment.

---

## Phase 5 — FastAPI server + TUI dashboard

OpenAI-compatible **`POST /v1/chat/completions`** (streaming and non-streaming), **`GET /health`**, and a JSON snapshot **`GET /v1/slots`**. The server loads the SAE, connects to **`pheromoned`**, and runs **Llama 3.2 1B** with **`ActiveSteeringWrapper`** from `hiveclaw_python/steering.py`. **`MAX_CONCURRENT = 1`** serializes MLX work (safe layer swap); scale-out is a follow-on.

### Install server + dashboard deps

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
pip install -r requirements/requirements-server.txt
```

Requires the same MLX spike venv + `make python` as Phase 4A, and **`models/hiveclaw_sae_v1.safetensors`**.

### Run daemon + API

```bash
cd ~/dev/HiveClaw
cargo build --release -p hiveclaw-daemon
make daemon-load
make daemon-status
```

**API (blocking tab):**

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
hiveclaw-server --host 127.0.0.1 --port 8080
```

Or with uvicorn directly (lifespan loads MLX + slab on startup):

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
uvicorn hiveclaw_python.openai_server:app --host 127.0.0.1 --port 8080
```

**Dashboard (another tab):**

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
hiveclaw-dashboard --refresh-ms 1000 --max-slots 64
```

Optional: point at a JSON-lines log for telemetry counts (e.g. redirect daemon stderr, or aggregate logs that contain `{"event":"poison_clamp",...}` / `torn_epoch_skip`):

```bash
hiveclaw-dashboard --max-slots 32 --telemetry-log /tmp/hiveclaw_telem.log
```

### curl examples

```bash
curl -s http://127.0.0.1:8080/health
```

```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hiveclaw-llama-1b","messages":[{"role":"user","content":"Tell me about bees"}],"max_tokens":128}'
```

Streaming (SSE):

```bash
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hiveclaw-llama-1b","messages":[{"role":"user","content":"Hi"}],"max_tokens":64,"stream":true}'
```

Optional body fields: **`temperature`** (default `0.8`), **`alpha`** steering strength (default `0.1`, matches `hiveclaw_steering`).

**Cursor / model listing:** Set **`HIVECLAW_CURSOR_MODEL_ALIAS`** (default **`hiveclaw-swarm-8b`**) so `/v1/models` advertises the model name users type in Cursor while the server keeps **`hiveclaw-llama-1b`** as the canonical id. Any request **`model`** id matching that alias, the canonical name, or a string starting with **`hiveclaw-`** is accepted. Related: **`HIVECLAW_TWO_AGENT=1`** enables the Coder+Reviewer pipeline ([`hiveclaw_python/swarm_agents.py`](../crates/hiveclaw-python/python/hiveclaw_python/swarm_agents.py)); **`HIVECLAW_TWO_AGENT_CODER_SLOT`** / **`HIVECLAW_TWO_AGENT_REVIEWER_SLOT`** (defaults **0** / **1**) pin slab slots; **`HIVECLAW_SHOW_THINKING=1`** emits an optional status chunk before swarm work in streaming mode.

### Client `base_url`

Use **`http://127.0.0.1:8080/v1`** as the OpenAI-compatible base URL for LangChain / OpenAI SDKs.
