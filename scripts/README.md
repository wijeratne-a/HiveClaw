# Intelligence spike (Phase 4A)

## Canonical workspace (macOS Metal)

Use **one** checkout path for all commands and for the **virtualenv** so PyO3, maturin, and MLX never mix binaries across folders.

**Recommended:** work only under **`~/dev/HiveClaw`** (e.g. `/Users/you/dev/HiveClaw`). Keep the venv at **`~/dev/HiveClaw/.venv`**. Avoid maintaining a second checkout (e.g. `~/Desktop/HiveClaw`) with a different `.venv` unless it is a throwaway clone.

All command blocks below assume:

```bash
cd ~/dev/HiveClaw
```

---

### Phase C scent dimension (`get_scent_dim`)

Phase C slot scents are **bf16 vectors** whose length is **`SCENT_ELEMS`** in [`crates/hiveclaw-core/src/math.rs`](../crates/hiveclaw-core/src/math.rs) (currently **2048**, aligned with **Llama 3.2 1B** hidden size). At runtime, Python calls **`SlabClient.get_scent_dim()`** (compiled into the PyO3 extension) so scripts never hardcode the width.

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

The script runs **Agent A** (prefill → L2-normalized last-token hidden state) writing an **L2-normalized bf16 scent** (length = `get_scent_dim()`, 2048 for the default 1B model) to IOSurface slot 0, then **Agent B** generates with the final layer wrapped to add `alpha * scent` to the last position each step (“active steering”). See `scripts/intelligence_spike.py`.

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
python -m pip install -r scripts/requirements-spike.txt
```

Build the PyO3 + MLX extensions (from repo root; also runs `pip install -r scripts/requirements-spike.txt`):

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
make python PYTHON="$(pwd)/.venv/bin/python3"
```

Always pass **`PYTHON="$(pwd)/.venv/bin/python3"`** when a **`.venv`** exists so maturin does not build for a different interpreter (**`RuntimeError: std::bad_cast`**).

**Conda + venv:** If maturin errors with “Both VIRTUAL_ENV and CONDA_PREFIX are set”, either run `conda deactivate` or rely on the Makefile, which runs maturin with `CONDA_PREFIX` / `CONDA_DEFAULT_ENV` unset.

**Wrong Python / “cross-compilation” / old path in errors:** If `make python` fails mentioning another directory’s `.venv` or “Unsupported Python interpreter for cross-compilation”, your shell may have **`VIRTUAL_ENV`** set to another checkout (left over from `activate` elsewhere) or **`CARGO_TARGET_DIR`** pointing at another checkout’s `target/` (PyO3 then reuses a stale interpreter path). The Makefile sets **`VIRTUAL_ENV`** from the chosen interpreter when it lives in a venv, and sets **`CARGO_TARGET_DIR`** to this repo’s `target/`. If problems persist, run `unset CARGO_TARGET_DIR CARGO_BUILD_TARGET`, then `make python-clean`, `cargo clean -p hiveclaw-python` (or `cargo clean`), and `make python PYTHON="$(pwd)/.venv/bin/python3"`. If **`.venv/pyvenv.cfg`**’s `command = ... -m venv <path>` points at the wrong folder, recreate `.venv` in this repo.

**`make python` vs repo `.venv`:** If **`$(pwd)/.venv`** exists, you must run **`make python PYTHON=$(pwd)/.venv/bin/python3`** (or activate that venv and use `python3` so it resolves to the same binary). Otherwise maturin can still **discover** `.venv` while building for a **different** interpreter → wrong arch / wrong CPython → **`RuntimeError: std::bad_cast`** at `write_scent` / `read_scent`. The Makefile **`python-check-maturin`** target enforces this match.

Alternative without `make python`: `cd crates/hiveclaw-mlx && python setup.py build_ext --inplace`, copy `hiveclaw_mlx_ext*.so` into `crates/hiveclaw-python/python/hiveclaw_python/`, then `cd ../hiveclaw-python && maturin develop --release` with the **same** `python` and an active **`VIRTUAL_ENV`** (still run `pip install -r scripts/requirements-spike.txt` once).

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
3. **Smoke:** `source .venv/bin/activate` → `python scripts/swarm_spike.py` — must get past the **first `read_scent`** without **`RuntimeError: std::bad_cast`**.

### 3. `RuntimeError: std::bad_cast` (MLX / nanobind ABI)

This happens when **`hiveclaw_mlx_ext`** was built against a **different MLX / nanobind / Python** than the one loaded at runtime: nanobind cannot cast Python **`mlx.core.array`** to C++ **`mlx::core::array`**.

1. **One interpreter:** Use a **single** venv (Python **≥ 3.11**). Avoid **`(conda base)` + `.venv`** — run `conda deactivate` until base is gone, then `source .venv/bin/activate`.
2. **Pinned MLX:** `scripts/requirements-spike.txt` pins **`mlx`**, **`mlx-metal`**, and **`mlx-lm`**. After **`pip install -U mlx`**, rebuild: **`make python-clean`** then **`make python PYTHON="$(pwd)/.venv/bin/python3"`**.
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
python scripts/intelligence_spike.py
```

If `SlabClient` cannot connect, the script prints to stderr:

`pheromoned is not running...` — ensure `make daemon-load` succeeded. The active plist is **`~/Library/LaunchAgents/com.hiveclaw.pheromoned.plist`** (a copy also appears as `com.hiveclaw.pheromoned.gen.plist` in the repo). `ProgramArguments` must point at **`…/target/release/pheromoned`** for this checkout.

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
python scripts/swarm_spike.py
```

In a **5th** terminal:

```bash
cd ~/dev/HiveClaw
source .venv/bin/activate
python scripts/overseer.py
```

Each `swarm_spike.py` instance races to claim slab slots, runs synthetic matmul FLOPs while holding the lock, then blends slot scent toward a random goal. `overseer.py` inhibits any slot whose geometry freezes (mean per-dimension variance &lt; 1e-5 over the last 5 ticks).

## LLM swarm integration (`scripts/llm_swarm.py`)

Multi-process test: **real** `mlx_lm` generation (default **`mlx-community/Llama-3.2-1B-Instruct-4bit`**) plus slab **sense → claim → generate (≤10 tokens) → write post-steer scent → release**. Each agent uses a **static goal vector** from a one-time prefill on its initial prompt; cosine pressure ranks unclaimed slots. **`--alpha`** tunes steering (default `0.1`). On EOS before 10 tokens, the agent picks a new prompt from a built-in pool.

**Requirements:** Mac + Apple Silicon + GPU + `pheromoned` under launchd (no headless mock). **`model.config` hidden size must equal `get_scent_dim()`** or the script exits with a clear error.

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
   python scripts/overseer.py
   ```

3. **Terminals 3–5** — LLM agents (use different `--prompt` strings per terminal):

   ```bash
   cd ~/dev/HiveClaw
   source .venv/bin/activate
   python scripts/llm_swarm.py --prompt "Tell a story about a dog."
   ```

   ```bash
   python scripts/llm_swarm.py --prompt "Write a poem about space."
   ```

   ```bash
   python scripts/llm_swarm.py --prompt "Explain quantum physics in simple terms."
   ```

### Action 5 observation checklist

- [ ] **Contention:** Agents sometimes sleep/retry when they cannot claim a slot.
- [ ] **Generation:** Each agent streams text to stdout in bursts (up to 10 tokens per hold).
- [ ] **Steering:** Narratives may drift as slots exchange blended hidden-state scents.
- [ ] **Immune system:** With default overseer timing (~500 ms × 5 samples), killing an agent mid-hold (**Ctrl+C**, no cleanup handler) should eventually lead to **`INHIBIT`** on that slot as variance collapses.

Do not merge the feature branch to `main` until this checklist is green in your environment.
