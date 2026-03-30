# Intelligence spike (Phase 4A)

The script runs **Agent A** (prefill → L2-normalized last-token hidden state) writing a **4096-D bf16 scent** to IOSurface slot 0, then **Agent B** generates with the final layer wrapped to add `alpha * scent` to the last position each step (“active steering”). See `scripts/intelligence_spike.py`.

Activate a venv first, then either install spike deps explicitly or use `make python` (which runs `pip install -r scripts/requirements-spike.txt` before maturin):

```bash
cd /Users/wijeratne/dev/HiveClaw
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r scripts/requirements-spike.txt
```

Build the PyO3 extension (from repo root; also installs the spike requirements above):

```bash
source .venv/bin/activate
make python
```

If `make spike-deps` installs into the wrong interpreter, pin it: `make python PYTHON=.venv/bin/python3`.

**Conda + venv:** If maturin errors with “Both VIRTUAL_ENV and CONDA_PREFIX are set”, either run `conda deactivate` or rely on the Makefile, which runs maturin with `CONDA_PREFIX` / `CONDA_DEFAULT_ENV` unset.

**Wrong Python / “cross-compilation” / old path in errors:** If `make python` fails mentioning another directory’s `.venv` or `Unsupported Python interpreter for cross-compilation`, check **`VIRTUAL_ENV`**: activating a venv in one checkout leaves it set; maturin can still use that path even when you pass `PYTHON=` from another repo. The Makefile **unsets `VIRTUAL_ENV`** for the maturin step so **`PYO3_PYTHON`** wins. Open **`.venv/pyvenv.cfg`**: if **`command = ... -m venv <path>`** points at a **different folder** than this repo, recreate the venv: `rm -rf .venv && python3 -m venv .venv && pip install -r scripts/requirements-spike.txt`, then `make python PYTHON=.venv/bin/python3`. The Makefile **sets `CARGO_TARGET_DIR` to this repo’s `target/`** and **`PYO3_PYTHON`** to match **`PYTHON`** (overrides stray `~/.cargo/config.toml` target dirs).

Alternative without `make python`: `cd crates/hiveclaw-python && maturin develop --release` (still run `pip install -r scripts/requirements-spike.txt` once for NumPy / MLX).

## Two-terminal run

**Terminal A** — register the XPC Mach service with launchd (starts `pheromoned`):

```bash
cd /Users/wijeratne/dev/HiveClaw
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
cd /Users/wijeratne/dev/HiveClaw
source .venv/bin/activate
make python
python scripts/intelligence_spike.py
```

If `SlabClient` cannot connect, the script prints to stderr:

`pheromoned is not running...` — ensure `make daemon-load` succeeded. The active plist is **`~/Library/LaunchAgents/com.hiveclaw.pheromoned.plist`** (a copy also appears as `com.hiveclaw.pheromoned.gen.plist` in the repo). `ProgramArguments` must point at **`…/target/release/pheromoned`** for this checkout.

**`make daemon-load` → `Bootstrap failed: 5: Input/output error`:** `make daemon-load` installs the plist under **`~/Library/LaunchAgents/`** and bootstraps that path (bootstrapping from a random repo path often hits this on macOS). If it still fails, run **`make daemon-load` from Terminal.app** (full GUI login session); integrated terminals in some IDEs cannot talk to `launchctl`’s `gui/$(id -u)` domain. Ensure the plist is **not world-writable** (`chmod 644` on the file).

## Integration test (`cargo test -p hiveclaw-daemon`)

The XPC test uses `launchctl bootstrap`. In restricted environments (some CI sandboxes), set:

`HIVECLAW_SKIP_LAUNCHD_TEST=1 cargo test -p hiveclaw-daemon`

to skip that test. On a normal macOS login session, leave this unset so the test runs end-to-end.
