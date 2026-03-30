# Intelligence spike (Phase 4A)

The script runs **Agent A** (prefill → L2-normalized last-token hidden state) writing a **4096-D bf16 scent** to IOSurface slot 0, then **Agent B** generates with the final layer wrapped to add `alpha * scent` to the last position each step (“active steering”). See `scripts/intelligence_spike.py`.

Activate a venv first, then either install spike deps explicitly or use `make python` (which runs `pip install -r scripts/requirements-spike.txt` before maturin):

```bash
cd /path/to/HiveClaw
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

Alternative without `make python`: `cd crates/hiveclaw-python && maturin develop --release` (still run `pip install -r scripts/requirements-spike.txt` once for NumPy / MLX).

## Two-terminal run

**Terminal A** — register the XPC Mach service with launchd (starts `pheromoned`):

```bash
cd /path/to/HiveClaw
cargo build --release -p hiveclaw-daemon
make daemon-load
```

To stop the service:

```bash
make daemon-unload
```

**Terminal B** — run the spike (do not run a bare `make` line that starts with `#`; that is interpreted as the target name `#`):

```bash
source .venv/bin/activate
make python
python scripts/intelligence_spike.py
```

If `SlabClient` cannot connect, the script prints to stderr:

`pheromoned is not running...` — ensure `make daemon-load` succeeded and the generated plist path in `com.hiveclaw.pheromoned.gen.plist` points at `target/release/pheromoned`.

## Integration test (`cargo test -p hiveclaw-daemon`)

The XPC test uses `launchctl bootstrap`. In restricted environments (some CI sandboxes), set:

`HIVECLAW_SKIP_LAUNCHD_TEST=1 cargo test -p hiveclaw-daemon`

to skip that test. On a normal macOS login session, leave this unset so the test runs end-to-end.
