# HiveClaw repository baseline (Rewind / causal-runtime discovery)

**Date:** 2026-08-30  
**Checkout:** `/Users/wijeratne/dev/HiveClaw`  
**HEAD:** `d577ed0` on `main` (`feat(demos): Repo Pulse audit demo …`), tracks `origin/main`.  
**Purpose:** Record verified facts vs inferences before any causal-runtime application code. This document is the discovery artifact for the Rewind build contract.

Legend:

- **Verified:** observed by running a command, reading a file, or inspecting git in this session.
- **Inferred:** reasonable engineering conclusion, not executed or not proven by a test.
- **Not run:** known suite or target that was skipped, with reason.

---

## 1. Product and stack (verified)

HiveClaw is a local Apple Silicon inference stack: Metal / IOSurface slab + MLX generation + OpenAI-compatible FastAPI. Multi-agent “stigmergy” in this repo means **VRAM-backed SAE latents on a shared slab**, not a typed causal trace.

| Fact | Evidence |
|------|----------|
| License AGPL-3.0 | `LICENSE`, `crates/hiveclaw-python/pyproject.toml` |
| Platform guard | `hiveclaw_python/__init__.py` raises `NotImplementedError` unless Darwin + arm64 |
| This host | `uname -sm` → `Darwin arm64` |
| Canonical venv | `.venv/bin/python3` → Python **3.11.1** (system `python3` is 3.14.5; do not use it) |
| Slab layout | Code is **v6** (`EXPECT_MAGIC = 0x48434C5700000006` in `tests/integration_test.py`; `SLAB_VERSION_V6` in `crates/hiveclaw-core/src/math.rs`). APIs still named `*_v5`. |
| Default SAE | `models/hiveclaw_sae_v1.safetensors` present locally (2.0M, 2026-04-05); gitignored |
| Living context | `CONTEXT.md` (uncommitted as of this discovery pass) |

Rust workspace members (`Cargo.toml`): `hiveclaw-core`, `hiveclaw-backend-metal`, `hiveclaw-daemon`, `hiveclaw-python`. `crates/hiveclaw-mlx` is CMake/nanobind, not a workspace member. `crates/hiveclaw-core/src/traits.rs` is an **empty placeholder** (“future phases”).

---

## 2. Existing agent / state / event abstractions (verified)

There is **no** typed claim/plan/action graph, append-only event log, reverse-dependency index, or policy gate for irreversible actions.

| Abstraction | What it actually is |
|-------------|---------------------|
| `SlabClient.claim_task` / `release_task` | IOSurface **slot** lifecycle (pid owner). Not a causal claim. |
| `Swarm` (`hiveclaw_python/swarm.py`) | Cosine ranking + slot claim helper. |
| `LocalSwarm` / `swarm_agents` | LLM agent subprocesses writing latents. |
| `HIVECLAW_STIGMERGY` | Enable slab claim/read/write on the generate path. |
| `quality_gate` | YAML-profile **generate → verify → repair** for Python fenced code. Types: `QualityProfile`, `Violation`, `GateReport`. No evidence IDs, no invalidation graph. |
| `catenar_tracing.py` | Optional Proof-of-Task tracer; no-ops without SDK. |
| `hiveclaw-core` traits | Empty. |

Grep for `event log`, `EventStore`, `invalidat` (application sense), and `causal` (non-KV-mask) found **no** causal-runtime module. The word “claim” in this codebase means **slab slot claimed**, not a justified hypothesis.

**Inferred:** Rewind cannot be bolted onto the IOSurface slab or `quality_gate` without fighting the existing product. A new CPU-only package at repo root (`hiveclaw_causal/`) is the smallest isolation that (a) does not import `hiveclaw_python` (mlx + Darwin guard), (b) does not require `pheromoned`, (c) follows the same `quality_gate/` + `tests/` pattern.

---

## 3. Language, build, test layout (verified)

| Layer | How |
|-------|-----|
| Rust | `make build` / `cargo build --workspace`; daemon `make build-release` |
| Python native | `make python PYTHON="$(pwd)/.venv/bin/python3"` (PyO3 + mlx ext). **Does not run tests.** |
| Daemon | `make daemon-load` → LaunchAgent `com.hiveclaw.pheromoned`. This session: **running**, program = this checkout’s `target/release/pheromoned`, `--latent-dim 256`. |
| Python tests | `tests/*.py` — mix of `unittest.main()`, argparse harnesses, and a `main()` that calls pytest-style functions. **pytest is not installed in `.venv`.** |
| Rust tests | `make test-ipc` → `cargo test -p hiveclaw-daemon -- --test-threads=1` (same Mach label as the live daemon). |
| Lint / type-check | **No** Makefile, pyproject, or CI target for ruff/mypy/pyright. `ruff` and `mypy` exist on the **system** PATH (`/Library/Frameworks/Python.framework/Versions/3.11/bin/`) but are **not** importable in `.venv`. |
| CI | Wheel build (macos-14) + ironclad burn-in (self-hosted Mac). No causal-runtime job. |

Existing Python test files:

- `tests/test_quality_controller.py` — CPU, unittest, no daemon
- `tests/test_continuous_batching.py` — MLX, no daemon (golden optional)
- `tests/test_sae_tied_weights.py` — MLX + SAE weights
- `tests/test_batched_steering.py` — daemon + SAE
- `tests/integration_test.py` — daemon (`--quick`, `--batched`, `--stress`)

---

## 4. Test baseline actually run (verified, 2026-08-30)

Interpreter: `/Users/wijeratne/dev/HiveClaw/.venv/bin/python` (3.11.1). Working directory: repo root. Daemon: running.

| Command | Result |
|---------|--------|
| `python tests/test_quality_controller.py` | **PASS** — 19 tests, 0.655s |
| `python tests/integration_test.py --quick` | **PASS** — `ok 256` |
| `python tests/integration_test.py --batched` | **PASS** — batched read/write ok |
| `python tests/test_continuous_batching.py` | **PASS** — golden skipped (`HIVECLAW_PHASE7_GOLDEN` unset); compiled-decode CI skipped |
| `python tests/test_sae_tied_weights.py` | **PASS** |
| `python tests/test_batched_steering.py` | **PASS** — duplicate, B=1 parity, B=2 shapes, torn epoch, clamp telemetry |
| `cargo test -p hiveclaw-core --offline` | **PASS** — 1 unit test (`default_layout_fits_legacy_slab_cap`) |
| `cargo test -p hiveclaw-backend-metal --offline` | **PASS** — 0 tests in crate |
| `cargo test -p hiveclaw-python --lib --offline` | **PASS** — 0 tests in crate |

**Not run (and why):**

| Target | Reason |
|--------|--------|
| `python tests/integration_test.py --stress` | Heavy claim/release + swarm_spike child; not required to establish baseline. |
| `HIVECLAW_PHASE7_GOLDEN=1` / `HIVECLAW_COMPILE_DECODE_CI=1` | Optional GPU/mlx-lm goldens; skipped by the test file by default. |
| `make test-ipc` / `cargo test -p hiveclaw-daemon` | Tests bootstrap the **same** Mach label `com.hiveclaw.pheromoned` as the live LaunchAgent. Would collide with the running daemon used by Python slab tests. |
| `bash .github/scripts/ci_ironclad_verify.sh` | Heavy self-hosted burn-in. |
| `make python` | Extensions already built (slab tests handshake). |

**Uncommitted working tree (do not mix into Rewind commits):** modified `demos/*`; untracked `demos/consensus_showdown.py`, `demos/llm_*.py`, `HEALTH_REPORT*.md`, `.DS_Store`, `CONTEXT.md`.

---

## 5. Architecture choice implication (inferred, for the ADR)

H1–H4 (external graph DB, message bus, CRDT, ad-hoc JSON files without an index) have **no working primitive in this tree**. Discovery found:

- SQLite is not used by the product.
- No Redis / NATS / Kafka client.
- No Neo4j / SQLite graph.
- Persistence today is IOSurface + optional Catenar HTTP.

Therefore the Rewind slice should implement **H5 by default**: append-only event log + current-state projection + indexed reverse-dependency lookup + leased task queue + deterministic policy check. A half-day comparison spike is **not** warranted: there is nothing real to compare against.

Package location (inferred, discovery-driven): **`hiveclaw_causal/` at repo root**, tests under `tests/test_hiveclaw_causal_*.py`, persistence SQLite (events append-never-overwrite). Must not import `hiveclaw_python` so CPU tests run without GPU/daemon.

---

## 6. Constraints the Rewind slice must respect (verified)

1. Do not change slab layout, XPC handshake, or `hiveclaw_python` imports in a way that breaks Metal tests.
2. Do not commit SAE weights, `.env`, `HEALTH_REPORT*.md`, `.DS_Store`, or unrelated demo WIP.
3. Prefer many small git commits; leave demo WIP unstaged.
4. No LLM required for Rewind e2e — investigator may be a deterministic fixture proposer.
5. Existing “stigmergy” name is occupied; causal objects should use explicit type names (`Claim`, `Artifact`, …) and not overload slab `claim_task`.
