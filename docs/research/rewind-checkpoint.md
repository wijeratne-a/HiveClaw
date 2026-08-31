# Rewind checkpoint — 2026-08-30

**HEAD at start of session:** `d577ed0`  
**This checkpoint is after the Rewind vertical slice.** Distinguish **proven** (command ran) vs **implemented but not separately measured**.

## Definition of done

| Item | Status |
|------|--------|
| Rewind e2e passes twice in a row | **Proven.** `python tests/test_hiveclaw_causal_rewind.py` twice, both 2/2 OK |
| Rollback blocked by policy code | **Proven.** Policy unit tests + e2e + demo: `allowed=False` with `edge=… rule=block_action` |
| Invalidation unit tests (propagation, idempotency, cycle, isolation) | **Proven.** `python tests/test_hiveclaw_causal_engine.py` 5/5 OK |
| Outage-explains-N% from fixture data | **Proven.** `hiveclaw_causal.stats.outage_explains_pct`; seed 42 → **92.0** (92/100 timestamps in window). Claim payload stores the float, not a string |
| ADR | **Written.** `docs/adr/CAUSAL_RUNTIME_H5.md` |
| Existing repo tests | **Proven** for the subset run below; daemon crate IPC **not** re-run (Mach collision) |
| Documented demo command | **Proven.** `python -m hiveclaw_causal.demo_rewind` |

## What was verified to work

Interpreter: `/Users/wijeratne/dev/HiveClaw/.venv/bin/python` (3.11.1). Daemon running for Metal tests.

| Command | Result |
|---------|--------|
| `python tests/test_hiveclaw_causal_engine.py` | 5 passed |
| `python tests/test_hiveclaw_causal_policy.py` | 3 passed |
| `python tests/test_hiveclaw_causal_rewind.py` (run 1) | 2 passed |
| `python tests/test_hiveclaw_causal_rewind.py` (run 2) | 2 passed |
| `python -m hiveclaw_causal.demo_rewind --db /tmp/hiveclaw-rewind-demo.sqlite` | exit 0; pct 92.0; rollback blocked |
| `python -m hiveclaw_causal.inspect --db … --id action-rollback-release` | status=blocked, edge+rule printed |
| `python tests/test_quality_controller.py` | 19 passed |
| `python tests/integration_test.py --quick` | pass |
| `python tests/test_continuous_batching.py` | pass (golden skipped) |
| `python tests/test_sae_tied_weights.py` | pass |
| `python tests/test_batched_steering.py` | pass |
| `ruff check hiveclaw_causal tests/test_hiveclaw_causal_*.py` | All checks passed (system ruff; not in venv) |

Demo printed (live query, not canned copy): cache claim `challenged` via `edge-contradicts-obs-provider-outage-claim-cache-regression` / `hard_challenge`; rollback `blocked` via `edge-justifies-…` / `block_action`; scheduled `task-verify-outage-pct`, `task-followup-cache`; not rerun: provider-crosscheck (gap filled) and unrelated-docs.

## What is implemented but unverified

- SQLite `events` table is append-only **by API** (`INSERT` only). No DB trigger forbidding `UPDATE`/`DELETE` was tested.
- Multi-writer leases: schema-ready in spirit (task statuses) but **single-process**; lease_owner columns were **not** added. Not required by Rewind tests.
- `mypy` / `pyright`: **not run** (not in `.venv`; no repo type-check target).
- `cargo test -p hiveclaw-daemon`: **not run** this session after discovery (would fight live `pheromoned`).
- `integration_test.py --stress`, Phase 7 goldens, ironclad burn-in: **not run**.

## What was deferred and why

- Patch worker, experiment planner, UI projection — contract appendix.
- LLM investigator — Rewind uses deterministic ingestor/investigator.
- Comparison spike H1–H4 — discovery found no in-tree graph DB/bus; ADR records that.
- Wiring causal objects onto the IOSurface slab — would couple CPU tests to GPU/daemon.

## Current build/test status

- Metal/MLX stack: **unchanged**; existing tests that were re-run still pass.
- New package: `hiveclaw_causal/` (CPU, SQLite). Does not import `hiveclaw_python`.
- Unrelated demo WIP (`demos/*`, `HEALTH_REPORT*.md`) left unstaged.

## Single next highest-value step

Add a SQLite trigger (or test) that **fails if `events` are updated/deleted**, so Guarantee C is enforced by the database rather than convention. Optional: a Makefile `test-causal` target so CI can run the CPU suite without pytest.
