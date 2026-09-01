# Rewind checkpoint

Distinguish **proven** (command ran) vs **implemented but not separately measured**.

**HEAD at start of Session 2:** `b456a74`  
**This file:** Session 2 (2026-08-31) update of the Session 1 (2026-08-30) checkpoint. Not a new file.

---

## Session 2 — 2026-08-31 (harden the Rewind slice)

Interpreter: `/Users/wijeratne/dev/HiveClaw/.venv/bin/python` (3.11.1).  
mypy: system `/Library/Frameworks/Python.framework/Versions/3.11/bin/mypy` 1.20.0 with `--python-executable .venv/bin/python` (mypy is **not** in `.venv`; no CI mypy target exists).

### Definition of done (this session)

| Item | Status |
|------|--------|
| Raw SQL UPDATE/DELETE on `events` fails after trigger; test was red first | **Proven.** Before trigger: 2 failed (`IntegrityError not raised`). After: store 3/3 OK |
| Full causal suite green after trigger | **Proven.** store 3/3, engine 5/5, policy 3/3, e2e 2/2 (e2e run twice after trigger) |
| Daemon crate tests with recorded output | **Proven.** `cargo test -p hiveclaw-daemon -- --test-threads=1` → ipc 5/5, phase_c stub 1/1, lib/bin/doctest 0 tests. Boot-out EIO 5 on launchctl (same as IDE/GUI domain noise); tests still **ok** |
| mypy recorded; new vs pre-existing | **Proven.** `hiveclaw_causal` + causal tests: 1 new error (`lastrowid` None) → fixed → **Success: no issues found in 16 source files**. `quality_gate`: 4 **pre-existing** errors (`yaml` stubs, assignment, AST vs Module). No CI mypy job |
| IOSurface slab wiring | **Deferred** (see Priority 3 below). No implementation this session |
| Rewind commits pushed | Recorded after push in git hygiene below |
| `demos/*` / `HEALTH_REPORT*.md` untouched | **Confirmed** at commit time: only causal store/test/docs in Session 2 commits |

### Priority 1 — append-only at the DB

**Red (before schema change):** `python tests/test_hiveclaw_causal_store.py`

```
.FF
FAIL: test_raw_delete_events_is_aborted — AssertionError: IntegrityError not raised
FAIL: test_raw_update_events_is_aborted — AssertionError: IntegrityError not raised
Ran 3 tests … FAILED (failures=2)
```

INSERT-only path (`test_insert_events_still_succeeds`) already passed.

**Fix:** `BEFORE UPDATE` / `BEFORE DELETE` triggers on `events` raising `RAISE(ABORT, 'events is append-only: …')` in `hiveclaw_causal/store.py`.

**Green:** store 3/3; engine 5/5; policy 3/3; rewind 2/2 then 2/2 again.

Commit: `19a112b` `test: enforce events append-only with SQLite UPDATE/DELETE triggers`

### Priority 2 — daemon crate tests and mypy

**Daemon (actual output, not inferred):**

```
cargo test -p hiveclaw-daemon -- --test-threads=1
  hiveclaw_daemon lib:  0 passed; 0 failed
  pheromoned bin:       0 passed; 0 failed
  ipc_test.rs:          5 passed; 0 failed  (1.60s)
  phase_c_test.rs:      1 passed; 0 failed  (phase_c_suite_is_ipc_macos_module)
  doc-tests:            0 passed; 0 failed
```

launchctl printed `Boot-out failed: 5: Input/output error` on each ipc test (test helper bootouts the shared Mach label). Tests still passed. Side effect: live LaunchAgent was unloaded after the suite; restored with `make daemon-load` (bootstrap OK).

No daemon **code** changes. No hiveclaw_causal import leak into the crate (would have been a compile/test failure; none observed).

**mypy hiveclaw_causal (new package):** 1 error at `store.py` `int(cur.lastrowid)` — sqlite `lastrowid` is `int | None`. Fixed with an explicit `None` check. Re-run: `Success: no issues found in 16 source files`.

**mypy quality_gate (pre-existing, not introduced by hiveclaw_causal):** 4 errors (`import-untyped` yaml, `ImportError` assignment, two AST vs Module). Not fixed this session (out of causal scope).

Commit: `42a759f` `fix: handle sqlite lastrowid None in causal event append`

### Priority 3 — IOSurface slab wiring (scope check only)

**Does Rewind require slab wiring?** No. The acceptance tests and demo persist typed records in SQLite and query that store. They do not import `hiveclaw_python`, do not talk to `pheromoned`, and do not read/write IOSurface slots.

**Named near-term consumer?** None in this tree. UI projection worker is an appendix deferral. Slab `claim_task` is slot ownership, not a causal claim. Putting claim JSON into 256-D bf16 latents would be a new encoding, not a drop-in.

**Why not a partial wire:** It would couple the CPU-only suite to Metal/daemon, reverse the isolation that makes Rewind CI-runnable, and is speculative without a consumer. Session 1 already noted this; Session 2 confirms and **does not implement**.

### Git hygiene

Session 2 commits (causal only): `19a112b`, `42a759f`, plus this docs commit. Unrelated unstaged WIP left as-is: `demos/*`, `HEALTH_REPORT*.md`, `.DS_Store`.

### What is implemented but unverified

- Triggers apply on `Store._init_schema` (`CREATE TRIGGER IF NOT EXISTS`). DBs created before Session 2 gain triggers on next `Store()` open; not separately tested against a pre-trigger file on disk.
- Multi-writer leases still single-process (unchanged).
- `integration_test.py --stress`, Phase 7 goldens, ironclad burn-in: still **not run**.

### Current build/test status

- `hiveclaw_causal/`: store + engine + policy + rewind green; mypy clean on that package.
- Daemon crate: ipc 5 + stub 1 green this session.
- Metal/MLX / `hiveclaw_python`: not modified this session.

### Single next highest-value step

A Makefile `test-causal` target (unittest, no pytest required) so CI can run the CPU suite on every push without GPU. Do **not** start slab wiring until a named consumer exists.

---

## Session 1 — 2026-08-30 (original slice)

**HEAD at start of session:** `d577ed0`

### Definition of done

| Item | Status |
|------|--------|
| Rewind e2e passes twice in a row | **Proven.** `python tests/test_hiveclaw_causal_rewind.py` twice, both 2/2 OK |
| Rollback blocked by policy code | **Proven.** Policy unit tests + e2e + demo: `allowed=False` with `edge=… rule=block_action` |
| Invalidation unit tests (propagation, idempotency, cycle, isolation) | **Proven.** `python tests/test_hiveclaw_causal_engine.py` 5/5 OK |
| Outage-explains-N% from fixture data | **Proven.** `hiveclaw_causal.stats.outage_explains_pct`; seed 42 → **92.0** (92/100 timestamps in window). Claim payload stores the float, not a string |
| ADR | **Written.** `docs/adr/CAUSAL_RUNTIME_H5.md` |
| Existing repo tests | **Proven** for the subset run below; daemon crate IPC **not** re-run in Session 1 (Mach collision). **Session 2 re-ran them (green).** |
| Documented demo command | **Proven.** `python -m hiveclaw_causal.demo_rewind` |

### What was verified to work (Session 1)

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
