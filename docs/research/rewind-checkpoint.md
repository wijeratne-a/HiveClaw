# Rewind checkpoint

Distinguish **proven** (command ran) vs **implemented but not separately measured**.

---

## State of evidence (Session 5, 2026-09-01)

HEAD when this section was written: local `main` after Session 5 measurements (pre-commit). Interpreter: `.venv/bin/python` 3.11.1. Causal suite: `make test-causal` → **29 tests OK**, mypy 22 files.

This is not a closeness-to-revolution score. It is what the tests and experiment logs actually show.

### Original four guarantees

| Guarantee | What it is | Status | Evidence |
|-----------|------------|--------|----------|
| **A — typed provenance** | Producer, source URI, version/hash, timestamp, trust class on records | **Evidenced** | `tests/test_hiveclaw_causal_rewind.py` (`_assert_guarantee_a` on claims, observations, actions, provider artifact). Types: `SourceRef` / `Provenance`. |
| **B — claims carry invalidation conditions** | Evidence ids, source snapshot, declared invalidation conditions | **Evidenced** | Same e2e (`_assert_guarantee_b_claim` on cache, outage, residual claims). |
| **C — causal edges + append-only history** | Typed edges with rules; status changes append events; events cannot be UPDATE/DELETE | **Evidenced** | Engine 5/5 (`tests/test_hiveclaw_causal_engine.py`: propagation, idempotency, cycle, isolation). Store 3/3: raw `UPDATE`/`DELETE` on `events` abort (`19a112b`). Reverse index used for invalidation *and*, as of Session 5, for task-cone lookup without a full task scan. |
| **D — deterministic policy gate** | In-process authorize/deny; LLM must not produce the decision | **Evidenced** | Policy 3/3 + e2e: rollback `allowed=False` twice with the same reason; `edge` + `rule=block_action` on the blocked action. Demo: `python -m hiveclaw_causal.demo_rewind`. |

### Section 4 / efficiency and coordination sub-hypotheses

These are the Rewind e2e “only affected work” step plus the Session 3–4 stigmergy-on-a-number claims.

| Claim | Status | Exact numbers / links |
|-------|--------|------------------------|
| Targeted repair reaches the **same conclusion** as naive full re-eval | **Evidenced** | 92.0% `outage_explains_pct` (seed 42), rollback blocked, follow-up present at N=12, 100, 500, 2000. exp-001, exp-002. |
| Targeted **touches fewer objects** | **Evidenced** | N=12: 9 vs 19. N=500: 11 vs 507. N=2000: 11 vs 2007. Touched stays ~11 while naive = full graph. |
| Targeted uses **fewer eval_steps** | **Evidenced, and the scaling story changed** | Session 4: targeted eval_steps **grew with task count** (7 → 38 → 173 at N=12/100/500) because every task was inspected to skip it. Session 5: tasks are `depends_on` their target in `reverse_deps`; repair uses `dependent_tasks` on the cone. Targeted eval_steps **6 → 8 → 10 → 10** at N=12/100/500/2000. Naive 19 / 109 / 511 / 2011. |
| Eval-step **gap grows faster than linearly** | **Not supported** | Difference naive − targeted ≈ N (13, 101, 501, 2001): **linear**, which is what O(N) vs O(cone) predicts. Ratio grows with N (~3× → ~201×). Superlinear eval-step savings were not observed. |
| **Wall-clock** targeted is faster | **Evidenced only at sufficient N, small absolute gap** | N=12: overlap / naive sometimes faster (~3 ms). N=500: targeted ~2.6–3.1 ms vs naive ~11.9 ms (~9 ms). N=2000: targeted ~3.0–3.2 ms vs naive ~41–46 ms (~38–43 ms). Still tens of milliseconds on this machine. exp-002. |
| Inspecting unrelated tasks is **required** by the index | **Falsified** | Existing `reverse_deps` was sufficient once task→target edges were written. Engine skips `TASK` on status propagation so those edges do not stale the queue. Not an open TODO; not a new index. |
| `apply_provider_overlap_rule` is O(cone) | **Assumed false / untested at claim scale** | It still inspects **every claim**. This fixture scales unrelated **tasks**, not claims (`R=1–2`). If claims grew like tasks, that scan would return. |
| Multi-process leases, no worker messaging | **Evidenced, single host** | exp-003: 2 workers drain Rewind pending; 5 workers × 3 tasks × 8 trials, **0 double-lease, 0 dropped**. WAL + `BEGIN IMMEDIATE` + CAS on `pending`. |
| Dead worker’s lease is **reclaimed** | **Evidenced, TTL not crash-detection** | exp-004: SIGKILL after lease; after 0.25 s TTL, survivor completes; `reclaimed_from=crasher`; `lease_reclaim` event. A slow live worker could be preempted (not tested). |
| Drain under **continuous insert** | **Evidenced, small N** | exp-004: 24 tasks inserted while 3 workers drain; 24 unique leases, all `done`. |
| Multi-host / IOSurface stigmergy is the same primitive | **Untested** | SQLite file on one machine. Slab `claim_task` is still slot ownership, not this queue. No second-host run. |
| LLM-free outage % | **Evidenced** | `stats.outage_explains_pct`; seed 42 → **92.0** (92/100 timestamps). Stored as float. |

### Smaller in practice than the original efficiency story

**Eval-step scaling (named):** Until Session 5, targeted repair’s eval_steps tracked **total task count**, not the invalidated cone, because skip eligibility scanned `objects_of(TASK)`. That capped how large the efficiency advantage could look. After indexing tasks in `reverse_deps`, that particular cap is gone on this fixture: targeted eval_steps stay ~10 from N=500 to N=2000 while naive tracks N.

What remains small: **wall-clock** (milliseconds, not a capacity-planning result); **claim overlap** still a full claim scan; **lease proof** still one SQLite file.

### Assumed but untested

- Pre-Session-2 SQLite files gaining append-only triggers on next open.
- Lease TTL vs a live worker slower than TTL.
- Overlap-rule cost when claim count ≈ task count.
- Multi-writer correctness beyond these process tests (not a model checker).
- Ironclad burn-in, `integration_test.py --stress`, Phase 7 goldens (out of Rewind scope).
- Any coupling of this causal graph to the Metal/IOSurface slab.

### Largest remaining risk to the core thesis

The thesis under test is that a typed reverse-indexed trace lets agents **coordinate and repair locally** instead of reprocessing the world, and that this is a real coordination primitive (not a single-process demo).

The largest remaining risk is **scope of the coordination evidence**, not the invalidation math: every lease and efficiency number is **one machine, one SQLite file**. Crash reclaim is a **timer**, not failure detection. The product’s other “stigmergy” (IOSurface slots) is still a different object. If the claim is “this replaces a manager / bus / slab for agents,” that is **not evidenced**. If the claim is “Rewind’s graph skips unrelated work and blocks a rollback for documented reasons, and a local CAS queue can drain without double-lease in these tests,” that **is** evidenced, including a now-flat eval-step curve vs N for extra unrelated tasks.

---

## Session 5 — 2026-09-01 (efficiency curve + harder leases)

### Part 1 — cone-indexed tasks, exp-002 re-run

Implemented: `depends_on` edges from tasks to `target_id`; `_after_provider` uses `Store.dependent_tasks`; engine does not status-propagate into `TASK`.

| N before | eval t/n | wall_s run1 t/n | wall_s run2 t/n |
|----------|----------|-----------------|-----------------|
| 12 | 6 / 19 | 0.003131 / 0.003376 | 0.003643 / 0.002916 |
| 100 | 8 / 109 | 0.002610 / 0.004298 | 0.002392 / 0.004426 |
| 500 | 10 / 511 | 0.003107 / 0.011973 | 0.002590 / 0.011888 |
| 2000 | 10 / 2011 | 0.003178 / 0.041284 | 0.003048 / 0.045704 |

Touched 9–11 vs naive full graph. Details: `docs/research/experiments/exp-002-scale-targeted-vs-naive.md`.

### Part 2 — crash reclaim + insert churn

`test_killed_worker_lease_is_reclaimed` and `test_continuous_insert_while_workers_drain` added. Full lease file 4/4. `docs/research/experiments/exp-004-lease-expiry-and-churn.md`.

### Part 3 — this file

Consolidated section above. No “revolutionary” framing.

### Local suite after Session 5

`make test-causal` → **29 tests OK**, mypy 22 files.

### Next highest-value step

If the thesis needs a coordination claim beyond one host: a second process isolation story that is not the same SQLite file (or an honest product decision that H5 is local-only). If the thesis needs efficiency: scale **claims** in the overlap rule, or stop. Do not wire the slab without a named consumer.

---

## Session 4 — 2026-08-31 (CI confirm + scale + concurrent leases)


### Part 1 — GitHub Actions (watched, not assumed)

| Field | Value |
|-------|--------|
| Workflow | Causal runtime |
| Run | https://github.com/wijeratne-a/HiveClaw/actions/runs/33478599893 |
| Conclusion | **success** |
| Event | push `main` |
| SHA | `368b3304223585feb8d7748cd404f094e17bf488` |
| Job `test-causal` | started 2026-09-01T06:40:14Z, completed 2026-09-01T06:40:26Z (~12 s) |
| Steps | checkout, setup-python, install mypy, **make test-causal success** |

Did not proceed to Part 2 until this URL returned `conclusion=success`.

### Part 2 — exp-002 scale table (this machine, seed 42, two runs)

| N before | touched t/n | eval t/n | wall_s run1 t/n | wall_s run2 t/n |
|----------|-------------|----------|-----------------|-----------------|
| 12 | 9 / 19 | 7 / 19 | 0.007134 / 0.007175 | 0.006982 / 0.006677 |
| 100 | 10 / 107 | 38 / 109 | 0.008567 / 0.009886 | 0.007753 / 0.010071 |
| 500 | 11 / 507 | 173 / 511 | 0.012144 / 0.019063 | 0.013902 / 0.019323 |

Both paths: 92.0% support, rollback blocked, follow-up present. Wall-clock gap is absent at N=12, present at N=100 and N=500 (~6–7 ms at 500). Targeted eval_steps still grow with task count (inspect-all-tasks to skip). Details: `docs/research/experiments/exp-002-scale-targeted-vs-naive.md`.

### Part 3 — exp-003 concurrent leases

`python tests/test_hiveclaw_causal_lease.py -v` → 2/2 in 1.612s.

- 2 processes drain Rewind pending tasks after provider injection: no duplicate ids; all `done`.
- 5 processes vs 3 tasks × 8 trials: **0 double-lease**, **0 dropped**.

Primitive: WAL + `BEGIN IMMEDIATE` + `UPDATE … WHERE status=pending`. Workers share only the db path. Not a multi-host proof. `docs/research/experiments/exp-003-concurrent-leases.md`.

### Local suite after Session 4

`make test-causal` → **17 tests OK**, mypy 21 files.

### What is unverified

- New CI run for `09ff106` / `1ed5036` not watched yet (will fire on push).
- Lease test is 8 oversubscribed trials, not an exhaustive race model.

### Next highest-value step

Watch the Session 4 Causal runtime Actions run. Then either more lease trials under load, or stop calling SQLite workers “stigmergy” until a second machine/process isolation story exists.

---

## Session 3 — 2026-08-31 (CI gate + first stigmergy benchmark)

Interpreter: `/Users/wijeratne/dev/HiveClaw/.venv/bin/python` (3.11.1).

### Definition of done

| Item | Status |
|------|--------|
| `make test-causal` + CI job | **Proven.** Target runs unittest `test_hiveclaw_causal_*.py` + mypy on `hiveclaw_causal`. Workflow `.github/workflows/causal.yml` (`ubuntu-latest`, no GPU). |
| Gate fails on a broken test then passes | **Proven.** Injected `self.fail('CI-gate probe: intentional failure')` → `FAILED (failures=1)`, make exit 2. Revert → 13/13 then 14/14 OK after Part 2. |
| Naive full-rerun path exists | **Proven.** `repair="naive"` actually inspects/touches every object; not a mocked count. |
| Comparison table | **Proven.** `python -m hiveclaw_causal.benchmark` twice. |
| Experiment log | **Written.** `docs/research/experiments/exp-001-targeted-vs-full-rerun.md` |
| Demo WIP untouched | **Confirmed** (`git status` still shows `demos/*`, `HEALTH_REPORT*.md`, `.DS_Store` only as leftover). |
| Pushed | Recorded after `git push` (see git hygiene). |

### Part 1 — `make test-causal`

Daemon was still `state = running` after `make daemon-unload` from this terminal (launchctl). Suite does **not** import `hiveclaw_python` / mlx / XPC (`grep` only comments). 13 tests OK + mypy 16 files before Part 2.

**Fail probe (store insert test):**

```
FAIL: test_insert_events_still_succeeds ... AssertionError: CI-gate probe: intentional failure
Ran 13 tests in 0.157s
FAILED (failures=1)
make: *** [test-causal] Error 1
```

Revert: 13 passed, mypy clean.

CI: `.github/workflows/causal.yml` — `on: push` and `pull_request`, job `test-causal`, `pip install mypy`, `make test-causal PYTHON=python3`. First live Actions run is after push (not executed inside this agent session).

Commit: `b1b3a50` `ci: add make test-causal and a CPU-only GitHub Actions job`

### Part 2 — exact numbers (no inflation)

`python -m hiveclaw_causal.benchmark` seed 42, two runs. Same conclusion: **support_pct 92.0**, rollback blocked, follow-up present.

| metric | targeted | naive |
|--------|----------|-------|
| objects_before | 12 | 12 |
| objects_after | 19 | 19 |
| objects_touched | 9 | 19 |
| objects_untouched | 10 | 0 |
| eval_steps | 7 | 19 |
| wall_s run 1 | 0.007147 | 0.006705 |
| wall_s run 2 | 0.007333 | 0.006677 |

Targeted left 10 objects untouched (listed in exp-001). Naive left 0.

Wall-clock: naive was slightly **faster** (~0.6 ms). Fixture is too small for a latency claim. Keep targeted for fewer eval steps; do not advertise speed.

After Part 2: `make test-causal` → **14 tests OK**, mypy **19 files**.

Commit: `4fe3c6d` `feat: measure targeted Rewind repair against a naive full rerun`

### What is implemented but unverified

- GitHub Actions `causal.yml` has not been observed green on github.com in this session (only local make).
- Wall-clock is not a stable metric at ~7 ms.

### Single next highest-value step

Watch the first `Causal runtime` Actions run on `main`. If green, scale the fixture (more unrelated claims/tasks) until wall-clock or eval_steps show a gap a reviewer would care about — or accept that this N=12 graph is only a skip-list demo.

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
| Rewind commits pushed | **Proven.** `git push origin HEAD` → `d577ed0..469796c  HEAD -> main` |
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

Session 2 commits (causal only): `19a112b`, `42a759f`, `469796c`. Pushed with Session 1 Rewind history: `d577ed0..469796c` on `origin/main`. Unrelated unstaged WIP left as-is: `demos/*`, `HEALTH_REPORT*.md`, `.DS_Store`.

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
