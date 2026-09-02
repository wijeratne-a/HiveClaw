# Rewind checkpoint

Distinguish **proven** (command ran) vs **implemented but not separately measured**.

---

## State of evidence (Session 10, 2026-09-01)

HEAD: `ff4b146e65f10407d3b3552ce3a9a2328faf1afa` on `origin/main` (Session 9 operator tooling). CI: https://github.com/wijeratne-a/HiveClaw/actions/runs/33596227872 (`test-causal` success on that SHA). Interpreter: `.venv/bin/python` 3.11.1.

Session 10 is documentation reconciliation after `docs/research/independent-audit-2026-09-01.md`. It does not change causal math. Historical session SHAs/run IDs below are left as they were for those sessions (verified 2026-09-01: they still resolve).

**Evidenced claim (precise):** Rewind is a **centralized causal store** (one SQLite file or one Postgres server) with **safe concurrent multi-process and networked clients**: bounded-cost invalidation on this fixture, append-only events, TTL/heartbeat leases, and a hard TTL ceiling so a client cannot strand work with an infinite/hour-long lease. It is **not** “no central manager” and **not** decentralized stigmergy.

This is not a closeness-to-revolution score.

### Original four guarantees

| Guarantee | What it is | Status | Evidence |
|-----------|------------|--------|----------|
| **A — typed provenance** | Producer, source URI, version/hash, timestamp, trust class on records | **Evidenced** | `tests/test_hiveclaw_causal_rewind.py` (`_assert_guarantee_a` on claims, observations, actions, provider artifact). Types: `SourceRef` / `Provenance`. |
| **B — claims carry invalidation conditions** | Evidence ids, source snapshot, declared invalidation conditions | **Evidenced** | Same e2e (`_assert_guarantee_b_claim` on cache, outage, residual claims). |
| **C — causal edges + append-only history** | Typed edges with rules; status changes append events; events cannot be UPDATE/DELETE | **Evidenced** | Engine 5/5. Store 3/3: raw `UPDATE`/`DELETE` on `events` abort. Reverse index used for invalidation, task-cone lookup, and provider-topic claim lookup. |
| **D — deterministic policy gate** | In-process authorize/deny; LLM must not produce the decision | **Evidenced** | Policy 3/3 + e2e: rollback `allowed=False` twice with the same reason; `edge` + `rule=block_action`. |

### Section 4 / efficiency and coordination sub-hypotheses

| Claim | Status | Exact numbers / links |
|-------|--------|------------------------|
| Targeted repair reaches the **same conclusion** as naive full re-eval | **Evidenced** | 92.0% `outage_explains_pct` (seed 42), rollback blocked, follow-up present at task-N=12/100/500/2000 and claim-C=0/500/2000. exp-001, exp-002. |
| Targeted **touches fewer objects** | **Evidenced** | Task-N=2000: 11 vs 2007. Claim-C=2000: 9 vs 2019. |
| Targeted uses **fewer eval_steps** (tasks) | **Evidenced** | SQLite Session 5: 6 → 8 → 10 → 10 vs naive 19 / 109 / 511 / 2011. Postgres Session 7: 6 → 7 → 8 → 8 vs 19 / 108 / 509 / 2009. |
| Targeted uses **fewer eval_steps** (unrelated claims) | **Evidenced (Session 6–7)** | Topic index: eval **6** vs naive 19 / 519 / 2019 at C=0/500/2000 on **both** SQLite and Postgres. |
| Eval-step **gap grows faster than linearly** | **Not supported** | Difference ≈ N (linear). |
| **Wall-clock** targeted is faster | **Evidenced only at sufficient N on SQLite; weak/noisy over TCP** | SQLite C=2000: ~3 ms vs ~30–35 ms. Postgres N=500: ~33–38 ms **both** paths. Postgres N=2000: ~30 ms vs ~73–78 ms. |
| Inspecting unrelated **tasks** is required by the index | **Falsified** | Task→target `depends_on`; `dependent_tasks` on the cone. |
| Inspecting unrelated **claims** is required because they cannot join reverse_deps | **Falsified as O(N) scan; scoped as topic-index** | A new provider observation is **not** a reverse-dep parent of existing claims until overlap fires. Claims are indexed on `topic-provider-status` at create time. Same `reverse_deps` table, different key than the task cone. |
| Multi-process leases, no worker messaging | **Evidenced** | SQLite exp-003: 5×3×8, **0 double-lease**. Postgres Session 7: same, 0 doubles. |
| Dead / silent worker’s lease is **reclaimed** | **Evidenced** | SQLite SIGKILL. Postgres SIGKILL and **TCP drop with process still alive**. |
| Slow-alive worker that **renews** is **not** reclaimed | **Evidenced (Session 6–7)** | SQLite and Postgres: 0.7 s work, 0.2 s TTL, renew every 0.05 s; poacher `None`. |
| Heartbeat delayed longer than TTL (stall, not death) | **Evidenced (Session 7)** | Proxy stall > TTL: poacher reclaims the live worker. Silence = reclaim. |
| Unbounded / huge client TTL can strand after TCP drop | **Closed (Session 8)** | `LEASE_TTL_CEILING_S = 30` in Python + `lease_config` + INSERT/UPDATE trigger. Client 3600s or `inf` is clamped. SQLite silence + Postgres TCP-drop tests reclaim within a 1s store max. |
| Event-log integrity / restore drill / lease dashboard | **Evidenced (Session 9)** | `verify-store` (read-only), SQLite backup API + restore to an isolated file, `store-status`, schema migrate `--confirm`, 8×5×3 contention still **0 doubles**. Failed lease attempts, reclaim latency, and process health are **not** persisted (`recorded: false`). |
| Drain under **continuous insert** | **Evidenced, small N, SQLite only** | 24 tasks / 3 workers. Not re-run on Postgres. |
| Multi-host stigmergy / no central store | **Out of scope** | Not “untested, try next session.” The data model **requires** one authoritative store. See `docs/research/decentralization-assessment.md`. |
| LLM-free outage % | **Evidenced** | seed 42 → **92.0** (SQLite and Postgres). |

### Smaller in practice than the original efficiency story

**Eval-step scaling (named):** Session 4–5 task-scan cap is gone. Session 6 claim-scan cap is gone **for provider-interest overlap**, via a topic key, not the observation cone. Session 7: the same pattern holds on Postgres over TCP; exact task-side integers drifted by 1–2 vs SQLite. Wall-clock on a local file remains milliseconds; over Docker TCP the gap is smaller and noisier. Naive is still O(N) by design.

### Assumed but untested

- Pre-Session-2 SQLite files gaining append-only triggers on next open.
- Multi-writer correctness beyond these process tests (not a model checker).
- Ironclad burn-in, `integration_test.py --stress`, Phase 7 goldens.
- Any coupling of this causal graph to the Metal/IOSurface slab.
- Continuous-insert drain on Postgres (SQLite only).
- Two independent Rewind stores merging without a shared server — **not an untested extra; Session 8 marks it a redesign, out of scope.**

### Largest remaining risks (severity order)

1. **Out of scope — decentralized / “no central manager” stigmergy.** Session 8’s answer is **no**: the event log, reverse-deps index, and lease CAS require a single authoritative store. SQLite and Postgres are the same architecture. This is no longer listed as the next experiment; it is a different product. Memo: `docs/research/decentralization-assessment.md`.
2. **Closed Session 6 — claim-side O(N) overlap scan.** Unrelated claims at C=500 and C=2000 keep targeted eval_steps at **6** vs naive 519 / 2019. Limit: `topic-provider-status`, not `dependent_claims(obs.id)`.
3. **Closed Session 8 — TTL-strand / unbounded lease.** Absolute ceiling 30s (schema CHECK + trigger + Python clamp). Store may set a stricter max (tests use 1s). Slow-alive + renew still holds. Remaining operational limit: silence (no heartbeat, including stalled network) is reclaimed after the **clamped** TTL, not after process death.
4. **Closed Session 9 as product ops, not as HA.** One store still means backup/restore *is* disaster recovery. Verifier + drill exist; there is no standby replica. Threat model is local developer / trusted internal, not multi-tenant.

If the claim is “this replaces a manager / bus / slab for agents,” that is **not evidenced and not in scope** for this runtime. If the claim is “Rewind is a centralized causal store: it skips unrelated tasks and non-provider claims on this fixture, blocks rollback for documented reasons, and a CAS queue can drain over a file or a TCP database, reclaim silence and dropped connections, respect heartbeats, and refuse an indefinite TTL,” that **is** evidenced. Session 9 adds: you can **verify**, **back up**, **inspect leases**, and **migrate** that one store without changing the coordination model.

### Demonstration UI (explicitly descoped)

The original product brief asked for a visual Rewind demo a non-technical reviewer could follow in under five minutes: a living work map, a rewind timeline, and a graphical “why?” inspector.

**None of that has been built in any session.** There is no Rewind HTML/TSX UI. What exists is CLI: `python -m hiveclaw_causal` / `demo_rewind.py`, `inspect.py`, `benchmark.py`, and Session 9 `store-status` / `verify-store`. HiveClaw’s SAE TUI (`hiveclaw-dashboard`) is a different product (slab latents), not this causal graph.

**Decision (Session 10):** descoped for now. A GUI sprint is not the next causal-runtime milestone. Proof remains CLI + tests. Revisit only if a named reviewer cannot use the CLI.

Discovery docs: `docs/research/architecture-map.md` and `docs/research/gap-analysis.md` (Session 10; the original pre-code pair was never written).

---

## Session 10 — 2026-09-01 (reconcile independent audit)

Independent audit: `docs/research/independent-audit-2026-09-01.md`. Causal core matched; paperwork and uncommitted Session 9 did not.

| Item | Action |
|------|--------|
| Session 9 on `origin/main` | Commit `ff4b146`. CI https://github.com/wijeratne-a/HiveClaw/actions/runs/33596227872 success. |
| Checkpoint HEAD / run ID | Banner updated from stale `f4496ee` / 33582036614 (that pair remains valid **for Session 8 docs** `f4496ee`). |
| exp-001 / exp-002 drift | Addenda only; original tables kept. |
| architecture-map / gap-analysis | Written now as current-state docs (they had never existed). |
| Rewind GUI | Descoped; section above. |

Historical run IDs left in place (checked against GitHub API 2026-09-01): 33582036614 (`f4496ee`), 33578062030 (`68d1f59`), 33575886839 (`9c64bd9`), 33478599893 (`368b330`). Session 8 tip-after-CI-record `c273453` is https://github.com/wijeratne-a/HiveClaw/actions/runs/33582094903.

---

## Session 9 — 2026-09-01 (centralized-store operational hardening)

Goal: make the one-authoritative-store architecture operable. Not a new backend, not CRDTs.

| Item | What shipped |
|------|----------------|
| **B verifier** | `python -m hiveclaw_causal verify-store --db …` read-only JSON + summary. Seq monotonic; gaps info-only; projection replay; reverse-deps; `has_applied`; illegal lease/complete transitions; append-only + TTL triggers; schema version. No event hash column (`event_checksums.recorded = false`). |
| **C backup/restore** | SQLite `Connection.backup` + `PRAGMA integrity_check`. Restore `--confirm` to an isolated path, then verify. Drill: fixture → backup while writer open → restore → lease + provider invalidation still work. Postgres: document `pg_dump`; in-process schema copy when DSN set. |
| **A store-status** | `store-status --db …` task counts, active leases (owner, acquired, expiry, remaining TTL, renewal/reclaim counts, clamp), near-expiry, clamped event count, reclaim event count, TTL vs ceiling. Honest gaps: failed attempts, reclaim latency, process health. |
| **D migrate** | Schema version 2 stamped. `migrate --to-latest --confirm`. No automatic downgrade. SQLite flock + Postgres advisory lock 872343. Post-migrate verifier. |
| **E contract** | `docs/research/deployment-contract.md` — SQLite vs Postgres, NFS warning, one DSN, clocks, retries vs CAS. |
| **F stress** | Extra 8 workers × 5 tasks × 3 trials, **0 doubles**. Duplicate `complete_task` retry does not append a second event. SIGKILL/TCP-drop remain Session 5–8 tests (not loosened). |
| **G threat model** | `docs/research/threat-model.md` — local developer / trusted internal. Not multi-tenant. |

CLI entry: `python -m hiveclaw_causal <command>` (`hiveclaw-causal` in prose). Bare `python -m hiveclaw_causal` still runs the demo.

DR write-up: `docs/research/backup-restore.md`.

### Local suite after Session 9

`make test-causal` → **44 tests OK**, 10 Postgres skipped, mypy **33 files**.  
`HIVECLAW_PG_DSN=... python -m unittest tests.test_hiveclaw_causal_pg -v` → **10/10 OK** (includes in-process schema copy drill).

---

## Session 8 — 2026-09-01 (TTL ceiling + scope-lock)

### Part 1 — strand risk

Failing test first: `test_oversized_client_ttl_does_not_strand_after_silence` (`Store(..., max_lease_ttl_s=1.0)` unexpected kwarg, then client 3600s would have left ~3600s on the row). Fix: `hiveclaw_causal/lease_policy.py` `LEASE_TTL_CEILING_S = 30`, `lease_config`, SQLite/Postgres triggers, clamp on `try_lease_one_task` / `renew_lease`. Same test then passes; inf clamped; raw SQL `lease_until` beyond ceiling aborted. Postgres: `test_oversized_ttl_tcp_drop_is_reclaimed_within_ceiling`. SQLite and Postgres lease suites re-run; slow-alive still green.

### Part 2 — memo

`docs/research/decentralization-assessment.md`: **No**, not without a fundamental redesign.

### Part 3 — framing

This file, the ADR, exp-004-multi-host, CONTEXT: “no central manager” is not a proven property and is not the near-term roadmap.

### Local suite after Session 8

`make test-causal` → **34 tests OK**, 9 Postgres skipped, mypy 26 files.  
`HIVECLAW_PG_DSN=... python -m unittest tests.test_hiveclaw_causal_pg -v` → **9/9 OK** (includes oversized-TTL TCP drop).

---

## Session 7 — 2026-09-01 (Postgres over TCP)

### Part 1 — scope before code

Named in `docs/research/experiments/exp-004-multi-host.md` before `PgStore` existed: **(a)** one Postgres over TCP, not true multi-host replication, not two threads on one SQLite file.

### Part 2 — what ran

Docker `postgres:16` on port 55432. `PgStore` + `TcpProxy`. Scaling table and 8/8 networked tests in that experiment file. Local `make test-causal` without DSN: 31 OK + 8 skipped, mypy 25 files.  
CI: https://github.com/wijeratne-a/HiveClaw/actions/runs/33578062030 job `test-causal` success.

### Part 3 — verdict

Bounded eval_steps and lease safety **held** on a real TCP hop to a single server. Wall-clock advantage **degraded**. “No central manager” was still unproven (Session 8 later: **out of scope**, not a pending backend).

### Part 4 — this risk ranking

Ordered 1–3 as above. Risk 1 is restated, not closed.

### Local suite after Session 7

`make test-causal` → **31 tests OK**, 8 skipped, mypy 25 files.  
`HIVECLAW_PG_DSN=... python -m unittest tests.test_hiveclaw_causal_pg -v` → **8/8 OK**.

---

## Session 6 — 2026-09-02 (commit Session 5, claim index, lease heartbeat)

### Part 1 — Session 5 committed and CI-green

Four commits on `main`, tip `9c64bd9`: `f9874a6` task index, `7eba213` exp-002, `e39e56e` lease crash/churn, `9c64bd9` evidence summary.

Local post-commit: `make test-causal` 29/29 + mypy.  
CI: https://github.com/wijeratne-a/HiveClaw/actions/runs/33575886839 job `test-causal` success (~15 s), step `make test-causal` success.

### Part 2 — claim-side overlap

Implemented topic index + `extra_unrelated_claims`. Numbers in exp-002 Session 6 table.

### Part 3 — heartbeat

`Store.renew_lease`; tests slow-alive vs SIGKILL. exp-004.

### Part 4 — this risk ranking

Ordered 1–3 as above.

### Local suite after Session 6

`make test-causal` → **31 tests OK**, mypy 22 files.

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
