# Independent audit — HiveClaw Rewind / causal runtime

**Auditor role:** verification only. This file is the report. No application code was changed, no commit, no push.  
**When:** 2026-09-01 (local), interpreter `.venv/bin/python3` → Python 3.11.1.  
**Important scope note:** Commands were run against the **dirty working tree**. Committed `HEAD` / `origin/main` is `c273453`. Session 9 operator tooling exists only as uncommitted files. Claims about Session 9 are therefore claims about the working tree unless marked otherwise.

Legend: **Match** = reproduced exactly as claimed. **Mismatch** = differs, even slightly. **Cannot verify** = not checkable here.

---

## Summary table

| # | Claim | How verified | Actual result | Status |
|---|--------|--------------|---------------|--------|
| 1 | Local `HEAD` is `f4496ee` on `origin/main` (checkpoint “State of evidence”) | `git rev-parse HEAD`; `git ls-remote origin refs/heads/main`; `git show HEAD:docs/research/rewind-checkpoint.md` | `HEAD` = `c273453449f6db2d8d9f91b3a661d646c6fc1dbb`. `origin/main` = same SHA. Committed checkpoint still says `HEAD: f4496ee`. Working-tree checkpoint still says `f4496ee` under a Session 9 header. | **Mismatch** |
| 2 | Branch is `main`, tracks origin, clean or only demo WIP | `git status`; `git status --porcelain=v1` | On `main`, up to date with `origin/main`, **not clean**. Unstaged + untracked listed in Part 1. | **Mismatch** vs “demo WIP only” |
| 3 | `docs/research/repository-baseline.md` exists, non-empty | `stat` | EXISTS, 8323 bytes | **Match** |
| 4 | `docs/research/architecture-map.md` exists | path check; `git log --all --` that path | **MISSING.** Never in git history. | **Mismatch** (absent) |
| 5 | `docs/research/gap-analysis.md` exists | path check; `git log --all --` | **MISSING.** Never in git history. | **Mismatch** (absent) |
| 6 | `docs/research/rewind-checkpoint.md` exists, non-empty | `stat` | EXISTS, 27792 bytes (working tree) | **Match** (file exists) |
| 7 | `docs/adr/CAUSAL_RUNTIME_H5.md` exists, non-empty | `stat` | EXISTS, 3710 bytes (working tree; also modified vs HEAD) | **Match** (file exists) |
| 8 | Every file under `docs/research/experiments/` exists | directory listing | 5 files, all non-empty: exp-001, exp-002, exp-003, exp-004-lease, exp-004-multi-host | **Match** |
| 9 | `make test-causal` Session 9: 44 OK, 10 skipped | `make test-causal PYTHON=.venv/bin/python3` on dirty tree | `Ran 54 tests in 5.719s` `OK (skipped=10)` → **44 passed + 10 skipped** | **Match** (working tree only) |
| 10 | Session 8: 34 OK, 9 skipped | Count `def test_` on `HEAD` causal files (not a clean-tree re-run) | HEAD: 43 methods = 34 + 9 Postgres. Did **not** check out a clean tree to re-execute. | **Cannot verify** by re-run; **count matches** HEAD sources |
| 11 | mypy: 33 files, no errors (Session 9 WT) | `.venv/bin/python3 -m mypy hiveclaw_causal tests/test_hiveclaw_causal_*.py` | `Success: no issues found in 33 source files` | **Match** (working tree) |
| 12 | Postgres suite 10/10 with DSN | `HIVECLAW_PG_DSN=postgresql://hiveclaw:hiveclaw@127.0.0.1:55432/hiveclaw python -m unittest tests.test_hiveclaw_causal_pg -v`; Docker `hiveclaw-exp004-pg` on 55432 | `Ran 10 tests in 14.097s` `OK` | **Match** (this machine, Docker up) |
| 13 | exp-004-multi-host: `make test-causal` skips **8** Postgres tests | `make test-causal` skip lines | **10** skipped (includes `test_logical_backup_restore_and_verify`) | **Mismatch** (file stale: 8 vs 10) |
| 14 | Daemon crate: ipc 5/5 + phase_c stub 1/1 | `cargo test -p hiveclaw-daemon -- --test-threads=1` | lib 0, bin 0, ipc **5 passed**, phase_c **1 passed**, doctest 0. `Boot-out failed: 5: Input/output error` on ipc tests. | **Match** (counts); side effect on live LaunchAgent as Session 2 described |
| 15 | exp-001: targeted eval_steps **7**, naive **19**, objects 9 vs 19, 92.0% | `python -m hiveclaw_causal.benchmark` twice | eval **6** / **19**; touched 9 / 19; untouched 10 / 0; support 92.0 both; rollback True; follow-up True. wall ~0.00285 / ~0.00326 (not ~0.007) | **Mismatch** (eval 6≠7; wall ≠ recorded) |
| 16 | exp-002 Session 5 N=500 SQLite: eval **10** / **511** | `python -m hiveclaw_causal.benchmark --extra-unrelated 162 --extra-related 2` | eval **8** / **509**; touched 11 / 507; before 500; 92.0%; rollback True | **Mismatch** (−2 / −2 eval vs Session 5 table) |
| 17 | exp-002 Session 6 C=500: eval **6** / **519** | `--extra-unrelated-claims 500` | eval **6** / **519**; touched 9 / 519; before 512; 92.0% | **Match** (eval/objects); wall 0.002826 / 0.011129 vs recorded ~0.003 / ~0.010 |
| 18 | exp-004 PG N=500: eval **8** / **509**, ~33–38 ms both paths | `measure_repair` on PgStore, U=162 R=2 | eval **8** / **509**; touched 11 / 507; wall **0.0344** / **0.0387**; 92.0% | **Match** (eval/touched); wall within a few ms of table |
| 19 | exp-004 PG C=500: eval **6** / **519** | same, C=500 | **6** / **519**; touched 9 / 519; wall 0.0247 / 0.034; 92.0% | **Match** (eval); wall naive 0.034 vs recorded ~0.032 |
| 20 | Append-only `BEFORE UPDATE`/`DELETE` on `events` | Read `store.py` schema; raw SQL on a temp Store | Triggers present; UPDATE/DELETE raise `IntegrityError` `events is append-only: …` | **Match** |
| 21 | `_put_task` writes `depends_on` into `reverse_deps` | Read `rewind.py` `_put_task`; `store.add_edge` + `_index_pair` | `add_edge` INSERT `reverse_deps`; `DEPENDS_ON` indexes `(dst, src)` so `dependent_tasks(target)` finds the task | **Match** |
| 22 | Topic key `topic-provider-status` at claim create; overlap uses `dependent_claims` not full claim scan | Read `engine.index_provider_interest`, `apply_provider_overlap_rule` | Indexed via `depends_on` to `TOPIC_PROVIDER_STATUS`; overlap iterates `dependent_claims(TOPIC_PROVIDER_STATUS)` | **Match** |
| 23 | Heartbeat is real renew; reclaim is TTL since last renewal, not lease start | Read `renew_lease`, `try_lease_one_task` SELECT, `work_slow_with_renew` | Reclaim iff `lease_until < now`; `renew_lease` sets `lease_until = now + ttl`. Heartbeat loop calls `renew_lease`. | **Match** (equivalent to last-renewal+TTL; no separate heartbeat timestamp) |
| 24 | Session 8 TTL ceiling implemented (`LEASE_TTL_CEILING_S=30`), not still open | Read `lease_policy.py`; triggers; tests ran green | Constant 30.0; clamp; SQLite/PG triggers; strand tests OK on this run | **Match** (in HEAD **and** WT) |
| 25 | Policy gate blocks rollback via claim/action **status fields**, not a UI string | Read `policy.authorize` | Checks `ActionStatus`, justifying `EdgeMode.JUSTIFIES`, claim `status` in `_FRESH` | **Match** |
| 26 | `outage_explains_pct` 92.0 is computed from fixture timestamps, not a constant | Read `stats.py`; `build_rewind_fixture(seed=42)` | 100 failures, 92 in window, `pct==92.0`; function is `100.0 * n / len(failures)` | **Match** |
| 27 | CI run `33582036614` success on Session 8 SHA | `curl` GitHub API (no `gh` binary) | conclusion `success`, `head_sha` = `f4496ee477a680a742ce0fc29bd6a829c16dcd8e`, job `test-causal` / step `make test-causal` success | **Match** |
| 28 | CI run `33578062030` Session 7 success | API | success; SHA `68d1f59e8e1afb63a74eae534599496df39d4c37` (Postgres feat, not the later docs commit) | **Match** (SHA is the feat commit) |
| 29 | CI run `33575886839` Session 6 / `9c64bd9` success | API | success; SHA `9c64bd9dbc3dbc81d71df5cca33641dcb864c819` | **Match** |
| 30 | CI run `33478599893` Session 4 / `368b330` success | API | success; SHA `368b3304223585feb8d7748cd404f094e17bf488`; event `push` | **Match** |
| 31 | Session 4 **action** pins are the SHAs now in `causal.yml` | `git show 368b330:.github/workflows/causal.yml` vs current file | At Session 4: `ubuntu-latest`, `actions/checkout@v4`, `setup-python@v5` (unpinned). **Now:** `ubuntu-24.04`, checkout `3d3c42e5…`, setup-python `5fda3b95…` | **Mismatch** if read as “Session 4 pins = current file”; **Match** vs hardening report `reports/github_actions_causal_ci_hardening.md` |
| 32 | Current `causal.yml` pins those two SHAs | Read `.github/workflows/causal.yml` | checkout `@3d3c42e5aac5ba805825da76410c181273ba90b1`; setup-python `@5fda3b95a4ea91299a34e894583c3862153e4b97`; `on: push, pull_request`; `python-version: "3.11"`; `make test-causal PYTHON=python3` | **Match** (current file vs hardening doc) |
| 33 | Cited commit SHAs exist with claimed messages | `git log -1` / `git cat-file` for every SHA listed in Part 4 | All listed SHAs exist; messages match the feat/docs titles claimed | **Match** |
| 34 | Local HEAD equals `origin/main` | `git ls-remote` vs `git rev-parse` | Both `c273453449f6db2d8d9f91b3a661d646c6fc1dbb` | **Match** (committed tips). Working tree is **ahead in files, not commits**. |
| 35 | Session 9 verifier/backup/migrate/store-status are in the repo product | `git ls-files hiveclaw_causal`; `git status` | Those modules are **untracked**. Not on `origin/main`. | **Mismatch** vs “shipped on main” |
| 36 | “No central manager” is not claimed as proven | grep docs | Caveated in checkpoint, ADR, decentralization-assessment, exp-004-multi-host. **Uncaveated leftover:** exp-003 **title** “no manager” | **Mismatch** (title leftover); body of exp-003 already says not distributed |
| 37 | Rewind “living work map / timeline / why? inspector UI” exists | glob under `hiveclaw_causal`; grep those phrases | **No** HTML/TSX UI. CLI: `demo_rewind.py`, `inspect.py` (and untracked `cli.py`). HiveClaw SAE TUI is a different product. | **Mismatch** vs original brief UI; **Match** if the claim is “CLI inspector only” |
| 38 | `gh` CLI available for Actions | `which gh` | `gh not found`. Used public GitHub HTTP API instead for the four run IDs. | **Cannot verify** via `gh`; **verified** via API |
| 39 | Full `python scripts/exp004_multi_host.py` (all scales × 2 + unittest) | Not executed in full (N=2000×2 + C=2000×2 skipped for time) | Subset: PG unittest 10/10; N=500 and C=500 one pair each | **Cannot verify** full script output vs table |
| 40 | Ironclad / `integration_test.py --stress` / Phase 7 goldens | Not run (checkpoint lists as untested) | Not run | **Cannot verify** (and checkpoint already says untested) |

---

## Part 1 — Local repository ground truth

### HEAD, branch, dirty tree

```
c273453449f6db2d8d9f91b3a661d646c6fc1dbb
BRANCH=main
Your branch is up to date with 'origin/main'.
```

`git log --oneline -15` (newest first):

```
c273453 docs: record Session 8 causal CI run URL
f4496ee docs: scope-lock Rewind as a centralized causal store
0f97a88 feat: clamp lease TTL so a dropped client cannot strand a task
1329abf docs: record Session 7 causal CI run URL
68d1f59 feat: run Rewind against Postgres over TCP and record the limits
0a557b2 docs: Session 6 claim-scale numbers, heartbeat evidence, and risk ranking
4e200ac feat: renew leases so slow-alive workers are not reclaimed
4a47330 feat: index provider-interest claims on a topic key
9c64bd9 docs: Session 5 evidence summary for Rewind guarantees and limits
e39e56e feat: reclaim expired leases after crash and drain under insert churn
7eba213 docs: record cone-indexed exp-002 eval_steps at N=500 and N=2000
f9874a6 feat: index tasks in reverse_deps and schedule from the cone
fe286d1 docs: record causal CI hardening GitHub Actions evidence
27bb7eb ci: pin Actions to node24 SHAs and harden causal workflow
2194fc5 docs: Session 4 checkpoint with CI run URL, exp-002 scales, exp-003 leases
```

### Unstaged (modified)

`CONTEXT.md`, `demos/README.md`, `demos/audit_swarm.py`, `demos/baseline_audit.py`, `demos/run_repo_pulse.py`, `docs/adr/CAUSAL_RUNTIME_H5.md`, `docs/causal/rewind-demo.md`, `docs/research/rewind-checkpoint.md`, `hiveclaw_causal/__main__.py`, `hiveclaw_causal/lease_policy.py`, `hiveclaw_causal/pg_store.py`, `hiveclaw_causal/store.py`, `tests/test_hiveclaw_causal_pg.py`

### Untracked

`.DS_Store`, `HEALTH_REPORT.md`, `HEALTH_REPORT_BASELINE.md`, `demos/consensus_showdown.py`, `demos/health_report_validate.py`, `demos/json_utils.py`, `demos/llm_ab.py`, `demos/llm_client.py`, `docs/research/backup-restore.md`, `docs/research/deployment-contract.md`, `docs/research/threat-model.md`, `hiveclaw_causal/backup.py`, `hiveclaw_causal/cli.py`, `hiveclaw_causal/migrate.py`, `hiveclaw_causal/ops_status.py`, `hiveclaw_causal/schema.py`, `hiveclaw_causal/verify.py`, `tests/test_hiveclaw_causal_ops.py`

(This audit file itself is an additional untracked report if written to `docs/research/`.)

### Top-level tracked tree (`git ls-files`, 180 paths)

Top-level files: `.cursorrules`, `.gitignore`, `CANONICAL.md`, `CONTEXT.md`, `Cargo.lock`, `Cargo.toml`, `LICENSE`, `Makefile`, `README.md`.  
Directories: `.github/`, `.vscode/`, `benchmarks/`, `crates/`, `demos/`, `docs/`, `examples/`, `hiveclaw_causal/`, `internal/`, `models/`, `quality_gate/`, `reports/`, `requirements/`, `scripts/`, `tests/`, `tools/`, `training/`.

### Docs required by the audit contract vs disk

| Path | Disk |
|------|------|
| `docs/research/repository-baseline.md` | present, 8323 B |
| `docs/research/architecture-map.md` | **does not exist** (never committed) |
| `docs/research/gap-analysis.md` | **does not exist** (never committed) |
| `docs/research/rewind-checkpoint.md` | present |
| `docs/adr/CAUSAL_RUNTIME_H5.md` | present |
| `docs/research/experiments/*` | 5 files, all non-empty |

**Present, not in the audit’s required list (tracked):** `docs/research/decentralization-assessment.md`, `docs/adr/BATCHED_STEERING_CONTRACT.md`, `docs/causal/rewind-demo.md`, `reports/github_actions_causal_ci_hardening.md`.  
**Present, untracked (Session 9):** `deployment-contract.md`, `backup-restore.md`, `threat-model.md`.

### `hiveclaw_causal/` modules (what the code does)

Tracked on HEAD:

| Module | What it does (from reading the file) |
|--------|--------------------------------------|
| `__init__.py` | Re-exports types only. |
| `__main__.py` | Dispatches CLI commands if first arg matches; otherwise runs `demo_rewind` (WT adds more commands). |
| `types.py` | Dataclasses/enums: records, edges, events, statuses, provenance. |
| `util.py` | Canonical JSON + SHA-256 `content_hash`. |
| `fixture.py` | Seeded Rewind scenario: failures, outage window, provider report. |
| `stats.py` | `outage_explains_pct` = fraction of failure timestamps inside the outage window. |
| `work.py` | `WorkCounter`: `eval_steps` and touched ids. |
| `store.py` | SQLite event log, objects, edges, reverse_deps, leases, append-only + TTL triggers. |
| `pg_store.py` | Same records over one Postgres database (TCP). |
| `engine.py` | Invalidation rules, topic index helper, `InvalidationEngine`. |
| `policy.py` | `authorize()` deterministic allow/deny for actions. |
| `rewind.py` | Orchestrator: ingest, `_put_task`, targeted/naive repair, verify %. |
| `lease.py` | Multiprocess drain, SIGKILL helper, renew loop, TCP-drop helper. |
| `lease_policy.py` | TTL default/ceiling/clamp (Session 8). |
| `netproxy.py` | Userspace TCP proxy: stall/drop without killing the client process. |
| `benchmark.py` | Targeted vs naive `measure_repair` CLI. |
| `inspect.py` | CLI: print one object’s status + last event reason. |
| `demo_rewind.py` | CLI demo writing a SQLite file and printing explanations. |

Untracked (working tree only): `cli.py` (operator subcommands), `verify.py` (read-only integrity report), `ops_status.py` (lease dashboard), `backup.py` (SQLite backup API + PG schema copy), `migrate.py` (schema version 2, `--confirm`), `schema.py` (`SCHEMA_VERSION = 2`).

---

## Part 2 — Tests and experiments (raw)

### `make test-causal` (dirty tree)

```
Ran 54 tests in 5.719s
OK (skipped=10)
Success: no issues found in 33 source files
OK: test-causal (unittest + mypy hiveclaw_causal)
```

Skipped reason (all 10): `HIVECLAW_PG_DSN not set` during this make invocation.

### mypy alone

```
Success: no issues found in 33 source files
```

### Postgres (Docker `hiveclaw-exp004-pg`, port 55432)

```
Ran 10 tests in 14.097s
OK
```

DSN used: `postgresql://hiveclaw:hiveclaw@127.0.0.1:55432/hiveclaw` (local Docker; not logged further).

### Daemon crate

`cargo test -p hiveclaw-daemon -- --test-threads=1`

- `hiveclaw_daemon` lib: 0 tests  
- `pheromoned` bin: 0 tests  
- `ipc_test.rs`: **5 passed**, 1.55s; many `Boot-out failed: 5: Input/output error`  
- `phase_c_test.rs`: **1 passed** (`phase_c_suite_is_ipc_macos_module`)  
- doc-tests: 0  

This **did** talk to the live Mach/LaunchAgent path (boot-out errors). Matches Session 2’s description. This audit did **not** restore `pheromoned` afterward.

### `python -m hiveclaw_causal.benchmark` (seed 42), two runs

| metric | run1 t/n | run2 t/n | exp-001 table |
|--------|----------|----------|---------------|
| objects_before | 12/12 | 12/12 | 12/12 |
| objects_after | 19/19 | 19/19 | 19/19 |
| objects_touched | 9/19 | 9/19 | 9/19 |
| objects_untouched | 10/0 | 10/0 | 10/0 |
| eval_steps | **6**/19 | **6**/19 | **7**/19 |
| support_pct | 92.0 | 92.0 | 92.0 |
| rollback_blocked | True | True | True |
| wall_s | 0.002848 / 0.003274 | 0.002915 / 0.003242 | ~0.007 / ~0.0067 |

Untouched targeted list **matches** exp-001 exactly.

Session 5 already recorded N=12 eval 7→**6**. **exp-001 was not updated.**

### Scaled SQLite vs exp-002

N=500 (`U=162,R=2`): eval **8/509** vs Session 5 table **10/511**. Touched 11/507 matches. 92.0% matches.

C=500: eval **6/519** matches Session 6 table.

### Postgres subset vs exp-004-multi-host

N=500: eval **8/509**, touched 11/507, wall 0.0344/0.0387 vs table run1 0.0364/0.0372.  
C=500: eval **6/519**, wall 0.0247/0.034 vs ~0.0248/0.0316.

Full `scripts/exp004_multi_host.py` (all scales × 2) was **not** run.

---

## Part 3 — Mechanisms (code, not docs)

### Append-only

`hiveclaw_causal/store.py` creates:

- `events_append_only_no_update` BEFORE UPDATE  
- `events_append_only_no_delete` BEFORE DELETE  

Live probe on a temp DB: both raise `sqlite3.IntegrityError` with `events is append-only`. Triggers listed also included `lease_until_absolute_ceiling_insert/update`.

### Task reverse-deps

`RewindRuntime._put_task` (approx. lines 564–600) `add_edge` with `mode=DEPENDS_ON`, `src=tid`, `dst=target`.  
`Store.add_edge` inserts `reverse_deps`.  
`_index_pair`: `DEPENDS_ON` → `(edge.dst, edge.src)` so the **target** is the lookup key.  
`_after_provider` uses `_tasks_in_cone` / reverse_deps, with an explicit comment not to `objects_of(TASK)`.

### Topic claims

`TOPIC_PROVIDER_STATUS = "topic-provider-status"`.  
`index_provider_interest` adds `depends_on` from the claim to that key if `_mentions_provider`.  
`apply_provider_overlap_rule` loops `self.store.dependent_claims(TOPIC_PROVIDER_STATUS)` and `inspect`s those rows only.

### Leases / heartbeat / reclaim

Acquisition/reclaim (`try_lease_one_task`): pending **or** (`status=leased` AND `lease_until < now`).  
`renew_lease`: requires current owner; writes `lease_until = time.time() + ttl`; **no event**.  
`work_slow_with_renew`: loop `renew_lease` until work deadline.  

Reclaim is **not** “age since first acquire.” It is expiry of the **current** `lease_until`, which renewals push forward.

### TTL strand (Session 8)

`LEASE_TTL_CEILING_S = 30.0` in `lease_policy.py`. Clamp on lease/renew. Schema CHECK + triggers. Tests `test_oversized_client_ttl_does_not_strand_after_silence` and PG `test_oversized_ttl_tcp_drop_is_reclaimed_within_ceiling` passed in this session. **Implemented in committed `0f97a88`, not an open item.**

### Policy

`authorize` loads the action record, rejects `EXECUTED`/`BLOCKED`/`executed` payload, requires `approved`, requires `JUSTIFIES` edges, requires justifying claim `status in {active, corroborated, verified}`. If blocked, reason includes last event `edge_id` and `rule`. No hardcoded `"rollback"` string.

### 92.0%

```
failures 100 in_window 92 pct 92.0
```

`outage_explains_pct` in `stats.py` is arithmetic over fixture timestamps.

---

## Part 4 — GitHub

`gh` CLI: **not installed**. Public API `https://api.github.com/repos/wijeratne-a/HiveClaw/actions/runs/...` returned HTTP 200.

| Run ID | conclusion | head_sha | title |
|--------|------------|----------|--------|
| 33582036614 | success | `f4496ee477a680a742ce0fc29bd6a829c16dcd8e` | docs: scope-lock… |
| 33578062030 | success | `68d1f59e8e1afb63a74eae534599496df39d4c37` | feat: Postgres over TCP… |
| 33575886839 | success | `9c64bd9dbc3dbc81d71df5cca33641dcb864c819` | docs: Session 5 evidence… |
| 33478599893 | success | `368b3304223585feb8d7748cd404f094e17bf488` | docs: Session 3 checkpoint… |

Latest causal run on `c273453`: **33582094903** success (`docs: record Session 8 causal CI run URL`). **Not cited** in the checkpoint HEAD line (still points at 33582036614 / `f4496ee`).

`origin/main` = `c273453…` = local committed HEAD.

**Tags:** none local, none on origin.

**Other branch:** local `feat/llm-swarm-integration` at `b785ba47` (2026-04-01), **0 commits ahead of main**, **63 behind**, **not on origin**. No unique unmerged work.

### Cited SHAs (all exist)

Including: `d577ed0`, `19a112b`, `42a759f`, `469796c`, `b1b3a50`, `4fe3c6d`, `368b330` (full `368b3304223585feb8d7748cd404f094e17bf488`), `09ff106`, `1ed5036`, `2194fc5`, `f9874a6`, `7eba213`, `e39e56e`, `9c64bd9`, `4e200ac`, `4a47330`, `0a557b2`, `68d1f59`, `1329abf`, `0f97a88`, `f4496ee`, `c273453`, `27bb7eb`, `fe286d1`, `3c09dc7`, `fece403`, `ad972e3`, `0aced8d`, `55e3bac`, `1ab90d3`, `2672472`, `3ee1054`, `b456a74`, `9e31ccf`, `fd1cb35`. Messages match the claimed feat/docs titles.

### Workflow `causal.yml` now vs Session 4

Now: `ubuntu-24.04`; pinned checkout `3d3c42e5aac5ba805825da76410c181273ba90b1`; pinned setup-python `5fda3b95a4ea91299a34e894583c3862153e4b97`.  
At `368b330`: `ubuntu-latest`; floating `@v4` / `@v5`.  
Session 3 checkpoint text still says `ubuntu-latest` (historical paragraph, never edited).

---

## Part 5 — Narrative vs evidence

### Experiment-file internal issues

- **exp-001** summary table still has targeted eval_steps **7**; current code/Session 5 N=12 is **6**. Wall-clock also stale. Conclusion checks (92.0, rollback, follow-up, 9 vs 19 touched) still hold.  
- **exp-002 Session 5 N=500** still lists **10/511**; this SQLite re-run is **8/509** (same integers as Session 7 Postgres / current topic-index world). Session 6 C=* tables still match.  
- **exp-003** title: “shared SQLite **(no manager)**”. Body correctly limits to local multi-process. Title conflicts with Session 8 language. Recorded “Ran 2 tests”; lease file now has more tests (historical).  
- **exp-004-lease** numbering jumps 1 → 3 (no item 2). Harmless. Full suite line still says 31 tests OK (Session 6).  
- **exp-004-multi-host** eval table for N=500/C=500 **matched** this re-run. Skip-count “8 Postgres tests” is stale (**10** now, including untracked backup test). Wall-clock is noisy; this run’s N=500 pair was close, not identical.

### Checkpoint risk ranking vs experiment caveats

| Rank | Checkpoint | Experiment support |
|------|------------|-------------------|
| 1 Out of scope / not “no central manager” | Supported by decentralization-assessment and exp-004 verdict | **Match**, except exp-003 title |
| 2 Claim O(N) closed via topic index | C=500 re-run 6 vs 519 eval_steps | **Match** for eval_steps; files still say not a production latency claim |
| 3 TTL strand closed | Code + tests this session | **Match** |
| 4 Session 9 ops / not HA | Code exists **uncommitted**; no standby | **Mismatch** if read as “on main”; **Match** that HA is still backup of one store |

### UI / original brief

No Rewind timeline, living work map, or graphical “why?” inspector. `inspect.py` prints status and last event fields. SAE `hiveclaw-dashboard` is the inference slab TUI, not Rewind.

---

## Confirmed claims (exact)

- Guarantees A–D still have passing tests in this run (rewind e2e, policy 3, engine 5, store append-only + ceiling).  
- 92.0% is `92/100` timestamps, seed 42.  
- Concurrent leases tests passed (including 5×3×8 and Session 9 8×5×3 on the dirty tree).  
- SIGKILL reclaim, slow-alive renew, PG TCP-drop, stall>TTL, oversized TTL clamp: passed where executed.  
- Four cited Actions run IDs exist, `conclusion=success`, SHAs as in the table above.  
- `origin/main` == local committed HEAD `c273453`.  
- Action pin SHAs in current `causal.yml` match the hardening report.  
- TTL ceiling is real in schema + Python, committed since `0f97a88`.

---

## Discrepancies (do not round away)

1. Checkpoint “current HEAD `f4496ee`” vs actual `c273453` (even the **committed** checkpoint is one docs commit behind). Latest CI for current tip is **33582094903**, not 33582036614.  
2. Working tree is a large uncommitted Session 9 + demo WIP; not “demos only.” Session 9 is **not** on GitHub `main`.  
3. `architecture-map.md` and `gap-analysis.md` do not exist.  
4. exp-001 eval_steps 7 vs live **6**.  
5. exp-002 Session 5 N=500 eval 10/511 vs live SQLite **8/509**.  
6. exp-004 skip count 8 vs live **10**.  
7. Session 3 text still `ubuntu-latest`; file is `ubuntu-24.04`. Session 4 used unpinned v4/v5 actions.  
8. exp-003 title still “no manager.”  
9. Wall-clock numbers in exp-001/002 differ from this host (expected noise **and** cheaper targeted path).  
10. Session 9 checkpoint claims operator product “evidenced” while those files are untracked.

---

## Cannot verify

- `gh` CLI (used HTTP API instead for the four runs; that part **was** verified).  
- Full `scripts/exp004_multi_host.py` matrix (N=2000 and all double runs).  
- Clean-tree `make test-causal` at `c273453` without Session 9 files (count of 34+9 is consistent with HEAD sources only).  
- Ironclad burn-in, `integration_test.py --stress`, Phase 7 goldens.  
- Whether cargo ipc tests left `pheromoned` unloaded (not checked; historically they do).  
- Private job logs beyond API `conclusion` / step names.  
- Multi-tenant / production TLS deployments (not claimed as evidenced).

---

## Net assessment

The **committed** Rewind product on `origin/main` (`c273453`) is a **centralized** SQLite-or-Postgres causal store with append-only events, reverse-dep / topic indexes, CAS leases, heartbeat, and a **hard TTL ceiling**. Those mechanisms are in the code, tests on this machine passed (including Docker Postgres), and the four historical Actions URLs are real successes on the SHAs claimed (with Session 7’s URL pointing at the Postgres **feat** commit, which is correct).

It is **not** accurate to treat the working-tree Session 9 write-up as GitHub state: verifier, backup, migrate, and `store-status` are local uncommitted files. The checkpoint’s own HEAD line is already wrong relative to `origin/main` by one commit. Two discovery docs the build contract named (`architecture-map`, `gap-analysis`) were never created. Efficiency logs **exp-001** and **exp-002 Session 5** still publish eval_step integers the current SQLite path no longer produces (6 not 7; 8/509 not 10/511), while **correctness** numbers (92.0%, rollback, object-touch counts at those scales) still reproduce.

Do not conclude “everything matches.” The causal core mostly does; the paperwork and the Session 9/main relationship do not.
