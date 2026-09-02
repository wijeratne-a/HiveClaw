# exp-003 — concurrent task leases from shared SQLite (no manager)

**Date:** 2026-08-31  
**Code:** `hiveclaw_causal/store.py` `try_lease_one_task` / `complete_task`; workers in `hiveclaw_causal/lease.py`.  
**Tests:** `python tests/test_hiveclaw_causal_lease.py`

## Hypothesis

Independent worker **processes** can lease pending tasks from the same SQLite file using only shared-state reads/writes (no sockets, queues, or direct worker-to-worker messages). Each pending task is leased by **exactly one** worker (`BEGIN IMMEDIATE` + `UPDATE … WHERE status='pending'`). Idle workers get nothing when the queue is empty. All tasks reach `done`.

## Baseline

Previous sessions had `TaskStatus.LEASED` in the type enum but **no** lease implementation and **no** concurrent test. ADR mentioned leases; they were unverified.

## Method

1. SQLite WAL + `busy_timeout=8000`. Each worker opens its **own** `Store` connection.
2. Lease: `BEGIN IMMEDIATE`; select one `kind=task AND status=pending`; `UPDATE … WHERE id=? AND status=pending`; append `lease_task` event; `COMMIT`. `rowcount != 1` → retry/abort (no lease).
3. Complete: `UPDATE … WHERE status=leased` and `payload.lease_owner` matches the worker.
4. Overlap: `multiprocessing` spawn pool; `time.sleep(0.003–0.004)` between lease and complete.
5. **Test A:** full Rewind after provider injection; 2 processes drain all pending tasks.
6. **Test B (break attempt):** 5 processes, 3 synthetic pending tasks, 8 independent trials.

Workers receive only `(db_path, worker_id, pause_s)`. They do not share Python objects.

## Race-condition results

`python tests/test_hiveclaw_causal_lease.py -v` (this session):

```
test_more_workers_than_tasks_no_double_lease ... ok
test_two_workers_after_rewind_injection ... ok
Ran 2 tests in 1.612s
OK
```

| check | result |
|-------|--------|
| Test A: 2 workers vs Rewind pending tasks | all pending ids leased once; all tasks `done` |
| Test B: 5 workers × 3 tasks × 8 trials | **double-lease trials = 0**; **dropped-task trials = 0**; all 3 tasks `done` every trial |

This is not a formal proof against every SQLite interleaving. It is 8 tight oversubscribed runs plus one Rewind run on this machine, all clean.

## Outcome

The CAS lease held under **more workers than tasks** in these runs. Coordination was file-backed SQLite, not a central in-process manager.

Limits: single-node SQLite, not multi-host; WAL required; `BEGIN IMMEDIATE` serializes writers (correctness over throughput). No LLM involved.

## Decision

**Keep** the IMMEDIATE + pending-status CAS lease as the queue primitive.

**Do not** treat this as a distributed stigmergy proof. It is a local multi-process queue test.

**Revise later** if a double-lease appears under heavier load (more trials, more tasks, shorter pauses). Crash/expiry and insert-while-drain: `docs/research/experiments/exp-004-lease-expiry-and-churn.md`.

## Reproduction

```bash
python tests/test_hiveclaw_causal_lease.py -v
```
