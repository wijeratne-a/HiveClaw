# exp-004 — lease expiry after crash, and drain under insert churn

**Date:** 2026-09-01  
**Code:** `Store.try_lease_one_task` reclaim via `payload.lease_until` (unix seconds); `hiveclaw_causal/lease.py` `lease_one_and_die`, `drain_until_idle`.  
**Tests:** `python tests/test_hiveclaw_causal_lease.py`

## Hypothesis

1. A worker that dies after `lease_task` and before `complete_task` (SIGKILL) does not permanently strand the task. After `lease_until`, another worker reclaims it (`lease_reclaim`) and completes it.
2. Inserting new pending tasks **while** workers are draining (not a pre-seeded list) does not produce double-leases or drops.

## Baseline

exp-003: clean queue, workers that always complete, 5×3×8 and Rewind drain. No crash. No producer.

## Method

1. **Crash:** seed 1 pending task. Spawn `lease_one_and_die` (`try_lease` then `SIGKILL`). Wait until the row is `leased` by `crasher`. Sleep `ttl_s + 0.15` (ttl = 0.25 s). Main process drains as `survivor`.
2. **Churn:** 3 spawn workers poll leases; test process inserts 24 tasks with 3 ms gaps; then writes a stop file. Workers exit after idle polls. Stop signal is a **file path** (pickle-safe under spawn); not a `multiprocessing.Event` through `Pool.map`.

`lease_until` is a unix float in JSON payload. Object `updated_at` strings are 1-second resolution and cannot express a 250 ms TTL.

## Race-condition results

`python tests/test_hiveclaw_causal_lease.py -v` (this session, after the spawn stop-file fix):

```
test_continuous_insert_while_workers_drain ... ok
test_killed_worker_lease_is_reclaimed ... ok
test_more_workers_than_tasks_no_double_lease ... ok
test_two_workers_after_rewind_injection ... ok
Ran 4 tests in ~2.1s (full causal suite 29 tests, 2.098s)
OK
```

| check | result |
|-------|--------|
| SIGKILL mid-lease | task `done`; `completed_by=survivor`; `reclaimed_from=crasher`; `lease_reclaim` event present |
| 24 inserts while 3 workers drain | 24 unique leases; all `done`; 0 doubles |

## Outcome

TTL reclaim works on this machine for a kill-9 after a committed lease. Churn drain held for 24 tasks / 3 workers.

Limits: reclaim is **time**, not crash detection. A live worker slower than TTL can lose the lease (not tested). Still single-node SQLite. Stop file is test harness, not a runtime protocol.

## Decision

**Keep** `lease_until` + CAS reclaim in `try_lease_one_task`.

**Do not** treat TTL as a liveness protocol for production workers without measuring slow-but-alive preemption.

## Reproduction

```bash
python tests/test_hiveclaw_causal_lease.py -v
```
