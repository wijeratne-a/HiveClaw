# exp-004 — lease expiry after crash, and drain under insert churn

**Date:** 2026-09-01  
**Code:** `Store.try_lease_one_task` reclaim via `payload.lease_until` (unix seconds); `hiveclaw_causal/lease.py` `lease_one_and_die`, `drain_until_idle`.  
**Tests:** `python tests/test_hiveclaw_causal_lease.py`

## Hypothesis

1. A worker that dies after `lease_task` and before `complete_task` (SIGKILL) does not permanently strand the task. After `lease_until`, another worker reclaims it (`lease_reclaim`) and completes it.
3. A worker that is **slow but alive** and **renews** `lease_until` must not be reclaimed. A worker that is **silent** (SIGKILL, no renew) must still be reclaimed after TTL.

## Baseline

exp-003: clean queue, workers that always complete, 5×3×8 and Rewind drain. No crash. No producer.

## Method

1. **Crash:** seed 1 pending task. Spawn `lease_one_and_die` (`try_lease` then `SIGKILL`). Wait until the row is `leased` by `crasher`. Sleep `ttl_s + 0.15` (ttl = 0.25 s). Main process drains as `survivor`.
3. **Heartbeat:** `work_slow_with_renew` leases, works 0.7 s with TTL 0.2 s, renews every 0.05 s. After TTL+0.2 s a poacher `try_lease` must get nothing. Worker then completes. `test_killed_worker_lease_is_reclaimed` is the no-renewal control.

`lease_until` is a unix float in JSON payload. Object `updated_at` strings are 1-second resolution and cannot express a 250 ms TTL.

## Race-condition results

`python tests/test_hiveclaw_causal_lease.py -v` (this session, after the spawn stop-file fix):

```
test_continuous_insert_while_workers_drain ... ok
test_killed_worker_lease_is_reclaimed ... ok
test_slow_alive_worker_that_renews_is_not_reclaimed ... ok
test_more_workers_than_tasks_no_double_lease ... ok
test_two_workers_after_rewind_injection ... ok
```

Full causal suite Session 6: **31 tests OK**.

| check | result |
|-------|--------|
| SIGKILL mid-lease (no renew) | task `done`; `completed_by=survivor`; `reclaimed_from=crasher`; `lease_reclaim` event present |
| Slow-alive + renew (0.7 s work, 0.2 s TTL) | poacher gets `None`; `completed_by=slow`; no `reclaimed_from`; no `lease_reclaim` |
| 24 inserts while 3 workers drain | 24 unique leases; all `done`; 0 doubles |

## Outcome

TTL reclaim works on this machine for a kill-9 after a committed lease. Churn drain held for 24 tasks / 3 workers.

Limits: silence still equals dead (a live worker that does not renew is preempted). Heartbeats do not append events. Still single-node SQLite. Stop file is test harness, not a runtime protocol.

## Decision

**Keep** `lease_until` + CAS reclaim + `renew_lease` heartbeat.

**Do not** treat missing heartbeats as a distinguished crash type — only as silence.

## Reproduction

```bash
python tests/test_hiveclaw_causal_lease.py -v
```
