# exp-004 — multi-host / networked backend

**Date:** 2026-09-01 (Session 7)  
**Interpreter:** `/Users/wijeratne/dev/HiveClaw/.venv/bin/python` 3.11.1  
**Backend:** Docker `postgres:16` on `127.0.0.1:55432` (Linux VM NAT on Docker Desktop — a real TCP hop, not a local file).  
**Code:** `hiveclaw_causal/pg_store.py`, `hiveclaw_causal/netproxy.py`  
**Tests:** `HIVECLAW_PG_DSN=... python -m unittest tests.test_hiveclaw_causal_pg -v`  
**Runner:** `HIVECLAW_PG_DSN=... python scripts/exp004_multi_host.py`

Existing `exp-004-lease-expiry-and-churn.md` is the SQLite crash/churn log. This file is the networked follow-up. The names collide; do not merge them.

---

## Part 1 — What “multi-host” can mean here

Written **before** implementation. Not edited after the run except to mark the decision as executed.

HiveClaw Rewind persisted in **SQLite WAL**. That is a single-writer file protocol. There is no replication, no RPC, no CRDT, and no second machine. Calling two local processes on one SQLite file “multi-host” would be a silent redefinition.

Honest options:

| Option | What it tests | What it does not test | Buildable this session? |
|--------|---------------|----------------------|-------------------------|
| **(a)** Port the schema to a **networked SQL server** (Postgres) and talk over TCP | Lease CAS, reverse_deps, overlap/topic index, eval_steps, when the bytes travel a TCP hop and the server is not a local file | Multi-master replication, partition-tolerant stigmergy, IOSurface, two independent causal stores merging | Yes, if Docker Postgres starts |
| **(b)** Keep SQLite, add “network latency” to local processes | Almost nothing new | Same as today’s exp-003/004 | Yes, but it would **not** answer the prompt |
| True multi-host stigmergy | Independent nodes, no shared server | — | **No.** No such layer exists |

### Decision (chosen before code, executed as written)

**Implement (a), named honestly as a subset of (b)’s wording:** *multi-process clients against one Postgres server over TCP, with a userspace proxy that can stall or cut the path.*

This is **not** decentralized stigmergy. Postgres is still a **central coordinator**. It is the smallest change that (1) stops using a shared local file handle, (2) uses a real TCP network, (3) can fail the **network path** without SIGKILL.

Still open after a green run: two hosts with no shared DB; replica lag; Raft/CRDT; slab stigmergy.

---

## Method

1. `docker run postgres:16` published on host port 55432.
2. `PgStore` with the same records/events/reverse_deps/TTL-lease API as SQLite `Store`. Dialect only: `%s` placeholders, `FOR UPDATE SKIP LOCKED`, plpgsql append-only triggers, schema-per-run isolation.
3. Userspace `TcpProxy` in front of Postgres for **drop** (close sockets, process stays up) and **stall** (stop forwarding without close — GC/jitter analogue).
4. Re-run exp-002-style targeted vs naive at N=12/100/500/2000 and C=0/500/2000, two runs each, seed 42. Causal rules unchanged.
5. Re-run 5 workers × 3 tasks × 8 trials; SIGKILL reclaim; slow-alive renew; TCP-drop reclaim; stall > TTL.

Default `make test-causal` does **not** require Postgres (those tests skip). This experiment is opt-in via `HIVECLAW_PG_DSN`.

---

## What held

### Causal conclusion

Unchanged: **92.0%** `outage_explains_pct`, rollback blocked, follow-up present, at every scale below. Same seed 42 fixture.

### Eval-steps — bounded vs linear still holds over TCP

Claim-side integers **matched SQLite** (topic index). Task-side integers **shifted down by 1–2** vs SQLite exp-002 (N=100: 7/108 vs 8/109; N=500: 8/509 vs 10/511; N=2000: 8/2009 vs 10/2011). Touched counts matched SQLite (9–11 targeted vs full graph naive). The structural pattern is the same: targeted stays flat, naive tracks N.

| scale | U,R or C | before | eval t/n | touched t/n | wall_s run1 t/n | wall_s run2 t/n |
|-------|----------|--------|----------|-------------|-----------------|-----------------|
| N~12 | 0,0 | 12 | **6** / 19 | 9 / 19 | 0.0337 / 0.0609 | 0.0395 / 0.0309 |
| N~100 | 29,1 | 100 | **7** / 108 | 10 / 107 | 0.0304 / 0.0934 | 0.0887 / 0.0336 |
| N~500 | 162,2 | 500 | **8** / 509 | 11 / 507 | 0.0364 / 0.0372 | 0.0328 / 0.0382 |
| N~2000 | 662,2 | 2000 | **8** / 2009 | 11 / 2007 | 0.0315 / 0.0781 | 0.0290 / 0.0725 |
| C=0 | 0 | 12 | **6** / 19 | 9 / 19 | 0.0249 / 0.0317 | 0.0235 / 0.0249 |
| C=500 | 500 | 512 | **6** / 519 | 9 / 519 | 0.0248 / 0.0316 | 0.0252 / 0.0336 |
| C=2000 | 2000 | 2012 | **6** / 2019 | 9 / 2019 | 0.0236 / 0.0594 | 0.0228 / 0.0588 |

Do not treat the 1–2 eval-step drift vs SQLite as a semantics change. Objects-before/after match. The claim is bounded-cost invalidation, not “the integer 10 survives every backend.”

### Leases over TCP

`tests.test_hiveclaw_causal_pg` — 8/8 OK (this session, after skipping concurrent `CREATE FUNCTION` on worker connect).

| check | result |
|-------|--------|
| 5 processes × 3 tasks × 8 trials | 0 double-lease, 0 dropped |
| SIGKILL mid-lease (no renew) | reclaimed after TTL; `completed_by=survivor`; `reclaimed_from=crasher` |
| Slow-alive + renew (0.7 s work, 0.2 s TTL) | poacher `None`; `completed_by=slow` |
| TCP drop mid-lease, **process still alive** | renew fails; after TTL survivor reclaims; worker was still `is_alive` |
| Proxy stall 0.45 s with TTL 0.2 s | poacher **does** reclaim the live worker (`reclaimed_from=slow`) |

Append-only triggers fire on Postgres (`UPDATE`/`DELETE` on `events` raise).

---

## What broke, degraded, or needed changes

1. **Wall-clock does not transfer.** On SQLite, N=500 was ~3 ms vs ~12 ms. Over Docker TCP, N=500 is ~33–38 ms for **both** paths (run1 0.036 vs 0.037). Network RTT swamps the eval-step gap until N is large (N=2000: ~30 ms vs ~73–78 ms; C=2000: ~23 ms vs ~59 ms). Run-to-run jitter is large (N=100 run2 targeted 89 ms vs naive 34 ms). Bounded eval_steps is still true; “targeted is faster” is **not** a reliable networked claim at these sizes.

2. **Concurrent schema DDL.** Five workers calling `CREATE OR REPLACE FUNCTION` on connect raised `tuple concurrently updated`. Fix: skip DDL when `schema.objects` already exists, advisory-lock the first init. Not a causal-semantics change.

3. **Lease lock dialect.** SQLite used `BEGIN IMMEDIATE` + CAS. Postgres uses `FOR UPDATE SKIP LOCKED` + the same `lease_until` JSON payload. Same TTL/heartbeat rules; different isolation primitive.

4. **Heartbeat delay is silence.** The stall test is the GC/jitter analogue Session 6 named but did not network-test. A live worker whose renews cannot get through for longer than TTL is reclaimed. That is the existing rule, now evidenced on a delayed network path, not a new bug.

5. **TCP drop does not itself release the lease.** Postgres session death is not wired to `complete_task`. The row stays `leased` until TTL. Heartbeat/TTL is the reclaim mechanism. If TTL were infinite, a dropped path would strand the task. **Session 8 closed this:** `LEASE_TTL_CEILING_S = 30` is enforced in Python and by a store trigger; a client-requested 3600s TTL is clamped. See `docs/research/decentralization-assessment.md` and `tests/test_hiveclaw_causal_lease.py` (`test_oversized_client_ttl_does_not_strand_after_silence`).

---

## Verdict

**Bounded-cost invalidation and safe concurrent coordination still hold once a real network is in the loop — against one Postgres server.** They are not a SQLite-file-only accident.

They are **not** proof of stigmergy without a central store. This session replaced “one SQLite file on one machine” with “one Postgres on a TCP hop, many client processes.” That is a real step (shared file handle is gone; connection drop is tested). It is still a **single shared server**.

Session 8’s assessment: **decentralization is out of current scope** (fundamental redesign), not an open experiment to keep stacking backends against. The evidenced claim is a centralized causal store with concurrent clients. Do not read a green TCP run as “no central manager.”

---

## Reproduction

```bash
docker run -d --name hiveclaw-exp004-pg \
  -e POSTGRES_PASSWORD=hiveclaw -e POSTGRES_USER=hiveclaw -e POSTGRES_DB=hiveclaw \
  -p 55432:5432 postgres:16
# wait until pg_isready
python -m pip install 'psycopg[binary]'   # or requirements/requirements-causal-pg.txt
export HIVECLAW_PG_DSN='postgresql://hiveclaw:hiveclaw@127.0.0.1:55432/hiveclaw'
python scripts/exp004_multi_host.py
```

`make test-causal` without the DSN: SQLite suite green, 8 Postgres tests skipped.
