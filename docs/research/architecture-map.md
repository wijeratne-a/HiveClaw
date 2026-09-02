# Architecture map — Rewind causal store (as built)

**Date:** 2026-09-01 (Session 10)  
**Decision:** This file did not exist during the original discovery protocol. It is written now as a **current-state** map of `hiveclaw_causal/`, not a speculative future architecture. See `docs/research/gap-analysis.md` for what was never built.

## What Rewind is

One **authoritative** event-sourced store. Clients (local processes or TCP) share **one** SQLite file or **one** Postgres database. SQLite and Postgres are the same records, indexes, and lease CAS — two deployments, not two consistency models.

It is not multi-master, not a CRDT, not “no central manager.” Memo: `docs/research/decentralization-assessment.md`. ADR: `docs/adr/CAUSAL_RUNTIME_H5.md`.

## Runtime pieces (code)

```
CLI / tests
  python -m hiveclaw_causal          demo if no subcommand
  store-status | verify-store | backup | restore | migrate | inspect
       │
       ▼
  RewindRuntime (rewind.py)          ingest, targeted/naive repair, policy
       │
       ├─ InvalidationEngine         reverse_deps walk; topic-provider-status overlap
       ├─ policy.authorize           action status + justifying claim freshness
       └─ stats.outage_explains_pct  92/100 timestamps at seed 42
       │
       ▼
  Store | PgStore                    events (append-only) + objects + edges
                                     reverse_deps + lease_config + schema_migrations
```

Leases: `try_lease_one_task` CAS; `renew_lease` heartbeat; reclaim when `lease_until < now`; `LEASE_TTL_CEILING_S = 30`.

## What this map is not

- Not the IOSurface slab / `pheromoned` / `hiveclaw-server` path. Those are a separate Apple Silicon inference stack. Rewind does not import `hiveclaw_python`.
- Not a GUI. No work map, timeline, or graphical inspector (`docs/research/rewind-checkpoint.md`, Session 10).
- Not a replica set. Backup/restore of the one store is the DR story (`docs/research/backup-restore.md`).
