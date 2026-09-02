# Rewind deployment contract

**Date:** 2026-09-01 (Session 9)  
**Scope:** The centralized causal store (`hiveclaw_causal`). SQLite and Postgres are two deployment mechanisms for the **same** architecture: one authoritative event log, reverse-dependency index, topic lookup, and lease CAS. They are not two consistency models.

This is not a multi-master, CRDT, or “no central manager” product. See `docs/research/decentralization-assessment.md`.

`hiveclaw-causal` in commands below means `python -m hiveclaw_causal`.

## Supported modes

| Mode | Intended scale | Authority model | Requirements |
|------|----------------|-----------------|--------------|
| **SQLite** | Single host / local concurrent processes | One shared database file | Filesystem locking that SQLite understands; backup procedure; **not** an arbitrary network filesystem |
| **Postgres** | Multi-host clients | One shared Postgres database | TLS in production, secret-managed credentials, migrations, backup/restore, connection limits |

Every client of a given Rewind deployment must use the **same** authoritative connection target (one file path, or one DSN + schema). Two files or two servers are two products, not a replica set.

## SQLite

- Use a local disk (APFS/HFS+, ext4, XFS, NTFS with a native SQLite build). WAL + `BEGIN IMMEDIATE` + `busy_timeout` is the concurrency story.
- **Not supported** on NFS, SMB, or other network filesystems unless locking semantics have been verified for that mount. “It opened” is not verification.
- Do not copy a live `*.sqlite` file with `cp` while writers exist. Use `python -m hiveclaw_causal backup --db <path> --out <file>` (`sqlite3.Connection.backup`, plus `PRAGMA integrity_check`).
- Multiple processes on one host are supported. Multiple hosts sharing one SQLite file over the network are not.

## Postgres

- Recommended when workers are on more than one host.
- Production should use TLS (`sslmode=require` or stricter) and credentials from a secret manager, not a shell history or committed `.env`.
- `PgStore` does not implement authentication of Rewind clients; Postgres roles do. See `docs/research/threat-model.md`.
- Connection limits: size `max_connections` for worker count plus operator sessions. `FOR UPDATE SKIP LOCKED` is the lease contention path; do not bypass CAS with a client-side cache of “I already own this task.”
- Operator dump/restore: `pg_dump` / `pg_restore`. In-process drill: `hiveclaw_causal.backup.postgres_logical_backup` (same server, new schema). The CLI `backup`/`restore` commands are SQLite-only.

## Clocks and TTL

- Lease expiry is `lease_until` compared to the **client’s** `time.time()` today, clamped by a store-configured max and `LEASE_TTL_CEILING_S` (30s) with a database trigger as a backstop.
- Prefer short TTLs plus heartbeats. Do not try to out-wait clock skew with a huge TTL; the ceiling exists so a silent owner cannot strand work.
- SQLite `strftime('%s','now')` in the ceiling trigger is **database** unix time. Python clamp uses the client clock. Keep hosts NTP-synced. Postgres trigger uses `EXTRACT(EPOCH FROM clock_timestamp())` on the server.

## Timeouts, retries, and CAS

- A retry after an uncertain network failure must not create a second owner or a second `complete_task` event.
- `try_lease_one_task` is compare-and-set on `pending` / expired `leased`. Retrying after a successful lease that the client did not see will not steal that same unexpired lease (the row is no longer pending).
- `complete_task` is idempotent for the same worker if the task is already `done` by that worker: no second event. A different worker completing the same id is still an error.
- Connection pooling is an operator concern (Postgres). Pooling must not share a transaction or a leased-task assumption across logical workers. One worker id per lease owner.

## Health

- Read-only: `python -m hiveclaw_causal store-status --db <path-or-url>` and `verify-store`.
- These are not Kubernetes liveness probes. A worker is “alive” iff it renews its lease; process-table health is **not** persisted (`owner_process_health.recorded = false`).
- Failed lease attempts (CAS miss) are not audit rows. Do not invent a dashboard number for them.

## What this contract does not authorize

- A second SQLite file that later “syncs.”
- A third storage backend.
- Vector clocks, CRDTs, or a distributed lease protocol on top of the current schema.
