# Rewind backup, restore, and disaster recovery

**Date:** 2026-09-01 (Session 9)  
**Why this is first-class:** Rewind has **one** authoritative store. There is no replica that can be promoted. Backup and restore *are* the HA story.

Commands use `python -m hiveclaw_causal`. Confirm flags are required for restore and migrate.

## RTO / RPO (honest)

These are **targets for this project**, not a measured SLA.

| | SQLite (local file) | Postgres (one server) |
|--|---------------------|------------------------|
| **RPO** | Last successful `backup` (drill uses `Connection.backup` while a writer is open). Aim: take backups at the cadence you can afford to lose (minutes to hours). | Last successful `pg_dump` (or WAL archive if you configure it — not shipped in-tree). In-process schema copy is a **drill**, not a PITR tool. |
| **RTO** | Time to copy the backup to a new path, `verify-store`, and point clients at that path. Drill on this machine is seconds for the Rewind fixture. | Time to restore into a new database/schema, `verify-store`, and cut clients to the new DSN. |

Do not quote cloud-vendor RTO numbers. This tree does not run a standby.

## SQLite procedure

1. **Integrity before backup:** `PRAGMA integrity_check` (the CLI `backup` command does this).
2. **Consistent copy while clients are active:** `python -m hiveclaw_causal backup --db <live> --out <backup.sqlite>`  
   This uses SQLite’s backup API, not `cp` of a WAL-mode file.
3. **Integrity after:** same command records `integrity_src_after` and `integrity_dest`.
4. **Restore into an isolated file:**  
   `python -m hiveclaw_causal restore --backup <backup.sqlite> --db <new.sqlite> --confirm`  
   Then `verify-store --db <new.sqlite>`. Do not overwrite the live file in place as the first restore.
5. **After restore:** leasing and invalidation must still work. The Session 9 drill creates a graph, backups, restores to a new path, verifies event/object counts, leases a task, and injects the provider artifact.

### Failure cases

| Situation | What to do |
|-----------|------------|
| Partially written event | Append-only triggers reject UPDATE/DELETE. A crashed INSERT is a rolled-back transaction; `seq` may skip (not a declared invariant). Run `verify-store`. |
| Failed migration | Restore from the pre-migration backup. There is **no automatic downgrade**. |
| Corrupted file | `integrity_check` ≠ `ok`. Restore the last good backup to a new path. Do not keep serving the corrupt file. |
| Lost worker | Not a store-restore problem. Lease TTL + reclaim. `store-status` shows active leases; process health is not recorded. |

## Postgres procedure

**Production:** `pg_dump` / `pg_restore` (or your platform’s backup). Prefer a **new** database for restore, then `verify-store`.

Example (operator laptop / CI DSN — do not log the password):

```bash
pg_dump --format=custom --no-password "$HIVECLAW_PG_DSN" -f rewind.dump
# restore into an empty database created for the drill
pg_restore --no-password --dbname="$HIVECLAW_PG_DSN_DRILL" rewind.dump
python -m hiveclaw_causal verify-store --db "$HIVECLAW_PG_DSN_DRILL"
```

**In-process drill** (same cluster, new schema, no `pg_dump` binary): `hiveclaw_causal.backup.postgres_logical_backup(dsn, src_schema, dest_schema)`. Initializes dest, copies tables, runs `verify_store`. Covered when `HIVECLAW_PG_DSN` is set.

Do not put DSNs in git, CI logs, or `store-status` JSON.

## Verify after restore

`python -m hiveclaw_causal verify-store --db <path-or-url> --json`

Must be `ok` before pointing production clients at the restored store. Checks include seq uniqueness, projection replay, reverse-deps vs edges, `has_applied`, append-only and TTL-ceiling triggers, schema version. Seq **gaps** are reported as info, not a failure.

## Secrets

- Postgres URLs are secrets. Pass them via env (`HIVECLAW_PG_DSN`), not argv that lands in process lists if you can avoid it.
- SQLite paths are not passwords; the file itself may contain payload data you treat as sensitive (see threat model).
