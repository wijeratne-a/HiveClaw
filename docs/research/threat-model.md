# Rewind threat model (Session 9)

**Date:** 2026-09-01  
**Audience:** operators deciding whether HiveClaw Rewind can leave a single trusted machine.

## Intended threat model (current)

Rewind is **not** a multi-tenant SaaS. Do not add login theater (an application password that does not change the data model).

| Deployment | Trust boundary | What we assume |
|------------|----------------|----------------|
| **Local developer tool** (default SQLite) | One user, one host | Anyone who can read/write the SQLite file or run processes as that user is an authorized operator. |
| **Trusted internal service** (Postgres) | One team, one Postgres | Network clients are trusted at the Postgres role layer. TLS and secret-managed DSNs are required in production. The application does not implement tenants. |
| **Multi-tenant product** | **Out of scope** | No tenant column, no RLS policy shipped, no per-tenant lease namespace. Building this is a product change, not a config flag. |

The IOSurface slab and `hiveclaw-server` are a different surface (local GPU/IPC). This document is only the causal store.

## What is in place

- **SQL injection (dynamic identifiers):** Postgres schema names are restricted to `[A-Za-z_][A-Za-z0-9_]*`. Object ids, topic keys, and payloads are bound parameters, not concatenated identifiers.
- **SQL injection (values):** event/object/edge writes use placeholders (`?` / `%s`).
- **Lease abuse:** client TTL is clamped; database triggers reject `lease_until` beyond the absolute ceiling.
- **Append-only log:** UPDATE/DELETE on `events` abort.
- **Connection-string handling:** CLI does not echo DSNs in verify/status summaries. Operators must still avoid `ps` exposure of `postgres://user:pass@...`.
- **Retry / CAS:** `complete_task` is idempotent for the completing worker; lease CAS does not mint a second owner on retry of an unexpired lease.

## What is not in place (do not pretend otherwise)

- Authentication or authorization inside `hiveclaw_causal` (no users, API keys, or audit ACL).
- Postgres role separation as code: we do not ship `REVOKE`/`GRANT` migrations. Operators should give workers DML on the Rewind schema and not superuser.
- Event payload size limits, dependency-cone DoS caps, or claim fan-out quotas. A trusted local user can write a large graph and pay CPU for invalidation.
- Encryption at rest (SQLite file encryption, pgcrypto). Use filesystem/volume encryption if needed.
- PII/secrets policy enforcement on causal payloads. Treat event `payload` JSON as **possibly sensitive**. Backups inherit that.
- Tenant/project namespace isolation.
- Audit-log access controls (the event log *is* the audit log; whoever has the DB has it).

## Postgres least privilege (operator checklist)

If this is a trusted internal service:

1. Dedicated role for Rewind workers: `CONNECT` + DML on the schema, no `CREATEDB`.
2. Separate operator role for `migrate` / `pg_dump`.
3. TLS to the server; no `sslmode=disable` in production (tests may disable TLS on loopback Docker).
4. Do not share one role across unrelated apps on the same cluster without schema isolation.

## Input validation

Object ids and topic keys are application strings stored in TEXT columns. The engine trusts ids that already exist in `objects` / `reverse_deps`. Garbage ids fail lookups; they are not a sandbox escape. Status transitions are checked in the verifier against declared rules for `status_transition` events; the write path is still the engine, not an open REST enum.

## If the audience changes

If Rewind must serve untrusted network clients or multiple tenants, stop and redesign: authn/z, RLS or separate databases per tenant, payload limits, and an explicit PII policy. That is not Session 9 work.
