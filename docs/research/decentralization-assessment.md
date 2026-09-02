# Decentralization assessment (Rewind / `hiveclaw_causal`)

**Date:** 2026-09-01 (Session 8)  
**Question:** Can HiveClaw’s current causal data model support stigmergic coordination *without* a single authoritative store, or is that a fundamental redesign?

## Direct answer

**No.** Independent stores that eventually reconcile cannot preserve the current causal-invalidation and lease guarantees without a **fundamental redesign**. This is not a small extension (another backend, a replica, a sync daemon). It is a different consistency model.

Do not describe Rewind as “no central manager” or as proven decentralized stigmergy. What has been proven is:

> **A centralized causal store with safe concurrent multi-process and networked clients** — bounded-cost invalidation, append-only events, and TTL/heartbeat leases — on one SQLite file or one Postgres server.

The IOSurface slab is a different object (latent slots, `claim_task`). It is also shared memory on one machine, not a mergeable multi-master log.

## 1. Does the current model require a single authoritative store?

**Yes, as implemented.**

The runtime is an **event-sourced projection with a reverse index and CAS leases**:

| Mechanism | Why it needs one writer-of-record |
|-----------|-----------------------------------|
| `events.seq` (SQLite AUTOINCREMENT / PG BIGSERIAL) | Total order of status changes. Two logs have two sequences; `has_applied(object_id, edge_id, new_status)` is per-store. |
| Append-only triggers | Local invariant. They do not commute across stores. |
| `reverse_deps` | Derived index of edges. Concurrent edge inserts on two stores produce divergent cones. Invalidation walks **this** index, not a merge of two graphs. |
| Overlap / topic key `topic-provider-status` | Lookup is `dependent_claims(topic)` on **one** table. Two stores can each miss the other’s claims. |
| `try_lease_one_task` CAS + `lease_until` | Exactly-one owner is a **single-row** compare-and-set. Two stores can both believe they leased the same task. TCP drop reclaim still talks to **one** `lease_until`. |

Causal invalidation here is “apply this edge’s rule to this object, record the event, fan out via reverse_deps.” That is linearizable (or at least serializable) mutation of one graph. Session 7’s Postgres port did not change that; it moved the same graph behind TCP.

Eventual reconciliation of two such graphs is **not** a property the schema has. There is no vector clock, no CRDT payload, no merge function for `status`, no rule for “both stores challenged the same claim for different reasons,” and no lease manager that is not the row itself.

## 2. If independent stores were possible in principle, what would have to change?

In principle, distributed event sourcing and CRDTs exist. Fitting **this** product into that world would require at least:

1. **Replace `seq` with a partial order** (vector clocks, hash-chained events, or a consensus log). Idempotency keys cannot stay `(object_id, edge_id, new_status)` alone if two nodes mint different edge ids for the same logical contradict.
2. **Merge rules for concurrent invalidations.** Example: store A marks a claim `challenged` from overlap; store B marks it `corroborated` from a different observation. The current engine has a single `next_status(kind, rule)` table, not a lattice join.
3. **Lease ownership without a single CAS target.** Options: consensus on the lease row (still a central log), or fencing tokens with bounded TTL (still needs a shared clock or a loosely synchronized ceiling — Session 8’s TTL cap is **not** a distributed lease protocol).
4. **Conflict policy for the projection.** Who wins if two nodes append contradictory `status_transition` events? That policy *is* the manager, even if it is a function instead of a process.

That is a **new runtime**, not a port of `Store`/`PgStore`. ADR H3 (CRDT / shared JSON) was already rejected for Rewind because it does not give reverse-index invalidation or append-only causal reasons. Reversing that decision is a project, not a session.

## 3. Recommendation

**Stop implying “no central manager” as a currently-true property of Rewind.**

Accurate framing:

- **Proven:** concurrent workers, no worker-to-worker messages, one authoritative causal store (file or TCP server); bounded eval_steps on this fixture; leases with heartbeat, SIGKILL reclaim, TCP-drop reclaim, and a hard TTL ceiling so a misconfigured client cannot strand work forever.
- **Out of scope for this codebase as designed:** multi-master stigmergy, partition-tolerant merge of two Rewind stores, “coordination without shared state.”

If the product thesis needs true decentralization, that is a separate design (consensus log or a real CRDT of claims) and should not be chased by adding more single-node backends. Session 7 already showed SQLite and Postgres are the **same architecture**. Session 8 should not add a third.

The IOSurface slab remains the on-device latent bus. Do not conflate it with Rewind’s event graph; do not claim either is manager-free in the distributed-systems sense.
