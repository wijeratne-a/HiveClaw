# ADR: Causal runtime storage for The Rewind (H5)

**Status:** Accepted  
**Date:** 2026-08-30  
**Context:** HiveClaw Rewind build contract — agents coordinate through a typed, versioned trace; the runtime must detect obsolete evidence, block unsafe actions, and repair the smallest affected subgraph.

## Decision

Implement **H5 hybrid** for the Rewind slice:

1. **Append-only event log** (SQLite `events` table, `INSERT` only — never `UPDATE`/`DELETE` events). Session 2: `BEFORE UPDATE`/`BEFORE DELETE` triggers `RAISE(ABORT)` so this is a DB invariant, not only a Python convention (`tests/test_hiveclaw_causal_store.py`).
2. **Current-state records** (SQLite `objects` projection of the log).
3. **Indexed reverse-dependency lookup** (`deps_by_target`: given evidence/object id, who depends on it and with which edge/rule).
4. **Task queue with leases** (`tasks.lease_owner`, `lease_until`).
5. **Deterministic policy check** in process (Python, no LLM) before any irreversible action.

Package: **`hiveclaw_causal/`** at repo root. Persistence: SQLite file. Tests: `tests/test_hiveclaw_causal_*.py`. No GPU, no `pheromoned`, no `import hiveclaw_python`.

## What was compared

The contract lists H1–H5. Discovery (`docs/research/repository-baseline.md`, HEAD at discovery: `d577ed0`) found **no working alternative primitive** that would make H5 redundant:

| Option | In this tree? | Why not chosen |
|--------|---------------|----------------|
| H1 — external graph DB (Neo4j, etc.) | No client, no service | Would add ops surface for a single e2e scenario |
| H2 — message bus (NATS/Kafka/Redis streams) | No | Overkill; Rewind is in-process and must be deterministic |
| H3 — CRDT / shared JSON document | No | Does not give reverse-index invalidation or append-only causal reasons |
| H4 — IOSurface / existing stigmergy slab | Yes, but wrong shape | Latent vectors + slot `claim_task`; not typed claims, no policy gate, requires daemon/GPU |
| H5 — event log + projection + reverse index + leases + policy | Not present; empty `hiveclaw-core` traits only | **Default.** Matches the four guarantees with one SQLite file |

`quality_gate` is generate→verify→repair for fenced Python, not a causal graph. `catenar_tracing` is an optional HTTP tracer. Neither is an event-sourced dependency index.

A half-day bake-off was **not** run: there was nothing real to measure against. Choosing H1–H4 would be greenfield plus extra moving parts.

## Consequences

- Status changes always append a `CausalEvent` with `old_status`, `new_status`, `reason`, `edge_id`, `rule` before updating the projection.
- Policy may **propose** nothing; workers may propose actions; only `policy.authorize()` flips an action to `approved` / keeps `blocked`.
- Existing Metal/MLX stack is untouched. Do not put this module under `hiveclaw_python` (Darwin+mlx import guard).
- Deferred: patch worker, experiment planner, UI projection.
- **Not deferred as a small follow-on:** multi-master / “no central manager.” Session 8: the event log + reverse_deps + lease CAS **require one authoritative store**. Independent reconciling stores would be a fundamental redesign (`docs/research/decentralization-assessment.md`). Proven shape: centralized causal store, concurrent clients.

## Evidence

Verified in discovery: no EventStore, no invalidation engine, slab “claim” ≠ hypothesis. Baseline test commands and results: `docs/research/repository-baseline.md`.
