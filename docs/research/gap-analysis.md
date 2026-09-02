# Gap analysis — Rewind / original discovery protocol

**Date:** 2026-09-01 (Session 10)  
**Context:** The HiveClaw Rewind build contract’s Section 6 discovery protocol required `architecture-map.md` and `gap-analysis.md` **before** application code. An independent audit (`docs/research/independent-audit-2026-09-01.md`) confirmed **neither file ever existed in git**. Code shipped anyway (Sessions 1–9). This document records that fact and the current gaps versus the original brief — not a speculative product.

## Decision on the two missing discovery docs

**They were never written.** That is a process miss, not a missing subsystem.

**Session 10 choice:** write the **minimum honest** versions now, grounded in what exists:

- `docs/research/architecture-map.md` — current `hiveclaw_causal` layout (created Session 10).
- This file — gaps versus the original brief and versus a complete operator/product story.

The original “two discovery docs must precede code” rule is **retired as a gate**. The living sources of truth for the causal store are the ADR (`docs/adr/CAUSAL_RUNTIME_H5.md`), `docs/research/rewind-checkpoint.md`, experiment logs, and tests. Do not block future work on recreating a pre-Session-1 discovery packet.

## What exists (verified)

Typed provenance, invalidation conditions, append-only events, reverse-dep + topic indexes, deterministic policy gate, SQLite and Postgres as one architecture, TTL/heartbeat leases with a hard ceiling, CLI demo/inspect/benchmark, Session 9 verify/status/backup/migrate. Evidence: checkpoint + `make test-causal`.

## What the original brief asked for that is not built

| Item | Status |
|------|--------|
| Graphical Rewind demo (work map, timeline, “why?” UI, sub-five-minute non-technical walkthrough) | **Not built. Descoped for now** (checkpoint Session 10). CLI only. |
| Multi-master / no central store / mergeable independent SQLite files | **Out of scope** (redesign). Not a gap in the current product. |
| Coupling to the IOSurface slab | **Not done; not required** for Rewind guarantees. |
| Standby replica / automatic HA | **Not built.** DR is backup/restore of the one store. |
| Multi-tenant authn/z | **Not built.** Threat model is local / trusted internal. |
| Event payload size / cone DoS limits | **Not built** (trusted-user assumption). |
| `architecture-map.md` / `gap-analysis.md` at discovery time | **Late.** Filled Session 10 as current-state docs. |

## What remains assumed / untested (from the checkpoint)

Pre-trigger SQLite files on next open; model-checking of all interleavings; ironclad burn-in; `integration_test.py --stress`; continuous-insert drain on Postgres.

## Do not treat as open “next experiment”

Adding another storage backend, CRDTs, vector clocks, or a sync daemon between SQLite files. Session 8 locked that as a different program.
