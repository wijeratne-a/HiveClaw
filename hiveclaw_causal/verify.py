"""Read-only integrity checks for the centralized causal store."""

from __future__ import annotations

from typing import Any

from .engine import next_status
from .lease_policy import LEASE_TTL_CEILING_S
from .store import _index_pair
from .types import InvalidationRule, ObjectKind, TaskStatus


def verify_store(store: Any) -> dict[str, Any]:
    """Inspect a store without writing. Returns a JSON-serializable report."""
    checks: list[dict[str, Any]] = []
    ok = True

    def add(name: str, passed: bool, detail: str, *, extra: dict[str, Any] | None = None) -> None:
        nonlocal ok
        if not passed:
            ok = False
        item: dict[str, Any] = {"name": name, "ok": passed, "detail": detail}
        if extra:
            item.update(extra)
        checks.append(item)

    events = store.all_events()
    objects = {r.id: r for r in store.all_objects()}
    edges = {e.id: e for e in store.all_edges()}

    seqs = [e.seq for e in events]
    add(
        "seq_monotonic",
        seqs == sorted(seqs) and len(seqs) == len(set(seqs)),
        "event seq must be unique and non-decreasing",
        extra={"count": len(seqs), "min": None if not seqs else seqs[0], "max": None if not seqs else seqs[-1]},
    )
    gaps = []
    if seqs:
        expected = list(range(seqs[0], seqs[-1] + 1))
        have = set(seqs)
        gaps = [n for n in expected if n not in have]
    add(
        "seq_gap_free",
        True,
        "gaps are not a declared invariant (failed inserts can skip AUTOINCREMENT/SERIAL); reported as info",
        extra={"gaps": gaps[:50], "gap_count": len(gaps), "invariant": False},
    )
    add(
        "event_checksums",
        True,
        "events table has no hash/checksum column; object source_snapshot is the content hash",
        extra={"recorded": False},
    )

    dangling_event_objects = [
        e.seq
        for e in events
        if e.object_id not in objects and not e.object_id.startswith("topic-")
    ]
    add(
        "events_reference_objects",
        not dangling_event_objects,
        "event.object_id should exist in objects (topic-* keys are allowed without a row)",
        extra={"missing_seqs": dangling_event_objects[:20]},
    )

    last_status: dict[str, str] = {}
    illegal: list[str] = []
    for ev in events:
        if ev.new_status:
            last_status[ev.object_id] = ev.new_status
        if (
            ev.event_type == "status_transition"
            and ev.rule
            and ev.object_id in objects
            and objects[ev.object_id].kind != ObjectKind.TASK
        ):
            rec = objects[ev.object_id]
            try:
                want = next_status(rec.kind, InvalidationRule(ev.rule))
            except (ValueError, KeyError):
                illegal.append(f"seq={ev.seq} unknown rule {ev.rule}")
                continue
            if ev.new_status != want:
                illegal.append(
                    f"seq={ev.seq} {ev.object_id} new_status={ev.new_status} != {want} for {ev.rule}"
                )
        if ev.event_type == "complete_task":
            if ev.old_status != TaskStatus.LEASED.value or ev.new_status != TaskStatus.DONE.value:
                illegal.append(f"seq={ev.seq} complete_task not leased->done")
        if ev.event_type in ("lease_task", "lease_reclaim"):
            if ev.new_status != TaskStatus.LEASED.value:
                illegal.append(f"seq={ev.seq} lease event not ->leased")
    add("illegal_transitions", not illegal, "status_transition/lease events match declared rules", extra={"examples": illegal[:20]})

    replay_mismatch = []
    for oid, rec in objects.items():
        if oid in last_status and last_status[oid] != rec.status:
            replay_mismatch.append(f"{oid}: events->{last_status[oid]} objects->{rec.status}")
    add(
        "projection_replay",
        not replay_mismatch,
        "each object's last event.new_status must match objects.status",
        extra={"examples": replay_mismatch[:20]},
    )

    has_applied_bad = []
    for ev in events:
        if ev.edge_id and ev.new_status and ev.object_id:
            if not store.has_applied(ev.object_id, ev.edge_id, ev.new_status):
                has_applied_bad.append(ev.seq)
    add(
        "has_applied_consistent",
        not has_applied_bad,
        "every (object_id, edge_id, new_status) event is visible to has_applied",
        extra={"missing_seqs": has_applied_bad[:20]},
    )

    index_missing = []
    extra_index = []
    expected_pairs: set[tuple[str, str, str, str]] = set()
    for edge in edges.values():
        target, dependent = _index_pair(edge)
        expected_pairs.add((target, dependent, edge.id, edge.rule.value))
        found = False
        for dep_id, eid, rule in store.reverse_lookup(target):
            if eid == edge.id and dep_id == dependent:
                found = True
                break
        if not found:
            index_missing.append(edge.id)
    seen: set[tuple[str, str, str]] = set()
    for rec in objects.values():
        for dep_id, eid, rule in store.reverse_lookup(rec.id):
            seen.add((rec.id, dep_id, eid))
            if eid not in edges and not eid.startswith("edge-depends-"):
                extra_index.append(eid)
    # topic keys are not objects; still check reverse_deps rows via SQL if available
    add(
        "reverse_deps_match_edges",
        not index_missing,
        "every edge has a reverse_deps row for _index_pair(edge)",
        extra={"missing_edge_ids": index_missing[:20], "orphan_index_edge_ids": extra_index[:20]},
    )

    triggers = _list_triggers(store)
    need_append = ["events_append_only_no_update", "events_append_only_no_delete"]
    have = set(triggers)
    missing_trig = [n for n in need_append if n not in have]
    add(
        "append_only_triggers",
        not missing_trig,
        "events UPDATE/DELETE triggers must be installed",
        extra={"triggers": triggers, "missing": missing_trig},
    )
    ceiling_ok = any("lease_until" in t or "ceiling" in t for t in triggers)
    add(
        "lease_ttl_ceiling_trigger",
        ceiling_ok,
        "lease_until absolute ceiling trigger should be present on current schema",
        extra={"ceiling_s": LEASE_TTL_CEILING_S},
    )

    from .schema import SCHEMA_VERSION, current_version

    ver = current_version(store)
    add(
        "schema_version",
        ver >= 1,
        f"schema_migrations version={ver} (code SCHEMA_VERSION={SCHEMA_VERSION})",
        extra={"version": ver, "code_version": SCHEMA_VERSION},
    )

    summary = "ok" if ok else "FAILED"
    return {
        "ok": ok,
        "summary": summary,
        "backend": getattr(store, "backend", "unknown"),
        "objects": len(objects),
        "events": len(events),
        "edges": len(edges),
        "checks": checks,
    }


def _list_triggers(store: Any) -> list[str]:
    backend = getattr(store, "backend", "sqlite")
    if backend == "postgres":
        rows = store._conn.execute(
            """
            SELECT tgname AS name
            FROM pg_trigger t
            JOIN pg_class c ON c.oid = t.tgrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE NOT t.tgisinternal
              AND n.nspname = current_schema()
            ORDER BY tgname
            """
        ).fetchall()
        return [str(r["name"]) for r in rows]
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name"
    ).fetchall()
    return [str(r["name"]) for r in rows]
