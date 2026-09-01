"""Append-only SQLite event log + current-state projection + reverse-dependency index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .types import (
    CausalEvent,
    Edge,
    EdgeMode,
    InvalidationCondition,
    InvalidationRule,
    ObjectKind,
    Provenance,
    Record,
    SourceRef,
    TaskStatus,
    TrustClass,
)


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def provenance_to_dict(p: Provenance) -> dict[str, Any]:
    return {
        "producer": p.producer,
        "timestamp": p.timestamp,
        "trust": p.trust.value,
        "sources": [
            {"uri": s.uri, "version": s.version, "content_hash": s.content_hash}
            for s in p.sources
        ],
    }


def provenance_from_dict(d: dict[str, Any]) -> Provenance:
    sources = tuple(
        SourceRef(
            uri=str(s["uri"]),
            version=s.get("version"),
            content_hash=str(s.get("content_hash") or ""),
        )
        for s in d.get("sources", [])
    )
    return Provenance(
        producer=str(d["producer"]),
        sources=sources,
        timestamp=str(d["timestamp"]),
        trust=TrustClass(d["trust"]),
    )


def conditions_to_json(conds: tuple[InvalidationCondition, ...]) -> str:
    return _json(
        [
            {
                "description": c.description,
                "evidence_ids": list(c.evidence_ids),
                "rule": c.rule.value,
            }
            for c in conds
        ]
    )


def conditions_from_json(raw: str) -> tuple[InvalidationCondition, ...]:
    data = json.loads(raw or "[]")
    return tuple(
        InvalidationCondition(
            description=str(c["description"]),
            evidence_ids=tuple(c.get("evidence_ids") or ()),
            rule=InvalidationRule(c["rule"]),
        )
        for c in data
    )


class Store:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=8000")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              event_type TEXT NOT NULL,
              object_id TEXT NOT NULL,
              old_status TEXT,
              new_status TEXT,
              reason TEXT NOT NULL,
              edge_id TEXT,
              rule TEXT,
              payload TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS events_append_only_no_update
            BEFORE UPDATE ON events
            BEGIN
              SELECT RAISE(ABORT, 'events is append-only: UPDATE is forbidden');
            END;
            CREATE TRIGGER IF NOT EXISTS events_append_only_no_delete
            BEFORE DELETE ON events
            BEGIN
              SELECT RAISE(ABORT, 'events is append-only: DELETE is forbidden');
            END;
            CREATE TABLE IF NOT EXISTS objects (
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL,
              status TEXT NOT NULL,
              provenance TEXT NOT NULL,
              payload TEXT NOT NULL,
              evidence_ids TEXT NOT NULL,
              source_snapshot TEXT NOT NULL,
              invalidation_conditions TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
              id TEXT PRIMARY KEY,
              src TEXT NOT NULL,
              dst TEXT NOT NULL,
              mode TEXT NOT NULL,
              rule TEXT NOT NULL,
              declared_effect TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reverse_deps (
              target_id TEXT NOT NULL,
              dependent_id TEXT NOT NULL,
              edge_id TEXT NOT NULL,
              rule TEXT NOT NULL,
              PRIMARY KEY (target_id, dependent_id, edge_id)
            );
            CREATE INDEX IF NOT EXISTS idx_reverse_target ON reverse_deps(target_id);
            CREATE INDEX IF NOT EXISTS idx_events_object ON events(object_id);
            """
        )
        self._conn.commit()

    def append_event(
        self,
        *,
        ts: str,
        event_type: str,
        object_id: str,
        reason: str,
        old_status: str | None = None,
        new_status: str | None = None,
        edge_id: str | None = None,
        rule: str | None = None,
        payload: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> CausalEvent:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO events (ts, event_type, object_id, old_status, new_status,
                                reason, edge_id, rule, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                event_type,
                object_id,
                old_status,
                new_status,
                reason,
                edge_id,
                rule,
                _json(payload or {}),
            ),
        )
        if commit:
            self._conn.commit()
        seq = cur.lastrowid
        if seq is None:
            raise RuntimeError("INSERT INTO events did not produce lastrowid")
        return CausalEvent(
            seq=seq,
            ts=ts,
            event_type=event_type,
            object_id=object_id,
            old_status=old_status,
            new_status=new_status,
            reason=reason,
            edge_id=edge_id,
            rule=rule,
            payload=payload or {},
        )

    def has_applied(self, object_id: str, edge_id: str, new_status: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM events
            WHERE object_id = ? AND edge_id = ? AND new_status = ?
            LIMIT 1
            """,
            (object_id, edge_id, new_status),
        ).fetchone()
        return row is not None

    def put_object(self, rec: Record, *, ts: str, event_type: str, reason: str) -> Record:
        existing = self.get_or_none(rec.id)
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO objects (
              id, kind, status, provenance, payload, evidence_ids,
              source_snapshot, invalidation_conditions, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              kind=excluded.kind,
              status=excluded.status,
              provenance=excluded.provenance,
              payload=excluded.payload,
              evidence_ids=excluded.evidence_ids,
              source_snapshot=excluded.source_snapshot,
              invalidation_conditions=excluded.invalidation_conditions,
              updated_at=excluded.updated_at
            """,
            (
                rec.id,
                rec.kind.value,
                rec.status,
                _json(provenance_to_dict(rec.provenance)),
                _json(rec.payload),
                _json(list(rec.evidence_ids)),
                rec.source_snapshot,
                conditions_to_json(rec.invalidation_conditions),
                rec.created_at,
                rec.updated_at,
            ),
        )
        old = None if existing is None else existing.status
        self.append_event(
            ts=ts,
            event_type=event_type,
            object_id=rec.id,
            old_status=old,
            new_status=rec.status,
            reason=reason,
        )
        return rec

    def get(self, object_id: str) -> Record:
        rec = self.get_or_none(object_id)
        if rec is None:
            raise KeyError(object_id)
        return rec

    def get_or_none(self, object_id: str) -> Record | None:
        row = self._conn.execute(
            "SELECT * FROM objects WHERE id = ?", (object_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def objects_of(self, kind: ObjectKind) -> list[Record]:
        rows = self._conn.execute(
            "SELECT * FROM objects WHERE kind = ? ORDER BY id", (kind.value,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def all_objects(self) -> list[Record]:
        rows = self._conn.execute("SELECT * FROM objects ORDER BY id").fetchall()
        return [self._row_to_record(r) for r in rows]

    def transition(
        self,
        object_id: str,
        new_status: str,
        *,
        ts: str,
        reason: str,
        edge_id: str | None,
        rule: str | None,
        payload_update: dict[str, Any] | None = None,
    ) -> Record:
        rec = self.get(object_id)
        old = rec.status
        if edge_id and self.has_applied(object_id, edge_id, new_status):
            return rec
        rec.status = new_status
        rec.updated_at = ts
        if payload_update:
            rec.payload = {**rec.payload, **payload_update}
        self._conn.execute(
            """
            UPDATE objects SET status = ?, updated_at = ?, payload = ?
            WHERE id = ?
            """,
            (new_status, ts, _json(rec.payload), object_id),
        )
        self.append_event(
            ts=ts,
            event_type="status_transition",
            object_id=object_id,
            old_status=old,
            new_status=new_status,
            reason=reason,
            edge_id=edge_id,
            rule=rule,
        )
        return rec

    def add_edge(self, edge: Edge) -> Edge:
        existing = self._conn.execute(
            "SELECT id FROM edges WHERE id = ?", (edge.id,)
        ).fetchone()
        if existing is not None:
            return edge
        self._conn.execute(
            """
            INSERT INTO edges (id, src, dst, mode, rule, declared_effect)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                edge.id,
                edge.src,
                edge.dst,
                edge.mode.value,
                edge.rule.value,
                edge.declared_effect,
            ),
        )
        target, dependent = _index_pair(edge)
        self._conn.execute(
            """
            INSERT OR IGNORE INTO reverse_deps (target_id, dependent_id, edge_id, rule)
            VALUES (?, ?, ?, ?)
            """,
            (target, dependent, edge.id, edge.rule.value),
        )
        self._conn.commit()
        return edge

    def reverse_lookup(self, target_id: str) -> list[tuple[str, str, str]]:
        """Return (dependent_id, edge_id, rule) for objects that depend on target_id."""
        rows = self._conn.execute(
            """
            SELECT dependent_id, edge_id, rule FROM reverse_deps
            WHERE target_id = ?
            """,
            (target_id,),
        ).fetchall()
        return [(str(r["dependent_id"]), str(r["edge_id"]), str(r["rule"])) for r in rows]

    def edges_to(self, dst: str, mode: EdgeMode | None = None) -> list[Edge]:
        if mode is None:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE dst = ?", (dst,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE dst = ? AND mode = ?",
                (dst, mode.value),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def events_for(self, object_id: str) -> list[CausalEvent]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE object_id = ? ORDER BY seq", (object_id,)
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def all_events(self) -> list[CausalEvent]:
        rows = self._conn.execute("SELECT * FROM events ORDER BY seq").fetchall()
        return [self._row_to_event(r) for r in rows]

    def try_lease_one_task(self, worker_id: str, ts: str) -> Record | None:
        """Atomically lease one pending task. Workers share only this SQLite file."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                """
                SELECT * FROM objects
                WHERE kind = ? AND status = ?
                ORDER BY id
                LIMIT 1
                """,
                (ObjectKind.TASK.value, TaskStatus.PENDING.value),
            ).fetchone()
            if row is None:
                self._conn.rollback()
                return None
            rec = self._row_to_record(row)
            payload = {**rec.payload, "lease_owner": worker_id}
            cur = self._conn.execute(
                """
                UPDATE objects SET status = ?, payload = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    TaskStatus.LEASED.value,
                    _json(payload),
                    ts,
                    rec.id,
                    TaskStatus.PENDING.value,
                ),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                return None
            self.append_event(
                ts=ts,
                event_type="lease_task",
                object_id=rec.id,
                reason=f"leased by {worker_id}",
                old_status=TaskStatus.PENDING.value,
                new_status=TaskStatus.LEASED.value,
                payload={"lease_owner": worker_id},
                commit=False,
            )
            self._conn.commit()
            rec.status = TaskStatus.LEASED.value
            rec.payload = payload
            rec.updated_at = ts
            return rec
        except Exception:
            self._conn.rollback()
            raise

    def complete_task(self, task_id: str, worker_id: str, ts: str) -> Record:
        rec = self.get(task_id)
        if rec.kind != ObjectKind.TASK:
            raise ValueError(f"{task_id} is not a task")
        if rec.status != TaskStatus.LEASED.value:
            raise ValueError(f"{task_id} status {rec.status} is not leased")
        if rec.payload.get("lease_owner") != worker_id:
            raise ValueError(f"{task_id} owned by {rec.payload.get('lease_owner')} not {worker_id}")
        payload = {**rec.payload, "completed_by": worker_id}
        cur = self._conn.execute(
            """
            UPDATE objects SET status = ?, payload = ?, updated_at = ?
            WHERE id = ? AND status = ?
            """,
            (
                TaskStatus.DONE.value,
                _json(payload),
                ts,
                task_id,
                TaskStatus.LEASED.value,
            ),
        )
        if cur.rowcount != 1:
            raise RuntimeError(f"lost lease on {task_id}")
        self.append_event(
            ts=ts,
            event_type="complete_task",
            object_id=task_id,
            reason=f"completed by {worker_id}",
            old_status=TaskStatus.LEASED.value,
            new_status=TaskStatus.DONE.value,
            payload={"lease_owner": worker_id},
        )
        rec.status = TaskStatus.DONE.value
        rec.payload = payload
        rec.updated_at = ts
        return rec

    def _row_to_record(self, row: sqlite3.Row) -> Record:
        return Record(
            id=str(row["id"]),
            kind=ObjectKind(row["kind"]),
            status=str(row["status"]),
            provenance=provenance_from_dict(json.loads(row["provenance"])),
            payload=json.loads(row["payload"] or "{}"),
            evidence_ids=tuple(json.loads(row["evidence_ids"] or "[]")),
            source_snapshot=str(row["source_snapshot"] or ""),
            invalidation_conditions=conditions_from_json(row["invalidation_conditions"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        return Edge(
            id=str(row["id"]),
            src=str(row["src"]),
            dst=str(row["dst"]),
            mode=EdgeMode(row["mode"]),
            rule=InvalidationRule(row["rule"]),
            declared_effect=str(row["declared_effect"]),
        )

    def _row_to_event(self, row: sqlite3.Row) -> CausalEvent:
        return CausalEvent(
            seq=int(row["seq"]),
            ts=str(row["ts"]),
            event_type=str(row["event_type"]),
            object_id=str(row["object_id"]),
            old_status=row["old_status"],
            new_status=row["new_status"],
            reason=str(row["reason"]),
            edge_id=row["edge_id"],
            rule=row["rule"],
            payload=json.loads(row["payload"] or "{}"),
        )


def _index_pair(edge: Edge) -> tuple[str, str]:
    """(target, dependent): when target changes, dependent is re-evaluated."""
    if edge.mode in (EdgeMode.DEPENDS_ON,):
        return edge.dst, edge.src
    if edge.mode in (EdgeMode.JUSTIFIES, EdgeMode.SUPPORTS, EdgeMode.PRODUCED_FROM):
        return edge.src, edge.dst
    if edge.mode in (EdgeMode.CONTRADICTS, EdgeMode.SUPERSEDES):
        return edge.src, edge.dst
    return edge.dst, edge.src
