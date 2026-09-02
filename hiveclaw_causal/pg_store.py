"""Postgres-backed causal store. Same records/events/leases as SQLite Store.

This is a networked *server*, not multi-master stigmergy. Clients share one
Postgres over TCP. Causal semantics (statuses, edges, reverse_deps, TTL leases)
are unchanged; SQL dialect and locking are the port.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from .store import (
    _index_pair,
    _json,
    conditions_from_json,
    conditions_to_json,
    provenance_from_dict,
    provenance_to_dict,
)
from .types import (
    CausalEvent,
    Edge,
    EdgeMode,
    InvalidationRule,
    ObjectKind,
    Record,
    TaskStatus,
)

_SCHEMA_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

try:
    import psycopg  # type: ignore[import-not-found]
    from psycopg.rows import dict_row  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional extra
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


def require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required for PgStore. Install: python -m pip install 'psycopg[binary]'"
        )


def new_schema_name(prefix: str = "hc") -> str:
    raw = f"{prefix}_{uuid.uuid4().hex[:10]}"
    if not _SCHEMA_OK.fullmatch(raw):
        raise ValueError(raw)
    return raw


def locator_json(dsn: str, schema: str) -> str:
    return json.dumps({"dsn": dsn, "schema": schema}, separators=(",", ":"))


def _check_schema(name: str) -> str:
    if not _SCHEMA_OK.fullmatch(name):
        raise ValueError(f"unsafe schema name: {name!r}")
    return name


class PgStore:
    """Duck-types Store. `db_path` is a label for logs, not a file."""

    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        require_psycopg()
        assert psycopg is not None
        assert dict_row is not None
        self.dsn = dsn
        self.schema = _check_schema(schema)
        self.db_path = Path(f"pg:{self.schema}")
        self._conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
        self._ensure_schema()
        if not self._schema_ready():
            self._conn.execute(
                "SELECT pg_advisory_lock(872341, hashtext(%s))", (self.schema,)
            )
            try:
                if not self._schema_ready():
                    self._init_schema()
            finally:
                self._conn.execute(
                    "SELECT pg_advisory_unlock(872341, hashtext(%s))",
                    (self.schema,),
                )

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
        self._conn.execute(f'SET search_path TO "{self.schema}", public')

    def _schema_ready(self) -> bool:
        row = self._conn.execute(
            "SELECT to_regclass(%s) AS rel",
            (f"{self.schema}.objects",),
        ).fetchone()
        return row is not None and row["rel"] is not None

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              seq BIGSERIAL PRIMARY KEY,
              ts TEXT NOT NULL,
              event_type TEXT NOT NULL,
              object_id TEXT NOT NULL,
              old_status TEXT,
              new_status TEXT,
              reason TEXT NOT NULL,
              edge_id TEXT,
              rule TEXT,
              payload TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE OR REPLACE FUNCTION hiveclaw_events_append_only()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
              RAISE EXCEPTION 'events is append-only: % is forbidden', TG_OP;
            END;
            $$
            """
        )
        self._conn.execute("DROP TRIGGER IF EXISTS events_append_only_no_update ON events")
        self._conn.execute(
            """
            CREATE TRIGGER events_append_only_no_update
            BEFORE UPDATE ON events
            FOR EACH ROW
            EXECUTE PROCEDURE hiveclaw_events_append_only()
            """
        )
        self._conn.execute("DROP TRIGGER IF EXISTS events_append_only_no_delete ON events")
        self._conn.execute(
            """
            CREATE TRIGGER events_append_only_no_delete
            BEFORE DELETE ON events
            FOR EACH ROW
            EXECUTE PROCEDURE hiveclaw_events_append_only()
            """
        )
        self._conn.execute(
            """
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
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
              id TEXT PRIMARY KEY,
              src TEXT NOT NULL,
              dst TEXT NOT NULL,
              mode TEXT NOT NULL,
              rule TEXT NOT NULL,
              declared_effect TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reverse_deps (
              target_id TEXT NOT NULL,
              dependent_id TEXT NOT NULL,
              edge_id TEXT NOT NULL,
              rule TEXT NOT NULL,
              PRIMARY KEY (target_id, dependent_id, edge_id)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reverse_target ON reverse_deps(target_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_object ON events(object_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_objects_kind_status ON objects(kind, status)"
        )

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
        cur = self._conn.execute(
            """
            INSERT INTO events (ts, event_type, object_id, old_status, new_status,
                                reason, edge_id, rule, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING seq
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
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("INSERT INTO events did not return seq")
        seq = int(row["seq"])
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
            WHERE object_id = %s AND edge_id = %s AND new_status = %s
            LIMIT 1
            """,
            (object_id, edge_id, new_status),
        ).fetchone()
        return row is not None

    def put_object(self, rec: Record, *, ts: str, event_type: str, reason: str) -> Record:
        existing = self.get_or_none(rec.id)
        self._conn.execute(
            """
            INSERT INTO objects (
              id, kind, status, provenance, payload, evidence_ids,
              source_snapshot, invalidation_conditions, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
              kind=EXCLUDED.kind,
              status=EXCLUDED.status,
              provenance=EXCLUDED.provenance,
              payload=EXCLUDED.payload,
              evidence_ids=EXCLUDED.evidence_ids,
              source_snapshot=EXCLUDED.source_snapshot,
              invalidation_conditions=EXCLUDED.invalidation_conditions,
              updated_at=EXCLUDED.updated_at
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
            "SELECT * FROM objects WHERE id = %s", (object_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def objects_of(self, kind: ObjectKind) -> list[Record]:
        rows = self._conn.execute(
            "SELECT * FROM objects WHERE kind = %s ORDER BY id", (kind.value,)
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
            UPDATE objects SET status = %s, updated_at = %s, payload = %s
            WHERE id = %s
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
            "SELECT id FROM edges WHERE id = %s", (edge.id,)
        ).fetchone()
        if existing is not None:
            return edge
        self._conn.execute(
            """
            INSERT INTO edges (id, src, dst, mode, rule, declared_effect)
            VALUES (%s, %s, %s, %s, %s, %s)
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
            INSERT INTO reverse_deps (target_id, dependent_id, edge_id, rule)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (target, dependent, edge.id, edge.rule.value),
        )
        return edge

    def reverse_lookup(self, target_id: str) -> list[tuple[str, str, str]]:
        rows = self._conn.execute(
            """
            SELECT dependent_id, edge_id, rule FROM reverse_deps
            WHERE target_id = %s
            """,
            (target_id,),
        ).fetchall()
        return [(str(r["dependent_id"]), str(r["edge_id"]), str(r["rule"])) for r in rows]

    def dependent_tasks(self, target_id: str) -> list[Record]:
        return self.dependent_of_kind(target_id, ObjectKind.TASK)

    def dependent_claims(self, target_id: str) -> list[Record]:
        return self.dependent_of_kind(target_id, ObjectKind.CLAIM)

    def dependent_of_kind(self, target_id: str, kind: ObjectKind) -> list[Record]:
        rows = self._conn.execute(
            """
            SELECT o.*
            FROM reverse_deps r
            JOIN objects o ON o.id = r.dependent_id
            WHERE r.target_id = %s AND o.kind = %s
            ORDER BY o.id
            """,
            (target_id, kind.value),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def edges_to(self, dst: str, mode: EdgeMode | None = None) -> list[Edge]:
        if mode is None:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE dst = %s", (dst,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE dst = %s AND mode = %s",
                (dst, mode.value),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def events_for(self, object_id: str) -> list[CausalEvent]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE object_id = %s ORDER BY seq", (object_id,)
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def all_events(self) -> list[CausalEvent]:
        rows = self._conn.execute("SELECT * FROM events ORDER BY seq").fetchall()
        return [self._row_to_event(r) for r in rows]

    def try_lease_one_task(
        self,
        worker_id: str,
        ts: str,
        *,
        lease_ttl_s: float = 30.0,
    ) -> Record | None:
        now = time.time()
        lease_until = now + lease_ttl_s
        with self._conn.transaction():
            row = self._conn.execute(
                """
                SELECT * FROM objects
                WHERE kind = %s
                  AND (
                    status = %s
                    OR (
                      status = %s
                      AND (payload::json->>'lease_until') IS NOT NULL
                      AND (payload::json->>'lease_until')::float < %s
                    )
                  )
                ORDER BY CASE status WHEN %s THEN 0 ELSE 1 END, id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                (
                    ObjectKind.TASK.value,
                    TaskStatus.PENDING.value,
                    TaskStatus.LEASED.value,
                    now,
                    TaskStatus.PENDING.value,
                ),
            ).fetchone()
            if row is None:
                return None
            rec = self._row_to_record(row)
            previous_owner = rec.payload.get("lease_owner")
            reclaim = rec.status == TaskStatus.LEASED.value
            payload = {
                **rec.payload,
                "lease_owner": worker_id,
                "lease_until": lease_until,
            }
            if reclaim:
                payload["reclaimed_from"] = previous_owner
            cas_status = (
                TaskStatus.LEASED.value if reclaim else TaskStatus.PENDING.value
            )
            cur = self._conn.execute(
                """
                UPDATE objects SET status = %s, payload = %s, updated_at = %s
                WHERE id = %s AND status = %s
                """,
                (
                    TaskStatus.LEASED.value,
                    _json(payload),
                    ts,
                    rec.id,
                    cas_status,
                ),
            )
            if cur.rowcount != 1:
                raise RuntimeError("lease CAS lost the row")
            event_type = "lease_reclaim" if reclaim else "lease_task"
            reason = (
                f"reclaimed by {worker_id} from {previous_owner}"
                if reclaim
                else f"leased by {worker_id}"
            )
            self.append_event(
                ts=ts,
                event_type=event_type,
                object_id=rec.id,
                reason=reason,
                old_status=cas_status,
                new_status=TaskStatus.LEASED.value,
                payload={
                    "lease_owner": worker_id,
                    "lease_until": lease_until,
                    "reclaimed": reclaim,
                },
                commit=False,
            )
            rec.status = TaskStatus.LEASED.value
            rec.payload = payload
            rec.updated_at = ts
            return rec

    def renew_lease(
        self,
        task_id: str,
        worker_id: str,
        ts: str,
        *,
        lease_ttl_s: float = 30.0,
    ) -> Record:
        now = time.time()
        lease_until = now + lease_ttl_s
        with self._conn.transaction():
            rec = self.get(task_id)
            if rec.kind != ObjectKind.TASK:
                raise ValueError(f"{task_id} is not a task")
            if rec.status != TaskStatus.LEASED.value:
                raise ValueError(f"{task_id} status {rec.status} is not leased")
            if rec.payload.get("lease_owner") != worker_id:
                raise ValueError(
                    f"{task_id} owned by {rec.payload.get('lease_owner')} not {worker_id}"
                )
            payload = {**rec.payload, "lease_until": lease_until}
            cur = self._conn.execute(
                """
                UPDATE objects SET payload = %s, updated_at = %s
                WHERE id = %s AND status = %s
                  AND payload::json->>'lease_owner' = %s
                """,
                (_json(payload), ts, task_id, TaskStatus.LEASED.value, worker_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError(f"lost lease on {task_id} during renew")
            rec.payload = payload
            rec.updated_at = ts
            return rec

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
            UPDATE objects SET status = %s, payload = %s, updated_at = %s
            WHERE id = %s AND status = %s
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

    def _row_to_record(self, row: dict[str, Any]) -> Record:
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

    def _row_to_edge(self, row: dict[str, Any]) -> Edge:
        return Edge(
            id=str(row["id"]),
            src=str(row["src"]),
            dst=str(row["dst"]),
            mode=EdgeMode(row["mode"]),
            rule=InvalidationRule(row["rule"]),
            declared_effect=str(row["declared_effect"]),
        )

    def _row_to_event(self, row: dict[str, Any]) -> CausalEvent:
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


# Fix _row_to_edge - InvalidationRule import
# (patched below if needed)
