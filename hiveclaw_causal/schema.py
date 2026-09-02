"""Monotonic schema versions for the centralized Rewind store.

Current version is 2 (lease_config + TTL ceiling triggers from Session 8).
Version 1 is objects/events/edges/reverse_deps + append-only event triggers.
There is no automatic downgrade.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 2


def _pg(store: Any) -> bool:
    return getattr(store, "backend", "") == "postgres"


def ensure_migrations_table(store: Any) -> None:
    if getattr(store, "read_only", False):
        return
    if _pg(store):
        store._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              version INTEGER PRIMARY KEY,
              applied_at TEXT NOT NULL
            )
            """
        )
        return
    store._conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          applied_at TEXT NOT NULL
        )
        """
    )
    store._conn.commit()


def stamp_current(store: Any) -> None:
    """Record SCHEMA_VERSION on a newly initialized store. Idempotent."""
    ensure_migrations_table(store)
    applied = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if _pg(store):
        store._conn.execute(
            """
            INSERT INTO schema_migrations(version, applied_at)
            VALUES (%s, %s)
            ON CONFLICT (version) DO NOTHING
            """,
            (SCHEMA_VERSION, applied),
        )
        return
    store._conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, applied),
    )
    store._conn.commit()


def current_version(store: Any) -> int:
    if not getattr(store, "read_only", False):
        ensure_migrations_table(store)
    stamped = _stamped_version(store)
    if stamped > 0:
        return stamped
    return infer_legacy_version(store)


def _stamped_version(store: Any) -> int:
    try:
        row = store._conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations"
        ).fetchone()
    except Exception:
        return 0
    if row is None:
        return 0
    try:
        val = row["v"] if hasattr(row, "keys") else row[0]
        return int(val)
    except (KeyError, IndexError, TypeError):
        return 0


def infer_legacy_version(store: Any) -> int:
    """Pre-schema_migrations files: 2 if lease_config exists, else 1 if objects exists."""
    if _pg(store):
        schema = getattr(store, "schema", "public")
        lease = store._conn.execute(
            "SELECT to_regclass(%s) AS rel", (f"{schema}.lease_config",)
        ).fetchone()
        objs = store._conn.execute(
            "SELECT to_regclass(%s) AS rel", (f"{schema}.objects",)
        ).fetchone()
        if lease is not None and lease["rel"] is not None:
            return 2
        if objs is not None and objs["rel"] is not None:
            return 1
        return 0
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='lease_config'"
    ).fetchone()
    if row is not None:
        return 2
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='objects'"
    ).fetchone()
    return 1 if row is not None else 0
