"""Transactional schema migrations. No automatic downgrade."""

from __future__ import annotations

import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, current_version, ensure_migrations_table, stamp_current
from .verify import verify_store


class MigrationError(RuntimeError):
    pass


def migrate_to_latest(store: Any, *, confirm: bool) -> dict[str, Any]:
    if not confirm:
        raise MigrationError("refusing to migrate without confirm=True (--confirm)")
    if getattr(store, "read_only", False):
        raise MigrationError("cannot migrate a read-only store")
    ensure_migrations_table(store)
    before = current_version(store)
    steps: list[str] = []
    _lock(store)
    try:
        ver = current_version(store)
        if ver < 1:
            raise MigrationError("no objects table; initialize a store instead of migrating")
        if ver < 2:
            _apply_v2(store)
            steps.append("1->2 lease_config and TTL ceiling triggers")
        stamp_current(store)
        after = current_version(store)
        if after < SCHEMA_VERSION:
            raise MigrationError(
                f"migration incomplete: store version {after} < code {SCHEMA_VERSION}"
            )
    finally:
        _unlock(store)
    report = verify_store(store)
    return {
        "ok": report["ok"] and current_version(store) >= SCHEMA_VERSION,
        "from_version": before,
        "to_version": current_version(store),
        "steps": steps,
        "downgrade": "not supported; restore from backup if event semantics would be lost",
        "verify": report,
    }


def _pg(store: Any) -> bool:
    return getattr(store, "backend", "") == "postgres"


def _lock(store: Any) -> None:
    if _pg(store):
        store._conn.execute("SELECT pg_advisory_lock(872343, 1)")
        return
    # Separate lock file: sqlite3.executescript COMMITs first, so BEGIN IMMEDIATE
    # on the store connection cannot be held across _init_schema.
    lock_path = Path(str(store.db_path) + ".migrate.lock")
    fh = open(lock_path, "a")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    store._migrate_lock_fh = fh


def _unlock(store: Any) -> None:
    if _pg(store):
        store._conn.execute("SELECT pg_advisory_unlock(872343, 1)")
        return
    fh = getattr(store, "_migrate_lock_fh", None)
    if fh is None:
        return
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()
        store._migrate_lock_fh = None


def _apply_v2(store: Any) -> None:
    applied = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if _pg(store):
        store._ensure_lease_schema()
        store._write_lease_config()
        store._conn.execute(
            """
            INSERT INTO schema_migrations(version, applied_at)
            VALUES (%s, %s)
            ON CONFLICT (version) DO NOTHING
            """,
            (2, applied),
        )
        return
    store._init_schema()
    store._write_lease_config()
    store._conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (2, applied),
    )
    store._conn.commit()
