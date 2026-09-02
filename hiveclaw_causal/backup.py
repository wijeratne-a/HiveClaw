"""Backup and restore for the one authoritative store."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .store import Store, open_store
from .verify import verify_store


def sqlite_integrity(path: Path | str) -> str:
    conn = sqlite3.connect(str(path), timeout=30.0)
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row else "missing"
    finally:
        conn.close()


def backup_sqlite(src: Path | str, dest: Path | str) -> dict[str, Any]:
    src_p = Path(src)
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    before = sqlite_integrity(src_p)
    src_conn = sqlite3.connect(str(src_p), timeout=30.0)
    try:
        dest_conn = sqlite3.connect(str(dest_p), timeout=30.0)
        try:
            src_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        src_conn.close()
    after_src = sqlite_integrity(src_p)
    after_dest = sqlite_integrity(dest_p)
    return {
        "ok": before == "ok" and after_src == "ok" and after_dest == "ok",
        "src": str(src_p),
        "dest": str(dest_p),
        "integrity_before": before,
        "integrity_src_after": after_src,
        "integrity_dest": after_dest,
    }


def restore_sqlite(backup: Path | str, dest: Path | str) -> dict[str, Any]:
    """Copy a backup file to dest (isolated path). Does not overwrite in place without copy."""
    return backup_sqlite(backup, dest)


def postgres_logical_backup(dsn: str, src_schema: str, dest_schema: str) -> dict[str, Any]:
    """Copy one schema's data into a newly initialized dest schema on the same server.

    Production operators should still use pg_dump/pg_restore; this is the
    in-process drill that does not require the pg_dump binary.
    """
    from .pg_store import PgStore, _check_schema

    src = _check_schema(src_schema)
    dst = _check_schema(dest_schema)
    dest_store = PgStore(dsn, schema=dst)
    src_store = PgStore(dsn, schema=src, read_only=True)
    try:
        dest_store._conn.execute("SET default_transaction_read_only = off")
        copy_order = [
            "objects",
            "edges",
            "reverse_deps",
            "events",
            "lease_config",
            "schema_migrations",
        ]
        for table in copy_order:
            present = src_store._conn.execute(
                "SELECT to_regclass(%s) AS rel", (f"{src}.{table}",)
            ).fetchone()
            if present is None or present["rel"] is None:
                continue
            if table == "lease_config":
                dest_store._conn.execute(f'DELETE FROM "{dst}".lease_config')
            sql = f'INSERT INTO "{dst}".{table} SELECT * FROM "{src}".{table}'
            if table == "schema_migrations":
                sql += " ON CONFLICT DO NOTHING"
            dest_store._conn.execute(sql)
        dest_store._conn.execute(
            """
            SELECT setval(
              pg_get_serial_sequence('events', 'seq'),
              COALESCE((SELECT MAX(seq) FROM events), 1)
            )
            """
        )
        report = verify_store(dest_store)
    finally:
        src_store.close()
        dest_store.close()
    return {"ok": report["ok"], "src_schema": src, "dest_schema": dst, "verify": report}


def verify_after_restore(locator: str) -> dict[str, Any]:
    store = open_store(locator, read_only=True)
    try:
        return verify_store(store)
    finally:
        store.close()
