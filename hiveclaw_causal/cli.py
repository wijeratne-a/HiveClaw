"""Operator CLI for the centralized Rewind store.

``python -m hiveclaw_causal <command>`` is the supported entrypoint.
``hiveclaw-causal`` in docs is this same module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .store import open_store


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hiveclaw-causal",
        description="Rewind centralized causal store (not a multi-master runtime).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run The Rewind fixture demo")
    d.add_argument("--db", default="output/rewind.sqlite")
    d.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("store-status", help="read-only lease/store operational view")
    s.add_argument("--db", required=True, help="SQLite path or postgres URL")
    s.add_argument("--json", action="store_true")

    v = sub.add_parser("verify-store", help="read-only event-log / projection integrity")
    v.add_argument("--db", required=True)
    v.add_argument("--json", action="store_true")

    b = sub.add_parser("backup", help="SQLite consistent backup (VACUUM/backup API)")
    b.add_argument("--db", required=True, help="source SQLite path")
    b.add_argument("--out", required=True, help="destination file")

    r = sub.add_parser("restore", help="copy a SQLite backup to an isolated dest path")
    r.add_argument("--backup", required=True)
    r.add_argument("--db", required=True, help="destination SQLite path")
    r.add_argument("--confirm", action="store_true")

    m = sub.add_parser("migrate", help="upgrade schema to the code version (write)")
    m.add_argument("--db", required=True)
    m.add_argument("--to-latest", action="store_true", required=True)
    m.add_argument("--confirm", action="store_true")

    i = sub.add_parser("inspect", help="explain one object")
    i.add_argument("--db", required=True)
    i.add_argument("--id", required=True)

    args = p.parse_args(argv)
    if args.cmd == "demo":
        from .demo_rewind import main as demo_main

        return demo_main(["--db", args.db, "--seed", str(args.seed)])
    if args.cmd == "store-status":
        return _status(args.db, as_json=args.json)
    if args.cmd == "verify-store":
        return _verify(args.db, as_json=args.json)
    if args.cmd == "backup":
        if _is_postgres_locator(args.db):
            print(
                "backup CLI is SQLite-only; Postgres: pg_dump or "
                "hiveclaw_causal.backup.postgres_logical_backup",
                file=sys.stderr,
            )
            return 2
        from .backup import backup_sqlite

        report = backup_sqlite(args.db, args.out)
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1
    if args.cmd == "restore":
        if not args.confirm:
            print("restore requires --confirm (will write dest)", file=sys.stderr)
            return 2
        if _is_postgres_locator(args.db) or _is_postgres_locator(args.backup):
            print(
                "restore CLI is SQLite-only; Postgres: pg_restore or "
                "hiveclaw_causal.backup.postgres_logical_backup",
                file=sys.stderr,
            )
            return 2
        from .backup import restore_sqlite, verify_after_restore

        report = restore_sqlite(args.backup, args.db)
        ver = verify_after_restore(args.db)
        print(json.dumps({"restore": report, "verify": ver}, indent=2))
        return 0 if report.get("ok") and ver.get("ok") else 1
    if args.cmd == "migrate":
        if not args.confirm:
            print("migrate requires --confirm", file=sys.stderr)
            return 2
        from .migrate import MigrationError, migrate_to_latest

        store = open_store(args.db, read_only=False)
        try:
            report = migrate_to_latest(store, confirm=True)
        except MigrationError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        finally:
            store.close()
        print(json.dumps({k: v for k, v in report.items() if k != "verify"}, indent=2))
        print(json.dumps(report["verify"], indent=2))
        return 0 if report.get("ok") else 1
    if args.cmd == "inspect":
        from .inspect import explain

        store = open_store(args.db, read_only=True)
        try:
            info = explain(store, args.id)
        except KeyError:
            print(f"unknown id: {args.id}", file=sys.stderr)
            return 1
        finally:
            store.close()
        print(f"{info['object_id']} kind={info['kind']} status={info['status']}")
        print(f"last_reason={info['last_reason']}")
        return 0
    return 2


def _is_postgres_locator(locator: str) -> bool:
    return locator.startswith("postgres") or locator.startswith("{")


def _status(locator: str, *, as_json: bool) -> int:
    from .ops_status import format_status, store_status

    store = open_store(locator, read_only=True)
    try:
        report = store_status(store)
    finally:
        store.close()
    if as_json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_status(report))
    return 0


def _verify(locator: str, *, as_json: bool) -> int:
    from .verify import verify_store

    store = open_store(locator, read_only=True)
    try:
        report = verify_store(store)
    finally:
        store.close()
    if as_json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"{report['summary']} backend={report['backend']} objects={report['objects']} events={report['events']}")
        for c in report["checks"]:
            mark = "ok" if c["ok"] else "FAIL"
            print(f"  [{mark}] {c['name']}: {c['detail']}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
