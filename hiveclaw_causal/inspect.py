"""Live queries: why is this object in this status?"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .store import Store


def explain(store: Store, object_id: str) -> dict[str, Any]:
    rec = store.get(object_id)
    events = store.events_for(object_id)
    last = events[-1] if events else None
    return {
        "object_id": object_id,
        "kind": rec.kind.value,
        "status": rec.status,
        "producer": rec.provenance.producer,
        "trust": rec.provenance.trust.value,
        "evidence_ids": rec.evidence_ids,
        "events": events,
        "last_reason": None if last is None else last.reason,
        "last_edge_id": None if last is None else last.edge_id,
        "last_rule": None if last is None else last.rule,
        "old_status": None if last is None else last.old_status,
        "new_status": None if last is None else last.new_status,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect HiveClaw causal object status from SQLite.")
    p.add_argument("--db", required=True)
    p.add_argument("--id", required=True, help="object id")
    args = p.parse_args(argv)
    store = Store(Path(args.db))
    try:
        info = explain(store, args.id)
    except KeyError:
        print(f"unknown id: {args.id}", file=sys.stderr)
        return 1
    finally:
        store.close()
    print(f"{info['object_id']} kind={info['kind']} status={info['status']}")
    print(f"producer={info['producer']} trust={info['trust']}")
    print(f"evidence={list(info['evidence_ids'])}")
    print(f"last_reason={info['last_reason']}")
    print(f"last_edge_id={info['last_edge_id']} rule={info['last_rule']}")
    print(f"old_status={info['old_status']} -> new_status={info['new_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
