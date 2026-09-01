"""Process-safe task lease drain. Workers share a SQLite path only — no messaging."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .store import Store


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def drain_pending_tasks(
    db_path: str,
    worker_id: str,
    pause_s: float = 0.002,
) -> list[str]:
    store = Store(db_path)
    leased: list[str] = []
    try:
        while True:
            rec = store.try_lease_one_task(worker_id, _ts())
            if rec is None:
                break
            leased.append(rec.id)
            if pause_s > 0:
                time.sleep(pause_s)
            store.complete_task(rec.id, worker_id, _ts())
    finally:
        store.close()
    return leased


def mp_drain(payload: tuple[str, str, float]) -> tuple[str, list[str]]:
    db_path, worker_id, pause_s = payload
    return worker_id, drain_pending_tasks(db_path, worker_id, pause_s)
