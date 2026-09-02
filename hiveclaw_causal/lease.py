"""Process-safe task lease drain. Workers share a store locator only — no messaging."""

from __future__ import annotations

import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from .store import open_store


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def drain_pending_tasks(
    db_path: str,
    worker_id: str,
    pause_s: float = 0.002,
    *,
    lease_ttl_s: float = 30.0,
) -> list[str]:
    store = open_store(db_path)
    leased: list[str] = []
    try:
        while True:
            rec = store.try_lease_one_task(
                worker_id, _ts(), lease_ttl_s=lease_ttl_s
            )
            if rec is None:
                break
            leased.append(rec.id)
            if pause_s > 0:
                time.sleep(pause_s)
            store.complete_task(rec.id, worker_id, _ts())
    finally:
        store.close()
    return leased


def drain_until_idle(
    db_path: str,
    worker_id: str,
    stop_path: str,
    pause_s: float = 0.002,
    *,
    lease_ttl_s: float = 30.0,
    idle_polls_after_stop: int = 8,
) -> list[str]:
    """Lease while a producer may still be inserting. Exit after stop file + idle polls."""
    store = open_store(db_path)
    leased: list[str] = []
    idle = 0
    try:
        while True:
            rec = store.try_lease_one_task(
                worker_id, _ts(), lease_ttl_s=lease_ttl_s
            )
            if rec is None:
                if Path(stop_path).exists():
                    idle += 1
                    if idle >= idle_polls_after_stop:
                        break
                time.sleep(max(pause_s, 0.001))
                continue
            idle = 0
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


def mp_drain_until_idle(
    payload: tuple[str, str, float, str],
) -> tuple[str, list[str]]:
    db_path, worker_id, pause_s, stop_path = payload
    return worker_id, drain_until_idle(db_path, worker_id, stop_path, pause_s)


def lease_one_and_die(payload: tuple[str, str, float]) -> None:
    """Lease one task, then SIGKILL this process before complete (crash mid-lease)."""
    db_path, worker_id, lease_ttl_s = payload
    store = open_store(db_path)
    try:
        rec = store.try_lease_one_task(
            worker_id, _ts(), lease_ttl_s=lease_ttl_s
        )
        if rec is None:
            return
    finally:
        store.close()
    os.kill(os.getpid(), signal.SIGKILL)


def work_slow_with_renew(payload: tuple[str, str, float, float, float]) -> None:
    """Hold a lease longer than TTL while renewing; then complete. Must stay owner."""
    db_path, worker_id, lease_ttl_s, work_s, renew_every_s = payload
    store = open_store(db_path)
    try:
        rec = store.try_lease_one_task(
            worker_id, _ts(), lease_ttl_s=lease_ttl_s
        )
        if rec is None:
            raise RuntimeError("no task to lease")
        deadline = time.time() + work_s
        while time.time() < deadline:
            try:
                store.renew_lease(
                    rec.id, worker_id, _ts(), lease_ttl_s=lease_ttl_s
                )
            except (ValueError, RuntimeError):
                # Lost the lease (TTL reclaim / poacher). Stay a clean exit.
                return
            remaining = deadline - time.time()
            time.sleep(max(0.0, min(renew_every_s, remaining)))
        try:
            store.complete_task(rec.id, worker_id, _ts())
        except (ValueError, RuntimeError):
            return
    finally:
        store.close()


def hold_lease_renew_until_fail(
    payload: tuple[str, str, float, str, str],
) -> None:
    """Lease and heartbeat until the store raises (e.g. TCP drop). Stay alive after.

    This is the network-failure analogue of SIGKILL: the process does not die;
    only the path to the server is cut. The lease row stays until TTL reclaim.
    """
    db_path, worker_id, lease_ttl_s, ready_path, fail_path = payload
    store = open_store(db_path)
    rec = store.try_lease_one_task(worker_id, _ts(), lease_ttl_s=lease_ttl_s)
    if rec is None:
        Path(fail_path).write_text("no task to lease\n", encoding="utf-8")
        store.close()
        return
    Path(ready_path).write_text(rec.id, encoding="utf-8")
    try:
        while True:
            time.sleep(0.05)
            store.renew_lease(rec.id, worker_id, _ts(), lease_ttl_s=lease_ttl_s)
    except Exception as exc:
        Path(fail_path).write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        try:
            store.close()
        except Exception:
            pass
        time.sleep(60.0)
