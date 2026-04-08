#!/usr/bin/env python3
"""
Repo Pulse split-screen demo runner.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from demos.audit_swarm import run_audit_swarm
from demos.baseline_audit import run_baseline
from demos.feature_dashboard import FeatureDashboard


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dict_path() -> Path:
    return _repo_root() / "demos" / "data" / "feature_dictionary.json"


def _state_panel(title: str, state: dict[str, Any]) -> Panel:
    t = Table.grid()
    t.add_column()
    t.add_row(f"status: {state.get('status', 'pending')}")
    t.add_row(f"agent: {state.get('agent', '-')}")
    t.add_row(f"elapsed: {state.get('elapsed_s', 0.0):.2f}s")
    t.add_row(f"coord_context_chars: {state.get('coord_context_chars', 0):,}")
    if "serialized_chars" in state:
        t.add_row(f"serialized_chars: {state.get('serialized_chars', 0):,}")
    if "slab_rw_count" in state:
        t.add_row(f"slab_rw_count: {state.get('slab_rw_count', 0)}")
    if "report_items" in state:
        t.add_row(f"report_items: {state.get('report_items', 0)}")
    return Panel(t, title=title)


def _scoreboard(baseline: dict[str, Any], hive: dict[str, Any]) -> Table:
    t = Table(title="Repo Pulse Scoreboard")
    t.add_column("Metric")
    t.add_column("Baseline", justify="right")
    t.add_column("HiveClaw", justify="right")
    b_wall = float(baseline.get("wall_s", 0.0) or 0.0)
    h_wall = float(hive.get("wall_s", 0.0) or 0.0)
    speed = (b_wall / h_wall) if h_wall > 1e-9 else 0.0
    t.add_row("wall_s", f"{b_wall:.2f}", f"{h_wall:.2f}")
    t.add_row("speedup (baseline/hive)", "-", f"{speed:.2f}x")
    t.add_row(
        "coord_context_chars",
        f"{int(baseline.get('coord_context_chars', 0)):,}",
        f"{int(hive.get('coord_context_chars', 0)):,}",
    )
    t.add_row(
        "report_items",
        str(int(baseline.get("report_items", 0))),
        str(int(hive.get("report_items", 0))),
    )
    return t


def run_demo(base_url: str, model: str, max_files: int, slot: int) -> int:
    baseline_state: dict[str, Any] = {"status": "pending", "agent": "A->B->C"}
    hive_state: dict[str, Any] = {"status": "pending", "agent": "A->B->C"}
    result_baseline: dict[str, Any] = {}
    result_hive: dict[str, Any] = {}

    lock = threading.Lock()
    done_baseline = threading.Event()
    done_hive = threading.Event()

    report_base = _repo_root() / "HEALTH_REPORT_BASELINE.md"
    report_hive = _repo_root() / "HEALTH_REPORT.md"

    def baseline_worker() -> None:
        t0 = time.perf_counter()
        with lock:
            baseline_state["status"] = "running"
        met = run_baseline(
            base_url=base_url, model=model, report_path=report_base, max_files=max_files
        )
        with lock:
            result_baseline.update(asdict(met))
            baseline_state.update(asdict(met))
            baseline_state["elapsed_s"] = time.perf_counter() - t0
            baseline_state["status"] = "done"
        done_baseline.set()

    def hive_worker() -> None:
        t0 = time.perf_counter()
        with lock:
            hive_state["status"] = "running"
        met = run_audit_swarm(
            base_url=base_url, model=model, report_path=report_hive, max_files=max_files
        )
        with lock:
            result_hive.update(asdict(met))
            hive_state.update(asdict(met))
            hive_state["elapsed_s"] = time.perf_counter() - t0
            hive_state["status"] = "done"
        done_hive.set()

    dash = FeatureDashboard(_dict_path())
    tb = threading.Thread(target=baseline_worker, daemon=True)
    th = threading.Thread(target=hive_worker, daemon=True)
    tb.start()
    th.start()

    with Live(refresh_per_second=4) as live:
        while not (done_baseline.is_set() and done_hive.is_set()):
            with lock:
                left = _state_panel("Baseline (JSON transcript)", baseline_state)
                right = _state_panel("HiveClaw (slab coordination)", hive_state)
            top = Table.grid(expand=True)
            top.add_column(ratio=1)
            top.add_column(ratio=1)
            top.add_row(left, right)
            try:
                rows = dash.snapshot(slot_index=slot, top_k=8)
                feat = FeatureDashboard.render_table(rows)
            except Exception:
                feat = FeatureDashboard.render_table([])
            with lock:
                score = _scoreboard(result_baseline, result_hive)
            live.update(Group(top, Panel(feat), Panel(score)))
            time.sleep(0.25)

    tb.join(timeout=1.0)
    th.join(timeout=1.0)
    print("Done. Wrote HEALTH_REPORT.md and HEALTH_REPORT_BASELINE.md")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Run Repo Pulse side-by-side demo")
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8080")
    p.add_argument("--model", type=str, default="hiveclaw-swarm-8b")
    p.add_argument("--max-files", type=int, default=60)
    p.add_argument("--feature-slot", type=int, default=0)
    args = p.parse_args()
    return run_demo(
        base_url=args.base_url,
        model=args.model,
        max_files=max(1, int(args.max_files)),
        slot=int(args.feature_slot),
    )


if __name__ == "__main__":
    raise SystemExit(main())
