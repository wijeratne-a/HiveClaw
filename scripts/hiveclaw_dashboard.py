#!/usr/bin/env python3
"""
Phase 5: Terminal dashboard for HiveClaw slab slot states (rich Live table).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mlx.core as mx
import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

FATAL_LINE = (
    "pheromoned is not running under launchd. From repo root: "
    "`cargo build --release -p hiveclaw-daemon` then `make daemon-load`. "
    "See scripts/README.md."
)


def _parse_telemetry_log(path: Path) -> tuple[int, int]:
    """Count poison_clamp and torn_epoch_skip events in a JSON-lines log."""
    if not path.is_file():
        return 0, 0
    poison = 0
    torn = 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0, 0
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = obj.get("event")
        if ev == "poison_clamp":
            poison += 1
        elif ev == "poison_clamp_batch":
            poison += len(obj.get("slots", []) or [])
        elif ev == "torn_epoch_skip":
            torn += 1
        elif ev == "torn_epoch_skip_batch":
            torn += len(obj.get("slots", []) or [])
    return torn, poison


def _top3_abs_preview(scent: mx.array) -> str:
    flat = np.array(scent.astype(mx.float32), dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return "—"
    idx = np.argsort(np.abs(flat))[-3:][::-1]
    parts = [f"{flat[i]:.2f}" for i in idx]
    return "[" + ", ".join(parts) + "]"


def _build_renderable(
    slab_client,
    max_slots: int,
    telemetry_path: Path | None,
    refresh_ms: int,
) -> Group:
    states = slab_client.get_slot_states()
    total = len(states)
    n = min(max_slots, total)

    torn_skip, poison_clamp = (0, 0)
    if telemetry_path is not None:
        torn_skip, poison_clamp = _parse_telemetry_log(telemetry_path)

    claimed_all = sum(1 for s in states if s["claimed"])
    free_all = total - claimed_all

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Slot", justify="right", style="dim", width=6)
    table.add_column("State", width=10)
    table.add_column("Owner", width=14)
    table.add_column("Scent (top-3 |value|)", overflow="ellipsis")

    for i in range(n):
        s = states[i]
        claimed = bool(s["claimed"])
        oid = int(s["owner_id"])
        state_txt = "CLAIMED" if claimed else "FREE"
        owner_txt = f"pid:{oid}" if claimed and oid else "—"
        scent_txt = "—"
        if claimed:
            try:
                scent = slab_client.read_slot_v5(i)
                mx.eval(scent)
                scent_txt = _top3_abs_preview(scent)
            except Exception:
                scent_txt = "(read err)"
        table.add_row(str(i), state_txt, owner_txt, scent_txt)

    title = Text()
    title.append("HiveClaw Slab Monitor", style="bold white")
    title.append("  v5  |  ", style="dim")
    title.append(f"{total} slots", style="yellow")
    title.append("  |  refresh: ", style="dim")
    title.append(f"{refresh_ms} ms", style="green")

    sep = "─" * 72
    footer_lines = [
        sep,
        f"  Claimed: {claimed_all}  Free: {free_all}  (rows: first {n} of {total})",
        f"  Telemetry:  torn_epoch_skip: {torn_skip}  poison_clamp: {poison_clamp}"
        + ("  (from --telemetry-log)" if telemetry_path else "  (use --telemetry-log to parse JSON lines)"),
    ]
    footer = Text("\n".join(footer_lines), style="dim")

    return Group(title, Text(sep, style="dim"), table, footer)


def main() -> None:
    p = argparse.ArgumentParser(description="HiveClaw slab TUI dashboard (Phase 5)")
    p.add_argument(
        "--refresh-ms",
        type=int,
        default=1000,
        help="UI refresh interval in milliseconds (default 1000)",
    )
    p.add_argument(
        "--max-slots",
        type=int,
        default=32,
        help="Show only the first N slots (default 32)",
    )
    p.add_argument(
        "--telemetry-log",
        type=Path,
        default=None,
        help="Optional path to a JSON-lines log (poison_clamp, torn_epoch_skip) to summarize",
    )
    args = p.parse_args()
    refresh_ms = max(50, int(args.refresh_ms))
    max_slots = max(1, int(args.max_slots))

    try:
        import hiveclaw_python

        slab_client = hiveclaw_python.SlabClient()
    except Exception:
        print(FATAL_LINE, file=sys.stderr)
        sys.exit(1)

    console = Console()
    telem_path: Path | None = args.telemetry_log

    with Live(
        console=console,
        auto_refresh=False,
        transient=False,
    ) as live:
        try:
            while True:
                live.update(
                    _build_renderable(
                        slab_client,
                        max_slots,
                        telem_path,
                        refresh_ms,
                    )
                )
                time.sleep(refresh_ms / 1000.0)
        except KeyboardInterrupt:
            console.print("\n[dashboard] stopped.", style="dim")


if __name__ == "__main__":
    main()
