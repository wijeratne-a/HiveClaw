#!/usr/bin/env python3
"""
Consensus Showdown: LangChain string committee vs HiveClaw latent committee.

Runs benchmarks/hiveclaw_consensus and benchmarks/langchain_string_swarm sequentially,
parses their stderr JSON events for a live Rich TUI, then prints a final scoreboard.

Usage (from repo root):
  python -m demos.consensus_showdown --rounds 10 --agents 5
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# benchmarks/ on sys.path for langchain_string_swarm, hiveclaw_consensus, string_swarm_baseline
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARKS = _REPO_ROOT / "benchmarks"
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))


class _JsonLineStderr:
    """Minimal file-like object: forward complete lines that parse as JSON to a queue."""

    def __init__(self, event_queue: queue.Queue) -> None:
        self._q = event_queue
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and line.endswith("}"):
                try:
                    self._q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return len(s)

    def flush(self) -> None:
        pass


def _drain_queue(q: queue.Queue, max_items: int = 500) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _ in range(max_items):
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


def _run_phase_in_thread(
    fn: Any,
    kwargs: dict[str, Any],
    result_box: list,
    error_box: list,
) -> None:
    try:
        result_box.append(fn(**kwargs))
    except Exception as e:  # pragma: no cover - surfaced in UI
        error_box.append(str(e))


def _live_panel(
    phase: str,
    phase_label: str,
    n_rounds: int,
    n_agents: int,
    state: dict[str, Any],
) -> Panel:
    r = int(state.get("round") or 0)
    a = int(state.get("agent") or 0)
    ctx = int(state.get("ctx_tokens") or 0)
    coord = int(state.get("coord_accum") or 0)
    role = str(state.get("role") or "-")
    t = Table.grid(padding=(0, 1))
    t.add_column()
    t.add_column()
    t.add_row("Phase", phase_label)
    t.add_row("Progress", f"Round {r}/{n_rounds} | Agent {a + 1}/{n_agents}")
    t.add_row("Role", role[:48] + ("…" if len(role) > 48 else ""))
    t.add_row("ctx_tokens (this prompt)", f"{ctx:,}")
    t.add_row("coord_tokens (cumulative)", f"{coord:,}")
    return Panel(t, title=f"[bold]{phase}[/bold]")


def _ctx_bar_table(
    label: str,
    per_round: list[int],
    max_w: int = 40,
) -> Table:
    t = Table(title=f"{label} — per-round max ctx_tokens", show_header=True)
    t.add_column("Round", justify="right")
    t.add_column("ctx_tokens", justify="right")
    t.add_column("Bar", ratio=1)
    mx = max(per_round) if per_round else 1
    for i, v in enumerate(per_round, start=1):
        frac = v / mx if mx else 0
        bar = "█" * max(1, int(frac * max_w))
        t.add_row(str(i), f"{v:,}", bar)
    return t


def _final_scoreboard(
    lc: Any | None,
    hc: Any | None,
    n_rounds: int,
) -> Group:
    tbl = Table(title="Consensus Showdown — Final", show_header=True)
    tbl.add_column("Metric")
    tbl.add_column("LangChain", justify="right")
    tbl.add_column("HiveClaw", justify="right")

    pr: list[int] = []
    hr: list[int] = []
    lc_ok = lc is not None and getattr(lc, "ok", True)
    hc_ok = hc is not None and getattr(hc, "ok", True)

    if lc_ok and lc is not None:
        lc_wall = lc.total_wall_ms / 1000.0
        lc_coord = lc.total_coord_tokens
        lc_ct = lc.total_content_tokens
        pr = list(lc.per_round_ctx_tokens)
        g0, g1 = (pr[0], pr[-1]) if pr else (0, 0)
    else:
        lc_wall = lc_coord = lc_ct = 0.0
        g0 = g1 = 0

    if hc_ok and hc is not None:
        hc_wall = hc.total_wall_ms / 1000.0
        hc_coord = hc.total_coord_tokens
        hc_ct = hc.total_content_tokens
        hr = list(hc.per_round_ctx_tokens)
        h0, h1 = (hr[0], hr[-1]) if hr else (0, 0)
    else:
        hc_wall = hc_coord = hc_ct = 0.0
        h0 = h1 = 0

    sp = (
        (lc_wall / hc_wall)
        if (lc_ok and hc_ok and hc_wall > 1e-9)
        else 0.0
    )

    tbl.add_row(
        "wall_s",
        f"{lc_wall:.2f}" if lc_ok else ("skipped" if lc is None else "failed"),
        f"{hc_wall:.2f}" if hc_ok else ("failed" if hc is not None else "-"),
    )
    tbl.add_row("speedup (LC wall / HC wall)", "-", f"{sp:.2f}x" if sp > 0 else "-")
    tbl.add_row(
        "total_coord_tokens",
        f"{int(lc_coord):,}" if lc_ok else "-",
        f"{int(hc_coord):,}" if hc_ok else "-",
    )
    tbl.add_row(
        "total_content_tokens",
        f"{int(lc_ct):,}" if lc_ok else "-",
        f"{int(hc_ct):,}" if hc_ok else "-",
    )
    tbl.add_row(
        f"ctx growth (round 1 → {n_rounds})",
        f"{int(g0):,} → {int(g1):,}" if pr else "-",
        f"{int(h0):,} → {int(h1):,}" if hr else "-",
    )

    parts: list[Any] = [Panel(tbl)]
    if pr:
        parts.append(Panel(_ctx_bar_table("LangChain (growing context)", pr)))
    if hr:
        parts.append(Panel(_ctx_bar_table("HiveClaw (~flat context)", hr)))
    return Group(*parts)


def run_showdown(
    *,
    rounds: int,
    agents: int,
    tokens_per_turn: int,
    model_id: str,
    skip_langchain: bool,
    json_out: str | None,
) -> int:
    common = dict(
        model_id=model_id,
        n_rounds=rounds,
        n_agents=agents,
        max_tokens_per_turn=tokens_per_turn,
    )

    lc_result: list = []
    hc_result: list = []
    lc_err: list = []
    hc_err: list = []

    # Deferred imports so --help works without langchain/mlx
    if not skip_langchain:
        try:
            from langchain_string_swarm import run_langchain_baseline
        except ImportError as e:
            print(
                "LangChain not installed. pip install -r requirements/requirements-bench-langchain.txt",
                file=sys.stderr,
            )
            return 1
    else:
        run_langchain_baseline = None  # type: ignore[assignment]

    try:
        from hiveclaw_consensus import run_hiveclaw_consensus
    except ImportError as e:
        print(f"Could not import hiveclaw_consensus: {e}", file=sys.stderr)
        return 1

    def process_events(events: list[dict[str, Any]], state: dict[str, Any], is_langchain: bool) -> None:
        for ev in events:
            evn = ev.get("event", "")
            if is_langchain and evn != "langchain_baseline_round":
                continue
            if not is_langchain and evn != "hiveclaw_round":
                continue
            state["round"] = int(ev.get("round", 0))
            state["agent"] = int(ev.get("agent", 0))
            state["role"] = str(ev.get("role", ""))
            state["ctx_tokens"] = int(ev.get("ctx_tokens", 0))
            ct = int(ev.get("coord_tokens", 0))
            if is_langchain:
                state["coord_accum"] = int(state.get("coord_accum", 0)) + ct
            else:
                state["coord_accum"] = 0

    lc_state: dict[str, Any] = {"round": 0, "agent": 0, "role": "", "ctx_tokens": 0, "coord_accum": 0}
    hc_state: dict[str, Any] = {"round": 0, "agent": 0, "role": "", "ctx_tokens": 0, "coord_accum": 0}

    event_q: queue.Queue = queue.Queue()
    intercept = _JsonLineStderr(event_q)
    old_stderr = sys.stderr

    with Live(refresh_per_second=12, console=None) as live:
        # Phase 1: LangChain
        if not skip_langchain and run_langchain_baseline is not None:
            sys.stderr = intercept  # type: ignore[assignment]
            th = threading.Thread(
                target=_run_phase_in_thread,
                args=(run_langchain_baseline, common, lc_result, lc_err),
                daemon=True,
            )
            th.start()
            while th.is_alive():
                process_events(_drain_queue(event_q), lc_state, is_langchain=True)
                live.update(
                    _live_panel("1/2", "LangChain string committee", rounds, agents, lc_state)
                )
                time.sleep(0.08)
            process_events(_drain_queue(event_q), lc_state, is_langchain=True)
            th.join(timeout=1.0)
            sys.stderr = old_stderr
            if lc_err:
                live.update(Panel(f"[red]LangChain phase failed: {lc_err[0]}[/red]", title="Error"))
                time.sleep(2.0)

        # Phase 2: HiveClaw
        hive_label = "1/1" if skip_langchain else "2/2"
        event_q = queue.Queue()
        intercept = _JsonLineStderr(event_q)
        sys.stderr = intercept  # type: ignore[assignment]
        th2 = threading.Thread(
            target=_run_phase_in_thread,
            args=(run_hiveclaw_consensus, common, hc_result, hc_err),
            daemon=True,
        )
        th2.start()
        while th2.is_alive():
            process_events(_drain_queue(event_q), hc_state, is_langchain=False)
            live.update(
                _live_panel(hive_label, "HiveClaw latent committee", rounds, agents, hc_state)
            )
            time.sleep(0.08)
        process_events(_drain_queue(event_q), hc_state, is_langchain=False)
        th2.join(timeout=1.0)
        sys.stderr = old_stderr

        lc = lc_result[0] if lc_result else None
        hc = hc_result[0] if hc_result else None

        if hc_err:
            live.update(
                Group(
                    _final_scoreboard(lc, None, rounds),
                    Panel(f"[red]HiveClaw phase failed: {hc_err[0]}[/red]", title="Error"),
                )
            )
        else:
            live.update(_final_scoreboard(lc, hc, rounds))

    lc = lc_result[0] if lc_result else None
    hc = hc_result[0] if hc_result else None

    summary = {
        "event": "consensus_showdown_summary",
        "langchain": lc.to_dict() if lc else None,
        "hiveclaw": hc.to_dict() if hc else None,
        "langchain_error": lc_err[0] if lc_err else None,
        "hiveclaw_error": hc_err[0] if hc_err else None,
    }
    print()
    print(json.dumps(summary, indent=2))

    if json_out:
        Path(json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {json_out}")

    if hc_err:
        return 1
    if hc and not getattr(hc, "ok", True):
        return 1
    if not skip_langchain and lc and not getattr(lc, "ok", True):
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Consensus Showdown: LangChain vs HiveClaw committee benchmark TUI")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--tokens-per-turn", type=int, default=24)
    p.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--no-langchain", action="store_true", help="Skip LangChain phase (HiveClaw only)")
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()

    return run_showdown(
        rounds=max(1, int(args.rounds)),
        agents=max(1, min(5, int(args.agents))),
        tokens_per_turn=max(1, int(args.tokens_per_turn)),
        model_id=args.model,
        skip_langchain=bool(args.no_langchain),
        json_out=args.json_out.strip() or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
