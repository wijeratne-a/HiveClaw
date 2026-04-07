#!/usr/bin/env python3
"""
Side-by-side terminal demo: JSON coordination tax vs HiveClaw latent slab (timing).

Run (mock timings, no daemon — good for screen recording)::

    python examples/hiveclaw_top.py --mock-only

Run (right pane uses real SlabClient write/read; left pane still simulates JSON)::

    python examples/hiveclaw_top.py

Requires: macOS Apple Silicon, ``make python``, ``pheromoned`` loaded (``make daemon-load``),
and ``pip install -r requirements/requirements-server.txt`` (for ``rich``).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError as e:
    print("Install rich: pip install -r requirements/requirements-server.txt", file=sys.stderr)
    raise SystemExit(1) from e


def _bar(frac: float, width: int = 22) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def _json_round_trip(transcript_chars: int) -> tuple[float, int]:
    """Simulate growing committee JSON: serialize + deserialize blob."""
    blob = "x" * transcript_chars
    t0 = time.perf_counter()
    s = json.dumps(
        {
            "round": 1,
            "agents": [{"id": i, "content": blob[:2000]} for i in range(5)],
            "history": blob,
        }
    )
    _ = json.loads(s)
    dt = time.perf_counter() - t0
    approx_tokens = transcript_chars // 4 + 800 * 5
    return dt, approx_tokens


def _slab_round_trip() -> float:
    """Real latent write/read via SlabClient (requires daemon)."""
    import mlx.core as mx

    from hiveclaw_python import Swarm

    swarm = Swarm()
    slot = swarm.try_claim_any()
    if slot < 0:
        raise RuntimeError("no slab slot available (daemon loaded? all busy?)")
    try:
        d = swarm.latent_dim()
        latent = mx.ones((1, 1, d), dtype=mx.bfloat16)
        t0 = time.perf_counter()
        w = swarm.client.write_slot_v5(slot, latent)
        mx.eval(w)
        r = swarm.client.read_slot_v5(slot)
        mx.eval(r)
        return time.perf_counter() - t0
    finally:
        swarm.release(slot)


def _mock_json_step(round_idx: int) -> tuple[float, int]:
    """Synthetic JSON path latencies inspired by committee token-tax harness."""
    base = 0.55 + round_idx * 0.18 + random.uniform(-0.04, 0.08)
    tokens = 2800 + round_idx * 950 + random.randint(-200, 400)
    return max(0.08, base), tokens


def _mock_slab_step(round_idx: int) -> float:
    base = 0.014 + round_idx * 0.006 + random.uniform(-0.003, 0.004)
    return max(0.004, base)


def _run_demo(*, rounds: int, mock_only: bool) -> int:
    console = Console()
    json_total = 0.0
    slab_total = 0.0
    transcript = 4000
    tokens_eliminated = 0

    def render(
        round_idx: int,
        json_step: float,
        slab_step: float,
        jtok: int,
        jlabel: str,
        slab_label: str,
    ) -> Group:
        nonlocal json_total, slab_total, tokens_eliminated
        json_total += json_step
        slab_total += slab_step
        tokens_eliminated += jtok

        speedup = json_total / max(slab_total, 1e-9)
        j_frac = min(1.0, json_step / 2.0)
        s_frac = min(1.0, slab_step / 0.25)

        left = Text.assemble(
            (f"Round {round_idx + 1} / {rounds}\n", "bold"),
            (f"{jlabel}\n", "cyan"),
            (f"{json_step:.3f} s  [{_bar(j_frac)}]\n", "yellow"),
            (f"Cumulative: {json_total:.3f} s", "dim"),
        )
        right = Text.assemble(
            (f"Round {round_idx + 1} / {rounds}\n", "bold"),
            (f"{slab_label}\n", "green"),
            (f"{slab_step:.3f} s  [{_bar(s_frac)}]\n", "green"),
            (f"Cumulative: {slab_total:.3f} s", "dim"),
        )
        tbl = Table.grid(expand=True)
        tbl.add_column(ratio=1)
        tbl.add_column(ratio=1)
        tbl.add_row(
            Panel(left, title="JSON coordination", border_style="yellow"),
            Panel(right, title="HiveClaw latent slab", border_style="green"),
        )
        footer = Text.assemble(
            ("SPEEDUP: ", "bold"),
            (f"{speedup:.1f}x  ", "bold magenta"),
            ("|  Coord-style work not billed as chat tokens on slab path  |  ", "dim"),
            ("Data stays on device", "bold white"),
        )
        return Group(tbl, Text(""), footer)

    if mock_only:
        with Live(console=console, refresh_per_second=8) as live:
            for r in range(rounds):
                js, jt = _mock_json_step(r)
                ss = _mock_slab_step(r)
                live.update(
                    render(
                        r,
                        js,
                        ss,
                        jt,
                        f"Serialize committee JSON (~{jt:,} tok-equiv)",
                        "Write/read 256-D bf16 latent (slab)",
                    )
                )
                time.sleep(0.35)
        console.print(
            "\n[dim]Sample benchmark (5 agents × 10 rounds): LangChain ~175.6 s / ~38k coord tokens "
            "vs HiveClaw ~45.8 s / 0 — see README and benchmarks/benchmark_external.py[/dim]"
        )
        return 0

    # Live slab + simulated JSON committee
    try:
        with Live(console=console, refresh_per_second=6) as live:
            for r in range(rounds):
                json_step, jtok = _json_round_trip(transcript)
                transcript = min(transcript + 3200, 120_000)

                try:
                    slab_step = _slab_round_trip()
                    slab_label = "Write/read 256-D bf16 latent (slab)"
                except Exception as e:
                    slab_step = 1e-6
                    slab_label = f"Slab error: {e}"
                live.update(
                    render(
                        r,
                        json_step,
                        slab_step,
                        jtok,
                        "json.dumps + loads (growing committee payload)",
                        slab_label,
                    )
                )
                time.sleep(0.12)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        console.print(
            "[dim]Tip: run with --mock-only for visuals without daemon, or "
            "`make daemon-load` then retry.[/dim]"
        )
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="HiveClaw vs JSON coordination — live terminal demo")
    p.add_argument("--rounds", type=int, default=5, help="Number of timed rounds")
    p.add_argument(
        "--mock-only",
        action="store_true",
        help="Synthetic latencies only (no SlabClient; for GIF / CI)",
    )
    args = p.parse_args()
    if args.rounds < 1:
        print("--rounds must be >= 1", file=sys.stderr)
        return 2

    if not args.mock_only:
        try:
            import hiveclaw_python  # noqa: F401 — platform guard
        except NotImplementedError as e:
            print(e, file=sys.stderr)
            return 2
        except ImportError as e:
            print(e, file=sys.stderr)
            return 2

    return _run_demo(rounds=args.rounds, mock_only=args.mock_only)


if __name__ == "__main__":
    sys.exit(main())
