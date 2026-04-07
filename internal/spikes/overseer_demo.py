#!/usr/bin/env python3
"""
Hero demo: Overseer detects frozen slab geometry (low time-variance on v5 scent),
calls inhibit(), agents observe reroute signals and switch synthetic goals.

Runs in one process: two agent threads (fixed slots) + one overseer thread.
No LLM — normalized bf16 latent goals only. Requires pheromoned + hiveclaw_python.

Usage:
  python internal/spikes/overseer_demo.py [--ticks 80] [--tick-ms 200] [--var-threshold 1e-5] [--slots 0,1]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
from typing import Any

import mlx.core as mx
import numpy as np

FATAL_LINE = (
    "pheromoned is not running under launchd. From repo root: "
    "`cargo build --release -p hiveclaw-daemon` then `make daemon-load`. "
    "See scripts/README.md."
)

_K = 5

_T0 = time.perf_counter()


def _log(msg: str) -> None:
    t = time.perf_counter() - _T0
    print(f"[t={t:8.2f}s] {msg}", flush=True)


def _make_goal(rng: np.random.Generator, d: int) -> mx.array:
    g = rng.standard_normal(d, dtype=np.float32)
    g /= float(np.linalg.norm(g) + 1e-7)
    # v5 write path expects [1,1,latent_dim] bf16 (same element count as 1-D).
    return mx.array(g, dtype=mx.bfloat16).reshape(1, 1, d)


def _agent_worker(
    *,
    name: str,
    slot: int,
    client: Any,
    d: int,
    rng: np.random.Generator,
    stop_evt: threading.Event,
    reroute_evt: threading.Event,
    stats: dict[str, int],
    stats_lock: threading.Lock,
    write_interval_s: float,
) -> None:
    goal_mx = _make_goal(rng, d)
    goal_idx = 0
    while not stop_evt.is_set():
        cands = mx.array([slot], dtype=mx.int32)
        res = client.claim_task(cands)
        mx.eval(res)
        got = int(np.asarray(res).reshape(-1)[0])
        if got != slot:
            time.sleep(0.01)
            continue

        _log(f"{name:8} CLAIM    slot={slot} goal_id={goal_idx}")
        try:
            while not stop_evt.is_set():
                write_res = client.write_scent(slot, goal_mx)
                mx.eval(write_res)
                time.sleep(write_interval_s)
                if reroute_evt.is_set():
                    reroute_evt.clear()
                    goal_idx += 1
                    goal_mx = _make_goal(rng, d)
                    with stats_lock:
                        stats["reroutes"] = stats.get("reroutes", 0) + 1
                    _log(f"{name:8} REROUTE  slot={slot} new_goal_id={goal_idx}")
                    break
        finally:
            try:
                client.release_task(slot)
            except Exception:
                pass
            _log(f"{name:8} RELEASE  slot={slot}")


def _overseer_worker(
    *,
    client: Any,
    d: int,
    slot_filter: set[int],
    tick_ms: int,
    var_threshold: float,
    max_ticks: int,
    stop_evt: threading.Event,
    reroute_events: dict[int, threading.Event],
    stats: dict[str, int],
    stats_lock: threading.Lock,
) -> None:
    history: dict[int, collections.deque[np.ndarray]] = {
        s: collections.deque(maxlen=_K) for s in slot_filter
    }
    ticks = 0
    while not stop_evt.is_set() and ticks < max_ticks:
        time.sleep(tick_ms / 1000.0)
        ticks += 1
        states = client.get_slot_states()
        for slot_idx in sorted(slot_filter):
            if slot_idx >= len(states):
                continue
            st = states[slot_idx]
            if not st["claimed"]:
                history[slot_idx].clear()
                continue

            scent_bf16 = client.read_slot_v5(slot_idx)
            scent_f32 = scent_bf16.astype(mx.float32)
            mx.eval(scent_f32)
            sample = np.array(scent_f32, dtype=np.float32).reshape(-1)
            history[slot_idx].append(sample)

            if len(history[slot_idx]) < _K:
                continue

            stack = np.stack(list(history[slot_idx]), axis=0)
            mean_var = float(np.mean(np.var(stack, axis=0)))

            if mean_var < var_threshold:
                owner_id = int(st["owner_id"])
                inhibit_res = client.inhibit(slot_idx, owner_id)
                mx.eval(inhibit_res)
                with stats_lock:
                    stats["inhibits"] = stats.get("inhibits", 0) + 1
                _log(
                    f"Overseer INHIBIT slot={slot_idx} owner_id={owner_id} "
                    f"mean_var={mean_var:.3e} tick={ticks}"
                )
                reroute_events[slot_idx].set()
                history[slot_idx].clear()


def main() -> int:
    p = argparse.ArgumentParser(description="Overseer + agent reroute demo (slab stigmergy)")
    p.add_argument("--ticks", type=int, default=80, help="Overseer loop iterations (default 80)")
    p.add_argument("--tick-ms", type=int, default=200, help="Overseer sleep per tick ms (default 200)")
    p.add_argument(
        "--var-threshold",
        type=float,
        default=1e-5,
        help="Inhibit if mean per-dim time variance below this (default 1e-5)",
    )
    p.add_argument(
        "--slots",
        type=str,
        default="0,1",
        help="Comma-separated slot indices to monitor (default 0,1)",
    )
    p.add_argument(
        "--write-interval-ms",
        type=int,
        default=40,
        help="Agent write period while holding claim (default 40)",
    )
    args = p.parse_args()

    try:
        slot_list = [int(x.strip()) for x in args.slots.split(",") if x.strip()]
    except ValueError:
        print("--slots must be comma-separated integers", file=sys.stderr)
        return 2
    if len(slot_list) < 1:
        print("need at least one slot", file=sys.stderr)
        return 2

    global _T0
    _T0 = time.perf_counter()

    slot_filter = set(slot_list)
    tick_ms = max(1, int(args.tick_ms))
    max_ticks = max(1, int(args.ticks))
    write_interval_s = max(0.005, int(args.write_interval_ms) / 1000.0)

    try:
        import hiveclaw_python

        client = hiveclaw_python.SlabClient()
    except Exception:
        print(FATAL_LINE, file=sys.stderr)
        return 1

    d = client.get_latent_dim()
    stop_evt = threading.Event()
    stats: dict[str, int] = {}
    stats_lock = threading.Lock()
    reroute_events = {s: threading.Event() for s in slot_list}

    seed = int(time.time_ns() & 0xFFFFFFFFFFFFFFFF)
    rngs = {s: np.random.default_rng(seed ^ (s * 0x9E3779B97F4A7C15)) for s in slot_list}

    threads: list[threading.Thread] = []
    for s in slot_list:
        name = f"Agent{s}"
        t = threading.Thread(
            target=_agent_worker,
            kwargs=dict(
                name=name,
                slot=s,
                client=client,
                d=d,
                rng=rngs[s],
                stop_evt=stop_evt,
                reroute_evt=reroute_events[s],
                stats=stats,
                stats_lock=stats_lock,
                write_interval_s=write_interval_s,
            ),
            daemon=True,
            name=name,
        )
        threads.append(t)

    overseer_t = threading.Thread(
        target=_overseer_worker,
        kwargs=dict(
            client=client,
            d=d,
            slot_filter=slot_filter,
            tick_ms=tick_ms,
            var_threshold=float(args.var_threshold),
            max_ticks=max_ticks,
            stop_evt=stop_evt,
            reroute_events=reroute_events,
            stats=stats,
            stats_lock=stats_lock,
        ),
        daemon=True,
        name="Overseer",
    )

    _log(
        f"DEMO start latent_dim={d} slots={sorted(slot_filter)} "
        f"overseer_ticks={max_ticks} tick_ms={tick_ms} var_threshold={args.var_threshold}"
    )
    for t in threads:
        t.start()
    overseer_t.start()
    overseer_t.join()
    stop_evt.set()
    for t in threads:
        t.join(timeout=2.0)

    with stats_lock:
        inh = stats.get("inhibits", 0)
        rer = stats.get("reroutes", 0)
    _log(f"SUMMARY  inhibits={inh}  agent_reroutes={rer}")
    print(
        json.dumps({"event": "overseer_demo_summary", "inhibits": inh, "reroutes": rer}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
