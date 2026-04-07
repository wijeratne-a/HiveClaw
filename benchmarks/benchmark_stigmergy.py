#!/usr/bin/env python3
"""
A/B benchmark: same SSE chat workload with stigmergy on (default server) vs off
(HIVECLAW_STIGMERGY=0 / --no-stigmergy). Spawns hiveclaw_server twice; prints JSON summary.

Requires: daemon + venv + models (same as scripts/verify_burn_in.sh).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_repo = Path(__file__).resolve().parent.parent


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


async def _one_sse_latency(
    client: Any,
    url: str,
    body: dict[str, Any],
) -> tuple[float, int]:
    """Returns (seconds until stream done, HTTP status)."""
    t0 = time.perf_counter()
    code = 0
    try:
        async with client.stream(
            "POST",
            url,
            json=body,
            headers={"Accept": "text/event-stream"},
            timeout=180.0,
        ) as resp:
            code = int(resp.status_code)
            if code == 200:
                async for _line in resp.aiter_lines():
                    pass
    except Exception:
        code = code or -1
    return time.perf_counter() - t0, code


async def _run_phase(
    base_url: str,
    concurrency: int,
    total_requests: int,
    prompt: str,
    model: str,
) -> dict[str, Any]:
    try:
        import httpx
    except ImportError as e:
        raise SystemExit(
            "httpx required: pip install -r requirements/requirements-server.txt"
        ) from e

    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 24,
        "temperature": 0.8,
    }
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    codes: dict[int, int] = {}

    async def one() -> None:
        async with sem:
            dt, code = await _one_sse_latency(client, url, body)
            latencies.append(dt)
            codes[code] = codes.get(code, 0) + 1

    t_wall0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[one() for _ in range(total_requests)])
    wall = time.perf_counter() - t_wall0

    latencies.sort()
    n_ok = codes.get(200, 0)
    rps = n_ok / wall if wall > 0 else 0.0
    return {
        "http_codes": {str(k): v for k, v in sorted(codes.items())},
        "n_ok": n_ok,
        "wall_seconds": wall,
        "requests_per_second": rps,
        "latency_p50_ms": _percentile(latencies, 0.50) * 1000.0,
        "latency_p95_ms": _percentile(latencies, 0.95) * 1000.0,
    }


def _wait_health(base_url: str, deadline_s: float = 300.0) -> bool:
    try:
        import httpx
    except ImportError:
        return False
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url.rstrip('/')}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _spawn_server(python_exe: str, port: int, *, stigmergy: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env["HIVECLAW_CONTINUOUS_BATCH"] = "1"
    env["HIVECLAW_COMPILE_WARMUP"] = "1"
    env["HIVECLAW_COMPILE_DECODE"] = env.get("HIVECLAW_COMPILE_DECODE", "1")
    srv = _repo / "scripts" / "hiveclaw_server.py"
    cmd = [python_exe, str(srv), "--host", "127.0.0.1", "--port", str(port)]
    if not stigmergy:
        cmd.append("--no-stigmergy")
    return subprocess.Popen(
        cmd,
        cwd=str(_repo),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Stigmergy on vs off SSE benchmark (two server runs)")
    p.add_argument("--port", type=int, default=8090, help="Base port (uses P and P+1 for two phases)")
    p.add_argument("--python", type=str, default=sys.executable, help="Python for server subprocess")
    p.add_argument("--concurrency", type=int, default=8, help="Concurrent SSE clients per phase")
    p.add_argument(
        "--requests",
        type=int,
        default=32,
        help="Total SSE completions per phase",
    )
    p.add_argument("--prompt", type=str, default="Say hello in one sentence.")
    p.add_argument("--model", type=str, default="hiveclaw-llama-1b")
    args = p.parse_args()

    phases: list[dict[str, Any]] = []

    def run_with_server(stig: bool, port: int) -> dict[str, Any]:
        base = f"http://127.0.0.1:{port}"
        proc: subprocess.Popen | None = None
        try:
            proc = _spawn_server(args.python, port, stigmergy=stig)
            if not _wait_health(base):
                return {"error": "health_timeout", "stigmergy": stig}

            import httpx

            h = httpx.get(f"{base}/health", timeout=5.0)
            health = h.json() if h.status_code == 200 else {}
            if health.get("stigmergy") is not None and bool(health["stigmergy"]) != stig:
                return {
                    "error": "health_stigmergy_mismatch",
                    "expected_stigmergy": stig,
                    "health": health,
                }

            stats = asyncio.run(
                _run_phase(
                    base,
                    args.concurrency,
                    args.requests,
                    args.prompt,
                    args.model,
                )
            )
            stats["stigmergy"] = stig
            stats["port"] = port
            return stats
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()

    phases.append(run_with_server(True, args.port))
    phases.append(run_with_server(False, args.port + 1))

    out = {
        "event": "benchmark_stigmergy_summary",
        "concurrency": args.concurrency,
        "requests_per_phase": args.requests,
        "phases": phases,
    }
    print(json.dumps(out, indent=2))

    for ph in phases:
        if "error" in ph:
            print(f"[benchmark_stigmergy] FAIL: {ph}", file=sys.stderr)
            return 1
        if ph.get("n_ok", 0) < args.requests:
            print(
                f"[benchmark_stigmergy] FAIL: expected {args.requests} OK, got {ph.get('n_ok')}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
