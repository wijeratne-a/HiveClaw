#!/usr/bin/env python3
"""
Ironclad Engine burn-in: concurrent SSE chat-completions + acceptance checks.

Criteria (from Phase 1 spec):
  1. Some HTTP 503 responses when concurrency exceeds server queue capacity (no crash).
  2. Zero ``eager_fallback`` JSON events on server stderr (requires --spawn-server or --stderr-file).
  3. vm_stat Swapins delta below threshold (macOS).

Examples::

  # External server (stderr not captured — criterion 2 skipped with a warning):
  HIVECLAW_CONTINUOUS_BATCH=1 HIVECLAW_COMPILE_WARMUP=1 python scripts/hiveclaw_server.py --port 8080
  python scripts/burn_in.py --base-url http://127.0.0.1:8080 --concurrency 55

  # Spawned server (all criteria; needs daemon + venv + models):
  python scripts/burn_in.py --spawn-server --concurrency 55
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_scripts = Path(__file__).resolve().parent
_repo = _scripts.parent


def _parse_swapins(vm_stat_text: str) -> int | None:
    for line in vm_stat_text.splitlines():
        m = re.search(r"Swapins:\s*([\d,]+)\.", line, re.I)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def _run_vm_stat() -> int | None:
    try:
        r = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _parse_swapins(r.stdout or "")


def _stderr_telemetry_worker(
    stream: Any,
    counts: dict[str, int],
    stop: threading.Event,
) -> None:
    try:
        for raw in iter(stream.readline, b""):
            if stop.is_set():
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = obj.get("event")
            if isinstance(ev, str):
                counts[ev] = counts.get(ev, 0) + 1
    finally:
        try:
            stream.close()
        except Exception:
            pass


async def _one_sse_client(
    client: Any,
    url: str,
    body: dict[str, Any],
    counters: defaultdict[str, int],
) -> None:
    try:
        async with client.stream(
            "POST",
            url,
            json=body,
            headers={"Accept": "text/event-stream"},
            timeout=120.0,
        ) as resp:
            code = int(resp.status_code)
            counters[f"http_{code}"] += 1
            if code == 200:
                async for _line in resp.aiter_lines():
                    pass
    except Exception:
        counters["client_exceptions"] += 1


async def _strike(
    base_url: str,
    concurrency: int,
    prompt: str,
    model: str,
) -> defaultdict[str, int]:
    try:
        import httpx
    except ImportError as e:
        raise SystemExit(
            "httpx required: pip install -r scripts/requirements-server.txt"
        ) from e

    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 24,
        "temperature": 0.8,
    }
    counters: defaultdict[str, int] = defaultdict(int)
    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(_one_sse_client(client, url, body, counters))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*tasks)
    return counters


def main() -> int:
    p = argparse.ArgumentParser(description="HiveClaw Ironclad burn-in (SSE load + checks)")
    p.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Server root URL (e.g. http://127.0.0.1:8080). Ignored if --spawn-server.",
    )
    p.add_argument(
        "--spawn-server",
        action="store_true",
        help="Start hiveclaw_server.py as subprocess and parse its stderr for telemetry.",
    )
    p.add_argument("--port", type=int, default=0, help="Port when using --spawn-server (0 = auto)")
    p.add_argument("--python", type=str, default=sys.executable, help="Python for --spawn-server")
    p.add_argument(
        "--concurrency",
        type=int,
        default=55,
        help="Concurrent SSE clients (use > max queue depth, default 50, to force 503s)",
    )
    p.add_argument("--prompt", type=str, default="Say hello in one sentence.")
    p.add_argument(
        "--model",
        type=str,
        default="hiveclaw-llama-1b",
        help="Chat model name (server default)",
    )
    p.add_argument(
        "--swapin-delta-max",
        type=int,
        default=100,
        help="Max allowed vm_stat Swapins increase during strike",
    )
    p.add_argument(
        "--stderr-file",
        type=str,
        default="",
        help="If set, tail-parse this file for eager_fallback instead of subprocess stderr",
    )
    args = p.parse_args()

    proc: subprocess.Popen | None = None
    telem_counts: dict[str, int] = {}
    stop_telem = threading.Event()
    telem_thread: threading.Thread | None = None

    port = args.port
    if args.spawn_server:
        if port <= 0:
            port = 8765 + (int(time.time()) % 500)
        base_url = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        env["HIVECLAW_CONTINUOUS_BATCH"] = "1"
        env["HIVECLAW_COMPILE_WARMUP"] = "1"
        srv_py = _repo / "scripts" / "hiveclaw_server.py"
        cmd = [
            args.python,
            str(srv_py),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(_repo),
            env=env,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )
        assert proc.stderr is not None
        telem_counts = {}
        telem_thread = threading.Thread(
            target=_stderr_telemetry_worker,
            args=(proc.stderr, telem_counts, stop_telem),
            daemon=True,
        )
        telem_thread.start()

        # Wait for /health
        deadline = time.time() + 300.0
        ok = False
        try:
            import httpx
        except ImportError:
            print("httpx missing; pip install -r scripts/requirements-server.txt", file=sys.stderr)
            if proc:
                proc.terminate()
            return 2
        while time.time() < deadline:
            if proc.poll() is not None:
                print("[burn_in] server process exited early", file=sys.stderr)
                return 1
            try:
                r = httpx.get(f"{base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not ok:
            print("[burn_in] timeout waiting for /health", file=sys.stderr)
            if proc:
                proc.terminate()
            return 1
        print(f"[burn_in] server up at {base_url}", flush=True)
    else:
        base_url = args.base_url.strip()
        if not base_url:
            print("Need --base-url or --spawn-server", file=sys.stderr)
            return 2

    swap0 = _run_vm_stat()
    counters = asyncio.run(_strike(base_url, args.concurrency, args.prompt, args.model))
    swap1 = _run_vm_stat()

    if proc is not None:
        stop_telem.set()
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        if telem_thread is not None:
            telem_thread.join(timeout=2.0)

    eager_fallback = 0
    if args.stderr_file:
        pth = Path(args.stderr_file)
        if pth.is_file():
            for line in pth.read_text(errors="replace").splitlines():
                if '"event"' in line and "eager_fallback" in line:
                    try:
                        obj = json.loads(line.strip())
                        if obj.get("event") == "eager_fallback":
                            eager_fallback += 1
                    except json.JSONDecodeError:
                        pass
    else:
        eager_fallback = int(telem_counts.get("eager_fallback", 0))

    n503 = counters.get("http_503", 0)
    n200 = counters.get("http_200", 0)
    swap_delta = None
    if swap0 is not None and swap1 is not None:
        swap_delta = swap1 - swap0

    print(json.dumps({"event": "burn_in_summary", "counters": dict(counters)}, indent=2))
    print(
        f"[burn_in] http_200={n200} http_503={n503} "
        f"eager_fallback={eager_fallback} swap_delta={swap_delta}"
    )

    ok1 = n503 >= 1
    ok2 = (
        eager_fallback == 0
        if (args.spawn_server or bool(args.stderr_file.strip()))
        else True
    )
    ok3 = (
        swap_delta is not None and swap_delta < args.swapin_delta_max
        if swap_delta is not None
        else True
    )

    if not args.spawn_server and not args.stderr_file.strip():
        print(
            "[burn_in] note: criterion 2 (eager_fallback==0) not checked "
            "(use --spawn-server or --stderr-file)",
            flush=True,
        )

    if swap_delta is None:
        print("[burn_in] note: vm_stat Swapins unavailable — criterion 3 skipped", flush=True)

    print(
        f"[burn_in] criterion1_503_when_overloaded: {'PASS' if ok1 else 'FAIL'} "
        f"(need >=1 503, got {n503})"
    )
    print(
        f"[burn_in] criterion2_zero_eager_fallback: {'PASS' if ok2 else 'FAIL'} "
        f"(count={eager_fallback})"
    )
    print(
        f"[burn_in] criterion3_swapins_delta: {'PASS' if ok3 else 'FAIL'} "
        f"(delta={swap_delta}, max={args.swapin_delta_max})"
    )

    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
