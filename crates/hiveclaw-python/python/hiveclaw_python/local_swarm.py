"""High-level local orchestration: optional daemon bootstrap + ``hiveclaw_server`` subprocess + SSE clients."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .catenar_tracing import make_tracer
from .init import init, resolve_manager_repo_root
from .manager import HiveClawManager


@dataclass
class AgentConfig:
    slot: int
    goal: str
    max_tokens: int = 256
    alpha: float = 0.1
    temperature: float = 0.8


class LocalSwarm:
    """
    Spawn ``hiveclaw_server`` (continuous batch + SSE) and run each registered agent
    as one ``/v1/chat/completions`` stream.

    **Coordination model:** multi-agent state is exchanged through **256-D bf16 latents**
    in the IOSurface slab (VRAM-backed); HTTP/SSE only carries user prompts and streamed
    text — it is not zero-copy for every chat byte. Slot indices pick slab rows for
    stigmergy; tune ``HIVECLAW_*`` env vars (see ``scripts/hiveclaw_server.py``) for
    advanced batching and stigmergy behavior.

    Optional **Catenar** Proof-of-Task: set ``catenar_enabled=True`` and install the SDK
    (see ``scripts/requirements-catenar.txt``). Each agent SSE turn emits one trace entry;
    the verifier at ``catenar_url`` must be reachable for signed receipts (local WAL still
    records traces if the network fails).

    Example::

        import hiveclaw_python as hc
        swarm = hc.LocalSwarm()
        swarm.add_agent(slot=1, goal="Optimize this kernel for Apple Silicon")
        swarm.add_agent(slot=2, goal="Check this kernel for correctness")
        swarm.run()
    """

    def __init__(
        self,
        model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit",
        *,
        sae_path: Path | str | None = None,
        alpha: float = 0.1,
        port: int = 8080,
        repo_root: Path | str | None = None,
        daemon_auto_start: bool = True,
        python_exe: str | None = None,
        build_if_missing: bool = False,
        extra_env: dict[str, str] | None = None,
        catenar_enabled: bool = False,
        catenar_url: str = "http://127.0.0.1:3000",
        catenar_agent_id: Optional[str] = None,
    ) -> None:
        self.model = model
        self.sae_path = sae_path
        self.default_alpha = float(alpha)
        self.port = int(port)
        self.repo_root = (
            Path(repo_root).resolve() if repo_root else resolve_manager_repo_root(None)
        )
        self.daemon_auto_start = bool(daemon_auto_start)
        self.python_exe = python_exe or sys.executable
        self.build_if_missing = bool(build_if_missing)
        self.extra_env = dict(extra_env) if extra_env else {}
        self.catenar_enabled = bool(catenar_enabled)
        self.catenar_url = str(catenar_url)
        self.catenar_agent_id = catenar_agent_id
        self._agents: list[AgentConfig] = []
        self._manager: HiveClawManager | None = None
        self._server_proc: subprocess.Popen | None = None
        self._catenar_tracer: Any = None

    def add_agent(
        self,
        slot: int,
        goal: str,
        *,
        max_tokens: int = 256,
        alpha: float | None = None,
        temperature: float = 0.8,
    ) -> None:
        self._agents.append(
            AgentConfig(
                slot=int(slot),
                goal=str(goal),
                max_tokens=int(max_tokens),
                alpha=float(alpha if alpha is not None else self.default_alpha),
                temperature=float(temperature),
            )
        )

    def _ensure_stack(self) -> None:
        if self.daemon_auto_start:
            self._manager = init(
                self.repo_root,
                python_exe=self.python_exe,
                build_if_missing=self.build_if_missing,
                bootstrap_daemon=True,
            )
        else:
            self._manager = HiveClawManager(self.repo_root, python_exe=self.python_exe)

        env: dict[str, str] = dict(self.extra_env)
        env["HIVECLAW_MODEL_ID"] = self.model
        if self.sae_path is not None:
            env["HIVECLAW_SAE_PATH"] = str(Path(self.sae_path).resolve())

        self._server_proc = self._manager.spawn_server(
            port=self.port,
            stigmergy=True,
            continuous_batch=True,
            env=env,
            timeout_s=300.0,
        )

        if self._catenar_tracer is None:
            self._catenar_tracer = make_tracer(
                self.catenar_enabled,
                base_url=self.catenar_url,
                agent_id=self.catenar_agent_id,
            )

    def run(
        self,
        *,
        stream_output: bool = True,
        timeout_per_agent_s: float = 120.0,
    ) -> list[str]:
        if not self._agents:
            raise ValueError("No agents registered; call add_agent() first.")

        self._ensure_stack()
        assert self._manager is not None
        base = f"http://127.0.0.1:{self.port}"
        out_texts: list[str] = []

        try:
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "httpx required: pip install -r scripts/requirements-server.txt"
            ) from e

        default_name = "hiveclaw-llama-1b"
        tracer = self._catenar_tracer
        with httpx.Client(timeout=timeout_per_agent_s) as client:
            for ag in self._agents:
                body = {
                    "model": default_name,
                    "messages": [{"role": "user", "content": ag.goal}],
                    "stream": True,
                    "max_tokens": ag.max_tokens,
                    "temperature": ag.temperature,
                    "alpha": ag.alpha,
                }
                pieces: list[str] = []
                t0 = time.perf_counter()
                with client.stream(
                    "POST",
                    f"{base}/v1/chat/completions",
                    json=body,
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if line == "data: [DONE]":
                            break
                        if line.startswith("data: "):
                            raw = line[6:].strip()
                            try:
                                obj = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            ch = (obj.get("choices") or [{}])[0]
                            delta = ch.get("delta") or {}
                            c = delta.get("content")
                            if c:
                                pieces.append(str(c))
                                if stream_output:
                                    print(c, end="", flush=True)
                if stream_output:
                    print()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                text = "".join(pieces)
                tracer.append_trace_entry(
                    {
                        "action": "agent_turn",
                        "target": f"hiveclaw://slot/{ag.slot}",
                        "amount": None,
                        "table": None,
                        "details": {
                            "slot": ag.slot,
                            "goal": ag.goal[:200],
                            "tokens_streamed": len(pieces),
                            "latency_ms": round(latency_ms, 3),
                            "model": self.model,
                        },
                        "reasoning_summary": ag.goal[:80],
                        "model_id": self.model,
                        "instruction_hash": None,
                        "parent_task_id": None,
                    }
                )
                out_texts.append(text)

        tracer.wait_for_results(len(self._agents), timeout_s=30.0)
        return out_texts

    def stop(self) -> None:
        if self._catenar_tracer is not None:
            try:
                self._catenar_tracer.close()
            except Exception:
                pass
            self._catenar_tracer = None
        if self._server_proc is not None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None

    def __enter__(self) -> LocalSwarm:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

