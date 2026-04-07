"""Optional Catenar Proof-of-Task tracing for :class:`LocalSwarm`. No-ops if disabled or SDK missing."""

from __future__ import annotations

import warnings
from typing import Any, Optional


class _NoopTracer:
    def append_trace_entry(self, entry: dict[str, Any]) -> None:
        _ = entry

    def wait_for_results(self, expected: int, timeout_s: float = 5.0) -> list[Any]:
        _ = (expected, timeout_s)
        return []

    def close(self) -> None:
        pass


class _CatenarTracer:
    """Thin wrapper so we do not rely on callers touching ``Catenar._append_trace``."""

    def __init__(self, catenar: Any) -> None:
        self._c = catenar

    def append_trace_entry(self, entry: dict[str, Any]) -> None:
        self._c._append_trace(entry)

    def wait_for_results(self, expected: int, timeout_s: float = 5.0) -> list[Any]:
        return self._c.wait_for_results(expected, timeout_s=timeout_s)

    def close(self) -> None:
        self._c.close()


def make_tracer(
    enabled: bool,
    *,
    base_url: str = "http://127.0.0.1:3000",
    agent_id: Optional[str] = None,
    domain: str = "hiveclaw",
    policy: Optional[dict[str, Any]] = None,
    public_values: Optional[dict[str, Any]] = None,
) -> _NoopTracer | _CatenarTracer:
    if not enabled:
        return _NoopTracer()
    try:
        from catenar_sdk import Catenar
    except ImportError:
        warnings.warn(
            "catenar_enabled=True but catenar_sdk is not installed. "
            "pip install per requirements/requirements-catenar.txt — tracing disabled.",
            stacklevel=2,
        )
        return _NoopTracer()

    # batch_size=1 so each agent_turn flushes quickly without waiting for 8 entries
    c = Catenar(base_url=base_url, agent_id=agent_id, batch_size=1, flush_interval_s=0.35)
    try:
        c.init(
            policy=policy if policy is not None else {"name": "hiveclaw-localswarm"},
            domain=domain,
            public_values=public_values if public_values is not None else {},
        )
    except Exception as e:
        warnings.warn(
            f"Catenar init failed ({e}); PoT tracing disabled for this session.",
            stacklevel=2,
        )
        try:
            c.close()
        except Exception:
            pass
        return _NoopTracer()
    return _CatenarTracer(c)
