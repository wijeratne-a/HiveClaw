"""Single-line JSON stderr telemetry for HiveClaw integration runs."""

from __future__ import annotations

import json
import sys
import time


def log_event(obj: dict) -> None:
    """Emit one JSON object per line (jq-friendly; never indent)."""
    out = dict(obj)
    out.setdefault("ts_ns", time.time_ns())
    sys.stderr.write(json.dumps(out, separators=(",", ":")) + "\n")
    sys.stderr.flush()
