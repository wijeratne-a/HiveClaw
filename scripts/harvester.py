#!/usr/bin/env python3
"""Shim: run ``python training/harvester.py`` from repo root."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "training" / "harvester.py"),
        run_name="__main__",
    )
