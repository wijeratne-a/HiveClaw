#!/usr/bin/env python3
"""Shim: run ``python training/train_sae.py`` from repo root."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "training" / "train_sae.py"),
        run_name="__main__",
    )
