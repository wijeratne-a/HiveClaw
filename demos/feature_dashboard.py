#!/usr/bin/env python3
"""
SAE feature proxy dashboard for Repo Pulse demo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from rich.table import Table


@dataclass
class FeatureRow:
    dim: int
    label: str
    score: float
    confidence: float


class FeatureDictionary:
    def __init__(self, payload: dict[str, dict]) -> None:
        self.payload = payload

    @classmethod
    def load(cls, path: Path) -> "FeatureDictionary":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data if isinstance(data, dict) else {})

    def label_for(self, dim: int) -> tuple[str, float, float, float]:
        item = self.payload.get(str(dim), {})
        label = str(item.get("label", f"Feature_{dim}"))
        conf = float(item.get("confidence", 0.30))
        cal = item.get("calibration", {}) if isinstance(item.get("calibration", {}), dict) else {}
        mean = float(cal.get("mean", 0.0))
        std = float(cal.get("std", 1.0))
        if std <= 1e-9:
            std = 1.0
        return label, conf, mean, std


class FeatureDashboard:
    def __init__(self, dictionary_path: Path) -> None:
        self.dictionary = FeatureDictionary.load(dictionary_path)
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            import hiveclaw_python as hc

            self._client = hc.SlabClient()
        return self._client

    def snapshot(self, slot_index: int, top_k: int = 8) -> list[FeatureRow]:
        c = self._client_lazy()
        latent = c.read_slot_v5(int(slot_index))
        mx.eval(latent)
        vec = np.array(latent.astype(mx.float32), dtype=np.float32).reshape(-1)
        if vec.size == 0:
            return []
        idx = np.argsort(np.abs(vec))[-max(1, int(top_k)) :][::-1]
        rows: list[FeatureRow] = []
        for i in idx.tolist():
            label, conf, mean, std = self.dictionary.label_for(int(i))
            z = (float(vec[i]) - mean) / std
            rows.append(
                FeatureRow(
                    dim=int(i),
                    label=label,
                    score=z,
                    confidence=conf,
                )
            )
        return rows

    @staticmethod
    def render_table(rows: list[FeatureRow]) -> Table:
        t = Table(title="Feature Dashboard (SAE proxy activations)")
        t.add_column("Feature")
        t.add_column("Dim", justify="right")
        t.add_column("Score", justify="right")
        t.add_column("Conf", justify="right")
        if not rows:
            t.add_row("no active slab signal", "-", "-", "-")
            return t
        for r in rows:
            # Bound score for cleaner display in TUI.
            s = max(-4.0, min(4.0, float(r.score)))
            t.add_row(f"{r.label} (proxy)", str(r.dim), f"{s:.2f}", f"{r.confidence:.2f}")
        return t
