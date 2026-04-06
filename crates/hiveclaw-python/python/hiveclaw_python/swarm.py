"""Slab / stigmergy helpers for multi-agent scripts.

Full LLM continuous batching stays in ``scripts/generate_batch.py``; this module only
wraps common ``SlabClient`` patterns (claim, cosine scoring, read/write v5).
"""

from __future__ import annotations

import random
from typing import Any

import mlx.core as mx
import numpy as np


class Swarm:
    """Thin facade over :class:`SlabClient` for slot discovery and claims."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import importlib

            _hp = importlib.import_module("hiveclaw_python")
            client = _hp.SlabClient()
        self.client = client

    def latent_dim(self) -> int:
        return int(self.client.get_latent_dim())

    def unclaimed_slot_indices(self) -> list[int]:
        states = self.client.get_slot_states()
        return [i for i, s in enumerate(states) if not s["claimed"]]

    @staticmethod
    def cosine_vs_goal(vec_bf16: mx.array, goal_f32_1d: np.ndarray) -> float:
        v = np.array(vec_bf16.astype(mx.float32), dtype=np.float64).reshape(-1)
        g = goal_f32_1d.astype(np.float64)
        nv = np.linalg.norm(v)
        ng = np.linalg.norm(g)
        if nv < 1e-12 or ng < 1e-12:
            return 0.0
        return float(np.dot(v, g) / (nv * ng))

    def score_unclaimed_by_cosine(
        self, unclaimed: list[int], goal_f32_1d: np.ndarray
    ) -> list[tuple[float, int]]:
        scored: list[tuple[float, int]] = []
        for slot in unclaimed:
            read_n = self.client.read_slot_v5(slot)
            mx.eval(read_n)
            scored.append((self.cosine_vs_goal(read_n, goal_f32_1d), slot))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def claim_ranked_slots(self, ordered_slots: list[int]) -> int:
        """Try ``claim_task`` in preference order; returns slot index or -1."""
        if not ordered_slots:
            return -1
        candidates = mx.array(ordered_slots, dtype=mx.int32)
        claim_res = self.client.claim_task(candidates)
        mx.eval(claim_res)
        return int(np.asarray(claim_res).reshape(-1)[0])

    def try_claim_any(self, max_candidates: int = 512) -> int:
        """Claim one of the unclaimed slots (random subset if many)."""
        unclaimed = self.unclaimed_slot_indices()
        if not unclaimed:
            return -1
        if len(unclaimed) > max_candidates:
            random.shuffle(unclaimed)
            unclaimed = unclaimed[:max_candidates]
        return self.claim_ranked_slots(unclaimed)

    def release(self, slot_index: int) -> None:
        self.client.release_task(int(slot_index))
