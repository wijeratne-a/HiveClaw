"""
HiveClaw KV cache with per-row attention masks for continuous batching.

Implements composite additive float16 masks (-1e4 = masked; MLX ``fast`` SDPA requirement) for:
- Left-padded prefill positions per real row
- Dummy batch rows (fully blinded)

See docs/adr/BATCHED_STEERING_CONTRACT.md Phase 7+.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache

# mlx fast SDPA requires mask dtype to promote to the attention output dtype (float16 for common LLMs).
_MASK_NEG = mx.array(-1e4, dtype=mx.float16)
_MASK_ZERO = mx.array(0.0, dtype=mx.float16)


class HiveClawKVCache(KVCache):
    """KVCache whose ``make_mask`` returns batched additive float16 masks.

    Attributes:
        hive_left_pad: int32 ``[B_bucket]``, count of left-pad tokens in the
            initial prefill for each row (0 for dummy rows).
        hive_row_active: float32 ``[B_bucket]``, 1.0 for real rows, 0.0 for dummies.
    """

    hive_left_pad: mx.array
    hive_row_active: mx.array

    def __init__(self) -> None:
        super().__init__()
        self.hive_left_pad = mx.array([0], dtype=mx.int32)
        self.hive_row_active = mx.array([1.0], dtype=mx.float32)

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        window_size: Optional[int] = None,
    ):
        if window_size is not None:
            return create_causal_mask(
                N, offset=self.offset, window_size=window_size
            )
        B = int(self.hive_left_pad.shape[0])
        offset = int(self.offset)

        if N > 1:
            return self._make_prefill_mask(N, B, offset)
        return self._make_decode_mask(B, offset)

    def _make_prefill_mask(self, N: int, B: int, offset: int) -> mx.array:
        """Additive float16 mask [B, 1, N, N] for prefill (offset should be 0)."""
        # Causal: query i may attend to key k iff i >= k (positions in this chunk)
        i_idx = mx.arange(N)
        k_idx = mx.arange(N)
        causal = i_idx[:, None] >= k_idx[None, :]
        causal = causal.reshape(1, 1, N, N)

        lp = self.hive_left_pad.reshape(B, 1, 1)
        k_ar = mx.arange(N, dtype=mx.int32).reshape(1, 1, N)
        key_ok = k_ar >= lp
        # [B,1,N] -> [B,1,N,N] repeating over query index i
        key_ok = mx.broadcast_to(key_ok[:, :, None, :], (B, 1, N, N))

        act = self.hive_row_active.reshape(B, 1, 1, 1)
        active_ok = act > 0.0
        allowed = mx.broadcast_to(causal, (B, 1, N, N)) & key_ok & active_ok

        return mx.where(allowed, _MASK_ZERO, _MASK_NEG)

    def _make_decode_mask(self, B: int, offset: int) -> mx.array:
        """Additive float16 mask [B, 1, 1, offset + 1] for single-token decode."""
        Sk = offset + 1
        k_ar = mx.arange(Sk, dtype=mx.int32).reshape(1, 1, 1, Sk)
        lp = self.hive_left_pad.reshape(B, 1, 1, 1)
        key_ok = k_ar >= lp

        act = self.hive_row_active.reshape(B, 1, 1, 1)
        active_ok = act > 0.0
        allowed = key_ok & active_ok

        return mx.where(allowed, _MASK_ZERO, _MASK_NEG)


def install_hiveclaw_kv_cache(
    model: object,
    cache: list,
    hive_left_pad: mx.array,
    hive_row_active: mx.array,
) -> bool:
    """Replace ``cache[fa_idx]`` with :class:`HiveClawKVCache` if supported.

    Returns False if the model uses sliding-window attention (not supported here).
    """
    inner = getattr(model, "model", model)
    if getattr(inner, "sliding_window", None) is not None:
        return False
    fa_idx = int(inner.fa_idx)
    old = cache[fa_idx]
    new_c = HiveClawKVCache()
    new_c.hive_left_pad = hive_left_pad
    new_c.hive_row_active = hive_row_active
    if getattr(old, "keys", None) is not None:
        new_c.keys = old.keys
        new_c.values = old.values
        new_c.offset = old.offset
    cache[fa_idx] = new_c
    return True


def rebuild_hive_kv_metadata(
    entries: list,
    B_bucket: int,
) -> tuple[mx.array, mx.array]:
    """Build ``hive_left_pad`` and ``hive_row_active`` from batch entries."""
    B_active = len(entries)
    max_len = max(int(e.prompt_tokens.size) for e in entries)
    left = [max_len - int(e.prompt_tokens.size) for e in entries]
    while len(left) < B_bucket:
        left.append(0)
    active = [1.0] * B_active + [0.0] * (B_bucket - B_active)
    return (
        mx.array(left, dtype=mx.int32),
        mx.array(active, dtype=mx.float32),
    )


def sync_hive_metadata_to_fa_cache(cache: list, model: object, lp: mx.array, act: mx.array) -> None:
    """Update hive arrays on the installed HiveClawKVCache (full-attention slot)."""
    inner = getattr(model, "model", model)
    fa_idx = int(inner.fa_idx)
    c = cache[fa_idx]
    if isinstance(c, HiveClawKVCache):
        c.hive_left_pad = lp
        c.hive_row_active = act
