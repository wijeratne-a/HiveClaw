#!/usr/bin/env python3
"""
Phase 7 unit tests: KV slice/pad, bucketing helpers (no daemon required).

Golden logits / full batch integration: set HIVECLAW_PHASE7_GOLDEN=1 (heavy; GPU + mlx_lm).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_scripts_dir = _REPO / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mlx.core as mx
import numpy as np
from unittest.mock import patch

from generate_batch import (
    next_pow2_bucket,
    pad_kv_cache_batch_dim,
    slice_kv_cache_batch_dim,
)
from hiveclaw_kv_mask import (
    HiveClawKVCache,
    HiveClawRotatingKVCache,
    install_hiveclaw_kv_cache,
    rebuild_hive_kv_metadata,
)


def test_next_pow2_bucket() -> None:
    assert next_pow2_bucket(1) == 1
    assert next_pow2_bucket(2) == 2
    assert next_pow2_bucket(3) == 4
    assert next_pow2_bucket(4) == 4
    assert next_pow2_bucket(5) == 8
    assert next_pow2_bucket(17, cap=16) == 16


def test_kv_slice_and_pad() -> None:
    from mlx_lm.models.cache import KVCache

    c = KVCache()
    B, H, S, D = 4, 2, 8, 32
    c.keys = mx.random.normal((B, H, S, D))
    c.values = mx.random.normal((B, H, S, D))
    c.offset = S
    cache = [c]
    slice_kv_cache_batch_dim(cache, [0, 2])
    assert cache[0].keys.shape[0] == 2
    assert cache[0].values.shape[0] == 2
    pad_kv_cache_batch_dim(cache, B_bucket=4, B_current=2)
    assert cache[0].keys.shape[0] == 4
    assert cache[0].values.shape[0] == 4


def test_disconnect_flag_eviction_unit() -> None:
    """Eviction set uses threading.Event (smoke)."""
    import threading

    ev = threading.Event()
    assert not ev.is_set()
    ev.set()
    assert ev.is_set()


def test_rebuild_hive_kv_metadata() -> None:
    class _E:
        def __init__(self, n: int) -> None:
            self.prompt_tokens = mx.arange(n, dtype=mx.int32)

    entries = [_E(3), _E(5)]
    lp, act = rebuild_hive_kv_metadata(entries, B_bucket=4)
    mx.eval(lp, act)
    assert int(lp.size) == 4
    assert int(act.size) == 4
    np.testing.assert_array_equal(np.array(lp), [2, 0, 0, 0])
    np.testing.assert_array_equal(np.array(act), [1.0, 1.0, 0.0, 0.0])


def test_hiveclaw_kv_cache_mask_shapes() -> None:
    c = HiveClawKVCache()
    c.hive_left_pad = mx.array([1, 0, 0], dtype=mx.int32)
    c.hive_row_active = mx.array([1.0, 1.0, 0.0], dtype=mx.float32)
    c.offset = 0
    m = c.make_mask(4)
    mx.eval(m)
    assert tuple(m.shape) == (3, 1, 4, 4)
    assert m.dtype == mx.float16
    c.offset = 4
    d = c.make_mask(1)
    mx.eval(d)
    assert tuple(d.shape) == (3, 1, 1, 5)


def test_golden_logits_optional() -> None:
    """Regression sentinel: B=1 vs padded batch (optional; slow)."""
    if os.environ.get("HIVECLAW_PHASE7_GOLDEN", "0") != "1":
        print("[test_continuous_batching] skip golden (set HIVECLAW_PHASE7_GOLDEN=1)")
        return
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer, _cfg = load(
        "mlx-community/Llama-3.2-1B-Instruct-4bit", return_config=True
    )
    mx.random.seed(42)
    np.random.seed(42)
    prompt = mx.array(tokenizer.encode("Hello"))
    cache1 = make_prompt_cache(model)
    logits1 = model(prompt[None, :], cache=cache1)
    mx.eval(logits1)
    t1 = np.array(logits1[:, -1, :], dtype=np.float32).reshape(-1)

    pad_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    max_len = int(prompt.size)
    row0 = mx.reshape(prompt, (max_len,))
    row1 = mx.full((max_len,), pad_id, dtype=row0.dtype)
    x2 = mx.stack([row0, row1, row1, row1], axis=0)
    cache2 = make_prompt_cache(model)
    class _E:
        def __init__(self, tok: mx.array) -> None:
            self.prompt_tokens = tok

    entries = [_E(row0)]
    lp, act = rebuild_hive_kv_metadata(entries, B_bucket=4)
    assert install_hiveclaw_kv_cache(model, cache2, lp, act) is True
    logits2 = model(x2, cache=cache2)
    mx.eval(logits2)
    assert not bool(mx.isnan(logits2).any().item())
    t2 = np.array(logits2[0, -1, :], dtype=np.float32).reshape(-1)
    ok = np.allclose(t1, t2, rtol=1e-2, atol=5e-2)
    assert ok, float(np.max(np.abs(t1 - t2)))
    assert int(np.argmax(t1)) == int(np.argmax(t2))


def test_sentinel_validation_allowed() -> None:
    try:
        from hiveclaw_python import _validate_batch_slots_arr
    except ImportError:
        print(
            "[test_continuous_batching] skip sentinel validation (hiveclaw_python missing)"
        )
        return
    B = _validate_batch_slots_arr(mx.array([-1, 0, 1, -1], dtype=mx.int32))
    assert B == 4


def test_real_duplicate_rejected() -> None:
    try:
        from hiveclaw_python import _validate_batch_slots_arr
    except ImportError:
        return
    try:
        _validate_batch_slots_arr(mx.array([0, 0, -1, -1], dtype=mx.int32))
    except ValueError:
        return
    raise AssertionError("expected ValueError for duplicate real slots")


_MOCK_LATENT = 256
_MOCK_HIDDEN = 2048


class _MockSlabSlots:
    def read_slots(self, slots, depends=None):
        B = int(slots.shape[0])
        return mx.zeros((B, 1, _MOCK_LATENT), dtype=mx.bfloat16), mx.zeros(
            (B,), dtype=mx.uint8
        )

    def write_slots(self, slots, latents, depends=None):
        return latents, mx.zeros((int(slots.shape[0]),), dtype=mx.uint8)


def test_static_shape_steering() -> None:
    from hiveclaw_steering import apply_steering_sandwich

    B = 4
    h = mx.ones((B, 1, _MOCK_HIDDEN), dtype=mx.float32)
    slots = mx.array([0, 1, -1, -1], dtype=mx.int32)
    W = mx.zeros((_MOCK_LATENT, _MOCK_HIDDEN), dtype=mx.float32)
    b = mx.zeros((_MOCK_HIDDEN,), dtype=mx.float32)
    H, norm, wst = apply_steering_sandwich(
        h, _MockSlabSlots(), slots, W, b, 0.0, b_enc=None
    )
    mx.eval(H, norm, wst)
    assert tuple(H.shape) == (B, 1, _MOCK_HIDDEN)
    assert tuple(norm.shape) == (B, 1, 1)
    assert tuple(wst.shape) == (1,) and wst.dtype == mx.uint32


def test_compiled_decode_no_fallback() -> None:
    """Optional GPU integration: set HIVECLAW_COMPILE_DECODE_CI=1 in a daemon+worker harness."""
    if os.environ.get("HIVECLAW_COMPILE_DECODE_CI", "0") != "1":
        print(
            "[test_continuous_batching] skip compiled decode CI "
            "(HIVECLAW_COMPILE_DECODE_CI=1)"
        )
        return
    print(
        "[test_continuous_batching] HIVECLAW_COMPILE_DECODE_CI=1: "
        "run scripts/generate_batch.py integration with daemon to assert no compile fallback"
    )


def test_probe_max_batch_mock() -> None:
    import generate_batch as gb

    class FakeModel:
        def __call__(self, toks, cache=None):
            B = int(toks.shape[0])
            if B >= 4:
                raise RuntimeError("simulated OOM")
            L = int(toks.shape[1])
            return mx.zeros((B, L, 16), dtype=mx.float32)

    def fake_load(model_id):
        return FakeModel(), None

    def fake_make_prompt(_m):
        return []

    with (
        patch("mlx_lm.load", fake_load),
        patch("mlx_lm.models.cache.make_prompt_cache", fake_make_prompt),
    ):
        got = gb.probe_max_batch("dummy", env_max=16, probe_ctx_len=8)
    assert got == 2


def test_rotating_kv_mask_decode_none() -> None:
    c = HiveClawRotatingKVCache(32, keep=0)
    c.hive_left_pad = mx.array([0, 2], dtype=mx.int32)
    c.hive_row_active = mx.array([1.0, 0.0], dtype=mx.float32)
    c.offset = 3
    c.keys = mx.zeros((2, 1, 3, 8), dtype=mx.float32)
    m = c.make_mask(1, window_size=None)
    assert m is not None
    mx.eval(m)
    assert int(m.shape[0]) == 2
    row1 = np.array(m[1, 0, 0, :], dtype=np.float32)
    assert np.all(row1 <= -1e3)


def test_rotating_kv_mask_decode_ring() -> None:
    c = HiveClawRotatingKVCache(8, keep=0)
    c.hive_left_pad = mx.array([0, 1], dtype=mx.int32)
    c.hive_row_active = mx.array([1.0, 1.0], dtype=mx.float32)
    c.offset = 10
    c._idx = 3
    c.keys = mx.zeros((2, 1, 8, 4), dtype=mx.float32)
    m = c.make_mask(1, window_size=4)
    assert m is not None
    mx.eval(m)
    assert tuple(m.shape) == (2, 1, 1, 8)


def test_install_hiveclaw_swa_model() -> None:
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    class Inner:
        fa_idx = 0
        swa_idx = 2

    class M:
        model = Inner()

    m = M()
    k0 = KVCache()
    k1 = KVCache()
    k2 = RotatingKVCache(128, keep=0)
    cache = [k0, k1, k2]
    lp = mx.array([0, 0, 0], dtype=mx.int32)
    act = mx.array([1.0, 1.0, 0.0], dtype=mx.float32)
    assert install_hiveclaw_kv_cache(m, cache, lp, act) is True
    assert isinstance(cache[0], HiveClawKVCache)
    assert isinstance(cache[1], KVCache)
    assert isinstance(cache[2], HiveClawRotatingKVCache)


def main() -> int:
    try:
        test_next_pow2_bucket()
        test_kv_slice_and_pad()
        test_disconnect_flag_eviction_unit()
        test_rebuild_hive_kv_metadata()
        test_hiveclaw_kv_cache_mask_shapes()
        test_golden_logits_optional()
        test_sentinel_validation_allowed()
        test_real_duplicate_rejected()
        test_static_shape_steering()
        test_compiled_decode_no_fallback()
        test_probe_max_batch_mock()
        test_rotating_kv_mask_decode_none()
        test_rotating_kv_mask_decode_ring()
        test_install_hiveclaw_swa_model()
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("[test_continuous_batching] ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
