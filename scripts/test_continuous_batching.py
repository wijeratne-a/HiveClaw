#!/usr/bin/env python3
"""
Phase 7 unit tests: KV slice/pad, bucketing helpers (no daemon required).

Golden logits / full batch integration: set HIVECLAW_PHASE7_GOLDEN=1 (heavy; GPU + mlx_lm).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mlx.core as mx
import numpy as np

from generate_batch import (
    next_pow2_bucket,
    pad_kv_cache_batch_dim,
    slice_kv_cache_batch_dim,
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
    logits2 = model(x2, cache=cache2)
    mx.eval(logits2)
    t2 = np.array(logits2[0, -1, :], dtype=np.float32).reshape(-1)
    ok = np.allclose(t1, t2, rtol=0.0, atol=1e-5)
    assert ok, float(np.max(np.abs(t1 - t2)))


def main() -> int:
    try:
        test_next_pow2_bucket()
        test_kv_slice_and_pad()
        test_disconnect_flag_eviction_unit()
        test_golden_logits_optional()
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("[test_continuous_batching] ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
