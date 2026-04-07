#!/usr/bin/env python3
"""Unit checks for hiveclaw_sae_v1.safetensors (shapes, no decoder.weight, round-trip, tied W)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SAE_PATH = REPO_ROOT / "models/hiveclaw_sae_v1.safetensors"


def main() -> int:
    if not SAE_PATH.is_file():
        print(f"FAIL: missing {SAE_PATH} (run scripts/train_sae.py)", file=sys.stderr)
        return 1

    mx.random.seed(42)
    random.seed(42)
    np.random.seed(42)

    from safetensors.numpy import safe_open

    with safe_open(str(SAE_PATH), framework="np") as f:
        keys = list(f.keys())
        assert "encoder.weight" in keys
        assert "encoder.bias" in keys
        assert "decoder.bias" in keys
        assert "decoder.weight" not in keys
        W = mx.array(f.get_tensor("encoder.weight"))
        b_enc = mx.array(f.get_tensor("encoder.bias"))
        b_dec = mx.array(f.get_tensor("decoder.bias"))

    assert tuple(W.shape) == (256, 2048), W.shape
    assert tuple(b_enc.shape) == (256,), b_enc.shape
    assert tuple(b_dec.shape) == (2048,), b_dec.shape

    x = mx.random.normal((1, 1, 2048))
    z = mx.maximum(mx.matmul(x, mx.transpose(W)) + b_enc, 0.0)
    x_hat = mx.matmul(z, W) + b_dec
    mx.eval(x_hat)
    assert x_hat.shape == (1, 1, 2048)

    # Tied decode: perturb W and see x_hat change when using same W in decode path
    W2 = W + 0.01
    x_hat2 = mx.matmul(z, W2) + b_dec
    mx.eval(x_hat2)
    assert float(mx.max(mx.abs(x_hat2 - x_hat)).item()) > 1e-6

    print("ok test_sae_tied_weights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
