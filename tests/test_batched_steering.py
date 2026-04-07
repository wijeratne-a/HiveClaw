#!/usr/bin/env python3
"""
Phase 6 regression: batched read_slots / steer_hidden_batched vs single-slot baseline.
Requires pheromoned + models/hiveclaw_sae_v1.safetensors.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_scripts_dir = _REPO / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import mlx.core as mx
import numpy as np

import hiveclaw_python as h
from hiveclaw_steering import (
    check_latent_dim,
    load_sae,
    steer_hidden,
    steer_hidden_batched,
)

SAE_PATH = _REPO / "models/hiveclaw_sae_v1.safetensors"

# hiveclaw_layout v6: slot header `front_epoch` u32 @ +12 from slot base; stride = 128 + 2*D.
_HCLW_GLOBAL_HDR = 4096
_HCLW_SLOT_HDR = 64
_HCLW_SLOT_FOOTER = 64
_OFF_S_FRONT_EPOCH = 12


def _slot_stride_bytes(latent_elems: int) -> int:
    return _HCLW_SLOT_HDR + int(latent_elems) * 2 + _HCLW_SLOT_FOOTER


def _slot_front_epoch_byte_off(slot: int, latent_elems: int) -> int:
    return _HCLW_GLOBAL_HDR + int(slot) * _slot_stride_bytes(latent_elems) + _OFF_S_FRONT_EPOCH


def test_parity_b1() -> None:
    mx.random.seed(42)
    np.random.seed(42)
    c = h.SlabClient()
    check_latent_dim(c)
    d = int(c.get_latent_dim())
    sae = load_sae(SAE_PATH)
    W_enc = sae["encoder.weight"]
    b_dec = sae["decoder.bias"]
    slot = 10
    r = c.claim_task(mx.array([slot], dtype=mx.int32))
    mx.eval(r)
    assert int(mx.array(r).item()) == slot
    latent = mx.ones((1, 1, d), dtype=mx.bfloat16) * 0.01
    mx.eval(c.write_slot_v5(slot, latent))
    h_step = mx.random.normal((1, 1, int(W_enc.shape[1])), dtype=mx.float32)
    mx.eval(h_step)
    alpha = 0.1
    a = steer_hidden(h_step, c, slot, W_enc, b_dec, alpha)
    bs = mx.array([slot], dtype=mx.int32)
    b = steer_hidden_batched(h_step, c, bs, W_enc, b_dec, alpha)
    mx.eval(a, b)
    da = np.array(a, dtype=np.float32).reshape(-1)
    db = np.array(b, dtype=np.float32).reshape(-1)
    assert np.allclose(da, db, rtol=0.0, atol=1e-2), float(np.max(np.abs(da - db)))
    _, st = c.read_slots(bs, depends=h_step)
    mx.eval(st)
    assert int(np.array(st, dtype=np.uint8).reshape(-1)[0]) == 0
    c.release_task(slot)
    print("[test_batched_steering] parity B=1 ok", flush=True)


def test_b2_shapes() -> None:
    c = h.SlabClient()
    check_latent_dim(c)
    d = int(c.get_latent_dim())
    slots = [20, 21]
    for s in slots:
        r = c.claim_task(mx.array([s], dtype=mx.int32))
        mx.eval(r)
        assert int(mx.array(r).item()) == s
        z = mx.ones((1, 1, d), dtype=mx.bfloat16) * (0.02 if s == 20 else 0.03)
        mx.eval(c.write_slot_v5(s, z))
    si = mx.array(slots, dtype=mx.int32)
    data, st = c.read_slots(si)
    mx.eval(data, st)
    assert list(data.shape) == [2, 1, d]
    assert list(st.shape) == [2]
    assert bool(np.all(np.array(st, dtype=np.uint8) == 0))
    latent = mx.ones((2, 1, d), dtype=mx.bfloat16) * 0.04
    out, wst = c.write_slots(si, latent)
    mx.eval(out, wst)
    assert list(wst.shape) == [2]
    assert bool(np.all(np.array(wst, dtype=np.uint8) == 0))
    for s in slots:
        c.release_task(s)
    print("[test_batched_steering] B=2 shapes ok", flush=True)


def test_torn_epoch_batched_read() -> None:
    c = h.SlabClient()
    check_latent_dim(c)
    d = int(c.get_latent_dim())
    slot = 14
    r = c.claim_task(mx.array([slot], dtype=mx.int32))
    mx.eval(r)
    assert int(mx.array(r).item()) == slot
    latent = mx.ones((1, 1, d), dtype=mx.bfloat16) * 0.05
    mx.eval(c.write_slot_v5(slot, latent))
    c.write_u32_at(_slot_front_epoch_byte_off(slot, d), 0xDEADBEEF)
    si = mx.array([slot], dtype=mx.int32)
    data, st = c.read_slots(si)
    mx.eval(data, st)
    assert int(np.array(st, dtype=np.uint8).reshape(-1)[0]) == 1
    assert bool(mx.all(data[0] == 0).item())
    c.release_task(slot)
    print("[test_batched_steering] torn epoch ok", flush=True)


def test_clamp_batch_telemetry() -> None:
    c = h.SlabClient()
    check_latent_dim(c)
    d = int(c.get_latent_dim())
    sae = load_sae(SAE_PATH)
    W_enc = sae["encoder.weight"]
    b_dec = sae["decoder.bias"]
    slot = 12
    r = c.claim_task(mx.array([slot], dtype=mx.int32))
    mx.eval(r)
    assert int(mx.array(r).item()) == slot
    latent = mx.ones((1, 1, d), dtype=mx.bfloat16) * 8.0
    mx.eval(c.write_slot_v5(slot, latent))
    h_step = mx.zeros((1, 1, int(W_enc.shape[1])), dtype=mx.float32)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        _ = steer_hidden_batched(
            h_step,
            c,
            mx.array([slot], dtype=mx.int32),
            W_enc,
            b_dec,
            64.0,
        )
        mx.eval(_)
    err = buf.getvalue()
    assert "poison_clamp_batch" in err, repr(err[:800])
    c.release_task(slot)
    print("[test_batched_steering] clamp batch telemetry ok", flush=True)


def test_duplicate_slots_rejected() -> None:
    c = h.SlabClient()
    si = mx.array([5, 5], dtype=mx.int32)
    try:
        c.read_slots(si)
    except ValueError as e:
        assert "duplicate" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for duplicate slots")
    print("[test_batched_steering] duplicate rejection ok", flush=True)


def main() -> int:
    try:
        test_duplicate_slots_rejected()
        test_parity_b1()
        test_b2_shapes()
        test_torn_epoch_batched_read()
        test_clamp_batch_telemetry()
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("[test_batched_steering] all passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
