"""
Shared SAE slab steering for HiveClaw LLM scripts (intelligence_spike, llm_swarm).

Decode: scent_2048 = matmul(scent_256, W_enc) + b_dec (tied SAE; no decoder.weight).
Clamp: L2 ball radius 2.0 on alpha * scent_2048; poison_clamp telemetry on stderr.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def check_latent_dim(slab_client) -> None:
    d = slab_client.get_latent_dim()
    if d != 256:
        print(
            f"[ERROR] expected get_latent_dim()==256, got {d}",
            file=sys.stderr,
        )
        sys.exit(1)


def load_sae(sae_path: Path) -> dict[str, mx.array]:
    if not sae_path.is_file():
        print(
            f"[ERROR] Missing SAE weights at {sae_path}. "
            "Run scripts/harvester.py then scripts/train_sae.py.",
            file=sys.stderr,
        )
        sys.exit(1)
    from safetensors.numpy import safe_open

    with safe_open(str(sae_path), framework="numpy") as f:
        out = {k: mx.array(f.get_tensor(k)) for k in f.keys()}
    W = out["encoder.weight"]
    if tuple(W.shape) != (256, 2048):
        print(
            f"[ERROR] encoder.weight shape {W.shape}, expected (256, 2048)",
            file=sys.stderr,
        )
        sys.exit(1)
    return out


class CaptureWrapper(nn.Module):
    """Captures final-layer hidden state in `captured_h`."""

    def __init__(self, layer):
        super().__init__()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "captured_h", None)

    def __getattr__(self, name: str):
        if name in ("layer", "captured_h"):
            return super().__getattr__(name)
        try:
            return getattr(self.layer, name)
        except AttributeError:
            return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        object.__setattr__(
            self, "captured_h", out[0] if isinstance(out, tuple) else out
        )
        return out


def steer_hidden(
    h_step: mx.array,
    slab_client,
    slot_index: int,
    W_enc: mx.array,
    b_dec: mx.array,
    alpha: float,
) -> mx.array:
    """read_slot_v5 → decode → L2 clamp on alpha*decoded → add to h_step."""
    scent_256 = slab_client.read_slot_v5(int(slot_index), depends=h_step)
    scent_f = scent_256.astype(mx.float32)
    scent_2048 = mx.matmul(scent_f, W_enc) + b_dec

    scaled = alpha * scent_2048
    norm = mx.linalg.norm(scaled, ord=2, axis=-1, keepdims=True)
    scale = mx.where(norm > 2.0, 2.0 / (norm + 1e-7), mx.ones_like(norm))
    safe = scaled * scale
    if bool(mx.any(norm > 2.0).item()):
        sys.stderr.write(
            f'{{"event":"poison_clamp","slot_id":{int(slot_index)},"ts_ns":{time.time_ns()}}}\n'
        )

    return (h_step.astype(mx.float32) + safe).astype(h_step.dtype)


def steer_hidden_batched(
    h_batch: mx.array,
    slab_client,
    batch_slots: mx.array,
    W_enc: mx.array,
    b_dec: mx.array,
    alpha: float,
) -> mx.array:
    """Batched read_slots → decode → L2 clamp per row → add to last-token hidden [B,1,2048]."""
    scents, _st = slab_client.read_slots(batch_slots, depends=h_batch)
    scent_f = scents.astype(mx.float32)
    scent_2048 = mx.matmul(scent_f, W_enc) + b_dec
    scaled = float(alpha) * scent_2048
    norm = mx.linalg.norm(scaled, ord=2, axis=-1, keepdims=True)
    scale = mx.where(norm > 2.0, 2.0 / (norm + 1e-7), mx.ones_like(norm))
    safe = scaled * scale
    mx.eval(norm)
    if bool(mx.any(norm > 2.0).item()):
        mx.eval(batch_slots)
        B = int(batch_slots.shape[0])
        slots_np = np.array(batch_slots, dtype=np.int32).reshape(-1)
        norm_flat = np.array(norm, dtype=np.float64).reshape(-1)
        clamped = [int(slots_np[i]) for i in range(B) if norm_flat[i] > 2.0]
        if clamped:
            sys.stderr.write(
                json.dumps(
                    {
                        "event": "poison_clamp_batch",
                        "slots": clamped,
                        "ts_ns": time.time_ns(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            sys.stderr.flush()
    return (h_batch.astype(mx.float32) + safe).astype(h_batch.dtype)


def apply_steering_sandwich(
    h_step_bucket: mx.array,
    slab_client,
    batch_slots: mx.array,
    W_enc: mx.array,
    b_dec: mx.array,
    alpha: float,
    *,
    b_enc: mx.array | None = None,
    d_latent: int = 256,
) -> tuple[mx.array, mx.array, mx.array]:
    """Full ``[B_bucket,1,2048]`` steering (no batch-axis slice). Dummy rows: ``batch_slots==-1``.

    Returns ``(H_steered, norm, wst)`` with ``H_steered``/``norm`` shapes
    ``[B_bucket,1,2048]`` and ``[B_bucket,1,1]``. ``wst`` is per-row write status
    ``[B_bucket]`` uint8 when ``b_enc`` is set; otherwise a shape-stable dummy
    ``(1,)`` uint32 for ``mx.compile`` tuple consistency (no ``mx.eval`` inside).
    Poison-clamp telemetry is emitted by the caller (e.g. host-side after ``mx.eval``).

    When ``b_enc`` is None, skips slab write-back (tests / partial integration).
    """
    B_bucket = int(h_step_bucket.shape[0])
    if int(batch_slots.shape[0]) != B_bucket:
        raise ValueError(
            f"batch_slots length {batch_slots.shape[0]} != B_bucket {B_bucket}"
        )

    scents, st = slab_client.read_slots(batch_slots, depends=h_step_bucket)
    st_exp = st.reshape(-1, 1, 1)
    scents = mx.where(st_exp == 0, scents, mx.zeros_like(scents))

    scent_f = scents.astype(mx.float32)
    scent_2048 = mx.matmul(scent_f, W_enc) + b_dec
    scaled = float(alpha) * scent_2048
    norm = mx.linalg.norm(scaled, ord=2, axis=-1, keepdims=True)
    scale = mx.where(norm > 2.0, 2.0 / (norm + 1e-7), mx.ones_like(norm))
    safe = scaled * scale

    active_mask = (batch_slots != -1).reshape(-1, 1, 1).astype(h_step_bucket.dtype)
    safe = safe.astype(h_step_bucket.dtype) * active_mask

    h32 = h_step_bucket.astype(mx.float32)
    steered = (h32 + safe.astype(mx.float32)).astype(h_step_bucket.dtype)
    H_steered = mx.where(active_mask > 0, steered, mx.zeros_like(h_step_bucket))

    dummy_wst = mx.zeros((1,), dtype=mx.uint32)
    if b_enc is not None:
        h_f = H_steered.astype(mx.float32)
        latent = mx.maximum(
            mx.matmul(h_f, mx.transpose(W_enc)) + b_enc, 0.0
        ).astype(mx.bfloat16)
        latent = latent.reshape(B_bucket, 1, d_latent)
        _, wst = slab_client.write_slots(batch_slots, latent, depends=H_steered)
        return H_steered, norm, wst

    return H_steered, norm, dummy_wst


class ActiveSteeringWrapper(nn.Module):
    """Final layer + read_slot_v5 → decode → clamp → inject into last token hidden."""

    def __init__(
        self,
        layer,
        slab_client,
        W_enc: mx.array,
        b_dec: mx.array,
        alpha: float,
        slot_index: int = 0,
    ):
        super().__init__()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "slab_client", slab_client)
        object.__setattr__(self, "W_enc", W_enc)
        object.__setattr__(self, "b_dec", b_dec)
        object.__setattr__(self, "alpha", float(alpha))
        object.__setattr__(self, "current_slot", int(slot_index))
        object.__setattr__(self, "last_steered_h", None)

    def __getattr__(self, name: str):
        if name in (
            "layer",
            "slab_client",
            "W_enc",
            "b_dec",
            "alpha",
            "current_slot",
            "last_steered_h",
        ):
            return super().__getattr__(name)
        try:
            return getattr(self.layer, name)
        except AttributeError:
            return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        h_step = h[:, -1:, :]

        h_modified = steer_hidden(
            h_step,
            self.slab_client,
            int(self.current_slot),
            self.W_enc,
            self.b_dec,
            self.alpha,
        )
        object.__setattr__(self, "last_steered_h", h_modified)

        if h.shape[1] > 1:
            h_final = mx.concatenate([h[:, :-1, :], h_modified], axis=1)
        else:
            h_final = h_modified

        return (h_final,) + out[1:] if isinstance(out, tuple) else h_final


class BatchedSteeringWrapper(nn.Module):
    """Final layer + Steering Sandwich: full ``B_bucket`` tensor; dummy rows use slot -1."""

    def __init__(
        self,
        layer,
        slab_client,
        W_enc: mx.array,
        b_dec: mx.array,
        alpha: float,
        batch_slots: mx.array,
        *,
        b_enc: mx.array | None = None,
        d_latent: int = 256,
        enable_slab: bool = True,
    ):
        super().__init__()
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "slab_client", slab_client)
        object.__setattr__(self, "W_enc", W_enc)
        object.__setattr__(self, "b_dec", b_dec)
        object.__setattr__(self, "b_enc", b_enc)
        object.__setattr__(self, "d_latent", int(d_latent))
        object.__setattr__(self, "alpha", float(alpha))
        object.__setattr__(self, "enable_slab", bool(enable_slab))
        object.__setattr__(self, "current_batch_slots", batch_slots)
        object.__setattr__(self, "last_steered_h", None)
        bb = int(batch_slots.shape[0])
        object.__setattr__(
            self,
            "last_steered_norm",
            mx.zeros((bb, 1, 1), dtype=mx.float32),
        )
        object.__setattr__(
            self,
            "last_steered_wst",
            mx.zeros((1,), dtype=mx.uint32),
        )

    def __getattr__(self, name: str):
        if name in (
            "layer",
            "slab_client",
            "W_enc",
            "b_dec",
            "b_enc",
            "d_latent",
            "alpha",
            "enable_slab",
            "current_batch_slots",
            "last_steered_h",
            "last_steered_norm",
            "last_steered_wst",
        ):
            return super().__getattr__(name)
        try:
            return getattr(self.layer, name)
        except AttributeError:
            return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        kwargs_fwd = dict(kwargs)
        batch_slots_kw = kwargs_fwd.pop("batch_slots", None)
        slots = (
            batch_slots_kw
            if batch_slots_kw is not None
            else self.current_batch_slots
        )
        out = self.layer(*args, **kwargs_fwd)
        if not self.enable_slab:
            h = out[0] if isinstance(out, tuple) else out
            h_step = h[:, -1:, :]
            object.__setattr__(self, "last_steered_h", h_step)
            bb = int(h.shape[0])
            object.__setattr__(
                self,
                "last_steered_norm",
                mx.zeros((bb, 1, 1), dtype=mx.float32),
            )
            object.__setattr__(
                self,
                "last_steered_wst",
                mx.zeros((1,), dtype=mx.uint32),
            )
            return out
        h = out[0] if isinstance(out, tuple) else out
        h_step = h[:, -1:, :]
        h_modified, norm, wst = apply_steering_sandwich(
            h_step,
            self.slab_client,
            slots,
            self.W_enc,
            self.b_dec,
            self.alpha,
            b_enc=self.b_enc,
            d_latent=int(self.d_latent),
        )
        object.__setattr__(self, "last_steered_h", h_modified)
        object.__setattr__(self, "last_steered_norm", norm)
        object.__setattr__(self, "last_steered_wst", wst)
        if h.shape[1] > 1:
            h_final = mx.concatenate([h[:, :-1, :], h_modified], axis=1)
        else:
            h_final = h_modified
        return (h_final,) + out[1:] if isinstance(out, tuple) else h_final
