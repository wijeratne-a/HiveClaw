"""
Shared SAE slab steering for HiveClaw LLM scripts (intelligence_spike, llm_swarm).

Decode: scent_2048 = matmul(scent_256, W_enc) + b_dec (tied SAE; no decoder.weight).
Clamp: L2 ball radius 2.0 on alpha * scent_2048; poison_clamp telemetry on stderr.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


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
