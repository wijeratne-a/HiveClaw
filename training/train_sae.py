#!/usr/bin/env python3
"""
Offline tied-weight SAE: encoder [256,2048] + biases, decoder uses W.T + decoder.bias.
AdamW lr=1e-4, batch 1024, MSE + lambda * L1(ReLU latent).

Validation split is 90/10 by shard file (not by row) to prevent data leakage: sequential
token hidden states within a shard are highly correlated; shard-level split guarantees the
validation set contains only unseen document contexts.

Early stop: absolute val MSE decrease < 1e-4 for 3 consecutive epochs; hard cap 100 epochs.
Saves models/hiveclaw_sae_v1.safetensors
"""

from __future__ import annotations

import glob
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from safetensors.numpy import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
OUT_PATH = MODELS / "hiveclaw_sae_v1.safetensors"
LAMBDA_L1 = 1e-3
LR = 1e-4
BATCH = 1024
LATENT = 256
INPUT_D = 2048
PLATEAU = 3
DELTA = 1e-4


class TiedSAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = mx.random.normal((LATENT, INPUT_D)) * 0.02
        self.b_enc = mx.zeros((LATENT,))
        self.b_dec = mx.zeros((INPUT_D,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        z = mx.maximum(mx.matmul(x, self.W.T) + self.b_enc, 0.0)
        x_hat = mx.matmul(z, self.W) + self.b_dec
        return x_hat, z


def _load_shard_paths() -> list[Path]:
    MODELS.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(MODELS / "latent_traces_*.npz")))
    if not files:
        raise SystemExit(
            f"No shards in {MODELS}. Run training/harvester.py first."
        )
    return [Path(f) for f in files]


def _iter_batches(paths: list[Path], rng: random.Random):
    """Infinite random batch iterator over training shards."""
    while True:
        p = rng.choice(paths)
        lat = np.load(p)["latents"].astype(np.float32)
        n = lat.shape[0]
        if n <= BATCH:
            idx = np.arange(n)
            rng.shuffle(idx)
            yield mx.array(lat[idx])
        else:
            pick = rng.sample(range(n), BATCH)
            yield mx.array(lat[np.array(pick)])


def main() -> None:
    mx.random.seed(42)
    random.seed(42)
    np.random.seed(42)

    all_paths = _load_shard_paths()
    rng = random.Random(42)
    rng.shuffle(all_paths)
    n = len(all_paths)
    n_tr = max(1, int(n * 0.9))
    train_paths = all_paths[:n_tr]
    val_paths = all_paths[n_tr:] or all_paths[:1]

    model = TiedSAE()
    mx.eval(model.parameters())

    def loss_fn(m: TiedSAE, x: mx.array) -> mx.array:
        x_hat, z = m(x)
        mse = mx.mean((x_hat - x) ** 2)
        l1 = mx.mean(mx.abs(z))
        return mse + LAMBDA_L1 * l1

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    optimizer = optim.AdamW(learning_rate=LR)
    optimizer.init(model.trainable_parameters())

    train_iter = _iter_batches(train_paths, rng)

    def val_mse() -> float:
        total = 0.0
        count = 0
        for vp in val_paths:
            lat = np.load(vp)["latents"].astype(np.float32)
            for i in range(0, lat.shape[0], BATCH):
                chunk = lat[i : i + BATCH]
                if chunk.shape[0] < BATCH:
                    continue
                x = mx.array(chunk)
                x_hat, _ = model(x)
                total += float(mx.mean((x_hat - x) ** 2).item())
                count += 1
        return total / max(count, 1)

    plateau_epochs = 0
    prev_val = None

    for epoch in range(100):
        batches_per_epoch = max(50, sum(np.load(p)["latents"].shape[0] for p in train_paths) // BATCH)
        for _ in range(batches_per_epoch):
            x = next(train_iter)
            loss, grads = loss_and_grad(model, x)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)

        v = val_mse()
        print(f"[train_sae] epoch {epoch} val_mse={v:.6f}", flush=True)

        if prev_val is not None:
            if prev_val - v < DELTA:
                plateau_epochs += 1
            else:
                plateau_epochs = 0
        prev_val = v
        if plateau_epochs >= PLATEAU:
            print("[train_sae] early stopping (val MSE plateau).", flush=True)
            break

    mx.eval(model.W, model.b_enc, model.b_dec)
    W = np.ascontiguousarray(np.array(model.W, dtype=np.float32))
    b_enc = np.ascontiguousarray(np.array(model.b_enc, dtype=np.float32))
    b_dec = np.ascontiguousarray(np.array(model.b_dec, dtype=np.float32))
    save_file(
        {
            "encoder.weight": W,
            "encoder.bias": b_enc,
            "decoder.bias": b_dec,
        },
        str(OUT_PATH),
    )
    print(f"[train_sae] saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
