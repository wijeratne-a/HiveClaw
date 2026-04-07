#!/usr/bin/env python3
"""
HiveClaw v5 latent harvester.
Llama 3.2 1B final layer, every 10th token, L2-normalize fp32, shards of [250000, 2048] float32.
Dataset: wikitext-103-raw-v1 train split. Stops at 2M vectors. Auto-resumes from models/.

On startup, incomplete shards (fewer than SHARD_SIZE rows) are deleted to avoid corrupted .npz
tails; the next shard index is max(existing index) + 1.
"""

from __future__ import annotations

import glob
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from mlx_lm import load

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "models"
SHARD_SIZE = 250_000
TOTAL_CAP = 2_000_000
CHUNK_LEN = 512
THROUGHPUT_LOG_EVERY = 1000


class CaptureWrapper(nn.Module):
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


def _parse_shard_index(path: str) -> int:
    m = re.search(r"latent_traces_(\d+)\.npz$", path)
    return int(m.group(1)) if m else -1


def _scan_existing() -> tuple[int, int]:
    """
    Delete partial shards; return (total rows on disk from full shards only, next shard index).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pattern = str(OUT_DIR / "latent_traces_*.npz")
    files = sorted(glob.glob(pattern), key=_parse_shard_index)
    for fpath in files:
        n = int(np.load(fpath)["latents"].shape[0])
        if n < SHARD_SIZE:
            os.remove(fpath)
            print(f"[harvester] removed incomplete shard {fpath} (n={n})", flush=True)
    files = sorted(glob.glob(pattern), key=_parse_shard_index)
    if not files:
        return 0, 0
    already = sum(int(np.load(f)["latents"].shape[0]) for f in files)
    indices = [_parse_shard_index(f) for f in files]
    return already, max(indices) + 1


def _flush_shard(buf: list[np.ndarray], shard_idx: int) -> None:
    if not buf:
        return
    arr = np.stack(buf, axis=0).astype(np.float32)
    path = OUT_DIR / f"latent_traces_{shard_idx:05d}.npz"
    np.savez_compressed(path, latents=arr)
    print(f"[harvester] wrote {path} shape={arr.shape}", flush=True)


def main() -> None:
    total_cap = TOTAL_CAP
    already, shard_idx = _scan_existing()
    room = total_cap - already
    if room <= 0:
        print(f"[harvester] already have {already} >= {total_cap}, nothing to do.")
        return

    print(
        f"[harvester] already={already} room={room} shard_idx={shard_idx} "
        f"total_cap={total_cap}",
        flush=True,
    )

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    model, tokenizer, _ = load(MODEL_ID, return_config=True)
    original = model.model.layers[-1]
    wrapper = CaptureWrapper(original)
    model.model.layers[-1] = wrapper

    token_buf: list[int] = []
    global_tok = 0
    skip = already
    new_done = 0
    shard_buf: list[np.ndarray] = []
    t0 = time.monotonic()
    last_log_n = 0

    try:
        for row in ds:
            if new_done >= room:
                break
            text = row.get("text") or ""
            if not text.strip():
                continue
            ids = tokenizer.encode(text)
            token_buf.extend(ids)
            while len(token_buf) >= CHUNK_LEN:
                chunk = token_buf[:CHUNK_LEN]
                token_buf = token_buf[CHUNK_LEN:]
                ids_mx = mx.array(chunk, dtype=mx.int32)[None, :]
                _ = model(ids_mx)
                mx.eval(wrapper.captured_h)
                h = np.array(wrapper.captured_h.astype(mx.float32))
                for pos in range(CHUNK_LEN):
                    if new_done >= room:
                        break
                    if global_tok % 10 == 0:
                        if skip > 0:
                            skip -= 1
                        else:
                            v = h[0, pos].astype(np.float64)
                            n = float(np.linalg.norm(v) + 1e-7)
                            shard_buf.append((v / n).astype(np.float32))
                            new_done += 1
                            if len(shard_buf) >= SHARD_SIZE:
                                _flush_shard(shard_buf, shard_idx)
                                shard_buf = []
                                shard_idx += 1
                            if new_done - last_log_n >= THROUGHPUT_LOG_EVERY:
                                dt = time.monotonic() - t0
                                rate = new_done / dt if dt > 0 else 0.0
                                print(
                                    f"[harvester] throughput: {rate:.2f} vectors/sec "
                                    f"(new_done={new_done})",
                                    flush=True,
                                )
                                last_log_n = new_done
                    global_tok += 1
                if new_done >= room:
                    break
    finally:
        model.model.layers[-1] = original

    if shard_buf:
        _flush_shard(shard_buf, shard_idx)

    print(f"[harvester] done. new this run={new_done}.", flush=True)


if __name__ == "__main__":
    main()
