#!/usr/bin/env python3
"""
One-time calibration script: map SAE dimensions to coarse concept labels.

This script is intentionally lightweight and heuristic-driven for demo purposes.
It computes latent activations for labeled snippets and selects top correlated
dimensions per concept.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from hiveclaw_python.steering import CaptureWrapper, load_sae

MODEL_DEFAULT = "mlx-community/Llama-3.2-1B-Instruct-4bit"

LABELED_SNIPPETS: list[tuple[str, str]] = [
    ("unsafe { ptr.read_unaligned() }", "Rust Safety"),
    ("value.unwrap()", "Unsafe Unwrap"),
    ("fn parse() -> Result<String, Error> { Ok(\"x\".into()) }", "Error Handling"),
    ("tokio::spawn(async move { work().await; });", "Async Pattern"),
    ("let mut data = Vec::with_capacity(1024);", "Memory Management"),
    ("def f(x):\n    return x", "Python Type Safety"),
    ("def f(x: int) -> int:\n    return x", "Python Type Safety"),
    ("try:\n    run()\nexcept:\n    pass", "Input Validation"),
    ("# no docstring\n\ndef public_fn(a: int) -> int:\n    return a", "Docstring Quality"),
    ("xpc_connection_send_message_with_reply_sync(conn, dict)", "IPC Slab Activity"),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_sae_path() -> Path:
    return _repo_root() / "models" / "hiveclaw_sae_v1.safetensors"


def _encode_for_model(tokenizer, text: str) -> mx.array:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return mx.array(tokenizer.encode(prompt))


def _latent_for_text(
    *,
    model: any,
    tokenizer: any,
    text: str,
    W_enc: mx.array,
    b_enc: mx.array,
    original_layer: any,
) -> np.ndarray:
    wrapper = CaptureWrapper(original_layer)
    model.model.layers[-1] = wrapper
    toks = _encode_for_model(tokenizer, text)
    _ = model(toks[None])
    mx.eval(wrapper.captured_h)
    model.model.layers[-1] = original_layer

    h = wrapper.captured_h[:, -1:, :].astype(mx.float32)
    z = mx.maximum(mx.matmul(h, mx.transpose(W_enc)) + b_enc, 0.0).reshape(-1)
    mx.eval(z)
    return np.array(z, dtype=np.float32)


def build_dictionary(model_id: str, sae_path: Path, out_json: Path) -> None:
    model, tokenizer, _cfg = load(model_id, return_config=True)
    sae = load_sae(sae_path)
    W_enc = sae["encoder.weight"]
    b_enc = sae["encoder.bias"]

    original_layer = model.model.layers[-1]
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)

    for text, label in LABELED_SNIPPETS:
        z = _latent_for_text(
            model=model,
            tokenizer=tokenizer,
            text=text,
            W_enc=W_enc,
            b_enc=b_enc,
            original_layer=original_layer,
        )
        by_label[label].append(z)

    label_centroids: dict[str, np.ndarray] = {}
    all_z: list[np.ndarray] = []
    for label, zs in by_label.items():
        stack = np.stack(zs, axis=0)
        label_centroids[label] = stack.mean(axis=0)
        all_z.append(stack)
    all_stack = np.concatenate(all_z, axis=0)
    mean = all_stack.mean(axis=0)
    std = all_stack.std(axis=0) + 1e-6

    dim_to_payload: dict[str, dict] = {}
    for label, centroid in label_centroids.items():
        zscore = (centroid - mean) / std
        dim = int(np.argmax(np.abs(zscore)))
        conf = float(min(0.95, max(0.30, abs(float(zscore[dim])) / 6.0)))
        dim_to_payload[str(dim)] = {
            "label": label,
            "confidence": round(conf, 2),
            "calibration": {
                "mean": round(float(mean[dim]), 6),
                "std": round(float(std[dim]), 6),
            },
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(dim_to_payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Build SAE feature dictionary for demo dashboard")
    p.add_argument("--model-id", type=str, default=MODEL_DEFAULT)
    p.add_argument("--sae-path", type=str, default=str(_default_sae_path()))
    p.add_argument(
        "--out",
        type=str,
        default=str(_repo_root() / "demos" / "data" / "feature_dictionary.json"),
    )
    args = p.parse_args()

    build_dictionary(
        model_id=args.model_id,
        sae_path=Path(args.sae_path),
        out_json=Path(args.out),
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
