# Model prep (optional)

These scripts build **`models/hiveclaw_sae_v1.safetensors`** for SAE-steered slab workflows. They are **not** required for users who only run inference with an existing SAE artifact.

| Script | Role |
|--------|------|
| [`harvester.py`](harvester.py) | Capture latent traces to `models/latent_traces_*.npz` |
| [`train_sae.py`](train_sae.py) | Train the sparse autoencoder from shards |

Run from repo root with the same venv as `make python` (Metal / MLX). Do not run heavy MLX tests while the harvester holds the GPU.
