# PR2 / Phase 4 v5 (Slab + SAE steering)

- **`FusedSteerScent` removed.** Steering is **Python-side MLX**: `read_slot_v5` → decode with tied SAE weights → L2 poison clamp → add to `h_step`. `intelligence_spike.py` / `llm_swarm.py` load `models/hiveclaw_sae_v1.safetensors`.
- **`ReadSlab` v5:** 512-byte payload, shape **`[1,1,256]`** bf16, **torn epoch → zeros**; GPU path uses embedded MSL **`read_slab_v5`** (32×8 threads) + **1-byte status** buffer; completion handler may emit **`torn_epoch_skip`** JSON to stderr when **`HIVECLAW_TELEMETRY != "0"`**.
- **`WriteSlab` v5 (stamped slots):** strict **`[1,1,256]`** bf16, bumps **`front_epoch`**, memcpy 512 B, sets **`back_epoch`** at slot **+576**.
- **Layout:** global header **4096** B; **4096** slots × **640** B stride; magic/version at bytes **0–11** (`0x48434C5700000005`, version **5**). XPC cmd **`get_surface_v5`** only; other commands → **`INVALID_COMMAND_OR_UNSUPPORTED_VERSION`**.
- **Training pipeline:** `training/harvester.py` → `models/latent_traces_*.npz` → `training/train_sae.py` → `models/hiveclaw_sae_v1.safetensors` (ignored by git). Verify with `tests/test_sae_tied_weights.py`.

See `scripts/README.md` for rebuild / daemon / integration gates.
