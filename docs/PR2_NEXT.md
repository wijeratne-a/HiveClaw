# PR2 (follow-up): zero-copy fused steering

Planned after PR1 (slab v4 + epoch telemetry) is validated on Apple Silicon:

- `hiveclaw-mlx` CustomOp: fused read + `alpha * scent` with runtime `D` from tensor shape.
- Hard L2 ceiling on `||alpha * s||` (e.g. 2.0) for poisoned reads that bypass epoch checks.
- C++ single-line JSON stderr telemetry for torn-read / clamp events (no Python GIL).
- Remove remaining Python hot-path scent copies where possible; keep `generate_step` in Python.

See the accepted internal roadmap / plan file for ordering vs Mach dead-name eviction.
