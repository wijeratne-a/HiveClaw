# hiveclaw_python

Python bindings and helpers for [HiveClaw](https://github.com/wijeratne-a/HiveClaw): Metal IOSurface slab access via PyO3, optional LaunchAgent bootstrap for `pheromoned`, and a `LocalSwarm` helper around the OpenAI-compatible server.

- **License:** AGPL-3.0 (see `LICENSE` in this crate).
- **Daemon:** On Apple Silicon, wheels may ship `native/macos_arm64/pheromoned` when built with CI (see `python/hiveclaw_python/native/macos_arm64/README.txt`). Otherwise build from the full repo: `cargo build --release -p hiveclaw-daemon`.
- **Server:** Run `hiveclaw-server` or `python -m hiveclaw_python.server_main` (FastAPI app: `hiveclaw_python.openai_server`). Set `HIVECLAW_REPO_ROOT` to your checkout when resolving bundled models/paths from the repo.
- **Operator CLIs:** `hiveclaw-dashboard`, `hiveclaw-overseer` (see `hiveclaw_python.cli`).
