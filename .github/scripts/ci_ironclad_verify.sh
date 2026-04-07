#!/usr/bin/env bash
# Ironclad exit-0 gate: make doctor (daemon + SlabClient) then verify_burn_in.sh (SSE + burn_in criteria + zero eager_fallback).
# macOS only. Requires: venv + make python + models as for hiveclaw_server; see scripts/README.md (Ironclad verification).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python3" ]]; then
  PYTHON="$ROOT/.venv/bin/python3"
fi
export PYTHON
make doctor PYTHON="$PYTHON"
exec bash "$ROOT/scripts/verify_burn_in.sh"
