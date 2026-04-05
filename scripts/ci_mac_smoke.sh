#!/usr/bin/env bash
# macOS CI / local smoke: doctor + quick integration (requires daemon + venv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python3" ]]; then
  PYTHON="$ROOT/.venv/bin/python3"
fi
export PYTHON
make doctor PYTHON="$PYTHON"
exec "$PYTHON" scripts/integration_test.py --quick
