#!/usr/bin/env bash
# Full Ironclad verification: SSE burn-in + burn_in.py criteria + zero eager_fallback.
#
# Defaults (override with env):
#   HEALTH_TIMEOUT_S=900         — wait for /health (Phase 7 probe + compile warmup can exceed 300s)
#   HIVECLAW_MAX_QUEUE_DEPTH=10  — force 503 admission control under load (vs 50 clients)
#   CONCURRENCY=50
#   SWAPIN_DELTA_MAX=500000      — realistic macOS unified-memory paging budget for MLX load
#
# Exit 0 only if: (1) 0 eager_fallback in server log, (2) burn_in exits 0 (503 + swap + HTTP checks).
#
# Usage (from repo root):
#   ./scripts/verify_burn_in.sh
#   PORT=8766 CONCURRENCY=60 ./scripts/verify_burn_in.sh
#   HIVECLAW_MAX_QUEUE_DEPTH=5 SWAPIN_DELTA_MAX=1000000 ./scripts/verify_burn_in.sh
#   HEALTH_TIMEOUT_S=1200 ./scripts/verify_burn_in.sh   # very slow cold compile / probe
#
# Requires: daemon loaded (make doctor), venv with mlx + httpx, models as for hiveclaw_server.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8080}"
CONCURRENCY="${CONCURRENCY:-50}"
SWAPIN_DELTA_MAX="${SWAPIN_DELTA_MAX:-500000}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-900}"
LOG="${VERIFY_LOG:-${TMPDIR:-/tmp}/hiveclaw_verify_$$.log}"

if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
  PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python3}"
else
  PYTHON="${PYTHON:-python3}"
fi

export HIVECLAW_CONTINUOUS_BATCH=1
export HIVECLAW_COMPILE_DECODE=1
export HIVECLAW_COMPILE_WARMUP=1
export HIVECLAW_MAX_QUEUE_DEPTH="${HIVECLAW_MAX_QUEUE_DEPTH:-10}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[verify_burn_in] log: ${LOG}"
rm -f "${LOG}"
touch "${LOG}"

echo "[verify_burn_in] starting server on 127.0.0.1:${PORT} (max_queue_depth=${HIVECLAW_MAX_QUEUE_DEPTH}) ..."
"${PYTHON}" -m hiveclaw_python.server_main --host 127.0.0.1 --port "${PORT}" >>"${LOG}" 2>&1 &
SERVER_PID=$!

echo "[verify_burn_in] waiting for /health (timeout=${HEALTH_TIMEOUT_S}s) ..."
deadline=$((SECONDS + HEALTH_TIMEOUT_S))
while (( SECONDS < deadline )); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "[verify_burn_in] FAIL: server did not become healthy (see ${LOG})" >&2
  tail -n 80 "${LOG}" >&2 || true
  exit 1
fi

echo "[verify_burn_in] running burn_in (concurrency=${CONCURRENCY}, swapin-delta-max=${SWAPIN_DELTA_MAX}) ..."
set +e
"${PYTHON}" scripts/burn_in.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --concurrency "${CONCURRENCY}" \
  --stderr-file "${LOG}" \
  --swapin-delta-max "${SWAPIN_DELTA_MAX}"
BURN_EXIT=$?
set -e

# Step A success metric: JSON lines with event eager_fallback (same logic as burn_in.py)
export VERIFY_LOG_PATH="${LOG}"
EAGER="$("${PYTHON}" <<'PY'
import json
import os

path = os.environ["VERIFY_LOG_PATH"]
n = 0
try:
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if '"event"' not in line or "eager_fallback" not in line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("event") == "eager_fallback":
                n += 1
except OSError:
    pass
print(n)
PY
)"
unset VERIFY_LOG_PATH

echo "[verify_burn_in] eager_fallback events in server log: ${EAGER}"
echo "[verify_burn_in] burn_in exit code: ${BURN_EXIT}"

if [[ "${EAGER}" -ne 0 ]]; then
  echo "[verify_burn_in] FAIL Step A: expected 0 eager_fallback, got ${EAGER}" >&2
  grep -F eager_fallback "${LOG}" >&2 || true
  exit 1
fi

echo "[verify_burn_in] PASS: 0 eager_fallback in server log."
if [[ "${BURN_EXIT}" -ne 0 ]]; then
  echo "[verify_burn_in] FAIL: burn_in exited ${BURN_EXIT} (need 503 under load, swap delta, and HTTP checks)." >&2
  exit "${BURN_EXIT}"
fi
echo "[verify_burn_in] PASS: burn_in full criteria (exit 0)."
exit 0
