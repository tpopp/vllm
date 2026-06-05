#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ROCM_TRACKER_ENV_FILE:-$HOME/.config/rocm-tracker/env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

: "${ROCM_TRACKER_REPO_PATH:=$HOME/src/vllm}"
: "${ROCM_TRACKER_DATA_DIR:=$HOME/.local/share/rocm-tracker}"

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROCM_TRACKER_DATA_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily-$(date +%Y-%m-%d).log"
VERBOSE_ARGS=()
if [[ "${ROCM_TRACKER_VERBOSE:-}" == "1" ]]; then
  VERBOSE_ARGS=(-v)
fi

run_daily() {
  echo "=== rocm-tracker daily start $(date -Is) ==="
  echo "log file: $LOG_FILE"
  cd "$ROCM_TRACKER_REPO_PATH"

  if [[ -x "$ROCM_TRACKER_REPO_PATH/.venv/bin/rocm-tracker" ]]; then
    "$ROCM_TRACKER_REPO_PATH/.venv/bin/rocm-tracker" daily "${VERBOSE_ARGS[@]}"
  elif command -v uv >/dev/null 2>&1; then
    uv run --project "$TOOL_DIR" rocm-tracker daily "${VERBOSE_ARGS[@]}"
  else
    PYTHONPATH="$TOOL_DIR" python3 -m rocm_tracker.cli daily "${VERBOSE_ARGS[@]}"
  fi

  echo "=== rocm-tracker daily end $(date -Is) ==="
}

if [[ "${ROCM_TRACKER_VERBOSE:-}" == "1" ]]; then
  run_daily 2>&1 | tee -a "$LOG_FILE"
else
  run_daily >>"$LOG_FILE" 2>&1
fi
