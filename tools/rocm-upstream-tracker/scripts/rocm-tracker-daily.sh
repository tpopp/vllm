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

{
  echo "=== rocm-tracker daily start $(date -Is) ==="
  cd "$ROCM_TRACKER_REPO_PATH"

  if [[ -x "$ROCM_TRACKER_REPO_PATH/.venv/bin/rocm-tracker" ]]; then
    "$ROCM_TRACKER_REPO_PATH/.venv/bin/rocm-tracker" daily
  elif command -v uv >/dev/null 2>&1; then
    uv run --project "$TOOL_DIR" rocm-tracker daily
  else
    PYTHONPATH="$TOOL_DIR" python3 -m rocm_tracker.cli daily
  fi

  echo "=== rocm-tracker daily end $(date -Is) ==="
} >>"$LOG_FILE" 2>&1
