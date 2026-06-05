#!/usr/bin/env bash
# One-shot setup for your laptop: install tool, config, and daily scheduler.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_PATH="$(cd "$TOOL_DIR/../.." && pwd)"

echo "Repo path: $REPO_PATH"
echo "Tool path: $TOOL_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required." >&2
  exit 1
fi

cd "$REPO_PATH"
if ! git remote | grep -q '^upstream$'; then
  git remote add upstream https://github.com/vllm-project/vllm.git
  echo "Added upstream remote."
fi

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv 2>/dev/null || true
  uv pip install -e "$TOOL_DIR"
else
  echo "uv not found. Install from https://astral.sh/uv/ then re-run." >&2
  exit 1
fi

mkdir -p "$HOME/.config/rocm-tracker"
ENV_FILE="$HOME/.config/rocm-tracker/env"
if [[ ! -f "$ENV_FILE" ]]; then
  cat >"$ENV_FILE" <<EOF
ROCM_TRACKER_REPO_PATH=$REPO_PATH
ROCM_TRACKER_DATA_DIR=$HOME/.local/share/rocm-tracker
ROCM_TRACKER_UPSTREAM_REPO=vllm-project/vllm
ROCM_TRACKER_FORK_REMOTE=origin
ROCM_TRACKER_MODEL=sonnet
ROCM_TRACKER_MAX_LLM_COMMITS_PER_RUN=20
ROCM_TRACKER_TIMEZONE=Europe/Berlin
ROCM_TRACKER_CURSOR_BIN=cursor
EOF
  echo "Wrote $ENV_FILE"
else
  echo "Config exists: $ENV_FILE (not overwritten)"
fi

bash "$SCRIPT_DIR/install-scheduler.sh" "$@"

echo ""
echo "Setup done. Verify:"
echo "  systemctl --user status rocm-tracker.timer   # preferred on Linux laptop"
echo "  crontab -l                                   # if cron fallback was used"
echo ""
echo "Test manually:"
echo "  $REPO_PATH/.venv/bin/rocm-tracker daily --dry-run"
