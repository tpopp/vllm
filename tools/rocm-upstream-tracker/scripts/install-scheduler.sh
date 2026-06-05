#!/usr/bin/env bash
set -euo pipefail

ENABLE_WAKE_ON_AC=false
CRON_ONLY=false
SYSTEMD_OK=false
CRON_OK=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-wake-on-ac) ENABLE_WAKE_ON_AC=true ;;
    --cron-fallback|--cron-only) CRON_ONLY=true ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${ROCM_TRACKER_DATA_DIR:-$HOME/.local/share/rocm-tracker}"
BIN_DIR="$DATA_DIR/bin"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
DAILY_BIN="$BIN_DIR/rocm-tracker-daily.sh"
CATCHUP_BIN="$BIN_DIR/rocm-tracker-catchup.sh"

mkdir -p "$BIN_DIR" "$SYSTEMD_USER_DIR" "$DATA_DIR/logs"
install -m 0755 "$SCRIPT_DIR/rocm-tracker-daily.sh" "$DAILY_BIN"

cat >"$CATCHUP_BIN" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export ROCM_TRACKER_ENV_FILE="\${ROCM_TRACKER_ENV_FILE:-\$HOME/.config/rocm-tracker/env}"
if [[ -f "\$ROCM_TRACKER_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "\$ROCM_TRACKER_ENV_FILE"
fi
: "\${ROCM_TRACKER_REPO_PATH:=\$HOME/src/vllm}"
TOOL_DIR="$TOOL_DIR"
if [[ -x "\$ROCM_TRACKER_REPO_PATH/.venv/bin/rocm-tracker" ]]; then
  "\$ROCM_TRACKER_REPO_PATH/.venv/bin/rocm-tracker" if-missed-today
elif command -v uv >/dev/null 2>&1; then
  uv run --project "\$TOOL_DIR" rocm-tracker if-missed-today
else
  PYTHONPATH="\$TOOL_DIR" python3 -m rocm_tracker.cli if-missed-today
fi
EOF
chmod +x "$CATCHUP_BIN"

mkdir -p "$HOME/.config/rocm-tracker"
if [[ ! -f "$HOME/.config/rocm-tracker/env" ]]; then
  cp "$TOOL_DIR/config.example.env" "$HOME/.config/rocm-tracker/env"
  echo "Created $HOME/.config/rocm-tracker/env — edit ROCM_TRACKER_REPO_PATH."
fi

install_systemd() {
  sed "s|%h/.local/share/rocm-tracker/bin/rocm-tracker-daily.sh|$DAILY_BIN|g" \
    "$SCRIPT_DIR/systemd/rocm-tracker.service" >"$SYSTEMD_USER_DIR/rocm-tracker.service"
  cp "$SCRIPT_DIR/systemd/rocm-tracker.timer" "$SYSTEMD_USER_DIR/rocm-tracker.timer"
  systemctl --user daemon-reload
  systemctl --user enable --now rocm-tracker.timer
  SYSTEMD_OK=true
  echo "Installed systemd user timer: rocm-tracker.timer"
  echo "Check status: systemctl --user status rocm-tracker.timer"
  echo "Optional linger (run when logged out): loginctl enable-linger \"$USER\""
}

install_cron() {
  if ! command -v crontab >/dev/null 2>&1; then
    echo "cron not available (crontab not found)." >&2
    return 1
  fi
  CRON_DAILY="0 8 * * * TZ=Europe/Berlin $DAILY_BIN"
  CRON_CATCHUP="@reboot sleep 120 && TZ=Europe/Berlin $CATCHUP_BIN"
  (
    crontab -l 2>/dev/null | grep -v rocm-tracker-daily.sh | grep -v rocm-tracker-catchup.sh || true
    echo "$CRON_DAILY"
    echo "$CRON_CATCHUP"
  ) | crontab -
  CRON_OK=true
  echo "Installed cron entries:"
  echo "  daily:   $CRON_DAILY"
  echo "  catchup: $CRON_CATCHUP"
}

if ! $CRON_ONLY && command -v systemctl >/dev/null 2>&1; then
  if systemctl --user daemon-reload >/dev/null 2>&1; then
    install_systemd || true
  else
    echo "systemd user bus unavailable; will try cron."
  fi
fi

if ! $SYSTEMD_OK; then
  install_cron || echo "Cron install skipped."
fi

if $ENABLE_WAKE_ON_AC; then
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Installing wake hook requires sudo:"
    echo "  sudo install -m 0755 $SCRIPT_DIR/sleep/rocm-tracker-wake.sh /usr/lib/systemd/system-sleep/rocm-tracker-wake.sh"
  else
    install -m 0755 "$SCRIPT_DIR/sleep/rocm-tracker-wake.sh" \
      /usr/lib/systemd/system-sleep/rocm-tracker-wake.sh
    echo "Installed RTC wake hook for AC suspend."
  fi
fi

if ! $SYSTEMD_OK && ! $CRON_OK; then
  echo ""
  echo "Scheduler not installed automatically in this environment."
  echo "Wrapper scripts are ready:"
  echo "  daily:   $DAILY_BIN"
  echo "  catchup: $CATCHUP_BIN"
  echo ""
  echo "On your laptop, run:"
  echo "  $SCRIPT_DIR/install-scheduler.sh"
  echo "Or add manually:"
  echo "  0 8 * * * TZ=Europe/Berlin $DAILY_BIN"
  exit 1
fi

echo "Install complete."
