#!/usr/bin/env bash
set -euo pipefail

ENABLE_WAKE_ON_AC=false
CRON_FALLBACK=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-wake-on-ac) ENABLE_WAKE_ON_AC=true ;;
    --cron-fallback) CRON_FALLBACK=true ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${ROCM_TRACKER_DATA_DIR:-$HOME/.local/share/rocm-tracker}"
BIN_DIR="$DATA_DIR/bin"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

mkdir -p "$BIN_DIR" "$SYSTEMD_USER_DIR" "$DATA_DIR/logs"
install -m 0755 "$SCRIPT_DIR/rocm-tracker-daily.sh" "$BIN_DIR/rocm-tracker-daily.sh"

# Patch service to use installed wrapper path.
sed "s|%h/.local/share/rocm-tracker/bin/rocm-tracker-daily.sh|$BIN_DIR/rocm-tracker-daily.sh|g" \
  "$SCRIPT_DIR/systemd/rocm-tracker.service" >"$SYSTEMD_USER_DIR/rocm-tracker.service"
cp "$SCRIPT_DIR/systemd/rocm-tracker.timer" "$SYSTEMD_USER_DIR/rocm-tracker.timer"

systemctl --user daemon-reload
systemctl --user enable --now rocm-tracker.timer

echo "Installed systemd user timer: rocm-tracker.timer"
echo "Check status: systemctl --user status rocm-tracker.timer"
echo "Optional linger (run when logged out): loginctl enable-linger \"$USER\""

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

if $CRON_FALLBACK; then
  CRON_LINE="0 8 * * * TZ=Europe/Berlin $BIN_DIR/rocm-tracker-daily.sh"
  (crontab -l 2>/dev/null | grep -v rocm-tracker-daily.sh; echo "$CRON_LINE") | crontab -
  echo "Installed cron fallback entry."
fi

mkdir -p "$HOME/.config/rocm-tracker"
if [[ ! -f "$HOME/.config/rocm-tracker/env" ]]; then
  cp "$TOOL_DIR/config.example.env" "$HOME/.config/rocm-tracker/env"
  echo "Created $HOME/.config/rocm-tracker/env — edit ROCM_TRACKER_REPO_PATH."
fi

echo "Install complete."
