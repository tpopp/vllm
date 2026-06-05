#!/usr/bin/env bash
# Optional systemd sleep hook: schedule RTC wake for next 08:00 Europe/Berlin on AC power.
# Install to /usr/lib/systemd/system-sleep/rocm-tracker-wake.sh (requires sudo).

set -euo pipefail

ACTION="${1:-}"
SUBSYSTEM="${2:-}"

on_ac_power() {
  for supply in /sys/class/power_supply/AC* /sys/class/power_supply/ADP*; do
    [[ -f "$supply/online" ]] || continue
    if [[ "$(cat "$supply/online")" == "1" ]]; then
      return 0
    fi
  done
  return 1
}

next_8am_berlin_epoch() {
  TZ=Europe/Berlin date -d "tomorrow 08:00" +%s
}

case "$ACTION/$SUBSYSTEM" in
  pre/suspend|pre/hibernate|pre/hybrid-sleep)
    if on_ac_power && command -v rtcwake >/dev/null 2>&1; then
      target="$(next_8am_berlin_epoch)"
      now="$(date +%s)"
      if (( target > now )); then
        rtcwake -m no -t "$target" || true
      fi
    fi
    ;;
esac

exit 0
