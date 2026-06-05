# ROCm Upstream Tracker

Local laptop scheduler and database for tracking upstream `vllm-project/vllm` changes relevant to ROCm/AMD work.

## What it does

1. Rebases your fork (`tpopp/vllm` by default) onto upstream `main`
2. Collects commits since the last successful watermark
3. Scores changed files for ROCm/NVIDIA relevance
4. Runs **Sonnet via Cursor CLI** with **fresh context per commit** (high/medium relevance only)
5. Stores results in SQLite for querying by model, category, or breaking-change flag

## Setup

```bash
# From your vLLM repo root
cd /path/to/vllm
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e tools/rocm-upstream-tracker

mkdir -p ~/.config/rocm-tracker
cp tools/rocm-upstream-tracker/config.example.env ~/.config/rocm-tracker/env
# Edit ROCM_TRACKER_REPO_PATH to your local fork clone
```

Ensure:

- `upstream` remote points to `https://github.com/vllm-project/vllm.git`
- `gh` CLI is authenticated
- **Cursor Agent CLI** installed and authenticated in WSL:

```bash
curl https://cursor.com/install -fsS | bash
agent login
# or set CURSOR_API_KEY in ~/.config/rocm-tracker/env
agent -p "Reply with JSON: {\"ok\": true}"
```

Set `ROCM_TRACKER_CURSOR_BIN=agent` (not `cursor` — the headless CLI is the `agent` command).

## Install laptop scheduler (systemd)

```bash
chmod +x tools/rocm-upstream-tracker/scripts/*.sh
tools/rocm-upstream-tracker/scripts/install-scheduler.sh
```

Optional:

```bash
# Wake from suspend at 08:00 when on AC (best-effort, needs sudo)
sudo tools/rocm-upstream-tracker/scripts/install-scheduler.sh --enable-wake-on-ac

# Plain cron fallback in addition to systemd
tools/rocm-upstream-tracker/scripts/install-scheduler.sh --cron-fallback
```

The systemd user timer uses `Persistent=true`:

- **Powered off at 08:00** → runs on next boot
- **Asleep at 08:00** → runs on resume
- **Idempotent guard** → skips if already succeeded today (Europe/Berlin)

Check timer:

```bash
systemctl --user status rocm-tracker.timer
systemctl --user list-timers rocm-tracker.timer
```

## Manual commands

```bash
rocm-tracker daily              # sync + analyze (scheduled job)
rocm-tracker daily -v           # verbose progress on stderr
rocm-tracker daily --dry-run    # no push
rocm-tracker sync
rocm-tracker analyze --commit <sha>
rocm-tracker query --model LlamaForCausalLM --breaking
rocm-tracker query --category perf_immediate --since 14d
rocm-tracker export --format jsonl --model DeepseekV2ForCausalLM
```

## Visibility / debugging

By default, scheduled runs via `rocm-tracker-daily.sh` write **only to a log file** (no terminal output):

```bash
tail -f ~/.local/share/rocm-tracker/logs/daily-$(date +%Y-%m-%d).log
```

For interactive testing, use verbose mode:

```bash
rocm-tracker daily -v --dry-run --force
```

Or enable verbose in the scheduler wrapper:

```bash
export ROCM_TRACKER_VERBOSE=1
~/.local/share/rocm-tracker/bin/rocm-tracker-daily.sh
```

Common silent outcomes:

- `skipped: already ran successfully today` — use `--force`
- `Found 0 new commit(s)` — watermark already at upstream tip
- Scheduler wrapper redirects stdout/stderr to the daily log file

## Data location

```
~/.local/share/rocm-tracker/
  rocm_tracker.db
  state.json
  logs/
  pending_analysis.jsonl
```

## Categories

- `api_breaking` — API/CLI/config changes to retest
- `perf_immediate` — ROCm-usable performance wins now
- `perf_with_work` — wins needing small ROCm porting
- `nvidia_replicate` — NVIDIA-only changes worth AMD follow-up

## AI assistance

This tool was implemented with AI assistance. Review scheduler install paths and Cursor CLI flags on your machine before relying on unattended runs.
