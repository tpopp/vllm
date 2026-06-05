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
- `cursor` CLI is on PATH with Sonnet enabled

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
rocm-tracker daily --dry-run    # no push
rocm-tracker sync
rocm-tracker analyze --commit <sha>
rocm-tracker query --model LlamaForCausalLM --breaking
rocm-tracker query --category perf_immediate --since 14d
rocm-tracker export --format jsonl --model DeepseekV2ForCausalLM
```

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
