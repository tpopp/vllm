from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class TrackerConfig:
    repo_path: Path
    data_dir: Path
    upstream_repo: str
    fork_remote: str
    model: str
    max_llm_commits_per_run: int
    timezone: str
    cursor_bin: str
    deep_model: str
    triage_max_items: int

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rocm_tracker.db"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def pending_path(self) -> Path:
        return self.data_dir / "pending_analysis.jsonl"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def prompts_dir(self) -> Path:
        return self.data_dir / "prompts"

    @property
    def registry_path(self) -> Path:
        return self.repo_path / "vllm/model_executor/models/registry.py"

    @property
    def system_prompt_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "prompts" / "analyze_change.txt"

    @property
    def triage_prompt_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "prompts" / "triage_rocm.txt"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


def load_config() -> TrackerConfig:
    env_file = Path(
        os.environ.get(
            "ROCM_TRACKER_ENV_FILE",
            Path.home() / ".config/rocm-tracker/env",
        )
    )
    _load_env_file(env_file)

    repo_path = Path(
        os.environ.get("ROCM_TRACKER_REPO_PATH", Path.home() / "src/vllm")
    ).expanduser()
    data_dir = Path(
        os.environ.get(
            "ROCM_TRACKER_DATA_DIR",
            Path.home() / ".local/share/rocm-tracker",
        )
    ).expanduser()

    return TrackerConfig(
        repo_path=repo_path,
        data_dir=data_dir,
        upstream_repo=os.environ.get("ROCM_TRACKER_UPSTREAM_REPO", "vllm-project/vllm"),
        fork_remote=os.environ.get("ROCM_TRACKER_FORK_REMOTE", "origin"),
        model=os.environ.get("ROCM_TRACKER_MODEL", "sonnet"),
        max_llm_commits_per_run=int(
            os.environ.get("ROCM_TRACKER_MAX_LLM_COMMITS_PER_RUN", "20")
        ),
        timezone=os.environ.get("ROCM_TRACKER_TIMEZONE", "Europe/Berlin"),
        cursor_bin=os.environ.get("ROCM_TRACKER_CURSOR_BIN", "agent"),
        deep_model=os.environ.get("ROCM_TRACKER_DEEP_MODEL", "opus"),
        triage_max_items=int(os.environ.get("ROCM_TRACKER_TRIAGE_MAX_ITEMS", "30")),
    )


def ensure_data_dirs(config: TrackerConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    config.prompts_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
