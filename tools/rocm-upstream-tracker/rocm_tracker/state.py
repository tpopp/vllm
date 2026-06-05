from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rocm_tracker.config import TrackerConfig


@dataclass
class TrackerState:
    last_upstream_sha: str | None = None
    last_sync_at: str | None = None
    last_fork_main_sha: str | None = None
    last_run_status: str | None = None
    last_successful_run_local_date: str | None = None

    @classmethod
    def load(cls, path: Path) -> TrackerState:
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            last_upstream_sha=data.get("last_upstream_sha"),
            last_sync_at=data.get("last_sync_at"),
            last_fork_main_sha=data.get("last_fork_main_sha"),
            last_run_status=data.get("last_run_status"),
            last_successful_run_local_date=data.get(
                "last_successful_run_local_date"
            ),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2) + "\n",
            encoding="utf-8",
        )


def local_today(config: TrackerConfig) -> str:
    tz = ZoneInfo(config.timezone)
    return datetime.now(tz).date().isoformat()


def already_ran_today(config: TrackerConfig, state: TrackerState) -> bool:
    return (
        state.last_successful_run_local_date == local_today(config)
        and state.last_run_status == "success"
    )


def mark_success(config: TrackerConfig, state: TrackerState) -> None:
    tz = ZoneInfo(config.timezone)
    now = datetime.now(tz).isoformat()
    state.last_sync_at = now
    state.last_run_status = "success"
    state.last_successful_run_local_date = local_today(config)
    state.save(config.state_path)


def mark_failure(config: TrackerConfig, state: TrackerState) -> None:
    state.last_run_status = "failure"
    state.save(config.state_path)
