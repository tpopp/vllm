from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from rocm_tracker.commits import get_upstream_sha
from rocm_tracker.config import TrackerConfig


UPSTREAM_URL = "https://github.com/vllm-project/vllm.git"


@dataclass(frozen=True)
class SyncResult:
    success: bool
    upstream_sha_before: str | None
    upstream_sha_after: str | None
    fork_sha: str | None
    message: str


def _run(repo_path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def ensure_upstream_remote(repo_path: Path) -> None:
    remotes = _run(repo_path, "remote").stdout.splitlines()
    if "upstream" not in remotes:
        _run(repo_path, "remote", "add", "upstream", UPSTREAM_URL)


def sync_fork(
    config: TrackerConfig,
    *,
    dry_run: bool = False,
) -> SyncResult:
    repo = config.repo_path
    if not repo.is_dir():
        return SyncResult(False, None, None, None, f"Repo not found: {repo}")

    ensure_upstream_remote(repo)
    _run(repo, "fetch", "upstream", "main")
    _run(repo, "fetch", config.fork_remote, "main")

    try:
        upstream_before = get_upstream_sha(repo, "upstream")
    except subprocess.CalledProcessError as exc:
        return SyncResult(False, None, None, None, exc.stderr or str(exc))

    if dry_run:
        fork_sha = _run(repo, "rev-parse", "main").stdout.strip()
        return SyncResult(
            True,
            upstream_before,
            upstream_before,
            fork_sha,
            "dry-run: sync skipped",
        )

    _run(repo, "checkout", "main")
    rebase = _run(repo, "rebase", "upstream/main", check=False)
    if rebase.returncode != 0:
        _run(repo, "rebase", "--abort", check=False)
        return SyncResult(
            False,
            upstream_before,
            None,
            None,
            rebase.stderr or rebase.stdout or "rebase failed",
        )

    push = _run(
        repo,
        "push",
        config.fork_remote,
        "main",
        check=False,
    )
    if push.returncode != 0:
        return SyncResult(
            False,
            upstream_before,
            get_upstream_sha(repo, "upstream"),
            _run(repo, "rev-parse", "HEAD").stdout.strip(),
            push.stderr or push.stdout or "push failed",
        )

    upstream_after = get_upstream_sha(repo, "upstream")
    fork_sha = _run(repo, "rev-parse", "HEAD").stdout.strip()
    return SyncResult(
        True,
        upstream_before,
        upstream_after,
        fork_sha,
        "sync ok",
    )
