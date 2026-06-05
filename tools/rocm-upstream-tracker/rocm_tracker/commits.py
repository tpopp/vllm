from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    date: str
    author: str
    subject: str
    changed_files: list[str]


def _run_git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_upstream_sha(repo_path: Path, upstream_remote: str = "upstream") -> str:
    return _run_git(repo_path, "rev-parse", f"{upstream_remote}/main")


def list_commits_between(
    repo_path: Path,
    start_sha: str | None,
    end_sha: str,
) -> list[CommitInfo]:
    if start_sha and start_sha == end_sha:
        return []
    range_spec = f"{start_sha}..{end_sha}" if start_sha else end_sha
    log = _run_git(
        repo_path,
        "log",
        "--reverse",
        "--pretty=format:%H%x1f%ad%x1f%an%x1f%s",
        "--date=iso-strict",
        range_spec,
    )
    if not log:
        return []
    commits: list[CommitInfo] = []
    for line in log.splitlines():
        sha, date_s, author, subject = line.split("\x1f", 3)
        files_raw = _run_git(
            repo_path,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            sha,
        )
        files = [f for f in files_raw.splitlines() if f]
        commits.append(
            CommitInfo(
                sha=sha,
                date=date_s,
                author=author,
                subject=subject,
                changed_files=files,
            )
        )
    return commits


def get_commit_info(repo_path: Path, sha: str) -> CommitInfo:
    log = _run_git(
        repo_path,
        "log",
        "-1",
        "--pretty=format:%H%x1f%ad%x1f%an%x1f%s",
        "--date=iso-strict",
        sha,
    )
    commit_sha, date_s, author, subject = log.split("\x1f", 3)
    files_raw = _run_git(
        repo_path,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit_sha,
    )
    files = [f for f in files_raw.splitlines() if f]
    return CommitInfo(
        sha=commit_sha,
        date=date_s,
        author=author,
        subject=subject,
        changed_files=files,
    )


def commit_diff_for_paths(
    repo_path: Path,
    sha: str,
    paths: list[str],
    max_chars: int = 32000,
) -> str:
    if not paths:
        paths = ["."]
    args = ["show", "--format=", sha, "--", *paths]
    try:
        diff = _run_git(repo_path, *args)
    except subprocess.CalledProcessError:
        return ""
    if len(diff) > max_chars:
        return diff[:max_chars] + "\n... [truncated]"
    return diff
