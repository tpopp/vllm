from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PullRequestMeta:
    number: int | None
    url: str | None
    title: str | None
    body: str | None
    comments: list[str]


def _gh_api(args: list[str]) -> Any | None:
    try:
        result = subprocess.run(
            ["gh", "api", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def fetch_pr_for_commit(
    upstream_repo: str,
    commit_sha: str,
    max_comments: int = 8,
) -> PullRequestMeta:
    pulls = _gh_api(
        [
            f"repos/{upstream_repo}/commits/{commit_sha}/pulls",
            "-H",
            "Accept: application/vnd.github.groot-preview+json",
        ]
    )
    if not pulls:
        return PullRequestMeta(None, None, None, None, [])

    pr = pulls[0]
    number = pr.get("number")
    url = pr.get("html_url")
    title = pr.get("title")
    body = pr.get("body") or ""

    comments: list[str] = []
    if number is not None:
        issue_comments = _gh_api(
            [
                f"repos/{upstream_repo}/issues/{number}/comments",
                "-f",
                "per_page=20",
            ]
        )
        if isinstance(issue_comments, list):
            for item in issue_comments[:max_comments]:
                user = (item.get("user") or {}).get("login", "unknown")
                text = (item.get("body") or "").strip()
                if text:
                    comments.append(f"{user}: {text[:500]}")

    return PullRequestMeta(number, url, title, body, comments)
