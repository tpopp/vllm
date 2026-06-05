from __future__ import annotations

from pathlib import Path

from rocm_tracker.commits import CommitInfo
from rocm_tracker.config import TrackerConfig
from rocm_tracker.github_meta import PullRequestMeta
from rocm_tracker.heuristics import HeuristicResult


def build_user_prompt(
    commit: CommitInfo,
    pr: PullRequestMeta,
    heuristic: HeuristicResult,
    candidate_impacts: list[dict[str, str]],
    diff_excerpt: str,
) -> str:
    comments = "\n".join(f"- {c}" for c in pr.comments) or "- (none)"
    impacts = "\n".join(
        f"- {i['architecture']}: {i.get('impact', 'possible')} ({i.get('rationale', '')})"
        for i in candidate_impacts[:30]
    ) or "- (none)"
    files = "\n".join(f"- {f}" for f in commit.changed_files[:80])

    return f"""Analyze this single upstream vLLM commit in isolation.

Commit: {commit.sha}
Date: {commit.date}
Author: {commit.author}
Subject: {commit.subject}

Pull request: {pr.url or 'unknown'}
PR title: {pr.title or 'unknown'}
PR body:
{pr.body or '(empty)'}

PR comments:
{comments}

Heuristic relevance: {heuristic.relevance} (score={heuristic.score:.1f})
Heuristic tags: {', '.join(heuristic.tags) or 'none'}
Suggested categories: {', '.join(heuristic.suggested_categories) or 'none'}

Changed files:
{files}

Candidate model impacts (verify conservatively):
{impacts}

Diff excerpt (ROCm-relevant paths):
{diff_excerpt or '(no diff)'}
"""


def write_prompt_package(
    config: TrackerConfig,
    commit_sha: str,
    user_prompt: str,
) -> Path:
    pkg_dir = config.prompts_dir / commit_sha
    pkg_dir.mkdir(parents=True, exist_ok=True)
    user_path = pkg_dir / "user_prompt.md"
    user_path.write_text(user_prompt, encoding="utf-8")
    system_path = config.system_prompt_path
    if system_path.is_file():
        (pkg_dir / "system_prompt.txt").write_text(
            system_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return user_path
