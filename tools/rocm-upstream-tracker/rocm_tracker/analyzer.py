from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rocm_tracker.backends.base import AnalysisResult
from rocm_tracker.backends.cursor_cli import CursorCliBackend
from rocm_tracker.commits import CommitInfo, commit_diff_for_paths
from rocm_tracker.config import TrackerConfig
from rocm_tracker.db import ChangeRecord, Database
from rocm_tracker.github_meta import fetch_pr_for_commit
from rocm_tracker.heuristics import score_changed_files, should_call_llm
from rocm_tracker.model_resolver import impacts_from_files, load_model_maps
from rocm_tracker.logutil import Logger
from rocm_tracker.prompt_builder import build_user_prompt, write_prompt_package


def _relevant_diff_paths(files: list[str]) -> list[str]:
    keywords = (
        "rocm",
        "aiter",
        "cuda",
        "platform",
        "attention",
        "fused_moe",
        "quantization",
        "kernels",
        "entrypoints",
        "config",
        "registry.py",
        "models/",
    )
    selected = [f for f in files if any(k in f for k in keywords)]
    return selected or files[:20]


def _heuristic_record(
    commit: CommitInfo,
    pr_meta,
    heuristic,
    impacts: list[dict[str, str]],
) -> ChangeRecord:
    summary = (
        f"{commit.subject}. "
        f"Relevance={heuristic.relevance}. "
        f"Files={len(commit.changed_files)}."
    )
    return ChangeRecord(
        commit_sha=commit.sha,
        commit_date=commit.date,
        author=commit.author,
        commit_subject=commit.subject,
        pr_number=pr_meta.number,
        pr_url=pr_meta.url,
        pr_title=pr_meta.title,
        summary=summary,
        is_breaking_api=False,
        categories=heuristic.suggested_categories,
        rocm_relevance=heuristic.relevance,
        changed_files=commit.changed_files,
        heuristic_score=heuristic.score,
        action_hint=None,
        model_used=None,
        analysis_backend="heuristic",
        analysis_cached_at=datetime.now(timezone.utc).isoformat(),
        model_impacts=impacts,
    )


def analyze_commits(
    config: TrackerConfig,
    db: Database,
    commits: list[CommitInfo],
    run_id: int | None,
    *,
    force_reanalyze: bool = False,
    log: Logger | None = None,
) -> tuple[int, int]:
    maps = load_model_maps(config.registry_path)
    backend = CursorCliBackend(
        cursor_bin=config.cursor_bin,
        model=config.model,
        workspace=config.repo_path,
    )
    system_prompt = config.system_prompt_path.read_text(encoding="utf-8")

    log = log or Logger()
    analyzed = 0
    llm_calls = 0
    skipped = 0
    pending: list[str] = []

    log.step(f"Analyzing {len(commits)} commit(s)")
    for index, commit in enumerate(commits, start=1):
        if db.has_commit(commit.sha) and not force_reanalyze:
            skipped += 1
            log.debug(f"[{index}/{len(commits)}] skip cached {commit.sha[:12]} {commit.subject}")
            continue

        log.info(f"[{index}/{len(commits)}] {commit.sha[:12]} {commit.subject}")
        heuristic = score_changed_files(commit.changed_files, commit.subject)
        log.debug(
            f"  relevance={heuristic.relevance} score={heuristic.score:.1f} "
            f"files={len(commit.changed_files)}"
        )
        impacts = impacts_from_files(maps, commit.changed_files)
        pr_meta = fetch_pr_for_commit(config.upstream_repo, commit.sha)

        if should_call_llm(heuristic.relevance) and llm_calls < config.max_llm_commits_per_run:
            log.debug(f"  calling Sonnet via Cursor CLI ({llm_calls + 1}/{config.max_llm_commits_per_run})")
            diff_paths = _relevant_diff_paths(commit.changed_files)
            diff_excerpt = commit_diff_for_paths(
                config.repo_path,
                commit.sha,
                diff_paths,
            )
            user_prompt = build_user_prompt(
                commit,
                pr_meta,
                heuristic,
                impacts,
                diff_excerpt,
            )
            write_prompt_package(config, commit.sha, user_prompt)
            try:
                result: AnalysisResult = backend.analyze(system_prompt, user_prompt)
            except Exception as exc:  # noqa: BLE001
                log.info(f"  LLM failed: {exc}")
                record = _heuristic_record(commit, pr_meta, heuristic, impacts)
                record.summary = (
                    f"{record.summary} LLM failed: {exc}"
                )
                db.upsert_change(run_id, record)
                analyzed += 1
                continue

            record = ChangeRecord(
                commit_sha=commit.sha,
                commit_date=commit.date,
                author=commit.author,
                commit_subject=commit.subject,
                pr_number=pr_meta.number,
                pr_url=pr_meta.url,
                pr_title=pr_meta.title,
                summary=result.summary,
                is_breaking_api=result.is_breaking_api,
                categories=result.categories or heuristic.suggested_categories,
                rocm_relevance=heuristic.relevance,
                changed_files=commit.changed_files,
                heuristic_score=heuristic.score,
                action_hint=result.action_hint,
                model_used=result.model_used,
                analysis_backend=result.backend,
                analysis_cached_at=datetime.now(timezone.utc).isoformat(),
                model_impacts=result.model_impacts or impacts,
            )
            db.upsert_change(run_id, record)
            analyzed += 1
            llm_calls += 1
            log.debug(f"  stored LLM analysis breaking={result.is_breaking_api}")
        else:
            if should_call_llm(heuristic.relevance):
                pending.append(commit.sha)
                log.debug("  queued for later (LLM budget exhausted)")
            else:
                log.debug("  stored heuristic-only analysis")
            record = _heuristic_record(commit, pr_meta, heuristic, impacts)
            db.upsert_change(run_id, record)
            analyzed += 1

    log.step(f"Analysis done: analyzed={analyzed} skipped={skipped} llm_calls={llm_calls}")
    if pending:
        config.pending_path.write_text(
            "\n".join(pending) + "\n",
            encoding="utf-8",
        )
    elif config.pending_path.is_file():
        config.pending_path.unlink()

    return analyzed, llm_calls
