from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rocm_tracker.analyzer import analyze_commits
from rocm_tracker.commits import get_upstream_sha, list_commits_between
from rocm_tracker.config import TrackerConfig, ensure_data_dirs, load_config
from rocm_tracker.db import Database
from rocm_tracker.state import (
    TrackerState,
    already_ran_today,
    mark_failure,
    mark_success,
)
from rocm_tracker.logutil import Logger
from rocm_tracker.sync import sync_fork
from rocm_tracker.triage import run_triage


def _logger(args: argparse.Namespace) -> Logger:
    return Logger(verbose=getattr(args, "verbose", False), stream=sys.stderr)


def _schema_path() -> Path:
    return Path(__file__).resolve().parent.parent / "schema.sql"


def _parse_since_days(value: str | None) -> int | None:
    if not value:
        return None
    if value.endswith("d"):
        return int(value[:-1])
    return int(value)


def cmd_daily(args: argparse.Namespace) -> int:
    log = _logger(args)
    config = load_config()
    ensure_data_dirs(config)
    state = TrackerState.load(config.state_path)

    log.debug(f"repo={config.repo_path}")
    log.debug(f"data_dir={config.data_dir}")
    log.debug(f"state={config.state_path}")
    log.debug(f"last_upstream_sha={state.last_upstream_sha}")

    if not args.force and already_ran_today(config, state):
        log.info("skipped: already ran successfully today (use --force to override)")
        return 0

    db = Database(config.db_path, _schema_path())
    try:
        log.step("Syncing fork with upstream")
        if args.dry_run:
            log.debug("dry-run: rebase and push will be skipped")
        sync = sync_fork(config, dry_run=args.dry_run)
        if not sync.success:
            mark_failure(config, state)
            log.info(sync.message)
            return 1
        log.info(sync.message)
        log.debug(f"upstream_before={sync.upstream_sha_before}")
        log.debug(f"upstream_after={sync.upstream_sha_after}")
        log.debug(f"fork_sha={sync.fork_sha}")

        upstream_after = sync.upstream_sha_after
        if upstream_after is None:
            mark_failure(config, state)
            log.info("missing upstream sha after sync")
            return 1

        start_sha = state.last_upstream_sha
        commits = list_commits_between(config.repo_path, start_sha, upstream_after)
        log.step(f"Found {len(commits)} new commit(s) since watermark")
        run_id = db.start_sync_run(sync.upstream_sha_before, upstream_after)

        analyzed, llm_calls = analyze_commits(
            config,
            db,
            commits,
            run_id,
            force_reanalyze=args.force_reanalyze,
            log=log,
        )

        db.finish_sync_run(
            run_id,
            commits_count=len(commits),
            fork_push_success=not args.dry_run,
            status="success",
            message=f"analyzed={analyzed} llm_calls={llm_calls}",
        )

        state.last_upstream_sha = upstream_after
        state.last_fork_main_sha = sync.fork_sha
        mark_success(config, state)
        log.info(
            f"daily ok: commits={len(commits)} analyzed={analyzed} "
            f"llm_calls={llm_calls} upstream={upstream_after[:12]}"
        )
        log.debug(f"db={config.db_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        mark_failure(config, state)
        log.info(f"daily failed: {exc}")
        if log.verbose:
            raise
        return 1
    finally:
        db.close()


def cmd_sync(args: argparse.Namespace) -> int:
    log = _logger(args)
    config = load_config()
    log.step("Syncing fork")
    result = sync_fork(config, dry_run=args.dry_run)
    log.info(result.message)
    return 0 if result.success else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    log = _logger(args)
    config = load_config()
    ensure_data_dirs(config)
    db = Database(config.db_path, _schema_path())
    state = TrackerState.load(config.state_path)
    try:
        end_sha = args.commit or get_upstream_sha(config.repo_path, "upstream")
        start_sha = state.last_upstream_sha if not args.commit else None
        if args.commit:
            from rocm_tracker.commits import get_commit_info

            commits = [get_commit_info(config.repo_path, args.commit)]
        else:
            commits = list_commits_between(config.repo_path, start_sha, end_sha)

        run_id = db.start_sync_run(start_sha, end_sha)
        analyzed, llm_calls = analyze_commits(
            config,
            db,
            commits,
            run_id,
            force_reanalyze=args.force_reanalyze,
            log=log,
        )
        db.finish_sync_run(
            run_id,
            commits_count=len(commits),
            fork_push_success=True,
            status="success",
            message=f"manual analyze analyzed={analyzed}",
        )
        log.info(f"analyze ok: commits={len(commits)} analyzed={analyzed} llm={llm_calls}")
        return 0
    finally:
        db.close()


def cmd_query(args: argparse.Namespace) -> int:
    config = load_config()
    db = Database(config.db_path, _schema_path())
    try:
        rows = db.query_changes(
            model=args.model,
            category=args.category,
            breaking=args.breaking,
            since_days=_parse_since_days(args.since),
        )
        for row in rows:
            print(
                f"{row['commit_date']} {row['commit_sha'][:12]} "
                f"breaking={bool(row['is_breaking_api'])} "
                f"{row['summary']}"
            )
            if row.get("pr_url"):
                print(f"  PR: {row['pr_url']}")
        return 0
    finally:
        db.close()


def cmd_export(args: argparse.Namespace) -> int:
    config = load_config()
    db = Database(config.db_path, _schema_path())
    try:
        rows = db.export_jsonl(
            model=args.model,
            since_days=_parse_since_days(args.since),
        )
        if args.format == "jsonl":
            for row in rows:
                print(json.dumps(row, ensure_ascii=True))
        else:
            print(json.dumps(rows, indent=2, ensure_ascii=True))
        return 0
    finally:
        db.close()


def cmd_triage(args: argparse.Namespace) -> int:
    log = _logger(args)
    config = load_config()
    ensure_data_dirs(config)
    db = Database(config.db_path, _schema_path())
    try:
        report, json_path, md_path = run_triage(
            config,
            db,
            focus_model=args.model,
            since_days=_parse_since_days(args.since),
            category=args.category,
            deep_model=args.deep_model,
            limit=args.limit,
            log=log,
        )
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            print((md_path).read_text(encoding="utf-8"))
        log.debug(f"report_json={json_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        log.info(f"triage failed: {exc}")
        if log.verbose:
            raise
        return 1
    finally:
        db.close()


def cmd_if_missed_today(args: argparse.Namespace) -> int:
    """Catch-up helper for login/cron-fallback paths."""
    config = load_config()
    state = TrackerState.load(config.state_path)
    if already_ran_today(config, state):
        print("skipped: already ran today")
        return 0
    if not hasattr(args, "dry_run"):
        args.dry_run = False
    if not hasattr(args, "force"):
        args.force = False
    if not hasattr(args, "force_reanalyze"):
        args.force_reanalyze = False
    return cmd_daily(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocm-tracker")
    sub = parser.add_subparsers(dest="command", required=True)

    daily = sub.add_parser("daily", help="Sync fork and analyze new commits")
    daily.add_argument("--dry-run", action="store_true")
    daily.add_argument("--force", action="store_true", help="Run even if already succeeded today")
    daily.add_argument("--force-reanalyze", action="store_true")
    daily.add_argument("-v", "--verbose", action="store_true", help="Progress and debug output on stderr")
    daily.set_defaults(func=cmd_daily, verbose=False)

    sync = sub.add_parser("sync", help="Sync fork only")
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("-v", "--verbose", action="store_true")
    sync.set_defaults(func=cmd_sync, verbose=False)

    analyze = sub.add_parser("analyze", help="Analyze commits")
    analyze.add_argument("--commit")
    analyze.add_argument("--force-reanalyze", action="store_true")
    analyze.add_argument("-v", "--verbose", action="store_true")
    analyze.set_defaults(func=cmd_analyze, verbose=False)

    query = sub.add_parser("query", help="Query database")
    query.add_argument("--model")
    query.add_argument("--category")
    query.add_argument("--breaking", action="store_true")
    query.add_argument("--since")
    query.set_defaults(func=cmd_query)

    export = sub.add_parser("export", help="Export records")
    export.add_argument("--model")
    export.add_argument("--since")
    export.add_argument("--format", choices=["jsonl", "json"], default="jsonl")
    export.set_defaults(func=cmd_export)

    triage = sub.add_parser(
        "triage",
        help="Deep ROCm triage: prioritize changes and propose evaluation plans",
    )
    triage.add_argument("--model", help="Focus architecture, e.g. Qwen3NextForCausalLM")
    triage.add_argument("--since", help="Limit to last N days, e.g. 30d")
    triage.add_argument("--category")
    triage.add_argument(
        "--deep-model",
        help="Capable model for triage (default: ROCM_TRACKER_DEEP_MODEL / opus)",
    )
    triage.add_argument("--limit", type=int, help="Max changes to triage (default 30)")
    triage.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format to stdout",
    )
    triage.add_argument("-v", "--verbose", action="store_true")
    triage.set_defaults(func=cmd_triage, verbose=False)

    missed = sub.add_parser("if-missed-today", help="Run daily only if not yet successful today")
    missed.add_argument("-v", "--verbose", action="store_true")
    missed.set_defaults(func=cmd_if_missed_today, verbose=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
