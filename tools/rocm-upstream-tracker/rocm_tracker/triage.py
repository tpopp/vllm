from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rocm_tracker.backends.cursor_cli import CursorCliBackend
from rocm_tracker.config import TrackerConfig
from rocm_tracker.db import Database
from rocm_tracker.logutil import Logger


def _build_triage_user_prompt(
    *,
    focus_model: str | None,
    changes: list[dict[str, Any]],
) -> str:
    lines = [
        "Triage these upstream vLLM changes for ROCm backend work.",
        f"Focus model / use-case: {focus_model or 'general ROCm serving'}",
        f"Change count: {len(changes)}",
        "",
        "Changes:",
    ]
    for index, change in enumerate(changes, start=1):
        categories = change.get("categories") or []
        impacts = change.get("model_impacts") or []
        impact_txt = ", ".join(
            f"{i.get('architecture')}:{i.get('impact_level')}"
            for i in impacts[:5]
        )
        lines.extend(
            [
                f"--- Change {index} ---",
                f"commit_sha: {change.get('commit_sha')}",
                f"date: {change.get('commit_date')}",
                f"pr_url: {change.get('pr_url')}",
                f"pr_title: {change.get('pr_title') or change.get('commit_subject')}",
                f"breaking_api: {bool(change.get('is_breaking_api'))}",
                f"categories: {', '.join(categories)}",
                f"rocm_relevance: {change.get('rocm_relevance')}",
                f"model_impacts: {impact_txt or 'none'}",
                f"summary: {change.get('summary')}",
                f"action_hint: {change.get('action_hint') or 'none'}",
                f"changed_files: {', '.join((change.get('changed_files') or [])[:12])}",
                "",
            ]
        )
    return "\n".join(lines)


def _slug(value: str | None) -> str:
    if not value:
        return "general"
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()[:40]


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ROCm triage report",
        "",
        f"**Focus:** {report.get('focus_model', 'general')}",
        f"**Generated:** {report.get('generated_at', '')}",
        f"**Deep model:** {report.get('deep_model', '')}",
        "",
        "## Overall",
        "",
        report.get("overall_summary", ""),
        "",
        "## Prioritized changes",
        "",
    ]
    priority_order = {"implement": 0, "evaluate": 1, "investigate": 2, "ignore": 3}
    items = list(report.get("items") or [])
    items.sort(key=lambda x: priority_order.get(x.get("priority", "ignore"), 9))

    for item in items:
        lines.extend(
            [
                f"### [{item.get('priority', '?').upper()}] "
                f"{(item.get('commit_sha') or '')[:12]}",
                "",
                f"- **PR:** {item.get('pr_url', 'n/a')}",
                f"- **ROCm benefit:** {item.get('rocm_benefit', '')}",
                f"- **Evaluation:** {item.get('evaluation_plan', '')}",
                f"- **vLLM serve:** {item.get('vllm_serve_notes', '')}",
                f"- **Rationale:** {item.get('rationale', '')}",
                "",
            ]
        )

    lines.extend(["## Next steps", ""])
    for step in report.get("next_steps") or []:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def run_triage(
    config: TrackerConfig,
    db: Database,
    *,
    focus_model: str | None = None,
    since_days: int | None = None,
    category: str | None = None,
    deep_model: str | None = None,
    limit: int | None = None,
    log: Logger | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    log = log or Logger()
    max_items = limit or config.triage_max_items
    model_name = deep_model or config.deep_model

    changes = db.export_jsonl(model=focus_model, since_days=since_days)
    if category:
        changes = [c for c in changes if category in (c.get("categories") or [])]

    if not changes:
        raise RuntimeError("No changes matched filters for triage")

    changes = changes[:max_items]
    log.step(f"Triage: {len(changes)} change(s), deep model={model_name}")

    system_prompt = config.triage_prompt_path.read_text(encoding="utf-8")
    user_prompt = _build_triage_user_prompt(focus_model=focus_model, changes=changes)

    backend = CursorCliBackend(
        cursor_bin=config.cursor_bin,
        model=model_name,
        workspace=config.repo_path,
        timeout_seconds=900,
    )
    log.info("Running deep ROCm triage via agent (fresh context)")
    payload = backend.run_json_prompt(system_prompt, user_prompt)

    report: dict[str, Any] = {
        "focus_model": payload.get("focus_model") or focus_model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deep_model": model_name,
        "input_count": len(changes),
        "overall_summary": payload.get("overall_summary", ""),
        "items": payload.get("items") or [],
        "next_steps": payload.get("next_steps") or [],
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = _slug(focus_model)
    json_path = config.reports_dir / f"triage-{slug}-{stamp}.json"
    md_path = config.reports_dir / f"triage-{slug}-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    log.info(f"Wrote {json_path}")
    log.info(f"Wrote {md_path}")
    return report, json_path, md_path
