from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from rocm_tracker.backends.base import AnalysisResult
from rocm_tracker.summary_util import truncate_summary


def resolve_agent_bin(configured: str) -> str:
    """Resolve agent executable: prefer standalone `agent` CLI over `cursor agent`."""
    if configured and configured != "cursor":
        return configured
    agent_path = shutil.which("agent")
    if agent_path:
        return agent_path
    cursor_path = shutil.which("cursor")
    if cursor_path:
        return cursor_path
    return configured or "agent"


class CursorCliBackend:
    def __init__(
        self,
        *,
        cursor_bin: str,
        model: str,
        workspace: Path,
        timeout_seconds: int = 600,
    ) -> None:
        self.agent_bin = resolve_agent_bin(cursor_bin)
        self.use_cursor_subcommand = (
            Path(self.agent_bin).name == "cursor"
            or self.agent_bin.endswith("Cursor.exe")
        )
        self.model = model
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds

    def run_json_prompt(self, system_prompt: str, user_prompt: str) -> dict:
        output = self._run_prompt(system_prompt, user_prompt)
        return _extract_json_from_agent_output(output)

    def analyze(self, system_prompt: str, user_prompt: str) -> AnalysisResult:
        payload = self.run_json_prompt(system_prompt, user_prompt)
        summary = truncate_summary(payload.get("summary", "").strip())
        return AnalysisResult(
            categories=list(payload.get("categories") or []),
            is_breaking_api=bool(payload.get("is_breaking_api")),
            summary=summary,
            model_impacts=list(payload.get("model_impacts") or []),
            action_hint=payload.get("action_hint"),
            backend="agent_cli",
            model_used=self.model,
        )

    def _run_prompt(self, system_prompt: str, user_prompt: str) -> str:
        prompt = (
            f"{system_prompt.strip()}\n\n"
            f"---\n\n"
            f"{user_prompt.strip()}\n\n"
            f"Respond with JSON only."
        )
        if self.use_cursor_subcommand:
            cmd = [
                self.agent_bin,
                "agent",
                "--model",
                self.model,
                "--print",
                "--workspace",
                str(self.workspace),
                prompt,
            ]
        else:
            cmd = [
                self.agent_bin,
                "-p",
                "--output-format",
                "json",
                "--mode",
                "ask",
                "--model",
                self.model,
                "--workspace",
                str(self.workspace),
                prompt,
            ]
        env = os.environ.copy()
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=self.workspace,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Cursor agent CLI not found: {self.agent_bin}. "
                "Install with: curl https://cursor.com/install -fsS | bash "
                "Then run: agent login (or set CURSOR_API_KEY)."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or "(no output before timeout)"
            raise RuntimeError(
                f"Cursor agent timed out after {self.timeout_seconds}s. "
                f"Last output: {detail[:500]}"
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or f"exit code {result.returncode}"
            if "Authentication required" in detail or "agent login" in detail:
                raise RuntimeError(
                    "Cursor agent authentication required. "
                    "Run `agent login` in WSL, or set CURSOR_API_KEY in "
                    "~/.config/rocm-tracker/env"
                )
            raise RuntimeError(
                f"Cursor agent failed ({result.returncode}): {detail[:800]}"
            )

        return (result.stdout or result.stderr).strip()


def _extract_json_from_agent_output(text: str) -> dict:
    # agent --output-format json wraps result; try direct parse first.
    try:
        envelope = json.loads(text)
        if isinstance(envelope, dict):
            if "result" in envelope and isinstance(envelope["result"], str):
                return _extract_json(envelope["result"])
            if "categories" in envelope:
                return envelope
    except json.JSONDecodeError:
        pass
    return _extract_json(text)


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON found in model output: {text[:500]}")

