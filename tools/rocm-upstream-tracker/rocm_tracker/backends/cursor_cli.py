from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from rocm_tracker.backends.base import AnalysisResult


class CursorCliBackend:
    def __init__(
        self,
        *,
        cursor_bin: str,
        model: str,
        workspace: Path,
        timeout_seconds: int = 600,
    ) -> None:
        self.cursor_bin = cursor_bin
        self.model = model
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds

    def analyze(self, system_prompt: str, user_prompt: str) -> AnalysisResult:
        prompt = (
            f"{system_prompt.strip()}\n\n"
            f"---\n\n"
            f"{user_prompt.strip()}\n\n"
            f"Respond with JSON only."
        )
        cmd = [
            self.cursor_bin,
            "agent",
            "--model",
            self.model,
            "--print",
            "--workspace",
            str(self.workspace),
            prompt,
        ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=self.workspace,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Cursor CLI not found: {self.cursor_bin}. "
                "Install Cursor and ensure `cursor` is on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Cursor agent timed out") from exc

        output = (result.stdout or result.stderr).strip()
        payload = _extract_json(output)
        summary = payload.get("summary", "").strip()
        _validate_summary(summary)
        return AnalysisResult(
            categories=list(payload.get("categories") or []),
            is_breaking_api=bool(payload.get("is_breaking_api")),
            summary=summary,
            model_impacts=list(payload.get("model_impacts") or []),
            action_hint=payload.get("action_hint"),
            backend="cursor_cli",
            model_used=self.model,
        )


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON found in model output: {text[:500]}")


def _validate_summary(summary: str) -> None:
    sentences = [s.strip() for s in re.split(r"[.!?]+", summary) if s.strip()]
    if len(sentences) > 8:
        raise ValueError(f"Summary exceeds 8 sentences ({len(sentences)})")
