from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AnalysisResult:
    categories: list[str]
    is_breaking_api: bool
    summary: str
    model_impacts: list[dict[str, str]]
    action_hint: str | None
    backend: str
    model_used: str


class AnalyzerBackend(Protocol):
    def analyze(self, system_prompt: str, user_prompt: str) -> AnalysisResult: ...
