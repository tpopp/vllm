from __future__ import annotations

import re
from dataclasses import dataclass


ROCM_PATTERNS = [
    re.compile(r"^csrc/rocm/"),
    re.compile(r"^docker/Dockerfile\.rocm"),
    re.compile(r"^requirements/rocm.*\.txt$"),
    re.compile(r"^vllm/model_executor/layers/fused_moe/rocm.*\.py$"),
    re.compile(r"^vllm/v1/attention/backends/rocm.*\.py$"),
    re.compile(r"^vllm/v1/attention/backends/mla/rocm.*\.py$"),
    re.compile(r"^vllm/v1/attention/ops/rocm.*\.py$"),
    re.compile(r"^tests/kernels/.*_rocm.*\.py$"),
    re.compile(r"^vllm/platforms/rocm\.py$"),
    re.compile(r"rocm_aiter"),
    re.compile(r"VLLM_ROCM"),
    re.compile(r"vllm/_aiter_ops\.py"),
]

NVIDIA_PATTERNS = [
    re.compile(r"cutlass"),
    re.compile(r"marlin"),
    re.compile(r"flashinfer"),
    re.compile(r"deep_gemm"),
    re.compile(r"nvfp4"),
    re.compile(r"^vllm/platforms/cuda\.py$"),
    re.compile(r"^csrc/quantization/machete/"),
    re.compile(r"^csrc/quantization/marlin/"),
]

SHARED_PATTERNS = [
    re.compile(r"current_platform"),
    re.compile(r"is_rocm\("),
    re.compile(r"is_cuda\("),
    re.compile(r"kernels/linear"),
    re.compile(r"^vllm/entrypoints/"),
    re.compile(r"^vllm/config/"),
    re.compile(r"^vllm/model_executor/models/"),
    re.compile(r"^vllm/model_executor/models/registry\.py$"),
]

DOCS_ONLY = re.compile(r"^(docs/|examples/|.*\.md$)")


@dataclass(frozen=True)
class HeuristicResult:
    score: float
    relevance: str
    tags: list[str]
    suggested_categories: list[str]


def score_changed_files(files: list[str], subject: str = "") -> HeuristicResult:
    if not files:
        return HeuristicResult(0.0, "none", [], [])

    rocm_hits = 0
    nvidia_hits = 0
    shared_hits = 0
    docs_hits = 0
    tags: list[str] = []
    categories: list[str] = []

    text = subject.lower()
    if "rocm" in text or "amd" in text:
        rocm_hits += 2
        tags.append("title-rocm")

    for path in files:
        if any(p.search(path) for p in ROCM_PATTERNS):
            rocm_hits += 2
            tags.append("rocm-path")
        if any(p.search(path) for p in NVIDIA_PATTERNS):
            nvidia_hits += 2
            tags.append("nvidia-path")
        if any(p.search(path) for p in SHARED_PATTERNS):
            shared_hits += 1
            tags.append("shared-path")
        if DOCS_ONLY.search(path):
            docs_hits += 1

    if nvidia_hits:
        categories.append("nvidia_replicate")
    if rocm_hits:
        categories.extend(["perf_immediate", "perf_with_work"])
    if shared_hits and "vllm/entrypoints/" in " ".join(files):
        categories.append("api_breaking")
    if shared_hits and "vllm/config/" in " ".join(files):
        categories.append("api_breaking")

    score = rocm_hits * 2.0 + shared_hits * 1.0 + nvidia_hits * 1.5
    if docs_hits == len(files):
        return HeuristicResult(0.0, "none", tags, categories)

    if rocm_hits >= 2 or (rocm_hits >= 1 and shared_hits >= 1):
        relevance = "high"
    elif rocm_hits >= 1 or shared_hits >= 2 or nvidia_hits >= 2:
        relevance = "medium"
    elif shared_hits >= 1 or nvidia_hits >= 1:
        relevance = "low"
    else:
        relevance = "none"

    return HeuristicResult(score, relevance, tags, list(dict.fromkeys(categories)))


def should_call_llm(relevance: str) -> bool:
    return relevance in {"high", "medium"}
