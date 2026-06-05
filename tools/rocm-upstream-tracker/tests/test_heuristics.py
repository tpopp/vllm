from rocm_tracker.heuristics import score_changed_files, should_call_llm


def test_rocm_paths_score_high():
    result = score_changed_files(
        ["vllm/platforms/rocm.py", "csrc/rocm/attention.cu"],
        subject="ROCm attention fix",
    )
    assert result.relevance == "high"
    assert should_call_llm(result.relevance)


def test_docs_only_is_none():
    result = score_changed_files(["docs/models/supported_models.md"])
    assert result.relevance == "none"
    assert not should_call_llm(result.relevance)
