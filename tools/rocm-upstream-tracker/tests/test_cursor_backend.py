import json

from rocm_tracker.backends.cursor_cli import (
    _extract_json,
    _extract_json_from_agent_output,
    _validate_summary,
    resolve_agent_bin,
)


def test_extract_json_from_fenced_block():
    text = """Here is the result:
```json
{"categories": ["perf_immediate"], "is_breaking_api": false, "summary": "One sentence.", "model_impacts": [], "action_hint": "none"}
```"""
    payload = _extract_json(text)
    assert payload["categories"] == ["perf_immediate"]


def test_extract_json_from_agent_envelope():
    inner = {
        "categories": ["api_breaking"],
        "is_breaking_api": True,
        "summary": "Breaking change.",
        "model_impacts": [],
        "action_hint": "test",
    }
    envelope = {"result": json.dumps(inner)}
    payload = _extract_json_from_agent_output(json.dumps(envelope))
    assert payload["is_breaking_api"] is True


def test_validate_summary_rejects_too_many_sentences():
    summary = ". ".join([f"Sentence {i}" for i in range(9)]) + "."
    try:
        _validate_summary(summary)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_resolve_agent_bin_prefers_agent():
    # When agent is on PATH in CI, should resolve to it
    resolved = resolve_agent_bin("cursor")
    assert resolved  # non-empty
