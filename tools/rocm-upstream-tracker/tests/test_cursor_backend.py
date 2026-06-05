import json

from rocm_tracker.backends.cursor_cli import _extract_json, _validate_summary


def test_extract_json_from_fenced_block():
    text = """Here is the result:
```json
{"categories": ["perf_immediate"], "is_breaking_api": false, "summary": "One sentence.", "model_impacts": [], "action_hint": "none"}
```"""
    payload = _extract_json(text)
    assert payload["categories"] == ["perf_immediate"]


def test_validate_summary_rejects_too_many_sentences():
    summary = ". ".join([f"Sentence {i}" for i in range(9)]) + "."
    try:
        _validate_summary(summary)
        raised = False
    except ValueError:
        raised = True
    assert raised
