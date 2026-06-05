from rocm_tracker.backends.cursor_cli import _extract_json_from_agent_output
from rocm_tracker.summary_util import truncate_summary


def test_truncate_summary_keeps_short_text():
    text = "One sentence."
    assert truncate_summary(text) == text


def test_truncate_summary_truncates_long_text():
    text = ". ".join([f"Sentence {i}" for i in range(12)]) + "."
    result = truncate_summary(text, max_sentences=8)
    assert result.count("Sentence") == 8


def test_long_summary_from_agent_is_accepted_via_analyze_path():
    inner = {
        "categories": ["perf_immediate"],
        "is_breaking_api": False,
        "summary": ". ".join([f"S{i}" for i in range(10)]) + ".",
        "model_impacts": [],
        "action_hint": "test",
    }
    import json

    payload = _extract_json_from_agent_output(json.dumps(inner))
    summary = truncate_summary(payload["summary"])
    assert summary.count("S") <= 8
