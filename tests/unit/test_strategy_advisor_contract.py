"""Dependency-free contract gate for StrategyAdvisor's serialized output.

This is the core regression gate for the LLM eval integration: it drives the
``advisor_target`` adapter offline (deterministic stub LLM, no API keys, no
``langfuse-eval-harness`` dependency) and asserts the JSON structure the harness
``json_keys`` / ``contains`` scorers rely on. It stays meaningful even when the
optional ``[eval]`` extra cannot be installed.

It also pins the contract that ``configs/eval/strategy_advisor.yaml`` encodes:
the ``json_keys`` ``required`` list must be a subset of the analysis output keys.
"""

import json

import pytest

from alpha_zero.evaluation.llm_harness.target import advisor_target

ANALYSIS_KEYS = {"summary", "strengths", "weaknesses", "suggestions", "confidence", "timestamp"}
REPORT_KEYS = {
    "game_id",
    "outcome",
    "total_turns",
    "key_moments",
    "strategy_assessment",
    "learning_points",
    "recommendations",
    "metrics",
}
# Mirrors scorers[0].params.required in configs/eval/strategy_advisor.yaml.
CONFIG_REQUIRED_KEYS = [
    "summary",
    "strengths",
    "weaknesses",
    "suggestions",
    "confidence",
    "timestamp",
]


@pytest.fixture(autouse=True)
def _force_offline(monkeypatch):
    """Guarantee the offline stub path regardless of ambient env."""
    monkeypatch.delenv("CIV6AZ_LLM_EVAL_ONLINE", raising=False)


def test_analyze_mode_emits_all_analysis_keys():
    out = advisor_target({"mode": "analyze", "state": {"turn": 5}, "context": None})
    data = json.loads(out)
    assert set(data) == ANALYSIS_KEYS
    assert data["summary"]
    assert data["strengths"]
    assert data["suggestions"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_default_mode_is_analyze():
    data = json.loads(advisor_target({"state": {}}))
    assert set(data) == ANALYSIS_KEYS


def test_output_contains_suggestions_substring():
    # The has_suggestions `contains` scorer matches "suggestion" case-insensitively.
    out = advisor_target({"mode": "analyze", "state": {"turn": 1}})
    assert "suggestion" in out.lower()


def test_report_mode_emits_report_keys():
    out = advisor_target(
        {
            "mode": "report",
            "trajectory": [{"reward": 1.0}, {"reward": 0.0}],
            "outcome": "win",
            "game_id": "g1",
        }
    )
    data = json.loads(out)
    assert set(data) == REPORT_KEYS
    assert data["outcome"] == "win"
    assert data["total_turns"] == 2
    assert data["game_id"] == "g1"


def test_output_is_deterministic_offline():
    # Everything except the wall-clock `timestamp` field is deterministic, which
    # is what the json_keys / contains scorers depend on.
    inputs = {"mode": "analyze", "state": {"turn": 7}, "context": {"phase": "mid"}}
    first = json.loads(advisor_target(inputs))
    second = json.loads(advisor_target(inputs))
    first.pop("timestamp")
    second.pop("timestamp")
    assert first == second


@pytest.mark.parametrize("state", [{}, {"turn": 1}, {"units": [], "cities": []}])
def test_sparse_states_are_graceful(state):
    data = json.loads(advisor_target({"mode": "analyze", "state": state}))
    assert set(data) == ANALYSIS_KEYS


def test_explicit_null_inputs_do_not_crash():
    # JSON null values present for known keys must coerce to defaults, not raise
    # TypeError on dict()/list()/str(None).
    analyze = json.loads(advisor_target({"mode": None, "state": None, "context": None}))
    assert set(analyze) == ANALYSIS_KEYS

    report = json.loads(
        advisor_target({"mode": "report", "trajectory": None, "outcome": None, "game_id": None})
    )
    assert set(report) == REPORT_KEYS
    assert report["outcome"] == "unknown"
    assert report["total_turns"] == 0


def test_config_required_keys_are_satisfied():
    """The keys the harness gate requires must all be present in the output."""
    data = json.loads(advisor_target({"state": {"turn": 1}}))
    assert all(key in data for key in CONFIG_REQUIRED_KEYS)
