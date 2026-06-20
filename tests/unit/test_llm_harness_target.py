"""Unit tests for the langfuse-eval-harness callable target adapter.

Covers the offline stub, online/offline mode selection, and an importorskip-guarded
end-to-end run of the harness CLI (skipped when the ``[eval]`` extra is absent).
"""

import json
import subprocess
import sys

import pytest

from alpha_zero.evaluation.llm_harness import advisor_target
from alpha_zero.evaluation.llm_harness import target as target_mod

ANALYSIS_KEYS = {"summary", "strengths", "weaknesses", "suggestions", "confidence", "timestamp"}


class TestOfflineStubLLM:
    def test_complete_contains_parser_keywords(self):
        text = target_mod._OfflineStubLLM().complete("any prompt")
        low = text.lower()
        for keyword in ("strength", "weakness", "suggest", "recommend", "learn"):
            assert keyword in low

    def test_complete_ignores_prompt(self):
        stub = target_mod._OfflineStubLLM()
        assert stub.complete("prompt A") == stub.complete("totally different prompt B")


class TestModeSelection:
    def test_is_online_truthy_values(self, monkeypatch):
        for value in ("1", "true", "YES", "On"):
            monkeypatch.setenv("CIV6AZ_LLM_EVAL_ONLINE", value)
            assert target_mod._is_online() is True

    def test_is_online_falsy_values(self, monkeypatch):
        for value in ("0", "false", "", "no", "off"):
            monkeypatch.setenv("CIV6AZ_LLM_EVAL_ONLINE", value)
            assert target_mod._is_online() is False

    def test_is_online_unset(self, monkeypatch):
        monkeypatch.delenv("CIV6AZ_LLM_EVAL_ONLINE", raising=False)
        assert target_mod._is_online() is False


class TestAdvisorTarget:
    def test_offline_does_not_build_real_client(self, monkeypatch):
        """The offline path must never construct a real (API-key) LLM client."""
        monkeypatch.delenv("CIV6AZ_LLM_EVAL_ONLINE", raising=False)
        import alpha_zero.utils.llm_client as llm_mod

        def boom(*args, **kwargs):
            raise AssertionError("offline path must not build a real LLM client")

        monkeypatch.setattr(llm_mod, "create_llm_client", boom)

        data = json.loads(advisor_target({"mode": "analyze", "state": {"turn": 3}}))
        assert set(data) == ANALYSIS_KEYS

    def test_online_uses_create_llm_client(self, monkeypatch):
        """Online mode drives the real client factory and parses its output."""
        monkeypatch.setenv("CIV6AZ_LLM_EVAL_ONLINE", "1")
        calls = {"n": 0}

        class FakeClient:
            def complete(self, prompt):
                calls["n"] += 1
                return "Online summary line\nSuggestion: expand quickly\n"

        import alpha_zero.utils.llm_client as llm_mod

        monkeypatch.setattr(llm_mod, "create_llm_client", lambda *a, **k: FakeClient())

        data = json.loads(advisor_target({"mode": "analyze", "state": {"turn": 3}}))
        assert calls["n"] == 1
        assert data["summary"] == "Online summary line"
        assert data["suggestions"]


@pytest.mark.e2e
def test_eval_harness_end_to_end(tmp_path, monkeypatch):
    """Run the full harness gate via the runner when the extra is installed."""
    pytest.importorskip("eval_harness")
    monkeypatch.setenv("CIV6AZ_EVAL_OUT", str(tmp_path))
    monkeypatch.delenv("CIV6AZ_LLM_EVAL_ONLINE", raising=False)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_llm_eval.py",
            "--config",
            "configs/eval/strategy_advisor.yaml",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
