"""Callable target adapter wrapping :class:`StrategyAdvisor` for the eval harness.

The ``langfuse-eval-harness`` ``callable`` target imports this module's
:func:`advisor_target` via the config path
``alpha_zero.evaluation.llm_harness.target:advisor_target`` and calls it with the
dataset item's ``inputs`` dict (the default ``pass_item: false``). The function
runs the advisor and returns its result dataclass serialized to JSON, so the
harness ``json_keys`` / ``contains`` scorers can grade the *structure* of the
output.

Modes (selected by ``inputs["mode"]``):
    - ``"analyze"`` (default): :meth:`StrategyAdvisor.analyze_game_state` ->
      ``AnalysisResult`` (keys: summary, strengths, weaknesses, suggestions,
      confidence, timestamp).
    - ``"report"``: :meth:`StrategyAdvisor.generate_game_report` -> ``GameReport``
      (keys: game_id, outcome, total_turns, key_moments, strategy_assessment,
      learning_points, recommendations, metrics).

By default the advisor is driven by a **deterministic offline stub LLM** so the
eval needs no API keys and is fully reproducible in CI. This is a
contract/serialization regression gate — it exercises prompt-building,
response-parsing and JSON serialization, *not* LLM output quality. Set
``CIV6AZ_LLM_EVAL_ONLINE=1`` to drive a real provider via
:func:`alpha_zero.utils.llm_client.create_llm_client` for genuine quality grading.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

from alpha_zero.orchestrator.config import StrategyAdvisorConfig
from alpha_zero.orchestrator.strategy_advisor import (
    AnalysisResult,
    GameReport,
    StrategyAdvisor,
)

_ONLINE_ENV = "CIV6AZ_LLM_EVAL_ONLINE"
_TRUTHY = {"1", "true", "yes", "on"}


class _OfflineStubLLM:
    """Deterministic offline stand-in exposing the ``complete(prompt)`` API.

    The advisor's parsers extract list items whose lines contain keywords
    ("strength", "weakness", "suggest", "recommend", "learn"), so the canned
    response includes those keywords. Output is independent of the prompt, which
    keeps the offline gate fully reproducible.
    """

    def complete(self, prompt: str) -> str:
        """Return canned, parseable analysis text (ignores the prompt)."""
        return (
            "Summary: stable developing position with room to expand.\n"
            "Strengths:\n"
            "- Strength: solid early economy and production base\n"
            "Weaknesses:\n"
            "- Weakness: limited military and exposed borders\n"
            "Suggestions:\n"
            "- Suggestion: prioritise expansion, then pivot to science\n"
            "Recommendations:\n"
            "- Recommend: secure a second city and scout neighbours\n"
            "Learning points:\n"
            "- Learn: early settlers compound into a long-term lead\n"
        )


def _is_online() -> bool:
    """Return True if the eval should call a real LLM provider."""
    return os.getenv(_ONLINE_ENV, "").strip().lower() in _TRUTHY


def _build_advisor() -> StrategyAdvisor:
    """Construct a StrategyAdvisor with an offline stub or a real LLM client.

    Mirrors the direct construction in ``orchestrator.py`` (``StrategyAdvisor(
    llm_client, rag, config)``); the factory ``create_strategy_advisor`` is
    deliberately avoided because it returns a non-functional placeholder dict.
    """
    config = StrategyAdvisorConfig(enabled=True)
    llm_client: Any
    if _is_online():
        # Lazy import: only needed for the online path, and keeps the offline
        # gate free of the openai/anthropic dependencies.
        from alpha_zero.utils.llm_client import create_llm_client

        llm_client = create_llm_client()
    else:
        llm_client = _OfflineStubLLM()
    return StrategyAdvisor(llm_client=llm_client, rag=None, config=config)


def advisor_target(inputs: dict[str, Any]) -> str:
    """Eval-harness ``callable`` target: run StrategyAdvisor, return JSON.

    Args:
        inputs: The dataset item's ``inputs`` dict. Recognised keys:
            ``mode`` ("analyze" | "report", default "analyze"); for analyze:
            ``state`` (dict) and ``context`` (dict | None); for report:
            ``trajectory`` (list[dict]), ``outcome`` (str) and ``game_id``
            (str | None).

    Returns:
        A JSON string of the advisor result dataclass, with sorted keys for
        deterministic output.
    """
    advisor = _build_advisor()
    # `or <default>` (not `.get(key, default)`) so explicit JSON null values are
    # coerced to the default instead of crashing dict()/list()/str() with None.
    mode = str(inputs.get("mode") or "analyze").strip().lower()

    # Both branches return a dataclass (AnalysisResult / GameReport) handled
    # uniformly by dataclasses.asdict below.
    result: AnalysisResult | GameReport
    if mode == "report":
        result = advisor.generate_game_report(
            trajectory=list(inputs.get("trajectory") or []),
            outcome=str(inputs.get("outcome") or "unknown"),
            game_id=inputs.get("game_id"),
        )
    else:
        result = advisor.analyze_game_state(
            state=dict(inputs.get("state") or {}),
            context=inputs.get("context"),
        )

    return json.dumps(dataclasses.asdict(result), default=str, sort_keys=True)
