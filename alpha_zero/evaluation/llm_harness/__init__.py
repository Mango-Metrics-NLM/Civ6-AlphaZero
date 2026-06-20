"""Adapter exposing Civ6-AlphaZero LLM components to ``langfuse-eval-harness``.

The harness (github.com/ianshank/Agents, distribution ``langfuse-eval-harness``)
drives a ``callable`` target by importing a module-level function via a
``"module:function"`` path. :func:`advisor_target` is that function: it runs the
:class:`~alpha_zero.orchestrator.strategy_advisor.StrategyAdvisor` and returns its
result serialized to JSON so the harness scorers can validate the structure.

This package has **no import-time dependency** on ``eval_harness`` — the harness
imports *us*, not the other way around — so :func:`advisor_target` is callable
(and unit-testable) whether or not the optional ``[eval]`` extra is installed.
"""

from alpha_zero.evaluation.llm_harness.target import advisor_target

__all__ = ["advisor_target"]
