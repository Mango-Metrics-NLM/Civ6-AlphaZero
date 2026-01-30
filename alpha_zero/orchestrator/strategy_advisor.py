"""Strategy advisor using RAG-augmented LLM analysis.

This module provides LLM-based game analysis and strategy suggestions,
using RAG retrieval to provide relevant context for more accurate analysis.

Features:
- Game state analysis for debugging/monitoring
- Move explanation generation
- Training progress summarization
- Hyperparameter improvement suggestions
- Full game report generation

Usage:
    from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor
    from alpha_zero.orchestrator.rag_integration import RAGIntegration

    advisor = StrategyAdvisor(llm_client, rag_integration, config)

    # Analyze a game state
    analysis = advisor.analyze_game_state(state, context)

    # Generate training summary
    summary = advisor.summarize_training_progress(metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alpha_zero.orchestrator.config import StrategyAdvisorConfig
    from alpha_zero.orchestrator.rag_integration import RAGIntegration

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from game state analysis."""

    summary: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class MoveExplanation:
    """Explanation of a move decision."""

    move_description: str
    reasoning: str
    alternatives: list[str] = field(default_factory=list)
    mcts_confidence: float = 0.0
    strategic_alignment: str = ""


@dataclass
class GameReport:
    """Complete game analysis report."""

    game_id: str
    outcome: str
    total_turns: int
    key_moments: list[str] = field(default_factory=list)
    strategy_assessment: str = ""
    learning_points: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingSummary:
    """Summary of training progress."""

    iteration_range: tuple[int, int]
    win_rate_trend: str
    loss_trend: str
    key_improvements: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


class StrategyAdvisor:
    """LLM-based strategy advisor with RAG augmentation.

    Provides analysis and suggestions for game strategy and training,
    using retrieval-augmented generation for context-aware responses.

    Attributes:
        llm_client: LLM client for text generation.
        rag: RAG integration for context retrieval.
        config: Strategy advisor configuration.
    """

    def __init__(
        self,
        llm_client: Any,
        rag: RAGIntegration | None,
        config: StrategyAdvisorConfig,
    ):
        """Initialize strategy advisor.

        Args:
            llm_client: LLM client for text generation.
            rag: RAG integration for context retrieval (optional).
            config: Strategy advisor configuration.
        """
        self.llm_client = llm_client
        self.rag = rag
        self.config = config
        self._analysis_count = 0
        self._last_analysis_iteration = 0

    def should_analyze(self, iteration: int) -> bool:
        """Check if analysis should run at this iteration.

        Args:
            iteration: Current training iteration.

        Returns:
            True if analysis should run.
        """
        if not self.config.enabled:
            return False

        # Check analysis interval
        if iteration - self._last_analysis_iteration >= self.config.analysis_interval:
            return True

        return False

    def analyze_game_state(
        self,
        state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """Analyze a game state.

        Args:
            state: Game state representation.
            context: Additional context (turn, phase, etc.).

        Returns:
            Analysis result with insights.
        """
        if not self.config.enabled or not self.llm_client:
            return AnalysisResult(summary="Analysis disabled")

        try:
            # Get RAG context if available
            rag_context = ""
            if self.rag and self.rag.initialized:
                state_desc = self._describe_state(state)
                results = self.rag.query_strategy(state_desc, game_context=context)
                if results:
                    rag_context = "\n".join(f"- {r.content[:200]}" for r in results[:3])

            # Build prompt
            prompt = self._build_analysis_prompt(state, context, rag_context)

            # Get LLM response
            response = self._query_llm(prompt)

            # Parse response into AnalysisResult
            return self._parse_analysis_response(response)

        except Exception as e:
            logger.error(f"Game state analysis failed: {e}")
            return AnalysisResult(summary=f"Analysis failed: {e}")

    def explain_move(
        self,
        state: dict[str, Any],
        action: int,
        mcts_stats: dict[str, Any] | None = None,
    ) -> MoveExplanation:
        """Generate explanation for a move.

        Args:
            state: Game state before move.
            action: Action taken.
            mcts_stats: MCTS statistics for this move.

        Returns:
            Move explanation.
        """
        if not self.config.explain_moves or not self.llm_client:
            return MoveExplanation(
                move_description=f"Action {action}",
                reasoning="Move explanation disabled",
            )

        try:
            # Build prompt for move explanation
            prompt = self._build_move_prompt(state, action, mcts_stats)

            # Get LLM response
            response = self._query_llm(prompt)

            # Parse response
            return self._parse_move_response(response, action, mcts_stats)

        except Exception as e:
            logger.error(f"Move explanation failed: {e}")
            return MoveExplanation(
                move_description=f"Action {action}",
                reasoning=f"Explanation failed: {e}",
            )

    def summarize_training_progress(
        self,
        metrics_history: list[dict[str, float]],
        iteration_range: tuple[int, int] | None = None,
    ) -> TrainingSummary:
        """Summarize training progress.

        Args:
            metrics_history: List of metrics from recent iterations.
            iteration_range: Range of iterations covered.

        Returns:
            Training progress summary.
        """
        if not self.config.enabled or not self.llm_client:
            return TrainingSummary(
                iteration_range=iteration_range or (0, 0),
                win_rate_trend="unknown",
                loss_trend="unknown",
            )

        try:
            # Analyze metrics trends
            trends = self._analyze_trends(metrics_history)

            # Build prompt
            prompt = self._build_summary_prompt(metrics_history, trends)

            # Get LLM response
            response = self._query_llm(prompt)

            # Parse into TrainingSummary
            return self._parse_summary_response(
                response,
                iteration_range or (0, len(metrics_history)),
                trends,
            )

        except Exception as e:
            logger.error(f"Training summary failed: {e}")
            return TrainingSummary(
                iteration_range=iteration_range or (0, 0),
                win_rate_trend="error",
                loss_trend="error",
            )

    def suggest_improvements(
        self,
        loss_history: list[float],
        current_hyperparams: dict[str, Any],
    ) -> list[str]:
        """Suggest hyperparameter improvements.

        Args:
            loss_history: Recent loss values.
            current_hyperparams: Current hyperparameter values.

        Returns:
            List of improvement suggestions.
        """
        if not self.config.suggest_improvements or not self.llm_client:
            return []

        try:
            # Analyze loss patterns
            avg_loss = sum(loss_history) / max(len(loss_history), 1)
            loss_trend = "decreasing" if loss_history[-1] < loss_history[0] else "increasing"

            # Build prompt
            prompt = f"""Analyze training progress and suggest hyperparameter improvements.

Current Loss: {loss_history[-1]:.4f}
Average Loss: {avg_loss:.4f}
Trend: {loss_trend}
Recent History: {loss_history[-10:]}

Current Hyperparameters:
{self._format_hyperparams(current_hyperparams)}

Provide 3-5 specific, actionable suggestions for improving training.
"""

            # Get LLM response
            response = self._query_llm(prompt)

            # Extract suggestions
            return self._extract_suggestions(response)

        except Exception as e:
            logger.error(f"Improvement suggestions failed: {e}")
            return []

    def generate_game_report(
        self,
        trajectory: list[dict[str, Any]],
        outcome: str,
        game_id: str | None = None,
    ) -> GameReport:
        """Generate a complete game analysis report.

        Args:
            trajectory: List of (state, action, reward) tuples or dicts.
            outcome: Game outcome (win/loss/draw).
            game_id: Optional game identifier.

        Returns:
            Complete game report.
        """
        game_id = game_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        if not self.config.enabled or not self.llm_client:
            return GameReport(
                game_id=game_id,
                outcome=outcome,
                total_turns=len(trajectory),
            )

        try:
            # Analyze trajectory for key moments
            key_moments = self._identify_key_moments(trajectory)

            # Build comprehensive prompt
            prompt = self._build_report_prompt(trajectory, outcome, key_moments)

            # Get LLM response
            response = self._query_llm(prompt)

            # Parse into GameReport
            report = self._parse_report_response(response, game_id, outcome, len(trajectory))

            # Store report if configured
            if self.config.store_reports and self.rag:
                self.rag.store_game_summary(
                    summary=report.strategy_assessment,
                    outcome=outcome,
                    metrics=report.metrics,
                    iteration=self._last_analysis_iteration,
                )

            self._analysis_count += 1
            return report

        except Exception as e:
            logger.error(f"Game report generation failed: {e}")
            return GameReport(
                game_id=game_id,
                outcome=outcome,
                total_turns=len(trajectory),
                strategy_assessment=f"Report generation failed: {e}",
            )

    def _describe_state(self, state: dict[str, Any]) -> str:
        """Convert state dict to textual description."""
        parts = []
        if "turn" in state:
            parts.append(f"Turn {state['turn']}")
        if "units" in state:
            parts.append(f"{len(state.get('units', []))} units")
        if "cities" in state:
            parts.append(f"{len(state.get('cities', []))} cities")
        if "score" in state:
            parts.append(f"Score: {state['score']}")
        return ", ".join(parts) if parts else "Game state"

    def _build_analysis_prompt(
        self,
        state: dict[str, Any],
        context: dict[str, Any] | None,
        rag_context: str,
    ) -> str:
        """Build prompt for game state analysis."""
        prompt = "Analyze the following game state:\n\n"

        # Add state information
        prompt += f"State: {self._describe_state(state)}\n"

        if context:
            prompt += f"Context: {context}\n"

        if rag_context:
            prompt += f"\nRelevant strategies from knowledge base:\n{rag_context}\n"

        prompt += """
Provide:
1. Brief summary of the position
2. Key strengths
3. Key weaknesses
4. Strategic suggestions

Keep the response concise and actionable.
"""
        return prompt

    def _build_move_prompt(
        self,
        state: dict[str, Any],
        action: int,
        mcts_stats: dict[str, Any] | None,
    ) -> str:
        """Build prompt for move explanation."""
        prompt = f"Explain why action {action} was chosen.\n\n"
        prompt += f"State: {self._describe_state(state)}\n"

        if mcts_stats:
            prompt += "MCTS Statistics:\n"
            prompt += f"- Visit count: {mcts_stats.get('visit_count', 'N/A')}\n"
            prompt += f"- Value: {mcts_stats.get('value', 'N/A')}\n"
            prompt += f"- Prior: {mcts_stats.get('prior', 'N/A')}\n"

        prompt += "\nExplain the reasoning behind this move choice."
        return prompt

    def _build_summary_prompt(
        self,
        metrics_history: list[dict[str, float]],
        trends: dict[str, str],
    ) -> str:
        """Build prompt for training summary."""
        prompt = "Summarize training progress:\n\n"
        prompt += f"Iterations analyzed: {len(metrics_history)}\n"
        prompt += f"Trends: {trends}\n"

        if metrics_history:
            latest = metrics_history[-1]
            prompt += f"Latest metrics: {latest}\n"

        prompt += "\nProvide a concise summary with key improvements and concerns."
        return prompt

    def _build_report_prompt(
        self,
        trajectory: list[dict[str, Any]],
        outcome: str,
        key_moments: list[str],
    ) -> str:
        """Build prompt for game report."""
        prompt = "Generate a game analysis report.\n\n"
        prompt += f"Outcome: {outcome}\n"
        prompt += f"Total turns: {len(trajectory)}\n"
        prompt += "Key moments:\n"
        for moment in key_moments[:5]:
            prompt += f"- {moment}\n"
        prompt += "\nProvide strategic assessment and learning points."
        return prompt

    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with a prompt."""
        if not self.llm_client:
            return ""

        try:
            # Truncate context if needed
            max_context = self.config.max_context_length
            if len(prompt) > max_context:
                prompt = prompt[:max_context] + "..."

            # Query LLM
            response = self.llm_client.complete(prompt)
            return response if isinstance(response, str) else str(response)

        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return ""

    def _parse_analysis_response(self, response: str) -> AnalysisResult:
        """Parse LLM response into AnalysisResult."""
        if not response:
            return AnalysisResult(summary="No response")

        # Simple parsing - extract sections
        lines = response.strip().split("\n")

        return AnalysisResult(
            summary=lines[0] if lines else "Analysis complete",
            strengths=self._extract_list_items(response, "strength"),
            weaknesses=self._extract_list_items(response, "weakness"),
            suggestions=self._extract_list_items(response, "suggest"),
            confidence=0.7,
        )

    def _parse_move_response(
        self,
        response: str,
        action: int,
        mcts_stats: dict[str, Any] | None,
    ) -> MoveExplanation:
        """Parse LLM response into MoveExplanation."""
        return MoveExplanation(
            move_description=f"Action {action}",
            reasoning=response[:500] if response else "No explanation available",
            mcts_confidence=mcts_stats.get("value", 0.0) if mcts_stats else 0.0,
        )

    def _parse_summary_response(
        self,
        response: str,
        iteration_range: tuple[int, int],
        trends: dict[str, str],
    ) -> TrainingSummary:
        """Parse LLM response into TrainingSummary."""
        return TrainingSummary(
            iteration_range=iteration_range,
            win_rate_trend=trends.get("win_rate", "unknown"),
            loss_trend=trends.get("loss", "unknown"),
            key_improvements=self._extract_list_items(response, "improve"),
            concerns=self._extract_list_items(response, "concern"),
            next_steps=self._extract_list_items(response, "next"),
        )

    def _parse_report_response(
        self,
        response: str,
        game_id: str,
        outcome: str,
        total_turns: int,
    ) -> GameReport:
        """Parse LLM response into GameReport."""
        return GameReport(
            game_id=game_id,
            outcome=outcome,
            total_turns=total_turns,
            strategy_assessment=response[:1000] if response else "No assessment",
            learning_points=self._extract_list_items(response, "learn"),
            recommendations=self._extract_list_items(response, "recommend"),
        )

    def _analyze_trends(
        self,
        metrics_history: list[dict[str, float]],
    ) -> dict[str, str]:
        """Analyze trends in metrics history."""
        if not metrics_history or len(metrics_history) < 2:
            return {"win_rate": "insufficient data", "loss": "insufficient data"}

        trends = {}

        # Analyze win rate trend
        if "win_rate" in metrics_history[0]:
            first_wr = metrics_history[0].get("win_rate", 0)
            last_wr = metrics_history[-1].get("win_rate", 0)
            if last_wr > first_wr + 0.05:
                trends["win_rate"] = "improving"
            elif last_wr < first_wr - 0.05:
                trends["win_rate"] = "declining"
            else:
                trends["win_rate"] = "stable"

        # Analyze loss trend
        if "loss" in metrics_history[0]:
            first_loss = metrics_history[0].get("loss", 0)
            last_loss = metrics_history[-1].get("loss", 0)
            if last_loss < first_loss * 0.9:
                trends["loss"] = "decreasing"
            elif last_loss > first_loss * 1.1:
                trends["loss"] = "increasing"
            else:
                trends["loss"] = "stable"

        return trends

    def _identify_key_moments(
        self,
        trajectory: list[dict[str, Any]],
    ) -> list[str]:
        """Identify key moments in game trajectory."""
        key_moments = []

        for i, step in enumerate(trajectory):
            reward = step.get("reward", 0)
            if abs(reward) > 0.5:
                key_moments.append(f"Turn {i}: significant reward ({reward:.2f})")

        return key_moments[:10]  # Limit to 10 key moments

    def _extract_list_items(self, text: str, keyword: str) -> list[str]:
        """Extract list items containing a keyword."""
        items = []
        for line in text.split("\n"):
            line = line.strip()
            if keyword.lower() in line.lower() and line:
                # Clean up bullet points and numbering
                clean_line = line.lstrip("-•*0123456789.) ")
                if clean_line:
                    items.append(clean_line)
        return items[:5]  # Limit to 5 items

    def _extract_suggestions(self, text: str) -> list[str]:
        """Extract suggestions from LLM response."""
        suggestions = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith(("-", "•", "*")) or (line and line[0].isdigit()):
                clean = line.lstrip("-•*0123456789.) ")
                if clean and len(clean) > 10:
                    suggestions.append(clean)
        return suggestions[:5]

    def _format_hyperparams(self, params: dict[str, Any]) -> str:
        """Format hyperparameters for display."""
        return "\n".join(f"  {k}: {v}" for k, v in params.items())

    def mark_analysis_done(self, iteration: int) -> None:
        """Mark that analysis was done at this iteration."""
        self._last_analysis_iteration = iteration

    def get_statistics(self) -> dict[str, Any]:
        """Get advisor statistics."""
        return {
            "enabled": self.config.enabled,
            "analysis_count": self._analysis_count,
            "last_analysis_iteration": self._last_analysis_iteration,
            "analysis_interval": self.config.analysis_interval,
            "explain_moves": self.config.explain_moves,
            "has_rag": self.rag is not None and self.rag.initialized,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StrategyAdvisor("
            f"enabled={self.config.enabled}, "
            f"analyses={self._analysis_count}, "
            f"has_rag={self.rag is not None})"
        )
