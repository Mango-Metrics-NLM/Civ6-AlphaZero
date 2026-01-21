"""Unit tests for the Strategy Advisor module.

Tests strategy advisor functionality including:
- Game state analysis
- Move explanations
- Training summaries
- Improvement suggestions
- Game reports
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


@dataclass
class MockStrategyAdvisorConfig:
    """Mock strategy advisor configuration."""

    enabled: bool = True
    analysis_interval: int = 100
    explain_moves: bool = False
    summarize_interval: int = 1000
    suggest_improvements: bool = True
    store_reports: bool = False
    max_context_length: int = 2000


class TestStrategyAdvisorInit:
    """Tests for StrategyAdvisor initialization."""

    def test_init_with_config(self):
        """Test initialization with config."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        llm_client = MagicMock()
        rag = MagicMock()
        config = MockStrategyAdvisorConfig()

        advisor = StrategyAdvisor(llm_client, rag, config)

        assert advisor.llm_client is llm_client
        assert advisor.rag is rag
        assert advisor.config is config

    def test_init_with_none_rag(self):
        """Test initialization with None RAG."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        llm_client = MagicMock()
        config = MockStrategyAdvisorConfig()

        advisor = StrategyAdvisor(llm_client, None, config)

        assert advisor.rag is None


class TestShouldAnalyze:
    """Tests for should_analyze method."""

    def test_returns_false_when_disabled(self):
        """Test returns False when advisor is disabled."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig(enabled=False)
        advisor = StrategyAdvisor(MagicMock(), None, config)

        assert not advisor.should_analyze(100)

    def test_returns_true_at_interval(self):
        """Test returns True at analysis interval."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig(analysis_interval=100)
        advisor = StrategyAdvisor(MagicMock(), None, config)

        assert advisor.should_analyze(100)
        assert advisor.should_analyze(200)

    def test_returns_false_before_interval(self):
        """Test returns False before interval is reached."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig(analysis_interval=100)
        advisor = StrategyAdvisor(MagicMock(), None, config)
        advisor._last_analysis_iteration = 50

        assert not advisor.should_analyze(100)


class TestAnalyzeGameState:
    """Tests for analyze_game_state method."""

    def test_returns_disabled_when_not_enabled(self):
        """Test returns disabled message when not enabled."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig(enabled=False)
        advisor = StrategyAdvisor(None, None, config)

        result = advisor.analyze_game_state({"turn": 50})

        assert "disabled" in result.summary.lower()

    def test_returns_analysis_with_llm(self):
        """Test returns analysis using LLM."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        llm_client = MagicMock()
        llm_client.complete.return_value = "Strong position with military advantage"

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(llm_client, None, config)

        # Use proper state format (lists for units/cities)
        state = {"turn": 50, "units": [1, 2, 3, 4, 5], "cities": [1, 2, 3], "score": 100}
        result = advisor.analyze_game_state(state)

        assert result.summary != ""
        llm_client.complete.assert_called_once()

    def test_uses_rag_context_when_available(self):
        """Test uses RAG context when available."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        llm_client = MagicMock()
        llm_client.complete.return_value = "Analysis with context"

        rag = MagicMock()
        rag.initialized = True
        rag.query_strategy.return_value = [
            MagicMock(content="Strategy tip 1"),
            MagicMock(content="Strategy tip 2"),
        ]

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(llm_client, rag, config)

        state = {"turn": 50}
        advisor.analyze_game_state(state)

        rag.query_strategy.assert_called_once()


class TestExplainMove:
    """Tests for explain_move method."""

    def test_returns_disabled_when_explain_moves_false(self):
        """Test returns disabled when explain_moves is False."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig(explain_moves=False)
        advisor = StrategyAdvisor(MagicMock(), None, config)

        result = advisor.explain_move({}, 5)

        assert "disabled" in result.reasoning.lower()

    def test_returns_explanation_with_llm(self):
        """Test returns explanation using LLM."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        llm_client = MagicMock()
        llm_client.complete.return_value = "This move expands territory"

        config = MockStrategyAdvisorConfig(explain_moves=True)
        advisor = StrategyAdvisor(llm_client, None, config)

        state = {"turn": 50}
        mcts_stats = {"visit_count": 100, "value": 0.7, "prior": 0.15}

        result = advisor.explain_move(state, 5, mcts_stats)

        assert result.move_description == "Action 5"
        assert result.mcts_confidence == 0.7
        llm_client.complete.assert_called_once()


class TestSummarizeTrainingProgress:
    """Tests for summarize_training_progress method."""

    def test_returns_unknown_when_disabled(self):
        """Test returns unknown trends when disabled."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig(enabled=False)
        advisor = StrategyAdvisor(None, None, config)

        metrics = [{"win_rate": 0.5, "loss": 0.1}]
        result = advisor.summarize_training_progress(metrics)

        assert result.win_rate_trend == "unknown"

    def test_returns_summary_with_llm(self):
        """Test returns summary using LLM."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        llm_client = MagicMock()
        llm_client.complete.return_value = "Training is improving. Key improvements: win rate up."

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(llm_client, None, config)

        metrics = [
            {"win_rate": 0.5, "loss": 0.1},
            {"win_rate": 0.55, "loss": 0.09},
            {"win_rate": 0.6, "loss": 0.08},
        ]

        result = advisor.summarize_training_progress(metrics, (0, 100))

        assert result.iteration_range == (0, 100)
        llm_client.complete.assert_called_once()


class TestSuggestImprovements:
    """Tests for suggest_improvements method."""

    def test_returns_empty_when_disabled(self):
        """Test returns empty list when suggestions disabled."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig(suggest_improvements=False)
        advisor = StrategyAdvisor(MagicMock(), None, config)

        suggestions = advisor.suggest_improvements([0.1, 0.09], {})

        assert suggestions == []

    def test_returns_suggestions_from_llm(self):
        """Test returns suggestions from LLM."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        llm_client = MagicMock()
        llm_client.complete.return_value = """
        1. Increase learning rate to 0.001
        2. Add more training data
        3. Try different batch sizes
        """

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(llm_client, None, config)

        loss_history = [0.1, 0.095, 0.09, 0.085]
        hyperparams = {"learning_rate": 0.0001, "batch_size": 64}

        suggestions = advisor.suggest_improvements(loss_history, hyperparams)

        assert len(suggestions) > 0
        llm_client.complete.assert_called_once()


class TestGenerateGameReport:
    """Tests for generate_game_report method."""

    def test_returns_basic_report_when_disabled(self):
        """Test returns basic report when disabled."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig(enabled=False)
        advisor = StrategyAdvisor(None, None, config)

        trajectory = [{"state": {}, "action": 1, "reward": 0.0}] * 10
        report = advisor.generate_game_report(trajectory, "win")

        assert report.outcome == "win"
        assert report.total_turns == 10

    def test_returns_full_report_with_llm(self):
        """Test returns full report using LLM."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        llm_client = MagicMock()
        llm_client.complete.return_value = """
        Strong early game with good expansion.
        Learning points: scout placement, timing attacks.
        Recommend: more aggressive expansion next time.
        """

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(llm_client, None, config)

        trajectory = [
            {"state": {}, "action": 1, "reward": 0.0},
            {"state": {}, "action": 2, "reward": 0.1},
            {"state": {}, "action": 3, "reward": 1.0},
        ]

        report = advisor.generate_game_report(trajectory, "win", "game_001")

        assert report.game_id == "game_001"
        assert report.outcome == "win"
        assert report.total_turns == 3
        assert report.strategy_assessment != ""


class TestAnalyzeTrends:
    """Tests for _analyze_trends method."""

    def test_insufficient_data(self):
        """Test returns insufficient data for short history."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(MagicMock(), None, config)

        trends = advisor._analyze_trends([{"win_rate": 0.5}])

        assert "insufficient" in trends.get("win_rate", "")

    def test_improving_win_rate(self):
        """Test detects improving win rate."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(MagicMock(), None, config)

        metrics = [
            {"win_rate": 0.5, "loss": 0.1},
            {"win_rate": 0.6, "loss": 0.08},
        ]

        trends = advisor._analyze_trends(metrics)

        assert trends["win_rate"] == "improving"

    def test_decreasing_loss(self):
        """Test detects decreasing loss."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(MagicMock(), None, config)

        metrics = [
            {"win_rate": 0.5, "loss": 0.1},
            {"win_rate": 0.55, "loss": 0.05},
        ]

        trends = advisor._analyze_trends(metrics)

        assert trends["loss"] == "decreasing"


class TestIdentifyKeyMoments:
    """Tests for _identify_key_moments method."""

    def test_identifies_significant_rewards(self):
        """Test identifies moments with significant rewards."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(MagicMock(), None, config)

        trajectory = [
            {"reward": 0.0},
            {"reward": 0.1},
            {"reward": 1.0},  # Significant
            {"reward": -0.8},  # Significant
        ]

        moments = advisor._identify_key_moments(trajectory)

        assert len(moments) == 2

    def test_limits_key_moments(self):
        """Test limits number of key moments."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(MagicMock(), None, config)

        trajectory = [{"reward": 1.0} for _ in range(20)]

        moments = advisor._identify_key_moments(trajectory)

        assert len(moments) <= 10


class TestMarkAnalysisDone:
    """Tests for mark_analysis_done method."""

    def test_updates_last_analysis_iteration(self):
        """Test updates last analysis iteration."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(MagicMock(), None, config)

        advisor.mark_analysis_done(500)

        assert advisor._last_analysis_iteration == 500


class TestGetStatistics:
    """Tests for get_statistics method."""

    def test_returns_statistics(self):
        """Test returns advisor statistics."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        rag = MagicMock()
        rag.initialized = True

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(MagicMock(), rag, config)
        advisor._analysis_count = 10

        stats = advisor.get_statistics()

        assert stats["enabled"] is True
        assert stats["analysis_count"] == 10
        assert stats["has_rag"] is True


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_contains_info(self):
        """Test __repr__ contains useful information."""
        from alpha_zero.orchestrator.strategy_advisor import StrategyAdvisor

        config = MockStrategyAdvisorConfig()
        advisor = StrategyAdvisor(MagicMock(), None, config)

        repr_str = repr(advisor)

        assert "StrategyAdvisor" in repr_str
        assert "enabled=" in repr_str


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_default_timestamp(self):
        """Test AnalysisResult gets default timestamp."""
        from alpha_zero.orchestrator.strategy_advisor import AnalysisResult

        result = AnalysisResult(summary="Test")

        assert result.timestamp != ""
        assert result.strengths == []
        assert result.weaknesses == []

    def test_custom_values(self):
        """Test AnalysisResult with custom values."""
        from alpha_zero.orchestrator.strategy_advisor import AnalysisResult

        result = AnalysisResult(
            summary="Strong position",
            strengths=["Good economy"],
            weaknesses=["Weak military"],
            suggestions=["Build more units"],
            confidence=0.85,
        )

        assert result.summary == "Strong position"
        assert len(result.strengths) == 1
        assert result.confidence == 0.85


class TestMoveExplanation:
    """Tests for MoveExplanation dataclass."""

    def test_default_values(self):
        """Test MoveExplanation default values."""
        from alpha_zero.orchestrator.strategy_advisor import MoveExplanation

        explanation = MoveExplanation(
            move_description="Action 5",
            reasoning="Expands territory",
        )

        assert explanation.alternatives == []
        assert explanation.mcts_confidence == 0.0


class TestGameReport:
    """Tests for GameReport dataclass."""

    def test_default_values(self):
        """Test GameReport default values."""
        from alpha_zero.orchestrator.strategy_advisor import GameReport

        report = GameReport(
            game_id="test_001",
            outcome="win",
            total_turns=50,
        )

        assert report.key_moments == []
        assert report.learning_points == []
        assert report.metrics == {}


class TestTrainingSummary:
    """Tests for TrainingSummary dataclass."""

    def test_default_values(self):
        """Test TrainingSummary default values."""
        from alpha_zero.orchestrator.strategy_advisor import TrainingSummary

        summary = TrainingSummary(
            iteration_range=(0, 100),
            win_rate_trend="improving",
            loss_trend="decreasing",
        )

        assert summary.key_improvements == []
        assert summary.concerns == []
        assert summary.next_steps == []
