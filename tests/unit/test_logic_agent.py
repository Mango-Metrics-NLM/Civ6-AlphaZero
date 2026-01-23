"""Tests for LogicAgent strategic reasoning system."""

import pytest
from unittest.mock import MagicMock, patch

from alpha_zero.agents import (
    LogicAgent,
    LogicAgentConfig,
    GameContext,
    StrategicDecision,
    UncertaintyReport,
)
from alpha_zero.agents.logic_agent import (
    GamePhase,
    ThreatLevel,
    UtilityPriority,
    StrategicObservation,
)


class TestLogicAgentConfig:
    """Tests for LogicAgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LogicAgentConfig()

        assert config.confidence_threshold == 0.85
        assert config.sacrifice_threshold == 0.3
        assert config.observation_memory_size == 100
        assert config.pattern_match_threshold == 0.7
        assert config.enable_uncertainty_disclosure is True
        assert config.enable_sacrifice_logic is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = LogicAgentConfig(
            confidence_threshold=0.95,
            sacrifice_threshold=0.5,
            observation_memory_size=50,
        )

        assert config.confidence_threshold == 0.95
        assert config.sacrifice_threshold == 0.5
        assert config.observation_memory_size == 50


class TestUncertaintyReport:
    """Tests for UncertaintyReport."""

    def test_is_confident_above_threshold(self):
        """Test confidence check above threshold."""
        report = UncertaintyReport(confidence_score=0.9)
        assert report.is_confident(0.85) is True

    def test_is_confident_below_threshold(self):
        """Test confidence check below threshold."""
        report = UncertaintyReport(confidence_score=0.7)
        assert report.is_confident(0.85) is False

    def test_explain_high_confidence(self):
        """Test explanation for high confidence."""
        report = UncertaintyReport(confidence_score=0.95)
        explanation = report.explain()
        assert "High confidence" in explanation
        assert "95" in explanation

    def test_explain_low_confidence_with_sources(self):
        """Test explanation with uncertainty sources."""
        report = UncertaintyReport(
            confidence_score=0.5,
            value_variance=0.15,
            policy_entropy=3.0,
            uncertainty_sources=["position is complex"],
        )
        explanation = report.explain()

        assert "50" in explanation
        assert "value estimates vary" in explanation or "many viable actions" in explanation

    def test_explain_generic_uncertainty(self):
        """Test explanation without specific sources."""
        report = UncertaintyReport(confidence_score=0.6)
        explanation = report.explain()
        assert "general uncertainty" in explanation.lower()


class TestStrategicDecision:
    """Tests for StrategicDecision."""

    def test_is_actionable_above_threshold(self):
        """Test actionable check above threshold."""
        decision = StrategicDecision(
            recommended_action=5,
            confidence=0.9,
            uncertainty_report=UncertaintyReport(confidence_score=0.9),
            reasoning="Test reasoning",
        )
        assert decision.is_actionable(0.85) is True

    def test_is_actionable_below_threshold(self):
        """Test actionable check below threshold."""
        decision = StrategicDecision(
            recommended_action=5,
            confidence=0.7,
            uncertainty_report=UncertaintyReport(confidence_score=0.7),
            reasoning="Test reasoning",
        )
        assert decision.is_actionable(0.85) is False

    def test_get_disclosure_high_confidence(self):
        """Test no disclosure for high confidence."""
        decision = StrategicDecision(
            recommended_action=5,
            confidence=0.99,
            uncertainty_report=UncertaintyReport(confidence_score=0.99),
            reasoning="Test",
        )
        assert decision.get_disclosure() == ""

    def test_get_disclosure_low_confidence(self):
        """Test disclosure for low confidence."""
        decision = StrategicDecision(
            recommended_action=5,
            confidence=0.7,
            uncertainty_report=UncertaintyReport(
                confidence_score=0.7,
                uncertainty_sources=["test source"],
            ),
            reasoning="Test",
        )
        disclosure = decision.get_disclosure()
        assert disclosure != ""
        assert "70" in disclosure

    def test_timestamp_auto_generated(self):
        """Test timestamp is auto-generated."""
        decision = StrategicDecision(
            recommended_action=1,
            confidence=0.8,
            uncertainty_report=UncertaintyReport(confidence_score=0.8),
            reasoning="Test",
        )
        assert decision.timestamp != ""


class TestLogicAgent:
    """Tests for LogicAgent main class."""

    @pytest.fixture
    def agent(self):
        """Create a basic agent for testing."""
        return LogicAgent(name="TestAgent")

    @pytest.fixture
    def agent_with_low_threshold(self):
        """Create agent with low confidence threshold."""
        config = LogicAgentConfig(confidence_threshold=0.5)
        return LogicAgent(config=config, name="LowThresholdAgent")

    def test_init_default(self, agent):
        """Test default initialization."""
        assert agent.name == "TestAgent"
        assert agent.config.confidence_threshold == 0.85
        assert agent.rag is None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = LogicAgentConfig(confidence_threshold=0.95)
        agent = LogicAgent(config=config)

        assert agent.config.confidence_threshold == 0.95

    def test_analyze_context_early_game(self, agent):
        """Test context analysis for early game."""
        state = {"turn": 10, "cities": [1, 2], "units": [1, 2, 3]}
        context = agent.analyze_context(state)

        assert context.phase == GamePhase.EARLY
        assert context.threat_level == ThreatLevel.NONE

    def test_analyze_context_mid_game(self, agent):
        """Test context analysis for mid game."""
        state = {
            "turn": 80,
            "cities": [1, 2, 3, 4, 5],
            "units": list(range(10)),
        }
        context = agent.analyze_context(state)

        assert context.phase == GamePhase.MID

    def test_analyze_context_late_game(self, agent):
        """Test context analysis for late game."""
        state = {
            "turn": 200,
            "cities": list(range(10)),
            "units": list(range(20)),
        }
        context = agent.analyze_context(state)

        assert context.phase == GamePhase.LATE

    def test_analyze_context_crisis(self, agent):
        """Test context analysis detects crisis."""
        state = {
            "turn": 50,
            "cities": [1, 2, 3, 4],
            "military_strength": 100,
            "enemy_military_strength": 250,
        }
        context = agent.analyze_context(state)

        assert context.phase == GamePhase.CRISIS
        assert context.threat_level == ThreatLevel.CRITICAL

    def test_analyze_context_detects_opportunity(self, agent):
        """Test opportunity detection."""
        state = {
            "turn": 50,
            "enemy_exposed_city": True,
            "tech_advantage": True,
        }
        context = agent.analyze_context(state)

        assert context.opportunity_detected is True
        assert "vulnerable_target" in context.detected_patterns
        assert "timing_attack_window" in context.detected_patterns

    def test_analyze_context_stores_observation(self, agent):
        """Test that context analysis stores observation."""
        state = {"turn": 50}
        agent.analyze_context(state)

        memory = agent.get_memory("context_analysis")
        assert len(memory) == 1

    def test_quantify_uncertainty_high_confidence(self, agent):
        """Test uncertainty quantification for confident search."""
        mcts_stats = {
            "visit_count": 200,
            "total_visits": 300,
            "value": 0.7,
            "value_variance": 0.01,
        }
        policy = [0.9, 0.05, 0.03, 0.02]  # Low entropy

        report = agent.quantify_uncertainty(mcts_stats, policy)

        assert report.confidence_score > 0.7
        assert report.visit_ratio > 0.5

    def test_quantify_uncertainty_low_confidence(self, agent):
        """Test uncertainty quantification for uncertain search."""
        mcts_stats = {
            "visit_count": 10,
            "total_visits": 100,
            "value_variance": 0.3,
        }
        # High entropy policy
        policy = [0.25, 0.25, 0.25, 0.25]

        report = agent.quantify_uncertainty(mcts_stats, policy)

        assert report.confidence_score < 0.7
        assert len(report.uncertainty_sources) > 0

    def test_quantify_uncertainty_generates_recommendation(self, agent):
        """Test that uncertainty report includes recommendation."""
        mcts_stats = {"visit_count": 50, "total_visits": 100}

        report = agent.quantify_uncertainty(mcts_stats)

        assert report.recommendation != ""

    def test_evaluate_sacrifice_not_involved(self, agent):
        """Test sacrifice evaluation when no sacrifice."""
        state = {"turn": 50}
        mcts_stats = {"action_cost": 0.1, "expected_value": 0.3}
        context = GameContext()

        involved, priority, reasoning = agent.evaluate_sacrifice(
            action=1, state=state, mcts_stats=mcts_stats, context=context
        )

        assert involved is False
        assert priority == UtilityPriority.TACTICAL_EFFICIENCY

    def test_evaluate_sacrifice_for_victory(self, agent):
        """Test sacrifice justified for victory condition."""
        state = {"turn": 450}
        mcts_stats = {"action_cost": 0.5, "expected_value": 0.8}
        context = GameContext(turn_pressure=0.9)

        involved, priority, reasoning = agent.evaluate_sacrifice(
            action=1, state=state, mcts_stats=mcts_stats, context=context
        )

        assert involved is True
        assert priority == UtilityPriority.VICTORY_CONDITION
        assert "Time pressure" in reasoning

    def test_evaluate_sacrifice_for_survival(self, agent):
        """Test sacrifice justified for survival."""
        state = {"turn": 100}
        mcts_stats = {"action_cost": 0.5, "expected_value": 0.3}
        context = GameContext(threat_level=ThreatLevel.CRITICAL)

        involved, priority, reasoning = agent.evaluate_sacrifice(
            action=1, state=state, mcts_stats=mcts_stats, context=context
        )

        assert involved is True
        assert priority == UtilityPriority.SURVIVAL

    def test_evaluate_sacrifice_increments_counter(self, agent):
        """Test sacrifice recommendation increments counter."""
        state = {}
        mcts_stats = {"action_cost": 0.5, "expected_value": 0.8}
        context = GameContext(turn_pressure=0.9)

        initial_count = agent._sacrifices_recommended
        agent.evaluate_sacrifice(1, state, mcts_stats, context)

        assert agent._sacrifices_recommended == initial_count + 1

    def test_evaluate_position_full_pipeline(self, agent):
        """Test full position evaluation pipeline."""
        state = {
            "turn": 100,
            "cities": list(range(5)),
            "units": list(range(10)),
            "military_strength": 100,
            "enemy_military_strength": 50,
        }
        mcts_stats = {
            "best_action": 42,
            "visit_count": 100,
            "total_visits": 200,
            "value": 0.6,
            "value_variance": 0.05,
        }
        policy = [0.7, 0.1, 0.1, 0.1]

        decision = agent.evaluate_position(state, mcts_stats, policy)

        assert isinstance(decision, StrategicDecision)
        assert decision.recommended_action == 42
        assert decision.context is not None
        assert decision.uncertainty_report is not None

    def test_evaluate_position_increments_counter(self, agent):
        """Test position evaluation increments decision counter."""
        state = {"turn": 50}
        mcts_stats = {"best_action": 1, "visit_count": 50, "total_visits": 100}

        initial_count = agent._decisions_made
        agent.evaluate_position(state, mcts_stats)

        assert agent._decisions_made == initial_count + 1

    def test_evaluate_position_uncertain_adds_disclosure(self, agent):
        """Test uncertain position adds uncertainty disclosure."""
        state = {"turn": 50}
        mcts_stats = {
            "best_action": 1,
            "visit_count": 5,  # Very low
            "total_visits": 100,
            "value_variance": 0.5,  # High variance
        }

        decision = agent.evaluate_position(state, mcts_stats)

        # Should have low confidence due to poor stats
        assert decision.confidence < 0.85
        assert "UNCERTAINTY" in decision.reasoning

    def test_check_honesty_constraint_confident(self, agent):
        """Test honesty constraint passes confident response."""
        response = "The answer is 42."
        result = agent.check_honesty_constraint(response, 0.95)

        assert result == response

    def test_check_honesty_constraint_uncertain(self, agent):
        """Test honesty constraint modifies uncertain response."""
        response = "The answer is 42."
        result = agent.check_honesty_constraint(response, 0.6)

        assert "cannot state this with certainty" in result
        assert "60" in result  # Confidence percentage
        assert "85" in result  # Threshold percentage

    def test_memory_storage_and_retrieval(self, agent):
        """Test strategic memory storage and retrieval."""
        # Store observations
        agent._store_observation("test1", "obs1", "type_a", 0.9)
        agent._store_observation("test2", "obs2", "type_b", 0.8)
        agent._store_observation("test3", "obs3", "type_a", 0.7)

        # Retrieve all
        all_memory = agent.get_memory()
        assert len(all_memory) == 3

        # Retrieve filtered
        type_a = agent.get_memory("type_a")
        assert len(type_a) == 2

    def test_memory_trimming(self):
        """Test memory is trimmed when exceeding limit."""
        config = LogicAgentConfig(observation_memory_size=5)
        agent = LogicAgent(config=config)

        # Store more than limit
        for i in range(10):
            agent._store_observation(f"test{i}", f"obs{i}", "test", 0.5)

        assert len(agent._strategic_memory) == 5
        # Should keep the most recent
        assert agent._strategic_memory[-1].trigger == "test9"

    def test_get_statistics(self, agent):
        """Test statistics retrieval."""
        # Make some decisions
        state = {"turn": 50}
        mcts_stats = {"best_action": 1, "visit_count": 50, "total_visits": 100}
        agent.evaluate_position(state, mcts_stats)

        stats = agent.get_statistics()

        assert stats["name"] == "TestAgent"
        assert stats["decisions_made"] == 1
        assert stats["confidence_threshold"] == 0.85
        assert "memory_size" in stats

    def test_repr(self, agent):
        """Test string representation."""
        repr_str = repr(agent)

        assert "LogicAgent" in repr_str
        assert "TestAgent" in repr_str


class TestLogicAgentWithRAG:
    """Tests for LogicAgent with RAG integration."""

    @pytest.fixture
    def mock_rag(self):
        """Create mock RAG integration."""
        rag = MagicMock()
        rag.initialized = True
        rag.query_strategy.return_value = [
            MagicMock(content="Test strategy content", score=0.8, source="test.md")
        ]
        rag.store_training_insight.return_value = True
        return rag

    @pytest.fixture
    def agent_with_rag(self, mock_rag):
        """Create agent with mock RAG."""
        return LogicAgent(rag=mock_rag, name="RAGAgent")

    def test_context_enrichment_from_rag(self, agent_with_rag, mock_rag):
        """Test context is enriched from RAG."""
        state = {"turn": 50}
        context = agent_with_rag.analyze_context(state)

        # RAG should have been queried
        mock_rag.query_strategy.assert_called()
        # Patterns should be detected from RAG results
        assert len(context.detected_patterns) > 0

    def test_imitate_strategy_success(self, agent_with_rag, mock_rag):
        """Test strategy imitation from RAG."""
        result = agent_with_rag.imitate_strategy("timing_attack")

        assert result is not None
        assert "Executing pattern" in result
        assert "timing_attack" in result

    def test_imitate_strategy_no_match(self, agent_with_rag, mock_rag):
        """Test strategy imitation with no match."""
        mock_rag.query_strategy.return_value = [
            MagicMock(content="content", score=0.3, source="test.md")  # Below threshold
        ]

        result = agent_with_rag.imitate_strategy("unknown_pattern")

        assert result is None

    def test_observation_stored_in_rag(self, agent_with_rag, mock_rag):
        """Test observations are stored in RAG."""
        state = {"turn": 50}
        agent_with_rag.analyze_context(state)

        # Should have called store_training_insight
        mock_rag.store_training_insight.assert_called()


class TestEnums:
    """Tests for enum classes."""

    def test_game_phase_values(self):
        """Test GamePhase enum values."""
        assert GamePhase.EARLY.value == "early"
        assert GamePhase.MID.value == "mid"
        assert GamePhase.LATE.value == "late"
        assert GamePhase.CRISIS.value == "crisis"

    def test_threat_level_ordering(self):
        """Test ThreatLevel enum ordering."""
        assert ThreatLevel.NONE.value < ThreatLevel.LOW.value
        assert ThreatLevel.LOW.value < ThreatLevel.MODERATE.value
        assert ThreatLevel.MODERATE.value < ThreatLevel.HIGH.value
        assert ThreatLevel.HIGH.value < ThreatLevel.CRITICAL.value

    def test_utility_priority_hierarchy(self):
        """Test UtilityPriority hierarchy."""
        assert UtilityPriority.VICTORY_CONDITION.value > UtilityPriority.SURVIVAL.value
        assert UtilityPriority.SURVIVAL.value > UtilityPriority.STRATEGIC_ADVANTAGE.value
        assert UtilityPriority.STRATEGIC_ADVANTAGE.value > UtilityPriority.TACTICAL_EFFICIENCY.value
        assert UtilityPriority.TACTICAL_EFFICIENCY.value > UtilityPriority.RESOURCE_PRESERVATION.value
        assert UtilityPriority.RESOURCE_PRESERVATION.value > UtilityPriority.AGENT_COMPUTATION.value
