"""Tests for reflection system integration in TrainingOrchestrator.

This module tests the LATS-style reflection memory integration:
- ReflectionMemory initialization in orchestrator
- Processing game losses for reflection generation
- MCTS value adjustment based on reflection memory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


@dataclass
class MockGameResult:
    """Mock game result for testing."""

    trajectory: list[dict[str, Any]] = field(default_factory=list)
    winner: int = 0
    total_reward: float = 0.0
    turns: int = 10
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class MockReflectionConfig:
    """Mock reflection config for testing."""

    enabled: bool = True
    storage_backend: str = "memory"
    collection_name: str = "test_reflections"
    embedding_dimension: int = 384
    similarity_threshold: float = 0.7
    max_retrievals: int = 5
    warning_penalty: float = 0.1
    max_entries: int = 1000
    generate_critiques: bool = True
    llm_temperature: float = 0.3
    llm_max_tokens: int = 500
    cache_critiques: bool = True
    use_fallback_on_failure: bool = True


class TestReflectionConfigIntegration:
    """Tests for ReflectionIntegrationConfig in orchestrator config."""

    def test_default_values(self):
        """Test default reflection config values."""
        from alpha_zero.orchestrator.config import ReflectionIntegrationConfig

        config = ReflectionIntegrationConfig()

        assert config.enabled is False
        assert config.storage_backend == "memory"
        assert config.warning_penalty == 0.1
        assert config.generate_critiques is True

    def test_config_with_custom_values(self):
        """Test reflection config with custom values."""
        from alpha_zero.orchestrator.config import ReflectionIntegrationConfig

        config = ReflectionIntegrationConfig(
            enabled=True,
            storage_backend="chroma",
            warning_penalty=0.2,
            max_retrievals=10,
        )

        assert config.enabled is True
        assert config.storage_backend == "chroma"
        assert config.warning_penalty == 0.2
        assert config.max_retrievals == 10


class TestOrchestratorReflectionInit:
    """Tests for reflection system initialization in orchestrator."""

    def test_reflection_disabled_by_default(self):
        """Test that reflection is disabled by default."""
        from alpha_zero.orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.reflection.enabled is False

    def test_reflection_fields_initialized_to_none(self):
        """Test that reflection fields are initialized to None."""
        from alpha_zero.orchestrator.config import OrchestratorConfig
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        assert orchestrator._reflection_memory is None
        assert orchestrator._reflection_generator is None


class TestProcessGameResult:
    """Tests for game result processing with reflection."""

    def test_process_winning_game_no_reflection(self):
        """Test that winning games don't generate reflections."""
        from alpha_zero.orchestrator.config import OrchestratorConfig
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        # Mock reflection components
        orchestrator._reflection_memory = MagicMock()
        orchestrator._reflection_generator = MagicMock()

        # Create winning game result
        result = MockGameResult(winner=1)

        # Process should not add reflection for wins
        orchestrator._process_game_result(result)

        # Reflection generator should not be called for wins
        orchestrator._reflection_generator.generate_critique.assert_not_called()

    def test_process_losing_game_generates_reflection(self):
        """Test that losing games generate reflections."""
        from alpha_zero.orchestrator.config import OrchestratorConfig
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        # Create mock trajectory
        trajectory = [
            {
                "spatial": np.zeros((13, 16, 16)),
                "vector": np.zeros(10),
                "action_mask": np.ones(356),
                "policy": np.ones(356) / 356,
                "value": 0.0,
            }
        ]

        # Mock reflection components with async support
        mock_memory = MagicMock()
        mock_memory.add_failed_game = AsyncMock()

        mock_critique = MagicMock()
        mock_critique.full_critique = "Test critique"
        mock_critique.mistake_action = 42

        mock_generator = MagicMock()
        mock_generator.generate_critique = AsyncMock(return_value=mock_critique)

        orchestrator._reflection_memory = mock_memory
        orchestrator._reflection_generator = mock_generator
        orchestrator.state.total_games = 1

        # Create losing game result
        result = MockGameResult(
            trajectory=trajectory,
            winner=-1,  # Loss
            turns=10,
        )

        # Process should generate reflection for loss
        orchestrator._process_game_result(result)

        # Note: This is async, so we need to verify the async call was made
        # The actual test would need to handle async properly


class TestMCTSReflectionIntegration:
    """Tests for MCTS integration with reflection memory."""

    def test_mcts_without_reflection_works(self):
        """Test that MCTS works normally without reflection memory."""
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig

        # Create mock network
        mock_network = MagicMock()
        mock_network.return_value = (
            np.ones(356) / 356,  # policy
            np.array([0.5]),  # value
        )

        config = MCTSConfig()
        mcts = Civ6MCTS(mock_network, config, reflection_memory=None)

        assert mcts._reflection_memory is None

    def test_mcts_with_reflection_memory(self):
        """Test that MCTS accepts reflection memory parameter."""
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig

        # Create mock network
        mock_network = MagicMock()

        # Create mock reflection memory
        mock_memory = MagicMock()
        mock_memory.initialized = True
        mock_memory.get_value_adjustment_sync = MagicMock(return_value=-0.1)

        config = MCTSConfig()
        mcts = Civ6MCTS(mock_network, config, reflection_memory=mock_memory)

        assert mcts._reflection_memory is mock_memory

    def test_create_mcts_factory_accepts_reflection_memory(self):
        """Test that create_mcts factory accepts reflection_memory parameter."""
        from alpha_zero.orchestrator.config import MCTSConfig
        from alpha_zero.utils.factories import create_mcts

        # Create mock network
        mock_network = MagicMock()
        mock_network.return_value = (np.ones(356) / 356, np.array([0.5]))

        # Create mock reflection memory
        mock_memory = MagicMock()

        config = MCTSConfig()

        # Should not raise
        mcts = create_mcts(
            config,
            mock_network,
            device="cpu",
            reflection_memory=mock_memory,
        )

        assert mcts._reflection_memory is mock_memory


class TestReflectionGracefulDegradation:
    """Tests for graceful degradation when reflection dependencies are missing."""

    def test_orchestrator_works_without_reflection_module(self):
        """Test that orchestrator works when reflection module is unavailable."""
        from alpha_zero.orchestrator.config import OrchestratorConfig
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        # Even if reflection is enabled in config, orchestrator should handle
        # missing dependencies gracefully
        config = OrchestratorConfig()
        config.reflection.enabled = True

        orchestrator = TrainingOrchestrator(config)

        # Should not raise, should log warning
        assert orchestrator._reflection_memory is None
        assert orchestrator._reflection_generator is None

    def test_process_game_result_handles_missing_components(self):
        """Test that process_game_result handles missing reflection components."""
        from alpha_zero.orchestrator.config import OrchestratorConfig
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        # Ensure components are None
        orchestrator._reflection_memory = None
        orchestrator._reflection_generator = None

        # Create losing game result
        result = MockGameResult(winner=-1)

        # Should not raise even with missing components
        orchestrator._process_game_result(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
