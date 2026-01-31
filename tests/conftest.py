"""Shared test fixtures for Civ6-AlphaZero test suite.

This module provides reusable pytest fixtures for:
- Environment instances (MicroCiv, mocks)
- Sample game states (early, mid, late game)
- Network instances (MicroCivNetwork, MuZeroNetwork, TransformerNetwork)
- Replay buffer fixtures
- MCTS fixtures
- Configuration fixtures

Usage:
    # In any test file, fixtures are automatically available:
    def test_something(microciv_env, microciv_network):
        obs = microciv_env.reset()
        # ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def microciv_env():
    """Create a MicroCiv environment for testing.

    Returns:
        MicroCivEnv with default settings (8x8 board, 50 max turns).
    """
    from alpha_zero.envs.microciv import MicroCivEnv

    return MicroCivEnv(board_size=8, max_turns=50, seed=42)


@pytest.fixture
def microciv_env_small():
    """Create a small MicroCiv environment for fast tests.

    Returns:
        MicroCivEnv with small board (4x4) and reduced turns.
    """
    from alpha_zero.envs.microciv import MicroCivEnv

    return MicroCivEnv(board_size=4, max_turns=20, seed=42)


@pytest.fixture
def microciv_env_large():
    """Create a larger MicroCiv environment for stress tests.

    Returns:
        MicroCivEnv with larger board (16x16).
    """
    from alpha_zero.envs.microciv import MicroCivEnv

    return MicroCivEnv(board_size=16, max_turns=100, seed=42)


@dataclass
class MockEnvironment:
    """Mock environment for unit testing.

    Provides controllable observations and actions for testing
    components that depend on environments.
    """

    observation: dict[str, np.ndarray] = field(default_factory=dict)
    done: bool = False
    reward: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)
    _step_count: int = 0

    def __post_init__(self):
        if not self.observation:
            self.observation = {
                "spatial": np.random.randn(13, 8, 8).astype(np.float32),
                "vector": np.random.randn(64).astype(np.float32),
                "action_mask": np.ones(200, dtype=np.float32),
            }

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset the environment."""
        self._step_count = 0
        self.done = False
        return self.observation

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Take a step in the environment."""
        self._step_count += 1
        return self.observation, self.reward, self.done, False, self.info

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation."""
        return self.observation

    def get_valid_actions(self) -> np.ndarray:
        """Get valid action mask."""
        return self.observation.get("action_mask", np.ones(200, dtype=np.float32))

    def clone(self) -> "MockEnvironment":
        """Clone the environment state."""
        return MockEnvironment(
            observation={k: v.copy() for k, v in self.observation.items()},
            done=self.done,
            reward=self.reward,
            info=self.info.copy(),
            _step_count=self._step_count,
        )


@pytest.fixture
def mock_env():
    """Create a mock environment for unit testing.

    Returns:
        MockEnvironment with controllable state.
    """
    return MockEnvironment()


# =============================================================================
# Sample Game State Fixtures
# =============================================================================


@pytest.fixture
def sample_game_state_early():
    """Sample early game state (turn 1-10).

    Returns:
        Dictionary representing an early game state.
    """
    return {
        "turn": 5,
        "current_player": 0,
        "players": [
            {
                "id": 0,
                "gold": 100,
                "science": 50,
                "culture": 30,
                "faith": 10,
                "era": "Ancient",
            },
            {
                "id": 1,
                "gold": 95,
                "science": 45,
                "culture": 25,
                "faith": 5,
                "era": "Ancient",
            },
        ],
        "cities": [
            {"id": 0, "player": 0, "x": 5, "y": 5, "name": "Capital", "population": 2},
            {"id": 1, "player": 1, "x": 10, "y": 10, "name": "EnemyCapital", "population": 2},
        ],
        "units": [
            {"id": 0, "player": 0, "type": "warrior", "x": 5, "y": 6, "health": 100},
            {"id": 1, "player": 0, "type": "settler", "x": 4, "y": 5, "health": 100},
            {"id": 2, "player": 1, "type": "warrior", "x": 10, "y": 11, "health": 100},
        ],
        "board_size": 16,
        "map": np.zeros((16, 16), dtype=np.int32),
    }


@pytest.fixture
def sample_game_state_mid():
    """Sample mid game state (turn 50-100).

    Returns:
        Dictionary representing a mid game state.
    """
    return {
        "turn": 75,
        "current_player": 0,
        "players": [
            {
                "id": 0,
                "gold": 500,
                "science": 300,
                "culture": 250,
                "faith": 100,
                "era": "Medieval",
            },
            {
                "id": 1,
                "gold": 450,
                "science": 280,
                "culture": 220,
                "faith": 80,
                "era": "Medieval",
            },
        ],
        "cities": [
            {"id": 0, "player": 0, "x": 5, "y": 5, "name": "Capital", "population": 8},
            {"id": 1, "player": 0, "x": 7, "y": 3, "name": "SecondCity", "population": 5},
            {"id": 2, "player": 0, "x": 3, "y": 7, "name": "ThirdCity", "population": 4},
            {"id": 3, "player": 1, "x": 10, "y": 10, "name": "EnemyCapital", "population": 7},
            {"id": 4, "player": 1, "x": 12, "y": 8, "name": "EnemySecond", "population": 4},
        ],
        "units": [
            {"id": 0, "player": 0, "type": "swordsman", "x": 5, "y": 6, "health": 100},
            {"id": 1, "player": 0, "type": "crossbowman", "x": 4, "y": 5, "health": 100},
            {"id": 2, "player": 0, "type": "knight", "x": 6, "y": 5, "health": 80},
            {"id": 3, "player": 1, "type": "swordsman", "x": 10, "y": 11, "health": 100},
            {"id": 4, "player": 1, "type": "crossbowman", "x": 9, "y": 10, "health": 100},
        ],
        "board_size": 16,
        "map": np.zeros((16, 16), dtype=np.int32),
    }


@pytest.fixture
def sample_game_state_late():
    """Sample late game state (turn 150+).

    Returns:
        Dictionary representing a late game state.
    """
    return {
        "turn": 180,
        "current_player": 0,
        "players": [
            {
                "id": 0,
                "gold": 2000,
                "science": 1200,
                "culture": 1000,
                "faith": 500,
                "era": "Modern",
            },
            {
                "id": 1,
                "gold": 1800,
                "science": 1100,
                "culture": 900,
                "faith": 400,
                "era": "Modern",
            },
        ],
        "cities": [
            {"id": i, "player": 0, "x": 5 + i, "y": 5, "name": f"City{i}", "population": 15 - i}
            for i in range(6)
        ]
        + [
            {"id": i + 6, "player": 1, "x": 10 + i, "y": 10, "name": f"EnemyCity{i}", "population": 14 - i}
            for i in range(5)
        ],
        "units": [
            {"id": i, "player": 0, "type": "infantry", "x": 5 + i, "y": 6, "health": 100}
            for i in range(8)
        ]
        + [
            {"id": i + 8, "player": 1, "type": "infantry", "x": 10 + i, "y": 11, "health": 100}
            for i in range(6)
        ],
        "board_size": 16,
        "map": np.zeros((16, 16), dtype=np.int32),
    }


@pytest.fixture
def sample_observation():
    """Sample observation dict with spatial/vector/action_mask.

    Returns:
        Dictionary with observation tensors matching MicroCiv format.
    """
    return {
        "spatial": np.random.randn(13, 8, 8).astype(np.float32),
        "vector": np.random.randn(64).astype(np.float32),
        "action_mask": np.ones(200, dtype=np.float32),
    }


@pytest.fixture
def sample_observation_large():
    """Sample observation dict for larger environments.

    Returns:
        Dictionary with observation tensors for 16x16 board.
    """
    return {
        "spatial": np.random.randn(13, 16, 16).astype(np.float32),
        "vector": np.random.randn(128).astype(np.float32),
        "action_mask": np.ones(500, dtype=np.float32),
    }


# =============================================================================
# Network Fixtures
# =============================================================================


@pytest.fixture
def microciv_network():
    """Create MicroCivNetwork for testing.

    Returns:
        MicroCivNetwork with small architecture for fast tests.
    """
    from alpha_zero.core.civ6_network import MicroCivNetwork

    return MicroCivNetwork(
        board_size=8,
        action_space_size=200,
        num_res_blocks=2,
        num_filters=16,
    )


@pytest.fixture
def microciv_network_medium():
    """Create medium-sized MicroCivNetwork.

    Returns:
        MicroCivNetwork with moderate architecture.
    """
    from alpha_zero.core.civ6_network import MicroCivNetwork

    return MicroCivNetwork(
        board_size=8,
        action_space_size=200,
        num_res_blocks=4,
        num_filters=32,
    )


@pytest.fixture
def muzero_network():
    """Create MuZeroNetwork for testing.

    Returns:
        MuZeroNetwork with small configuration.
    """
    try:
        from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork

        config = MuZeroConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=200,
            num_res_blocks=2,
            num_filters=16,
            hidden_state_dim=32,
            reward_support_size=51,
            value_support_size=51,
        )
        return MuZeroNetwork(config)
    except ImportError:
        pytest.skip("MuZeroNetwork not available")


@pytest.fixture
def transformer_network():
    """Create TransformerNetwork for testing.

    Returns:
        Civ6TransformerNetwork with small configuration.
    """
    try:
        from alpha_zero.core.transformer_network import (
            Civ6TransformerNetwork,
            TransformerConfig,
        )

        config = TransformerConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=200,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ff_dim=128,
            dropout=0.0,
        )
        return Civ6TransformerNetwork(config)
    except ImportError:
        pytest.skip("TransformerNetwork not available")


@pytest.fixture(params=["microciv", "muzero", "transformer"])
def any_network(request, microciv_network, muzero_network, transformer_network):
    """Parametrized fixture for any network type.

    Use this fixture when a test should run against all network types.

    Args:
        request: Pytest request object.

    Returns:
        One of: MicroCivNetwork, MuZeroNetwork, or TransformerNetwork.
    """
    networks = {
        "microciv": microciv_network,
        "muzero": muzero_network,
        "transformer": transformer_network,
    }
    return networks[request.param]


# =============================================================================
# Replay Buffer Fixtures
# =============================================================================


@pytest.fixture
def training_example():
    """Create a single TrainingExample.

    Returns:
        TrainingExample with random data.
    """
    from alpha_zero.distributed.shared_storage import TrainingExample

    return TrainingExample(
        spatial=np.random.randn(13, 8, 8).astype(np.float32),
        vector=np.random.randn(64).astype(np.float32),
        action_mask=np.ones(200, dtype=np.float32),
        policy=np.random.dirichlet(np.ones(200)).astype(np.float32),
        value=np.random.uniform(-1, 1),
    )


@pytest.fixture
def game_trajectory(training_example):
    """Create a GameTrajectory with multiple examples.

    Args:
        training_example: Base training example fixture.

    Returns:
        GameTrajectory with 10 training examples.
    """
    from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample

    examples = [
        TrainingExample(
            spatial=np.random.randn(13, 8, 8).astype(np.float32),
            vector=np.random.randn(64).astype(np.float32),
            action_mask=np.ones(200, dtype=np.float32),
            policy=np.random.dirichlet(np.ones(200)).astype(np.float32),
            value=np.random.uniform(-1, 1),
        )
        for _ in range(10)
    ]

    return GameTrajectory(
        examples=examples,
        game_id=1,
        actor_id=0,
        game_length=10,
        winner=0,
    )


@pytest.fixture
def replay_buffer_config():
    """Create a ReplayBufferConfig for testing.

    Returns:
        ReplayBufferConfig with small sizes for fast tests.
    """
    from alpha_zero.core.buffer_config import ReplayBufferConfig

    return ReplayBufferConfig.small()


# =============================================================================
# MCTS Fixtures
# =============================================================================


@pytest.fixture
def mcts_config():
    """Standard MCTS configuration for testing.

    Returns:
        MCTSConfig with reduced simulations for fast tests.
    """
    from alpha_zero.core.civ6_mcts import MCTSConfig

    return MCTSConfig(
        num_simulations=10,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        exploration_fraction=0.25,
    )


@pytest.fixture
def simple_mcts():
    """Create SimpleMCTS for testing.

    Returns:
        SimpleMCTS with small number of simulations.
    """
    from alpha_zero.core.civ6_mcts import SimpleMCTS

    return SimpleMCTS(
        num_simulations=10,
        c_puct=1.5,
        rollout_depth=5,
    )


@pytest.fixture
def civ6_mcts(microciv_network, mcts_config):
    """Create Civ6MCTS with network.

    Args:
        microciv_network: Network fixture.
        mcts_config: MCTS config fixture.

    Returns:
        Civ6MCTS instance ready for use.
    """
    from alpha_zero.core.civ6_mcts import Civ6MCTS

    return Civ6MCTS(microciv_network, mcts_config, device="cpu")


@pytest.fixture
def gumbel_mcts(microciv_network):
    """Create GumbelMCTS for testing.

    Args:
        microciv_network: Network fixture.

    Returns:
        GumbelMCTS instance.
    """
    try:
        from alpha_zero.core.civ6_mcts import GumbelMCTS, GumbelMCTSConfig

        config = GumbelMCTSConfig(
            num_simulations=8,
            num_sampled_actions=4,
            c_visit=50.0,
            c_scale=1.0,
        )
        return GumbelMCTS(microciv_network, config, device="cpu")
    except ImportError:
        pytest.skip("GumbelMCTS not available")


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def timeout_config():
    """Create TimeoutConfig for testing.

    Returns:
        TimeoutConfig with fast timeout values.
    """
    from alpha_zero.core.timeout_config import TimeoutConfig

    return TimeoutConfig.fast()


@pytest.fixture
def timeout_config_production():
    """Create production TimeoutConfig for testing.

    Returns:
        TimeoutConfig with production values.
    """
    from alpha_zero.core.timeout_config import TimeoutConfig

    return TimeoutConfig.production()


# =============================================================================
# Hierarchical Fixtures
# =============================================================================


@pytest.fixture
def strategic_encoder():
    """Create StrategicStateEncoder.

    Returns:
        StrategicStateEncoder instance.
    """
    try:
        from alpha_zero.hierarchical.encoders.state_encoder import StrategicStateEncoder

        return StrategicStateEncoder()
    except ImportError:
        pytest.skip("Hierarchical encoders not available")


@pytest.fixture
def tactical_encoder():
    """Create TacticalStateEncoder.

    Returns:
        TacticalStateEncoder instance.
    """
    try:
        from alpha_zero.hierarchical.encoders.state_encoder import TacticalStateEncoder

        return TacticalStateEncoder()
    except ImportError:
        pytest.skip("Hierarchical encoders not available")


@pytest.fixture
def hierarchical_encoder():
    """Create HierarchicalStateEncoder.

    Returns:
        HierarchicalStateEncoder instance.
    """
    try:
        from alpha_zero.hierarchical.encoders.state_encoder import HierarchicalStateEncoder

        return HierarchicalStateEncoder()
    except ImportError:
        pytest.skip("Hierarchical encoders not available")


@pytest.fixture
def sample_goal():
    """Create a sample Goal for testing.

    Returns:
        Goal instance for testing.
    """
    try:
        from alpha_zero.hierarchical.goals.goal_types import Goal, StrategicGoalType

        return Goal.strategic(StrategicGoalType.SCIENCE_VICTORY)
    except ImportError:
        pytest.skip("Hierarchical goals not available")


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility.

    Sets numpy random seed to 42.

    Returns:
        The seed value (42).
    """
    np.random.seed(42)
    return 42


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints.

    Args:
        tmp_path: Pytest tmp_path fixture.

    Returns:
        Path to temporary checkpoint directory.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir
