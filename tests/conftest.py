"""Shared test fixtures for Civ6 AlphaZero test suite.

This module provides reusable fixtures for testing across the project.
All fixtures use appropriate cleanup to prevent resource leaks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Import project modules
from alpha_zero.core.civ6_mcts import MCTSConfig, SimpleMCTS
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.distributed.actor import ActorConfig
from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    GameTrajectory,
    ReplayBufferConfig,
    SharedStorage,
    SharedStorageConfig,
    TrainingExample,
)
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.utils.constants import ENCODER, GAME


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def small_env() -> Generator[MicroCivEnv, None, None]:
    """Create a small environment for fast testing."""
    env = MicroCivEnv(board_size=8, max_turns=20, seed=42)
    yield env
    env.close()


@pytest.fixture
def minimal_env() -> Generator[MicroCivEnv, None, None]:
    """Create minimal environment for unit tests."""
    env = MicroCivEnv(board_size=6, max_turns=10, seed=42)
    yield env
    env.close()


@pytest.fixture
def default_env() -> Generator[MicroCivEnv, None, None]:
    """Create default-sized environment."""
    env = MicroCivEnv(board_size=GAME.DEFAULT_BOARD_SIZE, max_turns=50, seed=42)
    yield env
    env.close()


# =============================================================================
# Network Fixtures
# =============================================================================


@pytest.fixture
def small_network() -> MicroCivNetwork:
    """Create a small network for testing."""
    env = MicroCivEnv(board_size=8)
    action_space_size = env.action_space_handler.total_size
    env.close()

    return MicroCivNetwork(
        board_size=8,
        action_space_size=action_space_size,
        num_res_blocks=2,
        num_filters=16,
    )


@pytest.fixture
def tiny_network() -> MicroCivNetwork:
    """Create minimal network for unit tests."""
    env = MicroCivEnv(board_size=6)
    action_space_size = env.action_space_handler.total_size
    env.close()

    return MicroCivNetwork(
        board_size=6,
        action_space_size=action_space_size,
        num_res_blocks=1,
        num_filters=8,
    )


# =============================================================================
# MCTS Fixtures
# =============================================================================


@pytest.fixture
def simple_mcts() -> SimpleMCTS:
    """Create SimpleMCTS for testing."""
    return SimpleMCTS(num_simulations=10, c_puct=1.5)


@pytest.fixture
def mcts_config() -> MCTSConfig:
    """Create MCTS configuration."""
    return MCTSConfig(
        num_simulations=10,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )


# =============================================================================
# Distributed Component Fixtures
# =============================================================================


@pytest.fixture
def shared_storage() -> Generator[SharedStorage, None, None]:
    """Create shared storage with cleanup."""
    config = SharedStorageConfig(compress_weights=False)
    storage = SharedStorage(config=config)
    yield storage
    storage.shutdown()


@pytest.fixture
def replay_buffer() -> Generator[DistributedReplayBuffer, None, None]:
    """Create replay buffer with cleanup."""
    config = ReplayBufferConfig(max_size=1000, min_size=10)
    buffer = DistributedReplayBuffer(config=config)
    yield buffer
    buffer.shutdown()


@pytest.fixture
def small_replay_buffer() -> Generator[DistributedReplayBuffer, None, None]:
    """Create small replay buffer for overflow testing."""
    config = ReplayBufferConfig(max_size=100, min_size=5, trajectory_queue_size=10)
    buffer = DistributedReplayBuffer(config=config)
    yield buffer
    buffer.shutdown()


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def fast_actor_config() -> ActorConfig:
    """Actor config optimized for fast tests."""
    return ActorConfig(
        actor_id=0,
        board_size=8,
        max_turns=20,
        num_res_blocks=2,
        num_filters=16,
        num_simulations=5,
        max_games=2,
        log_interval=1,
    )


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """Create temporary log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_spatial() -> np.ndarray:
    """Create sample spatial observation."""
    return np.random.rand(ENCODER.MICROCIV_CHANNELS, 8, 8).astype(np.float32)


@pytest.fixture
def sample_vector() -> np.ndarray:
    """Create sample vector observation."""
    return np.random.rand(ENCODER.MICROCIV_VECTOR_DIM).astype(np.float32)


@pytest.fixture
def sample_action_mask() -> np.ndarray:
    """Create sample action mask."""
    env = MicroCivEnv(board_size=8)
    action_space_size = env.action_space_handler.total_size
    env.close()

    mask = np.zeros(action_space_size, dtype=np.float32)
    # Make some actions valid
    valid_indices = np.random.choice(action_space_size, size=50, replace=False)
    mask[valid_indices] = 1.0
    return mask


@pytest.fixture
def sample_training_example(
    sample_spatial: np.ndarray,
    sample_vector: np.ndarray,
    sample_action_mask: np.ndarray,
) -> TrainingExample:
    """Create sample training example."""
    action_space_size = len(sample_action_mask)
    policy = np.random.rand(action_space_size).astype(np.float32)
    policy = policy / policy.sum()  # Normalize

    return TrainingExample(
        spatial=sample_spatial,
        vector=sample_vector,
        action_mask=sample_action_mask,
        policy=policy,
        value=0.5,
    )


@pytest.fixture
def sample_game_trajectory(
    sample_training_example: TrainingExample,
) -> GameTrajectory:
    """Create sample game trajectory."""
    examples = [sample_training_example for _ in range(10)]
    return GameTrajectory(
        examples=examples,
        game_id=1,
        actor_id=0,
        game_length=10,
        winner=1,
    )


def generate_training_examples(
    count: int,
    board_size: int = 8,
    seed: int = 42,
) -> list[TrainingExample]:
    """Generate multiple training examples for testing.

    Args:
        count: Number of examples to generate.
        board_size: Board size for spatial dimensions.
        seed: Random seed.

    Returns:
        List of training examples.
    """
    rng = np.random.default_rng(seed)
    env = MicroCivEnv(board_size=board_size)
    action_space_size = env.action_space_handler.total_size
    env.close()

    examples = []
    for _ in range(count):
        spatial = rng.random((ENCODER.MICROCIV_CHANNELS, board_size, board_size)).astype(
            np.float32
        )
        vector = rng.random(ENCODER.MICROCIV_VECTOR_DIM).astype(np.float32)
        action_mask = np.ones(action_space_size, dtype=np.float32)

        policy = rng.random(action_space_size).astype(np.float32)
        policy = policy / policy.sum()

        examples.append(
            TrainingExample(
                spatial=spatial,
                vector=vector,
                action_mask=action_mask,
                policy=policy,
                value=rng.choice([-1.0, 0.0, 1.0]),
            )
        )

    return examples


def generate_game_trajectories(
    count: int,
    examples_per_game: int = 5,
    board_size: int = 8,
    seed: int = 42,
) -> list[GameTrajectory]:
    """Generate multiple game trajectories for testing.

    Args:
        count: Number of trajectories to generate.
        examples_per_game: Examples per trajectory.
        board_size: Board size for spatial dimensions.
        seed: Random seed.

    Returns:
        List of game trajectories.
    """
    rng = np.random.default_rng(seed)
    trajectories = []

    for i in range(count):
        examples = generate_training_examples(
            examples_per_game, board_size, seed=seed + i
        )
        trajectories.append(
            GameTrajectory(
                examples=examples,
                game_id=i,
                actor_id=i % 4,
                game_length=examples_per_game,
                winner=rng.choice([None, 0, 1]),
            )
        )

    return trajectories


# =============================================================================
# Timeout and Retry Helpers
# =============================================================================


@pytest.fixture
def short_timeout() -> float:
    """Short timeout for failure tests."""
    return 0.5


@pytest.fixture
def very_short_timeout() -> float:
    """Very short timeout for immediate failure tests."""
    return 0.05


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_network() -> MagicMock:
    """Create a mock network for isolated testing."""
    mock = MagicMock(spec=MicroCivNetwork)

    # Setup default return values
    def predict_side_effect(spatial, vector, action_mask=None):
        batch_size = spatial.shape[0] if len(spatial.shape) > 3 else 1
        action_size = 376  # Default action space size

        policy = torch.zeros(batch_size, action_size)
        policy[:, 0] = 1.0  # Always predict first action

        value = torch.zeros(batch_size, 1)
        return policy, value

    mock.return_value = predict_side_effect
    mock.forward = predict_side_effect

    return mock


# =============================================================================
# Pytest Markers
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "stress: marks tests as stress/load tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests requiring GPU")
