"""Shared fixtures for SDK core unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.config import (
    CurriculumConfig,
    CurriculumStage,
    MCTSConfig,
    NetworkConfig,
    SDKConfig,
    TrainingConfig,
)
from src.core.types import (
    DeviceType,
    GameOutcome,
    GameTrajectory,
    ObservationSpec,
    TrainingExample,
)


@pytest.fixture
def default_sdk_config() -> SDKConfig:
    """SDKConfig with all defaults."""
    return SDKConfig()


@pytest.fixture
def small_mcts_config() -> MCTSConfig:
    """Minimal MCTS config for fast tests."""
    return MCTSConfig(num_simulations=8, c_puct=1.0)


@pytest.fixture
def tiny_network_config() -> NetworkConfig:
    """Tiny network config for fast tests."""
    return NetworkConfig(
        num_residual_blocks=1,
        num_filters=8,
        fc_hidden_dim=16,
        value_head_hidden_dim=8,
        policy_head_hidden_dim=8,
    )


@pytest.fixture
def smoke_training_config() -> TrainingConfig:
    """Minimal training config for smoke tests."""
    return TrainingConfig(
        batch_size=4,
        learning_rate=1e-3,
        max_iterations=1,
        games_per_iteration=1,
        replay_buffer_size=100,
        checkpoint_interval=1,
        eval_interval=1,
        eval_games=1,
        max_steps_per_game=10,
        num_actors=1,
        device=DeviceType.cpu,
    )


@pytest.fixture
def sample_observation_spec() -> ObservationSpec:
    """Standard observation spec for testing."""
    return ObservationSpec(
        spatial_shape=(13, 16, 16),
        vector_dim=64,
        temporal_depth=1,
    )


@pytest.fixture
def sample_training_example() -> TrainingExample:
    """A single training example with realistic shapes."""
    rng = np.random.default_rng(42)
    return TrainingExample(
        spatial=rng.random((13, 16, 16), dtype=np.float32),
        vector=rng.random(64, dtype=np.float32),
        action_mask=rng.integers(0, 2, size=356, dtype=np.bool_),
        policy=np.ones(356, dtype=np.float32) / 356,
        value=0.5,
    )


@pytest.fixture
def sample_game_trajectory(sample_training_example: TrainingExample) -> GameTrajectory:
    """A short game trajectory for testing."""
    return GameTrajectory(
        examples=[sample_training_example] * 3,
        outcome=GameOutcome.win,
        iteration=1,
    )


@pytest.fixture
def three_stage_curriculum() -> CurriculumConfig:
    """A three-stage curriculum for testing."""
    return CurriculumConfig(
        stages=[
            CurriculumStage(name="beginner", environment="mini_resource"),
            CurriculumStage(name="intermediate", environment="mini_combat"),
            CurriculumStage(name="advanced", environment="full_game"),
        ],
        promotion_elo_threshold=100.0,
        promotion_window=50,
    )


@pytest.fixture
def fixed_seed() -> int:
    """Fixed seed value for reproducible tests."""
    return 42
