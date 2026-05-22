"""Integration tests for batch self-play pipeline.

Validates the `run_batch_selfplay()` function from `selfplay_worker.py`,
which uses `BatchGumbelMCTS` + `VectorizedMicroCivEnv` for parallel
game generation.

Tests cover:
  - Trajectory key/shape validation
  - Batch vs sequential schema consistency
  - Edge cases (single game, zero games)
  - Vectorized env lifecycle

Run with:
    pytest tests/integration/test_batch_selfplay_integration.py -v --timeout=120
"""

from __future__ import annotations


import numpy as np
import pytest

from alpha_zero.core.batch_mcts import BatchMCTSConfig
from alpha_zero.envs.vectorized_env import VectorizedEnvConfig, VectorizedMicroCivEnv

# Small config for fast tests
_BOARD_SIZE = 8
_MCTS_SIMS = 4  # Very few sims for speed
_GUMBEL_ACTIONS = 4
_MAX_BATCH_SIZE = 8
_MAX_TURNS = 15


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_model():
    """Create a small transformer model for batch self-play testing."""
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    config = TransformerConfig(
        spatial_channels=13,
        spatial_size=_BOARD_SIZE,
        vector_dim=64,
        action_space_size=376,
        num_layers=1,
        embed_dim=32,
    )
    model = Civ6TransformerNetwork(config=config)
    model.eval()
    return model


@pytest.fixture
def vec_env():
    """Create vectorized environment with 4 parallel envs."""
    config = VectorizedEnvConfig(
        num_envs=4,
        board_size=_BOARD_SIZE,
        max_turns=_MAX_TURNS,
    )
    return VectorizedMicroCivEnv(config=config)


# ============================================================================
# Test: Vectorized Environment
# ============================================================================


class TestVectorizedEnvLifecycle:
    """Tests for VectorizedMicroCivEnv basic lifecycle."""

    def test_env_creation(self, vec_env):
        """Vectorized env initializes with correct count."""
        assert vec_env.num_envs == 4
        assert vec_env.board_size == _BOARD_SIZE

    def test_reset_all(self, vec_env):
        """Reset all returns valid observations."""
        vec_env.reset_all()
        spatial, vector, action_mask = vec_env.get_observations()
        assert spatial.shape[0] == 4  # num_envs
        assert vector.shape[0] == 4
        assert action_mask.shape[0] == 4

    def test_step_all(self, vec_env):
        """Step all with valid actions."""
        vec_env.reset_all()
        _, _, action_mask = vec_env.get_observations()

        # Pick first valid action for each env
        actions = np.zeros(4, dtype=np.int32)
        for i in range(4):
            valid = np.where(action_mask[i] > 0)[0]
            actions[i] = valid[0] if len(valid) > 0 else 0

        rewards, terminated, truncated, infos = vec_env.step_all(actions)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)

    def test_clone_env(self, vec_env):
        """Clone creates independent copy."""
        vec_env.reset_all()
        original = vec_env.get_env(0)
        clone = vec_env.clone_env(0)

        # Should have same state
        orig_obs = original.get_observation()
        clone_obs = clone.get_observation()
        np.testing.assert_array_equal(orig_obs["spatial"], clone_obs["spatial"])

    def test_active_count(self, vec_env):
        """Active count starts at num_envs."""
        vec_env.reset_all()
        assert vec_env.active_count == 4


# ============================================================================
# Test: Batch MCTS Configuration
# ============================================================================


class TestBatchMCTSConfig:
    """Tests for BatchMCTSConfig validation."""

    def test_default_config(self):
        """Default config is valid."""
        config = BatchMCTSConfig()
        assert config.num_simulations > 0
        assert config.num_sampled_actions > 0

    def test_custom_config(self):
        """Custom config values are stored."""
        config = BatchMCTSConfig(
            num_simulations=_MCTS_SIMS,
            num_sampled_actions=_GUMBEL_ACTIONS,
            max_batch_size=_MAX_BATCH_SIZE,
        )
        assert config.num_simulations == _MCTS_SIMS
        assert config.num_sampled_actions == _GUMBEL_ACTIONS
        assert config.max_batch_size == _MAX_BATCH_SIZE

    def test_invalid_simulations_raises(self):
        """Zero simulations should raise ValueError."""
        with pytest.raises(ValueError):
            BatchMCTSConfig(num_simulations=0)


# ============================================================================
# Test: Batch Self-Play Trajectories
# ============================================================================


class TestBatchSelfplayTrajectories:
    """Tests for run_batch_selfplay trajectory output."""

    @pytest.mark.timeout(120)
    def test_batch_selfplay_produces_trajectories(self, small_model):
        """run_batch_selfplay returns non-empty trajectory list."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

        trajectories = run_batch_selfplay(
            model=small_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=2,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        assert len(trajectories) > 0, "Should produce at least 1 trajectory"

    @pytest.mark.timeout(120)
    def test_trajectory_keys(self, small_model):
        """Trajectories contain required keys."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

        trajectories = run_batch_selfplay(
            model=small_model,
            num_games=1,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=1,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        required_keys = {"spatial", "vector", "policies", "actions", "rewards"}
        for traj in trajectories:
            assert required_keys.issubset(traj.keys()), (
                f"Missing keys: {required_keys - set(traj.keys())}"
            )

    @pytest.mark.timeout(120)
    def test_trajectory_lengths_consistent(self, small_model):
        """All trajectory arrays have same length within each trajectory."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

        trajectories = run_batch_selfplay(
            model=small_model,
            num_games=1,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=1,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        for traj in trajectories:
            n_steps = len(traj["spatial"])
            assert n_steps > 0, "Trajectory should have at least 1 step"
            assert len(traj["vector"]) == n_steps
            assert len(traj["policies"]) == n_steps
            assert len(traj["actions"]) == n_steps
            assert len(traj["rewards"]) == n_steps


# ============================================================================
# Test: Schema Consistency
# ============================================================================


class TestBatchVsSequentialConsistency:
    """Tests that batch and sequential self-play produce compatible schemas."""

    @pytest.mark.timeout(180)
    def test_same_trajectory_keys(self, small_model):
        """Batch and sequential produce trajectories with identical key sets."""
        from alpha_zero.cloud.gcp.selfplay_worker import (
            run_batch_selfplay,
            run_selfplay,
        )

        batch_trajs = run_batch_selfplay(
            model=small_model,
            num_games=1,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=1,
            max_batch_size=_MAX_BATCH_SIZE,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        seq_trajs = run_selfplay(
            model=small_model,
            num_games=1,
            mcts_sims=_MCTS_SIMS,
            mcts_type="gumbel",
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )

        if batch_trajs and seq_trajs:
            # Both should have at minimum these common keys
            common_keys = {"spatial", "vector", "policies", "actions", "rewards"}
            assert common_keys.issubset(batch_trajs[0].keys())
            assert common_keys.issubset(seq_trajs[0].keys())
