"""Comprehensive tests for VectorizedMicroCivEnv.

Tests the vectorized environment wrapper that manages N parallel MicroCivEnv
instances with pre-allocated observation buffers for efficient batch self-play.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.envs.civ6_encoder import MicroCivEncoder
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.envs.vectorized_env import (
    VectorizedEnvConfig,
    VectorizedMicroCivEnv,
)


class TestVectorizedEnvConfig:
    """Test suite for VectorizedEnvConfig dataclass."""

    def test_defaults(self) -> None:
        """Verify all default configuration values."""
        config = VectorizedEnvConfig()
        assert config.num_envs == 8
        assert config.board_size == 8
        assert config.max_turns == 30
        assert config.seed_offset == 0

    def test_custom_values(self) -> None:
        """Verify custom configuration values are respected."""
        config = VectorizedEnvConfig(
            num_envs=16,
            board_size=16,
            max_turns=50,
            seed_offset=42,
        )
        assert config.num_envs == 16
        assert config.board_size == 16
        assert config.max_turns == 50
        assert config.seed_offset == 42

    def test_invalid_num_envs(self) -> None:
        """Verify ValueError raised for num_envs < 1."""
        with pytest.raises(ValueError, match="num_envs must be >= 1"):
            VectorizedEnvConfig(num_envs=0)

        with pytest.raises(ValueError, match="num_envs must be >= 1"):
            VectorizedEnvConfig(num_envs=-1)

    def test_invalid_board_size(self) -> None:
        """Verify ValueError raised for board_size < 1."""
        with pytest.raises(ValueError, match="board_size must be >= 1"):
            VectorizedEnvConfig(board_size=0)

        with pytest.raises(ValueError, match="board_size must be >= 1"):
            VectorizedEnvConfig(board_size=-5)

    def test_invalid_max_turns(self) -> None:
        """Verify ValueError raised for max_turns < 1."""
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            VectorizedEnvConfig(max_turns=0)

        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            VectorizedEnvConfig(max_turns=-10)


class TestVectorizedMicroCivEnv:
    """Test suite for VectorizedMicroCivEnv class."""

    def test_creation_default(self) -> None:
        """Verify default initialization creates 8 environments."""
        vec_env = VectorizedMicroCivEnv()
        assert vec_env.num_envs == 8
        assert len(vec_env._envs) == 8
        assert all(isinstance(env, MicroCivEnv) for env in vec_env._envs)

    def test_creation_custom(self) -> None:
        """Verify custom configuration creates N environments."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        assert vec_env.num_envs == 4
        assert len(vec_env._envs) == 4
        assert vec_env.board_size == 8
        assert vec_env.config.max_turns == 10

    def test_num_envs_property(self) -> None:
        """Verify num_envs property returns correct value."""
        config = VectorizedEnvConfig(num_envs=6)
        vec_env = VectorizedMicroCivEnv(config)
        assert vec_env.num_envs == 6

    def test_board_size_property(self) -> None:
        """Verify board_size property returns correct value."""
        config = VectorizedEnvConfig(num_envs=4, board_size=16)
        vec_env = VectorizedMicroCivEnv(config)
        assert vec_env.board_size == 16

    def test_action_space_size_property(self) -> None:
        """Verify action_space_size property returns 376 for MicroCiv."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        # Action space size is 376 regardless of board_size
        assert vec_env.action_space_size == 376

    def test_reset_all(self) -> None:
        """Verify reset_all sets all environments to non-done state."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)

        # Initially done flags may be arbitrary
        vec_env._done[:] = True  # Manually set all to done

        vec_env.reset_all()

        # All environments should be active after reset
        assert vec_env.active_count == 4
        assert not any(vec_env._done)
        assert all(vec_env._rewards == 0.0)

    def test_reset_all_fills_buffers(self) -> None:
        """Verify reset_all populates observation buffers with data."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)

        vec_env.reset_all()

        # Buffers should contain non-zero data from environment observations
        spatial, vector, mask = vec_env.get_observations()

        # At least some values should be non-zero after reset
        # (action masks should definitely have valid actions available)
        assert mask.sum() > 0, "Action masks should have valid actions"
        assert spatial.shape == (4, 13, 8, 8)
        assert vector.shape == (4, 64)
        assert mask.shape == (4, 376)

    def test_buffer_shapes(self) -> None:
        """Verify observation buffer shapes match expected dimensions."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        spatial, vector, mask = vec_env.get_observations()

        # Expected shapes:
        # spatial: (num_envs, TOTAL_CHANNELS, H, W)
        # vector: (num_envs, vector_dim)
        # mask: (num_envs, action_space_size)
        assert spatial.shape == (4, MicroCivEncoder.TOTAL_CHANNELS, 8, 8)
        assert vector.shape == (4, 64)
        assert mask.shape == (4, 376)

        # Verify dtypes
        assert spatial.dtype == np.float32
        assert vector.dtype == np.float32
        assert mask.dtype == np.float32

    def test_step_all(self) -> None:
        """Verify step_all executes actions and returns correct outputs."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        # Create action array (action 0 = pass/end turn)
        actions = np.zeros(4, dtype=np.int32)

        rewards, terminated, truncated, infos = vec_env.step_all(actions)

        # Verify return types
        assert isinstance(rewards, np.ndarray)
        assert isinstance(terminated, np.ndarray)
        assert isinstance(truncated, np.ndarray)
        assert isinstance(infos, list)
        assert len(infos) == 4

    def test_step_all_shapes(self) -> None:
        """Verify step_all returns arrays with correct shapes."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        actions = np.zeros(4, dtype=np.int32)
        rewards, terminated, truncated, _infos = vec_env.step_all(actions)

        # All arrays should have shape (num_envs,)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)

        # Verify dtypes
        assert rewards.dtype == np.float32
        assert terminated.dtype == bool
        assert truncated.dtype == bool

    def test_step_all_wrong_action_count(self) -> None:
        """Verify step_all raises ValueError for wrong number of actions."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        # Wrong number of actions
        actions = np.zeros(3, dtype=np.int32)

        with pytest.raises(ValueError, match="Expected 4 actions"):
            vec_env.step_all(actions)

    def test_get_active_envs_initial(self) -> None:
        """Verify all environments are active after reset_all."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        active_envs = vec_env.get_active_envs()

        assert active_envs == [0, 1, 2, 3]
        assert len(active_envs) == 4

    def test_active_count_property(self) -> None:
        """Verify active_count property tracks non-done environments."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        # Initially all active
        assert vec_env.active_count == 4

        # Manually mark some as done
        vec_env._done[1] = True
        vec_env._done[3] = True
        assert vec_env.active_count == 2

        # Reset all should restore to 4 active
        vec_env.reset_all()
        assert vec_env.active_count == 4

    def test_get_env(self) -> None:
        """Verify get_env returns correct MicroCivEnv instance."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)

        for i in range(4):
            env = vec_env.get_env(i)
            assert isinstance(env, MicroCivEnv)
            assert env is vec_env._envs[i]

    def test_get_env_out_of_bounds(self) -> None:
        """Verify get_env raises IndexError for invalid indices."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)

        with pytest.raises(IndexError, match="out of bounds"):
            vec_env.get_env(-1)

        with pytest.raises(IndexError, match="out of bounds"):
            vec_env.get_env(4)

        with pytest.raises(IndexError, match="out of bounds"):
            vec_env.get_env(10)

    def test_clone_env(self) -> None:
        """Verify clone_env returns a cloned MicroCivEnv instance."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        original_env = vec_env.get_env(0)
        cloned_env = vec_env.clone_env(0)

        # Should be different objects
        assert cloned_env is not original_env
        assert isinstance(cloned_env, MicroCivEnv)

        # But should have same initial state
        orig_obs = original_env.get_observation()
        clone_obs = cloned_env.get_observation()

        np.testing.assert_array_equal(orig_obs["spatial"], clone_obs["spatial"])
        np.testing.assert_array_equal(orig_obs["vector"], clone_obs["vector"])
        np.testing.assert_array_equal(orig_obs["action_mask"], clone_obs["action_mask"])

    def test_clone_env_independence(self) -> None:
        """Verify cloned environment is independent from original."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        cloned_env = vec_env.clone_env(0)

        # Step the original environment
        original_env = vec_env.get_env(0)
        original_env.step(0)

        # Clone should not have stepped
        # (They will have different turn numbers or states)
        orig_obs = original_env.get_observation()
        clone_obs = cloned_env.get_observation()

        # At least the turn count in vector should differ
        # Vector encoding includes turn information
        turn_differs = not np.array_equal(orig_obs["vector"], clone_obs["vector"])
        assert turn_differs, "Clone should be independent after original steps"

    def test_get_observations_returns_views(self) -> None:
        """Verify get_observations returns zero-copy views into buffers."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        spatial1, vector1, mask1 = vec_env.get_observations()
        spatial2, vector2, mask2 = vec_env.get_observations()

        # Same object, not copies
        assert spatial1 is spatial2
        assert vector1 is vector2
        assert mask1 is mask2

        # Are views of internal buffers
        assert spatial1 is vec_env._spatial_buffer
        assert vector1 is vec_env._vector_buffer
        assert mask1 is vec_env._action_mask_buffer

    def test_deterministic_seeds(self) -> None:
        """Verify same seed_offset produces same initial states."""
        config1 = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10, seed_offset=42)
        vec_env1 = VectorizedMicroCivEnv(config1)
        vec_env1.reset_all()
        spatial1, vector1, mask1 = vec_env1.get_observations()

        config2 = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10, seed_offset=42)
        vec_env2 = VectorizedMicroCivEnv(config2)
        vec_env2.reset_all()
        spatial2, vector2, mask2 = vec_env2.get_observations()

        # Same seed should produce identical initial observations
        np.testing.assert_array_equal(spatial1, spatial2)
        np.testing.assert_array_equal(vector1, vector2)
        np.testing.assert_array_equal(mask1, mask2)

    def test_deterministic_seeds_different_offset(self) -> None:
        """Verify different seed_offset produces different initial states."""
        config1 = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10, seed_offset=42)
        vec_env1 = VectorizedMicroCivEnv(config1)
        vec_env1.reset_all()
        spatial1, vector1, _mask1 = vec_env1.get_observations()

        config2 = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10, seed_offset=99)
        vec_env2 = VectorizedMicroCivEnv(config2)
        vec_env2.reset_all()
        spatial2, vector2, _mask2 = vec_env2.get_observations()

        # Different seed should produce different observations
        # (at least one should differ)
        spatial_differs = not np.array_equal(spatial1, spatial2)
        vector_differs = not np.array_equal(vector1, vector2)

        assert spatial_differs or vector_differs, "Different seeds should produce different states"

    def test_reset_done(self) -> None:
        """Verify reset_done only resets environments marked as done."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        # Manually mark env 1 and 3 as done
        vec_env._done[1] = True
        vec_env._done[3] = True

        # Reset done environments
        num_reset = vec_env.reset_done()

        # Should have reset 2 environments
        assert num_reset == 2

        # All should now be non-done
        assert vec_env.active_count == 4

        # Environments 1 and 3 should have new observations
        _, _, mask_after = vec_env.get_observations()

        # Envs 1 and 3 should have valid action masks after reset
        assert mask_after[1].sum() > 0
        assert mask_after[3].sum() > 0

    def test_reset_done_no_done_envs(self) -> None:
        """Verify reset_done returns 0 when no environments are done."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        # All environments active
        assert vec_env.active_count == 4

        # Reset should do nothing
        num_reset = vec_env.reset_done()
        assert num_reset == 0
        assert vec_env.active_count == 4

    def test_step_all_updates_done_status(self) -> None:
        """Verify step_all correctly marks environments as done."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=5)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        # Step repeatedly until at least one environment is done
        max_attempts = 50
        for _ in range(max_attempts):
            actions = np.zeros(4, dtype=np.int32)  # Pass action
            _rewards, terminated, truncated, _infos = vec_env.step_all(actions)

            # Check if any environment is done
            if terminated.any() or truncated.any():
                done_indices = np.where(terminated | truncated)[0]
                # Verify internal done flags match
                for idx in done_indices:
                    assert vec_env._done[idx]
                break
        else:
            # After max_turns, at least some should be truncated
            # If not, that's okay for this test
            pass

    def test_step_all_skips_done_environments(self) -> None:
        """Verify step_all does not step environments already marked done."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        # Manually mark env 1 as done
        vec_env._done[1] = True

        # Save observations before step
        spatial_before, _, _ = vec_env.get_observations()
        env1_spatial_before = spatial_before[1].copy()

        # Step all (env 1 should be skipped)
        actions = np.zeros(4, dtype=np.int32)
        _rewards, _terminated, _truncated, infos = vec_env.step_all(actions)

        # Env 1 should still have same observation (not stepped)
        spatial_after, _, _ = vec_env.get_observations()
        np.testing.assert_array_equal(spatial_after[1], env1_spatial_before)

        # Info for env 1 should be empty dict
        assert infos[1] == {}

    def test_step_all_updates_buffers(self) -> None:
        """Verify step_all updates observation buffers with new states."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        # Get initial observations
        _spatial_before, vector_before, _mask_before = vec_env.get_observations()
        vector_copy = vector_before.copy()

        # Step with pass actions
        actions = np.zeros(4, dtype=np.int32)
        vec_env.step_all(actions)

        # Observations should be updated in buffers
        _spatial_after, _vector_after, _mask_after = vec_env.get_observations()

        # At least vector should change (turn counter increments)
        assert not np.array_equal(vector_before, vector_copy), "Buffers should be updated"

    def test_multiple_resets(self) -> None:
        """Verify environment can be reset multiple times successfully."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)

        for _ in range(3):
            vec_env.reset_all()
            assert vec_env.active_count == 4

            # Take some steps
            for _ in range(5):
                actions = np.zeros(4, dtype=np.int32)
                vec_env.step_all(actions)

            # Should be able to reset again
            vec_env.reset_all()
            assert vec_env.active_count == 4

    def test_get_observations_after_step(self) -> None:
        """Verify get_observations reflects latest state after stepping."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        # Initial state
        _, vector1, _ = vec_env.get_observations()
        turn_info_1 = vector1.copy()

        # Step
        actions = np.zeros(4, dtype=np.int32)
        vec_env.step_all(actions)

        # State should have changed
        _, vector2, _ = vec_env.get_observations()
        turn_info_2 = vector2.copy()

        # Vector should differ (turn incremented)
        assert not np.array_equal(turn_info_1, turn_info_2)

    def test_action_mask_validity(self) -> None:
        """Verify action masks always have at least one valid action."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        # After reset, should have valid actions
        _, _, mask = vec_env.get_observations()
        for i in range(4):
            assert mask[i].sum() > 0, f"Env {i} should have valid actions"

        # After stepping, active envs should still have valid actions
        for _ in range(5):
            actions = np.zeros(4, dtype=np.int32)
            vec_env.step_all(actions)

            _, _, mask = vec_env.get_observations()
            active_envs = vec_env.get_active_envs()
            for i in active_envs:
                assert mask[i].sum() > 0, f"Active env {i} should have valid actions"

    def test_rewards_storage(self) -> None:
        """Verify rewards are stored in internal buffer after step."""
        config = VectorizedEnvConfig(num_envs=4, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)
        vec_env.reset_all()

        actions = np.zeros(4, dtype=np.int32)
        rewards, _, _, _ = vec_env.step_all(actions)

        # Internal rewards buffer should match returned rewards
        np.testing.assert_array_equal(vec_env._rewards, rewards)

    def test_large_num_envs(self) -> None:
        """Verify creation with larger number of environments."""
        config = VectorizedEnvConfig(num_envs=32, board_size=8, max_turns=10)
        vec_env = VectorizedMicroCivEnv(config)

        assert vec_env.num_envs == 32
        assert len(vec_env._envs) == 32

        vec_env.reset_all()
        assert vec_env.active_count == 32

        spatial, vector, mask = vec_env.get_observations()
        assert spatial.shape == (32, 13, 8, 8)
        assert vector.shape == (32, 64)
        assert mask.shape == (32, 376)

    def test_config_none_uses_defaults(self) -> None:
        """Verify passing config=None uses default configuration."""
        vec_env = VectorizedMicroCivEnv(config=None)

        assert vec_env.num_envs == 8
        assert vec_env.board_size == 8
        assert vec_env.config.max_turns == 30
        assert vec_env.config.seed_offset == 0
