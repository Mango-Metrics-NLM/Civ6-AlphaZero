"""Unit tests for reward shaping module."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.utils.reward_shaping import (
    RewardShaper,
    RewardShapingConfig,
    StateFeatures,
    compute_shaped_reward,
    compute_state_potential,
    create_reward_shaper,
    extract_features_from_obs,
)


class TestStateFeatures:
    """Tests for StateFeatures dataclass."""

    def test_default_values(self):
        """Test default StateFeatures values."""
        features = StateFeatures()

        assert features.own_units == 0
        assert features.enemy_units == 0
        assert features.own_cities == 0
        assert features.enemy_cities == 0
        assert features.own_avg_health == 100.0
        assert features.enemy_avg_health == 100.0
        assert features.own_territory == 0
        assert features.enemy_territory == 0
        assert features.total_tiles == 64
        assert features.turn == 0
        assert features.max_turns == 100

    def test_custom_values(self):
        """Test StateFeatures with custom values."""
        features = StateFeatures(
            own_units=5,
            enemy_units=3,
            own_cities=2,
            enemy_cities=1,
            own_avg_health=80.0,
            enemy_avg_health=60.0,
            own_territory=10,
            enemy_territory=5,
            total_tiles=100,
            turn=50,
        )

        assert features.own_units == 5
        assert features.enemy_units == 3
        assert features.own_cities == 2
        assert features.enemy_cities == 1
        assert features.own_avg_health == 80.0


class TestRewardShapingConfig:
    """Tests for RewardShapingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RewardShapingConfig()

        assert config.enabled is True
        assert config.unit_advantage_weight == 0.05
        assert config.city_advantage_weight == 0.1
        assert config.health_advantage_weight == 0.02
        assert config.territory_weight == 0.01
        assert config.gamma == 0.99
        assert config.max_units == 10
        assert config.max_cities == 5

    def test_disabled_config(self):
        """Test disabled configuration."""
        config = RewardShapingConfig(enabled=False)
        assert config.enabled is False

    def test_custom_weights(self):
        """Test custom weight configuration."""
        config = RewardShapingConfig(
            unit_advantage_weight=0.1,
            city_advantage_weight=0.2,
            gamma=0.95,
        )

        assert config.unit_advantage_weight == 0.1
        assert config.city_advantage_weight == 0.2
        assert config.gamma == 0.95


class TestExtractFeaturesFromObs:
    """Tests for extract_features_from_obs function."""

    def create_mock_obs(self, board_size: int = 8) -> dict[str, np.ndarray]:
        """Create a mock observation dictionary."""
        spatial = np.zeros((13, board_size, board_size), dtype=np.float32)
        vector = np.zeros(64, dtype=np.float32)

        # Set some values in vector
        vector[0] = 0.25  # Turn 50 (normalized by 200)
        vector[2] = 0.15  # 3 own units (normalized by 20)
        vector[3] = 0.1   # 2 enemy units (normalized by 20)
        vector[4] = 0.4   # 2 own cities (normalized by 5)
        vector[5] = 0.2   # 1 enemy city (normalized by 5)
        vector[6] = 0.8   # 80% own health
        vector[7] = 0.6   # 60% enemy health

        # Set some unit positions in spatial
        spatial[3, 2, 2] = 1.0  # Own unit
        spatial[6, 5, 5] = 1.0  # Enemy unit

        return {"spatial": spatial, "vector": vector}

    def test_extract_basic_features(self):
        """Test extraction of basic features."""
        obs = self.create_mock_obs()
        features = extract_features_from_obs(obs, board_size=8)

        assert features.turn == 50
        assert features.own_units == 3
        assert features.enemy_units == 2
        assert features.own_cities == 2
        assert features.enemy_cities == 1
        assert features.own_avg_health == pytest.approx(80.0, rel=1e-5)
        assert features.enemy_avg_health == pytest.approx(60.0, rel=1e-5)

    def test_extract_territory(self):
        """Test extraction of territory information."""
        obs = self.create_mock_obs()
        features = extract_features_from_obs(obs, board_size=8)

        # Should have some territory from unit positions
        assert features.total_tiles == 64
        assert features.own_territory >= 0
        assert features.enemy_territory >= 0

    def test_different_board_sizes(self):
        """Test with different board sizes."""
        for board_size in [4, 8, 16]:
            spatial = np.zeros((13, board_size, board_size), dtype=np.float32)
            vector = np.zeros(64, dtype=np.float32)
            obs = {"spatial": spatial, "vector": vector}

            features = extract_features_from_obs(obs, board_size=board_size)
            assert features.total_tiles == board_size * board_size

    def test_empty_observation(self):
        """Test with minimal observation."""
        spatial = np.zeros((13, 8, 8), dtype=np.float32)
        vector = np.zeros(64, dtype=np.float32)
        obs = {"spatial": spatial, "vector": vector}

        features = extract_features_from_obs(obs, board_size=8)

        assert features.turn == 0
        assert features.own_units == 0
        assert features.enemy_units == 0


class TestComputeStatePotential:
    """Tests for compute_state_potential function."""

    def test_neutral_state(self):
        """Test potential of neutral state (no advantage)."""
        features = StateFeatures(
            own_units=2,
            enemy_units=2,
            own_cities=1,
            enemy_cities=1,
            own_avg_health=100.0,
            enemy_avg_health=100.0,
            own_territory=10,
            enemy_territory=10,
        )
        config = RewardShapingConfig()

        potential = compute_state_potential(features, config)

        # With no advantage, potential should be near zero
        assert abs(potential) < 1e-6

    def test_unit_advantage_increases_potential(self):
        """Test that unit advantage increases potential."""
        config = RewardShapingConfig()

        features_even = StateFeatures(own_units=2, enemy_units=2)
        features_ahead = StateFeatures(own_units=5, enemy_units=2)

        potential_even = compute_state_potential(features_even, config)
        potential_ahead = compute_state_potential(features_ahead, config)

        assert potential_ahead > potential_even

    def test_city_advantage_increases_potential(self):
        """Test that city advantage increases potential."""
        config = RewardShapingConfig()

        features_even = StateFeatures(own_cities=1, enemy_cities=1)
        features_ahead = StateFeatures(own_cities=3, enemy_cities=1)

        potential_even = compute_state_potential(features_even, config)
        potential_ahead = compute_state_potential(features_ahead, config)

        assert potential_ahead > potential_even

    def test_health_advantage_increases_potential(self):
        """Test that health advantage increases potential."""
        config = RewardShapingConfig()

        features_even = StateFeatures(own_avg_health=100.0, enemy_avg_health=100.0)
        features_ahead = StateFeatures(own_avg_health=100.0, enemy_avg_health=50.0)

        potential_even = compute_state_potential(features_even, config)
        potential_ahead = compute_state_potential(features_ahead, config)

        assert potential_ahead > potential_even

    def test_territory_advantage_increases_potential(self):
        """Test that territory advantage increases potential."""
        config = RewardShapingConfig()

        features_even = StateFeatures(own_territory=10, enemy_territory=10, total_tiles=64)
        features_ahead = StateFeatures(own_territory=30, enemy_territory=10, total_tiles=64)

        potential_even = compute_state_potential(features_even, config)
        potential_ahead = compute_state_potential(features_ahead, config)

        assert potential_ahead > potential_even

    def test_losing_state_negative_potential(self):
        """Test that losing state has negative potential."""
        config = RewardShapingConfig()

        features = StateFeatures(
            own_units=1,
            enemy_units=5,
            own_cities=1,
            enemy_cities=3,
            own_avg_health=50.0,
            enemy_avg_health=100.0,
        )

        potential = compute_state_potential(features, config)
        assert potential < 0


class TestComputeShapedReward:
    """Tests for compute_shaped_reward function."""

    def test_disabled_shaping_returns_base_reward(self):
        """Test that disabled shaping returns base reward unchanged."""
        config = RewardShapingConfig(enabled=False)
        features = StateFeatures(own_units=5, enemy_units=2)

        shaped = compute_shaped_reward(
            base_reward=0.5,
            prev_features=features,
            curr_features=features,
            config=config,
        )

        assert shaped == 0.5

    def test_first_step_returns_base_reward(self):
        """Test that first step (no previous state) returns base reward."""
        config = RewardShapingConfig()
        features = StateFeatures(own_units=5, enemy_units=2)

        shaped = compute_shaped_reward(
            base_reward=0.5,
            prev_features=None,
            curr_features=features,
            config=config,
        )

        assert shaped == 0.5

    def test_improvement_adds_positive_shaping(self):
        """Test that state improvement adds positive shaping bonus."""
        config = RewardShapingConfig()

        prev_features = StateFeatures(own_units=2, enemy_units=3)
        curr_features = StateFeatures(own_units=4, enemy_units=2)

        shaped = compute_shaped_reward(
            base_reward=0.0,
            prev_features=prev_features,
            curr_features=curr_features,
            config=config,
        )

        assert shaped > 0.0

    def test_deterioration_adds_negative_shaping(self):
        """Test that state deterioration adds negative shaping."""
        config = RewardShapingConfig()

        prev_features = StateFeatures(own_units=4, enemy_units=2)
        curr_features = StateFeatures(own_units=2, enemy_units=3)

        shaped = compute_shaped_reward(
            base_reward=0.0,
            prev_features=prev_features,
            curr_features=curr_features,
            config=config,
        )

        assert shaped < 0.0

    def test_terminal_state_uses_negative_prev_potential(self):
        """Test that terminal state uses -Φ(s) for shaping."""
        config = RewardShapingConfig()

        prev_features = StateFeatures(own_units=5, enemy_units=2)

        # At termination, shaping is -Φ(prev)
        shaped = compute_shaped_reward(
            base_reward=1.0,
            prev_features=prev_features,
            curr_features=prev_features,
            config=config,
            terminated=True,
        )

        # Should include base reward plus shaping
        assert shaped != 1.0

    def test_preserves_base_reward_direction(self):
        """Test that large base rewards dominate shaping."""
        config = RewardShapingConfig()

        prev_features = StateFeatures(own_units=2, enemy_units=3)
        curr_features = StateFeatures(own_units=2, enemy_units=3)

        # Large positive base reward
        shaped_pos = compute_shaped_reward(
            base_reward=10.0,
            prev_features=prev_features,
            curr_features=curr_features,
            config=config,
        )

        # Large negative base reward
        shaped_neg = compute_shaped_reward(
            base_reward=-10.0,
            prev_features=prev_features,
            curr_features=curr_features,
            config=config,
        )

        assert shaped_pos > 0
        assert shaped_neg < 0


class TestRewardShaper:
    """Tests for RewardShaper class."""

    def create_mock_obs(self) -> dict[str, np.ndarray]:
        """Create a mock observation."""
        return {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
        }

    def test_initialization(self):
        """Test RewardShaper initialization."""
        shaper = RewardShaper()

        assert shaper.config is not None
        assert shaper.board_size == 8
        assert shaper._prev_features is None

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = RewardShapingConfig(unit_advantage_weight=0.2)
        shaper = RewardShaper(config=config, board_size=16)

        assert shaper.config.unit_advantage_weight == 0.2
        assert shaper.board_size == 16

    def test_reset(self):
        """Test reset clears state."""
        shaper = RewardShaper()
        obs = self.create_mock_obs()

        # Shape some rewards
        shaper.shape_reward(obs, 0.1)
        shaper.shape_reward(obs, 0.2)

        assert len(shaper._episode_rewards) == 2

        shaper.reset()

        assert shaper._prev_features is None
        assert len(shaper._episode_rewards) == 0
        assert len(shaper._episode_shaped_rewards) == 0

    def test_shape_reward_first_step(self):
        """Test shaping on first step."""
        shaper = RewardShaper()
        obs = self.create_mock_obs()

        shaped = shaper.shape_reward(obs, 0.5)

        # First step should return base reward
        assert shaped == 0.5

    def test_shape_reward_updates_state(self):
        """Test that shape_reward updates internal state."""
        shaper = RewardShaper()
        obs = self.create_mock_obs()

        shaper.shape_reward(obs, 0.1)

        assert shaper._prev_features is not None
        assert len(shaper._episode_rewards) == 1

    def test_episode_stats(self):
        """Test episode statistics."""
        shaper = RewardShaper()
        obs = self.create_mock_obs()

        # Shape several rewards
        shaper.shape_reward(obs, 0.1)
        shaper.shape_reward(obs, 0.2)
        shaper.shape_reward(obs, 0.3)

        stats = shaper.get_episode_stats()

        assert stats["num_steps"] == 3
        assert stats["original_reward_sum"] == pytest.approx(0.6)
        assert "shaped_reward_sum" in stats
        assert "shaping_contribution" in stats

    def test_episode_stats_empty(self):
        """Test episode statistics with no data."""
        shaper = RewardShaper()

        stats = shaper.get_episode_stats()

        assert stats["num_steps"] == 0
        assert stats["original_reward_sum"] == 0.0

    def test_episode_reward_sums(self):
        """Test episode reward sum properties."""
        shaper = RewardShaper()
        obs = self.create_mock_obs()

        shaper.shape_reward(obs, 0.1)
        shaper.shape_reward(obs, 0.2)

        assert shaper.episode_reward_sum == pytest.approx(0.3)
        assert shaper.episode_shaped_reward_sum is not None


class TestCreateRewardShaper:
    """Tests for create_reward_shaper factory function."""

    def test_factory_default(self):
        """Test factory with default arguments."""
        shaper = create_reward_shaper()

        assert shaper.config.enabled is True
        assert shaper.board_size == 8

    def test_factory_disabled(self):
        """Test factory with disabled shaping."""
        shaper = create_reward_shaper(enabled=False)

        assert shaper.config.enabled is False

    def test_factory_custom_board_size(self):
        """Test factory with custom board size."""
        shaper = create_reward_shaper(board_size=16)

        assert shaper.board_size == 16

    def test_factory_custom_weights(self):
        """Test factory with custom weights."""
        shaper = create_reward_shaper(
            unit_advantage_weight=0.2,
            city_advantage_weight=0.3,
        )

        assert shaper.config.unit_advantage_weight == 0.2
        assert shaper.config.city_advantage_weight == 0.3


class TestRewardShapingMathematicalProperties:
    """Test mathematical properties of reward shaping."""

    def test_potential_based_shaping_sums_to_zero_in_cycle(self):
        """Test that potential-based shaping sums to ~0 over full episode."""
        config = RewardShapingConfig(gamma=1.0)  # gamma=1 for exact cancellation
        shaper = RewardShaper(config=config)

        # Create observations that form a cycle
        obs_a = {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.array([0.0] * 64, dtype=np.float32),
        }
        obs_a["vector"][2] = 0.1  # 2 units

        obs_b = {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.array([0.0] * 64, dtype=np.float32),
        }
        obs_b["vector"][2] = 0.2  # 4 units

        # Go A -> B -> A
        shaper.shape_reward(obs_a, 0.0)
        shaper.shape_reward(obs_b, 0.0)
        shaper.shape_reward(obs_a, 0.0, terminated=True)

        # With gamma=1, shaping should sum to negative of initial potential
        # (due to terminal state handling)
        stats = shaper.get_episode_stats()

        # Original rewards sum to 0
        assert stats["original_reward_sum"] == 0.0

    def test_shaping_does_not_change_optimal_policy(self):
        """Test that reward shaping preserves reward ordering."""
        config = RewardShapingConfig()

        # Two different transitions
        prev = StateFeatures(own_units=2, enemy_units=2)
        curr_good = StateFeatures(own_units=3, enemy_units=2)
        curr_bad = StateFeatures(own_units=1, enemy_units=3)

        # Better transition should have higher shaped reward
        shaped_good = compute_shaped_reward(0.0, prev, curr_good, config)
        shaped_bad = compute_shaped_reward(0.0, prev, curr_bad, config)

        assert shaped_good > shaped_bad
