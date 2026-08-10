"""Unit tests for environment tuning module."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.utils.environment_tuning import (
    CombatConfig,
    EarlyTerminationChecker,
    EarlyTerminationConfig,
    EnvironmentTuningConfig,
    ProductionConfig,
    TunedEnvironmentWrapper,
    create_tuned_environment,
    get_default_balanced_config,
    get_default_fast_config,
    get_default_realistic_config,
)


class TestEnvironmentTuningConfig:
    """Tests for EnvironmentTuningConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnvironmentTuningConfig()

        assert config.enabled is True
        assert config.max_turns == 100
        assert config.fog_of_war_enabled is True
        assert config.early_termination.enabled is True
        assert config.production.multiplier == 1.0
        assert config.combat.simplified is False

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "enabled": True,
            "max_turns": 50,
            "fog_of_war_enabled": False,
            "early_termination": {
                "enabled": True,
                "advantage_threshold": 2,
                "min_turns": 5,
            },
            "production": {
                "multiplier": 2.0,
            },
            "combat": {
                "simplified": True,
                "damage_multiplier": 1.5,
            },
        }

        config = EnvironmentTuningConfig.from_dict(data)

        assert config.max_turns == 50
        assert config.fog_of_war_enabled is False
        assert config.early_termination.advantage_threshold == 2
        assert config.production.multiplier == 2.0
        assert config.combat.simplified is True

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = EnvironmentTuningConfig(
            max_turns=75,
            early_termination=EarlyTerminationConfig(advantage_threshold=5),
        )

        data = config.to_dict()

        assert data["max_turns"] == 75
        assert data["early_termination"]["advantage_threshold"] == 5
        assert "production" in data
        assert "combat" in data

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = EnvironmentTuningConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_max_turns(self):
        """Test validation catches invalid max_turns."""
        config = EnvironmentTuningConfig(max_turns=0)
        errors = config.validate()
        assert any("max_turns" in e for e in errors)

    def test_validate_invalid_production_multiplier(self):
        """Test validation catches invalid production multiplier."""
        config = EnvironmentTuningConfig(production=ProductionConfig(multiplier=-1.0))
        errors = config.validate()
        assert any("production" in e.lower() or "multiplier" in e.lower() for e in errors)


class TestEarlyTerminationChecker:
    """Tests for EarlyTerminationChecker."""

    def test_disabled_termination(self):
        """Test that disabled checker never terminates."""
        config = EarlyTerminationConfig(enabled=False)
        checker = EarlyTerminationChecker(config)

        should_terminate, _winner, _reason = checker.check(
            turn=100,
            player_units=10,
            player_cities=3,
            opponent_units=0,
            opponent_cities=0,
        )

        assert should_terminate is False

    def test_min_turns_respected(self):
        """Test that advantage-based termination doesn't occur before min_turns."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=20,
            advantage_threshold=2,
            check_interval=5,
        )
        checker = EarlyTerminationChecker(config)

        # Big advantage but both players have cities (not elimination)
        # Should NOT terminate before min_turns
        should_terminate, _, _ = checker.check(
            turn=5,  # Turn is on interval but before min_turns
            player_units=10,
            player_cities=3,
            opponent_units=1,
            opponent_cities=1,  # Opponent has cities so not elimination
        )

        assert should_terminate is False

    def test_player_advantage_termination(self):
        """Test termination on player advantage (not elimination)."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=5,
            advantage_threshold=3,
            check_interval=5,
        )
        checker = EarlyTerminationChecker(config)

        # Big advantage but both have cities (not elimination)
        should_terminate, winner, reason = checker.check(
            turn=10,
            player_units=6,
            player_cities=3,
            opponent_units=1,
            opponent_cities=1,  # Opponent still has cities so not elimination
        )

        assert should_terminate is True
        assert winner == 0  # Player wins
        assert "advantage" in reason.lower()

    def test_opponent_advantage_termination(self):
        """Test termination on opponent advantage."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=5,
            advantage_threshold=3,
            check_interval=5,
        )
        checker = EarlyTerminationChecker(config)

        should_terminate, winner, _reason = checker.check(
            turn=10,
            player_units=1,
            player_cities=0,
            opponent_units=5,
            opponent_cities=2,
        )

        assert should_terminate is True
        assert winner == 1  # Opponent wins

    def test_stalemate_detection(self):
        """Test stalemate detection."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=5,
            stalemate_threshold=3,
            check_interval=1,
        )
        checker = EarlyTerminationChecker(config)

        # Same state repeatedly
        for i in range(4):
            should_terminate, winner, reason = checker.check(
                turn=10 + i,
                player_units=2,
                player_cities=1,
                opponent_units=2,
                opponent_cities=1,
            )

        assert should_terminate is True
        assert winner is None  # Draw
        assert "stalemate" in reason.lower()

    def test_reset(self):
        """Test checker reset."""
        config = EarlyTerminationConfig(enabled=True, stalemate_threshold=2)
        checker = EarlyTerminationChecker(config)

        # Build up stalemate
        for _ in range(2):
            checker.check(10, 2, 1, 2, 1)

        checker.reset()

        # After reset, stalemate counter should be zero
        should_terminate, _, _ = checker.check(10, 2, 1, 2, 1)
        assert should_terminate is False


class TestEliminationConditions:
    """Tests for immediate elimination detection (Phase 3 fix)."""

    def test_opponent_elimination_immediate(self):
        """Test opponent elimination is detected immediately, not just on interval."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=5,
            check_interval=10,  # Interval is 10, but elimination should be immediate
        )
        checker = EarlyTerminationChecker(config)

        # Turn 7 is not on interval (7 % 10 != 0), but elimination should still trigger
        # Elimination requires BOTH 0 cities AND 0 units
        should_terminate, winner, reason = checker.check(
            turn=7,
            player_units=5,
            player_cities=2,
            opponent_units=0,  # No units
            opponent_cities=0,  # No cities = fully eliminated
        )

        assert should_terminate is True
        assert winner == 0  # Player wins
        assert "eliminated" in reason.lower()

    def test_player_elimination_immediate(self):
        """Test player elimination is detected immediately."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=5,
            check_interval=10,
        )
        checker = EarlyTerminationChecker(config)

        # Turn 13 is not on interval
        should_terminate, winner, reason = checker.check(
            turn=13,
            player_units=0,
            player_cities=0,  # Player eliminated
            opponent_units=5,
            opponent_cities=2,
        )

        assert should_terminate is True
        assert winner == 1  # Opponent wins
        assert "eliminated" in reason.lower()

    def test_mutual_elimination_immediate(self):
        """Test mutual elimination is detected immediately."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=5,
            check_interval=10,
        )
        checker = EarlyTerminationChecker(config)

        # Mutual elimination requires BOTH players to have 0 cities AND 0 units
        should_terminate, winner, reason = checker.check(
            turn=8,
            player_units=0,  # No units
            player_cities=0,  # No cities
            opponent_units=0,  # No units
            opponent_cities=0,  # No cities
        )

        assert should_terminate is True
        assert winner is None  # Draw
        assert "mutual" in reason.lower()

    def test_elimination_detected_before_min_turns(self):
        """Test elimination is NOT detected before min_turns (only advantage is blocked)."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=20,
            check_interval=5,
        )
        checker = EarlyTerminationChecker(config)

        # Turn 5 is before min_turns, but elimination should still trigger
        should_terminate, winner, reason = checker.check(
            turn=5,
            player_units=5,
            player_cities=2,
            opponent_units=0,
            opponent_cities=0,  # Opponent eliminated
        )

        # Elimination should be detected even before min_turns
        assert should_terminate is True
        assert winner == 0
        assert "eliminated" in reason.lower()

    def test_advantage_still_respects_interval(self):
        """Test advantage-based termination still respects check_interval."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=5,
            advantage_threshold=3,
            check_interval=5,
        )
        checker = EarlyTerminationChecker(config)

        # Turn 7 is not on interval - advantage should NOT trigger
        should_terminate, winner, _reason = checker.check(
            turn=7,
            player_units=10,
            player_cities=5,
            opponent_units=1,
            opponent_cities=1,  # Big advantage but both have cities
        )

        assert should_terminate is False

        # Turn 10 is on interval - advantage SHOULD trigger
        should_terminate, winner, _reason = checker.check(
            turn=10,
            player_units=10,
            player_cities=5,
            opponent_units=1,
            opponent_cities=1,
        )

        assert should_terminate is True
        assert winner == 0


class TestHashBasedStalemate:
    """Tests for hash-based stalemate detection (Phase 3 fix)."""

    def test_stalemate_with_same_state_hash(self):
        """Test stalemate detection uses hash comparison."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=5,
            stalemate_threshold=3,
            check_interval=1,
        )
        checker = EarlyTerminationChecker(config)

        # Same state multiple times
        for i in range(4):
            should_terminate, winner, reason = checker.check(
                turn=10 + i,
                player_units=2,
                player_cities=1,
                opponent_units=2,
                opponent_cities=1,
            )

        assert should_terminate is True
        assert winner is None
        assert "stalemate" in reason.lower()

    def test_stalemate_reset_on_state_change(self):
        """Test stalemate counter resets when state changes."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=5,
            stalemate_threshold=5,
            check_interval=1,
        )
        checker = EarlyTerminationChecker(config)

        # Build up stalemate counter
        for i in range(3):
            checker.check(10 + i, 2, 1, 2, 1)

        # State changes - counter should reset
        checker.check(13, 3, 1, 2, 1)  # Different player_units

        # Continue with new state - should need full threshold again
        # Need 5 consecutive same states to trigger stalemate (threshold=5)
        for i in range(5):
            should_terminate, _, _ = checker.check(14 + i, 3, 1, 2, 1)

        # After 6 turns with same state (13-18), should be stalemate
        assert should_terminate is True

    def test_hash_consistent_for_same_values(self):
        """Test that hash is consistent for same game state values."""
        config = EarlyTerminationConfig(
            enabled=True,
            min_turns=5,
            stalemate_threshold=2,
            check_interval=1,
        )
        checker = EarlyTerminationChecker(config)

        # Same values should produce same hash
        checker.check(10, 5, 2, 3, 1)
        checker.check(11, 5, 2, 3, 1)  # Exact same values

        should_terminate, _, _ = checker.check(12, 5, 2, 3, 1)
        assert should_terminate is True  # Stalemate detected


class TestTunedEnvironmentWrapper:
    """Tests for TunedEnvironmentWrapper."""

    @pytest.fixture
    def base_env(self):
        """Create base MicroCiv environment."""
        from alpha_zero.envs.microciv import MicroCivEnv

        return MicroCivEnv(board_size=8, max_turns=100)

    def test_wrapper_creation(self, base_env):
        """Test wrapper can be created."""
        config = EnvironmentTuningConfig(max_turns=50)
        wrapper = TunedEnvironmentWrapper(base_env, config)

        assert wrapper.env.max_turns == 50

    def test_wrapper_reset(self, base_env):
        """Test wrapper reset."""
        config = EnvironmentTuningConfig()
        wrapper = TunedEnvironmentWrapper(base_env, config)

        obs, info = wrapper.reset()

        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs
        assert info["tuning_enabled"] is True

    def test_wrapper_step(self, base_env):
        """Test wrapper step."""
        config = EnvironmentTuningConfig()
        wrapper = TunedEnvironmentWrapper(base_env, config)

        wrapper.reset()

        # Get valid action
        action_mask = wrapper.env.get_valid_actions()
        valid_actions = np.where(action_mask > 0)[0]
        action = valid_actions[0] if len(valid_actions) > 0 else 0

        obs, _reward, _terminated, _truncated, info = wrapper.step(action)

        assert "spatial" in obs
        assert "episode_stats" in info

    def test_disabled_tuning(self, base_env):
        """Test wrapper with tuning disabled."""
        config = EnvironmentTuningConfig(enabled=False)
        wrapper = TunedEnvironmentWrapper(base_env, config)

        _obs, info = wrapper.reset()

        assert info["tuning_enabled"] is False

    def test_episode_stats(self, base_env):
        """Test episode statistics tracking."""
        config = EnvironmentTuningConfig()
        wrapper = TunedEnvironmentWrapper(base_env, config)

        wrapper.reset()

        stats = wrapper.get_episode_stats()

        assert "turns" in stats
        assert "early_termination" in stats


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tuned_environment(self):
        """Test create_tuned_environment factory."""
        config = EnvironmentTuningConfig(max_turns=30, board_size=8)
        env = create_tuned_environment(config)

        assert env.env.max_turns == 30
        assert env.env.board_size == 8

    def test_create_with_invalid_env_type(self):
        """Test create_tuned_environment with invalid env type."""
        with pytest.raises(ValueError, match="Unsupported"):
            create_tuned_environment(env_type="invalid")

    def test_create_with_invalid_config(self):
        """Test create_tuned_environment with invalid config."""
        config = EnvironmentTuningConfig(max_turns=-1)

        with pytest.raises(ValueError, match="Invalid"):
            create_tuned_environment(config)

    def test_get_default_fast_config(self):
        """Test fast config factory."""
        config = get_default_fast_config()

        assert config.max_turns == 50
        assert config.early_termination.enabled is True
        assert config.production.multiplier > 1.0

    def test_get_default_balanced_config(self):
        """Test balanced config factory."""
        config = get_default_balanced_config()

        assert config.max_turns == 100
        assert config.early_termination.enabled is True

    def test_get_default_realistic_config(self):
        """Test realistic config factory."""
        config = get_default_realistic_config()

        assert config.max_turns == 200
        assert config.early_termination.enabled is False
        assert config.production.multiplier == 1.0


class TestProductionConfig:
    """Tests for ProductionConfig."""

    def test_default_unit_costs(self):
        """Test default unit costs."""
        config = ProductionConfig()

        assert "warrior" in config.unit_costs
        assert "settler" in config.unit_costs
        assert "archer" in config.unit_costs

    def test_custom_unit_costs(self):
        """Test custom unit costs."""
        config = ProductionConfig(unit_costs={"warrior": 1, "settler": 2})

        assert config.unit_costs["warrior"] == 1
        assert config.unit_costs["settler"] == 2


class TestCombatConfig:
    """Tests for CombatConfig."""

    def test_default_values(self):
        """Test default combat values."""
        config = CombatConfig()

        assert config.simplified is False
        assert config.damage_multiplier == 1.0
        assert config.min_damage == 10

    def test_simplified_mode(self):
        """Test simplified combat mode."""
        config = CombatConfig(simplified=True, damage_multiplier=2.0)

        assert config.simplified is True
        assert config.damage_multiplier == 2.0
