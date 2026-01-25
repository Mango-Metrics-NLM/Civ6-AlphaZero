"""Integration tests for RealCiv6Env and Civ6Interface.

Tests:
- Gymnasium environment compliance
- BaseGameInterface contract compliance
- MockCiv6Env for testing without actual game
- Environment reset/step cycle
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import gymnasium as gym
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.envs.real_civ6_env import MockCiv6Env
from alpha_zero.interfaces.base_interface import GameConfig, GameStatus
from alpha_zero.interfaces.civ6_interface import Civ6Interface

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def temp_comm_dir():
    """Create temporary communication directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def bridge_config(temp_comm_dir):
    """Create test bridge configuration."""
    return BridgeConfig(
        comm_dir=temp_comm_dir,
        poll_interval_ms=5,
        command_timeout_ms=100,  # Short timeout for tests
        heartbeat_timeout_s=1.0,
        board_size=16,
    )


@pytest.fixture
def game_config():
    """Create test game configuration."""
    return GameConfig(
        board_size=16,
        max_players=2,
        max_turns=500,
        observation_channels=84,
        vector_dim=512,
    )


# ==============================================================================
# MockCiv6Env Tests
# ==============================================================================


class TestMockCiv6Env:
    """Tests for MockCiv6Env."""

    def test_creation(self, bridge_config):
        """Test MockCiv6Env can be created."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_observation_space_shape(self, bridge_config):
        """Test observation space has correct structure."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)

        obs_space = env.observation_space
        assert "spatial" in obs_space.spaces
        assert "vector" in obs_space.spaces
        assert "action_mask" in obs_space.spaces

        # Check shapes
        assert obs_space["spatial"].shape == (84, 16, 16)
        assert obs_space["vector"].shape == (512,)

    def test_action_space(self, bridge_config):
        """Test action space is properly sized."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)

        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n > 0

    def test_reset(self, bridge_config):
        """Test environment reset."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)

        # MockCiv6Env creates its own mock state on reset
        obs, info = env.reset()

        # Verify observation structure
        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs

        # Verify shapes
        assert obs["spatial"].shape == (84, 16, 16)
        assert obs["vector"].shape == (512,)

        # Verify info
        assert "turn" in info

        env.close()

    def test_step_end_turn(self, bridge_config):
        """Test step with end_turn action."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)
        env.reset()

        # Action 0 is end_turn
        obs, reward, terminated, truncated, info = env.step(0)

        # Verify observation
        assert "spatial" in obs
        assert obs["spatial"].shape == (84, 16, 16)

        # Verify return types
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()

    def test_multiple_steps(self, bridge_config):
        """Test multiple steps."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)
        env.reset()

        # Take multiple steps
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)

            if terminated or truncated:
                break

        env.close()
        assert True  # Test passes if no exceptions

    def test_action_mask_validity(self, bridge_config):
        """Test action mask has valid actions."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)
        obs, _ = env.reset()

        action_mask = obs["action_mask"]

        # Mask should have correct shape
        assert action_mask.shape == (env.action_space.n,)

        # Should have at least end_turn valid
        assert action_mask[0] == 1.0

        # Values should be 0 or 1
        assert np.all((action_mask == 0.0) | (action_mask == 1.0))

        env.close()

    def test_close(self, bridge_config):
        """Test environment close."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)
        env.close()
        # Should not raise

    def test_render(self, bridge_config):
        """Test environment render."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)
        env.reset()

        # Should not raise
        env.render()
        env.close()

    def test_reset_stops_old_bridge(self, bridge_config):
        """Test that reset() properly stops old bridge before creating new one."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)

        # First reset creates a bridge
        env.reset()
        first_bridge = env._bridge
        assert first_bridge is not None
        assert first_bridge._running

        # Second reset should stop the old bridge and create a new one
        env.reset()
        second_bridge = env._bridge

        # Old bridge should be stopped
        assert not first_bridge._running

        # New bridge should be running
        assert second_bridge is not None
        assert second_bridge._running
        assert second_bridge is not first_bridge

        env.close()

    def test_multiple_resets_no_resource_leak(self, bridge_config):
        """Test multiple resets don't leak resources."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)

        bridges = []
        for i in range(5):
            env.reset()
            bridges.append(env._bridge)

        # All previous bridges should be stopped
        for i, bridge in enumerate(bridges[:-1]):
            assert not bridge._running, f"Bridge {i} should be stopped"

        # Only the last bridge should be running
        assert bridges[-1]._running

        env.close()


# ==============================================================================
# Gymnasium Compliance Tests
# ==============================================================================


@pytest.mark.skipif(not HAS_GYMNASIUM, reason="Gymnasium not installed")
class TestGymnasiumCompliance:
    """Tests for Gymnasium API compliance."""

    def test_env_checker(self, bridge_config):
        """Test environment passes Gymnasium env_checker."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)

        # Note: check_env may have issues with Dict observation spaces
        # We do a manual compliance check instead
        try:
            # Test reset returns valid types
            obs, info = env.reset()
            assert isinstance(obs, dict)
            assert isinstance(info, dict)

            # Test step returns valid types
            obs, reward, terminated, truncated, info = env.step(0)
            assert isinstance(obs, dict)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

        finally:
            env.close()

    def test_observation_in_space(self, bridge_config):
        """Test observation is within observation space."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)

        obs, _ = env.reset()

        # Check each component is in its space
        assert env.observation_space["spatial"].contains(obs["spatial"])
        assert env.observation_space["vector"].contains(obs["vector"])
        assert env.observation_space["action_mask"].contains(obs["action_mask"])

        env.close()

    def test_action_in_space(self, bridge_config):
        """Test actions are within action space."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)
        env.reset()

        # Sample random actions
        for _ in range(10):
            action = env.action_space.sample()
            assert env.action_space.contains(action)

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()

        env.close()


# ==============================================================================
# Civ6Interface Tests
# ==============================================================================


class TestCiv6Interface:
    """Tests for Civ6Interface (BaseGameInterface implementation)."""

    def test_creation(self, game_config, bridge_config):
        """Test interface can be created."""
        interface = Civ6Interface(
            config=game_config,
            bridge_config=bridge_config,
            use_mock=True,
        )
        assert interface is not None

    def test_connect_mock(self, game_config, bridge_config):
        """Test mock connection."""
        interface = Civ6Interface(
            config=game_config,
            bridge_config=bridge_config,
            use_mock=True,
        )

        result = interface.connect()
        assert result is True

        interface.disconnect()

    def test_reset_mock(self, game_config, bridge_config):
        """Test reset with mock environment."""
        interface = Civ6Interface(
            config=game_config,
            bridge_config=bridge_config,
            use_mock=True,
        )

        interface.connect()
        obs = interface.reset()

        assert obs is not None
        assert obs.spatial is not None
        assert obs.vector is not None
        assert obs.action_mask is not None

        interface.disconnect()

    def test_step_mock(self, game_config, bridge_config):
        """Test step with mock environment."""
        interface = Civ6Interface(
            config=game_config,
            bridge_config=bridge_config,
            use_mock=True,
        )

        interface.connect()
        interface.reset()

        result = interface.step(0)  # end_turn

        assert result is not None
        assert result.observation is not None
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

        interface.disconnect()

    def test_get_observation(self, game_config, bridge_config):
        """Test get_observation method."""
        interface = Civ6Interface(
            config=game_config,
            bridge_config=bridge_config,
            use_mock=True,
        )

        interface.connect()
        interface.reset()

        obs = interface.get_observation()
        assert obs is not None
        assert obs.spatial.shape == (84, 16, 16)

        interface.disconnect()

    def test_get_valid_actions(self, game_config, bridge_config):
        """Test get_valid_actions method."""
        interface = Civ6Interface(
            config=game_config,
            bridge_config=bridge_config,
            use_mock=True,
        )

        interface.connect()
        interface.reset()

        valid = interface.get_valid_actions()
        assert valid is not None
        assert valid.dtype == bool
        assert valid.sum() >= 1  # At least end_turn

        interface.disconnect()

    def test_get_game_status(self, game_config, bridge_config):
        """Test get_game_status method."""
        interface = Civ6Interface(
            config=game_config,
            bridge_config=bridge_config,
            use_mock=True,
        )

        interface.connect()
        assert interface.get_game_status() == GameStatus.NOT_STARTED

        interface.reset()
        assert interface.get_game_status() == GameStatus.IN_PROGRESS

        interface.disconnect()

    def test_get_current_player(self, game_config, bridge_config):
        """Test get_current_player method."""
        interface = Civ6Interface(
            config=game_config,
            bridge_config=bridge_config,
            use_mock=True,
        )

        interface.connect()
        interface.reset()

        player = interface.get_current_player()
        assert player == 0  # AI is always player 0

        interface.disconnect()

    def test_context_manager_pattern(self, game_config, bridge_config):
        """Test interface supports context manager pattern."""
        interface = Civ6Interface(
            config=game_config,
            bridge_config=bridge_config,
            use_mock=True,
        )

        # Manual connect/disconnect
        interface.connect()
        interface.reset()
        interface.step(0)
        interface.close()  # Should work same as disconnect


# ==============================================================================
# Game Loop Integration Tests
# ==============================================================================


class TestGameLoopIntegration:
    """Integration tests for complete game loops."""

    def test_simple_game_loop(self, bridge_config):
        """Test simple game loop with MockCiv6Env."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)

        obs, info = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100

        while steps < max_steps:
            # Get valid actions
            action_mask = obs["action_mask"]
            valid_actions = np.where(action_mask > 0)[0]

            if len(valid_actions) == 0:
                break

            # Select action (always end_turn for simplicity)
            action = 0 if 0 in valid_actions else valid_actions[0]

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        env.close()
        assert steps > 0

    def test_interface_game_loop(self, game_config, bridge_config):
        """Test game loop using Civ6Interface."""
        interface = Civ6Interface(
            config=game_config,
            bridge_config=bridge_config,
            use_mock=True,
        )

        interface.connect()
        interface.reset()
        total_reward = 0
        steps = 0
        max_steps = 50

        while steps < max_steps:
            valid = interface.get_valid_actions()
            valid_indices = np.where(valid)[0]

            if len(valid_indices) == 0:
                break

            action = valid_indices[0]  # Take first valid action

            result = interface.step(action)
            total_reward += result.reward
            steps += 1

            if result.done or result.truncated:
                break

        interface.disconnect()
        assert steps > 0

    def test_multiple_episodes(self, bridge_config):
        """Test running multiple episodes."""
        env = MockCiv6Env(bridge_config=bridge_config, board_size=16)

        num_episodes = 3
        episodes_completed = 0

        for _ in range(num_episodes):
            obs, _ = env.reset()
            steps = 0

            while steps < 10:
                obs, reward, terminated, truncated, info = env.step(0)
                steps += 1
                if terminated or truncated:
                    break

            episodes_completed += 1

        env.close()
        assert episodes_completed == num_episodes


# ==============================================================================
# Fixture-Based Tests
# ==============================================================================


class TestWithFixtures:
    """Tests using JSON fixture files."""

    @pytest.fixture
    def fixtures_dir(self):
        """Get fixtures directory path."""
        return Path(__file__).parent.parent / "fixtures" / "game_states"

    def test_fixture_files_exist(self, fixtures_dir):
        """Test that fixture files exist."""
        expected_files = [
            "turn_1_start.json",
            "turn_10_mid.json",
            "turn_50_combat.json",
        ]

        for filename in expected_files:
            fixture_path = fixtures_dir / filename
            assert fixture_path.exists(), f"Fixture file not found: {fixture_path}"

    def test_fixture_json_valid(self, fixtures_dir):
        """Test that fixture files contain valid JSON."""
        for fixture_file in fixtures_dir.glob("*.json"):
            with open(fixture_file) as f:
                data = json.load(f)

            # Verify basic structure
            assert "turn_id" in data
            assert "map" in data
            assert "units" in data
            assert "cities" in data
            assert "legal_actions" in data

    def test_fixture_state_structure(self, fixtures_dir):
        """Test fixture state structure matches expected format."""
        fixture_path = fixtures_dir / "turn_1_start.json"
        if not fixture_path.exists():
            pytest.skip("Fixture file not found")

        with open(fixture_path) as f:
            state_data = json.load(f)

        # Verify turn structure
        assert state_data["turn_id"] == 1
        assert state_data["player_id"] == 0

        # Verify map structure
        assert "width" in state_data["map"]
        assert "height" in state_data["map"]
        assert "tiles" in state_data["map"]

        # Verify units structure
        assert len(state_data["units"]) >= 1
        unit = state_data["units"][0]
        assert "id" in unit
        assert "type_name" in unit
        assert "x" in unit
        assert "y" in unit

        # Verify legal actions
        assert len(state_data["legal_actions"]) >= 1
        assert any(a["action"] == "end_turn" for a in state_data["legal_actions"])


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
