"""Additional tests targeting specific uncovered branches.

These tests fill coverage gaps in atari_env.py, config.py, hierarchical.py,
and trainer.py identified by the coverage report.

Note: ``FakeAtariGym`` and ``mock_gymnasium`` are provided by conftest.py.
"""

from __future__ import annotations

from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from demos.atari.config import (
    AtariConfig,
    AtariEnvConfig,
    AtariHierarchicalConfig,
    AtariMCTSConfig,
    AtariNetworkConfig,
    AtariTrainingConfig,
    _env_override,
)
from tests.unit.test_atari.conftest import FakeAtariGym

# ============================================================================
# _env_override edge cases
# ============================================================================


class TestEnvOverride:
    def test_bool_true_values(self, monkeypatch):
        monkeypatch.setenv("ATARI_DEMO_TEST_BOOL", "true")
        assert _env_override("TEST_BOOL", False, bool) is True
        monkeypatch.setenv("ATARI_DEMO_TEST_BOOL", "1")
        assert _env_override("TEST_BOOL", False, bool) is True
        monkeypatch.setenv("ATARI_DEMO_TEST_BOOL", "yes")
        assert _env_override("TEST_BOOL", False, bool) is True

    def test_bool_false_values(self, monkeypatch):
        monkeypatch.setenv("ATARI_DEMO_TEST_BOOL", "false")
        assert _env_override("TEST_BOOL", True, bool) is False
        monkeypatch.setenv("ATARI_DEMO_TEST_BOOL", "0")
        assert _env_override("TEST_BOOL", True, bool) is False

    def test_cast_failure_returns_default(self, monkeypatch):
        monkeypatch.setenv("ATARI_DEMO_BAD_INT", "not_a_number")
        result = _env_override("BAD_INT", 42, int)
        assert result == 42

    def test_unset_returns_default(self):
        result = _env_override("NONEXISTENT_KEY_12345", "default_value")
        assert result == "default_value"


# ============================================================================
# Config - YAML loading edge cases
# ============================================================================


class TestConfigYAML:
    def test_yaml_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            AtariConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_yaml_with_atari_key(self, tmp_path):
        """YAML files can wrap config under an 'atari' key."""
        yaml_file = tmp_path / "wrapped.yaml"
        yaml_file.write_text(
            "atari:\n"
            "  environment:\n"
            "    game_name: TestGame\n"
        )
        cfg = AtariConfig.from_yaml(yaml_file)
        assert cfg.environment.game_name == "TestGame"

    def test_yaml_empty_file(self, tmp_path):
        """Empty YAML files should produce default config."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        cfg = AtariConfig.from_yaml(yaml_file)
        assert cfg.environment.game_name == "Pong"  # default

    def test_yaml_import_error_when_unavailable(self):
        """from_yaml raises ImportError when PyYAML is unavailable."""
        with mock.patch("demos.atari.config.YAML_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyYAML"):
                AtariConfig.from_yaml("test.yaml")


# ============================================================================
# Config - for_game preset and defaults
# ============================================================================


class TestConfigPresets:
    def test_for_game_unknown_game(self):
        """Unknown game gets default environment settings."""
        cfg = AtariConfig.for_game("UnknownGame123")
        assert cfg.environment.game_name == "UnknownGame123"

    def test_for_game_with_overrides(self):
        """for_game accepts additional overrides."""
        cfg = AtariConfig.for_game("Pong", network={"num_filters": 32})
        assert cfg.environment.game_name == "Pong"
        assert cfg.network.num_filters == 32


# ============================================================================
# Config - device detection
# ============================================================================


class TestTrainingConfigDevice:
    def test_device_auto_detection_cpu(self):
        """Default device is auto-detected (CPU in test env)."""
        cfg = AtariTrainingConfig()
        assert cfg.device in ("cpu", "cuda")

    def test_device_explicit(self):
        """Explicit device string is preserved."""
        cfg = AtariTrainingConfig(device="cpu")
        assert cfg.device == "cpu"


# ============================================================================
# Config - env overrides across sub-configs
# ============================================================================


class TestSubConfigEnvOverrides:
    def test_env_config_overrides(self, monkeypatch):
        monkeypatch.setenv("ATARI_DEMO_GAME_NAME", "TestGame")
        monkeypatch.setenv("ATARI_DEMO_FRAME_HEIGHT", "42")
        monkeypatch.setenv("ATARI_DEMO_FRAME_WIDTH", "42")
        monkeypatch.setenv("ATARI_DEMO_FRAME_STACK", "2")
        monkeypatch.setenv("ATARI_DEMO_FRAME_SKIP", "2")
        monkeypatch.setenv("ATARI_DEMO_GRAYSCALE", "false")
        monkeypatch.setenv("ATARI_DEMO_CLIP_REWARDS", "false")

        cfg = AtariEnvConfig()
        cfg.apply_env_overrides()
        assert cfg.game_name == "TestGame"
        assert cfg.frame_height == 42
        assert cfg.frame_width == 42
        assert cfg.frame_stack == 2
        assert cfg.frame_skip == 2
        assert cfg.grayscale is False
        assert cfg.clip_rewards is False

    def test_network_config_overrides(self, monkeypatch):
        monkeypatch.setenv("ATARI_DEMO_NUM_RES_BLOCKS", "3")
        monkeypatch.setenv("ATARI_DEMO_NUM_FILTERS", "32")
        monkeypatch.setenv("ATARI_DEMO_FC_HIDDEN_DIM", "128")

        cfg = AtariNetworkConfig()
        cfg.apply_env_overrides()
        assert cfg.num_res_blocks == 3
        assert cfg.num_filters == 32
        assert cfg.fc_hidden_dim == 128

    def test_mcts_config_overrides(self, monkeypatch):
        monkeypatch.setenv("ATARI_DEMO_NUM_SIMULATIONS", "100")
        monkeypatch.setenv("ATARI_DEMO_C_PUCT", "2.0")

        cfg = AtariMCTSConfig()
        cfg.apply_env_overrides()
        assert cfg.num_simulations == 100
        assert cfg.c_puct == 2.0

    def test_training_config_overrides(self, monkeypatch):
        monkeypatch.setenv("ATARI_DEMO_NUM_ITERATIONS", "200")
        monkeypatch.setenv("ATARI_DEMO_EPISODES_PER_ITER", "20")
        monkeypatch.setenv("ATARI_DEMO_BATCH_SIZE", "256")
        monkeypatch.setenv("ATARI_DEMO_LEARNING_RATE", "0.01")
        monkeypatch.setenv("ATARI_DEMO_DISCOUNT_FACTOR", "0.95")
        monkeypatch.setenv("ATARI_DEMO_DEVICE", "cpu")

        cfg = AtariTrainingConfig(device="cuda")
        cfg.apply_env_overrides()
        assert cfg.num_iterations == 200
        assert cfg.episodes_per_iteration == 20
        assert cfg.batch_size == 256
        assert cfg.learning_rate == 0.01
        assert cfg.discount_factor == 0.95
        assert cfg.device == "cpu"

    def test_hierarchical_config_overrides(self, monkeypatch):
        monkeypatch.setenv("ATARI_DEMO_HIERARCHICAL_ENABLED", "true")
        monkeypatch.setenv("ATARI_DEMO_STRATEGIC_UPDATE_FREQ", "100")
        monkeypatch.setenv("ATARI_DEMO_TACTICAL_UPDATE_FREQ", "10")

        cfg = AtariHierarchicalConfig()
        cfg.apply_env_overrides()
        assert cfg.enabled is True
        assert cfg.strategic_update_freq == 100
        assert cfg.tactical_update_freq == 10


# ============================================================================
# AtariEnv - uncovered branches
# ============================================================================


class TestAtariEnvBranches:
    def test_color_frames(self):
        """Test non-grayscale (color) frame processing."""
        from demos.atari.atari_env import AtariEnv

        env_cfg = AtariEnvConfig(
            game_name="Fake",
            frame_height=42,
            frame_width=42,
            frame_stack=2,
            grayscale=False,
        )
        env = AtariEnv(env_cfg)
        env.reset()  # populate frame stack
        obs = env.get_observation()
        # Color: stack * 3 channels
        assert obs["spatial"].shape[0] == env_cfg.frame_stack * 3
        env.close()

    def test_max_episode_steps_config(self):
        """Test that max_episode_steps > 0 path is exercised via config."""
        env_cfg = AtariEnvConfig(
            game_name="Fake",
            frame_height=42,
            frame_width=42,
            frame_stack=2,
            max_episode_steps=50,
        )
        # When max_episode_steps > 0, _build_env wraps with TimeLimit.
        # Our fake env isn't a real gymnasium.Env so TimeLimit wrapper will fail.
        # We just test the config setting is preserved.
        assert env_cfg.max_episode_steps == 50

    def test_sticky_actions(self):
        """Test sticky actions parameter passed to env."""
        from demos.atari.atari_env import AtariEnv

        env_cfg = AtariEnvConfig(
            game_name="Fake",
            frame_height=42,
            frame_width=42,
            frame_stack=2,
            sticky_actions=0.25,
        )
        env = AtariEnv(env_cfg)
        env.reset()
        env.close()

    def test_env_fallback_on_name_error(self, monkeypatch):
        """Test fallback when ALE/ env is not found."""
        from demos.atari.atari_env import AtariEnv

        call_count = [0]

        def fake_make(*args, **kwargs):
            call_count[0] += 1
            env_id = args[0] if args else kwargs.get("id", "")
            if "ALE/" in str(env_id):
                raise gym.error.NameNotFound(env_id)
            return FakeAtariGym()

        monkeypatch.setattr(gym, "make", fake_make)

        env_cfg = AtariEnvConfig(game_name="Fake", frame_height=42, frame_width=42, frame_stack=2)
        env = AtariEnv(env_cfg)
        assert call_count[0] == 2  # First try ALE/, then fallback
        env.close()

    def test_vector_dim_less_than_4(self):
        """Test with small vector dim where stat encoding is skipped."""
        from demos.atari.atari_env import AtariEnv

        env_cfg = AtariEnvConfig(
            game_name="Fake",
            frame_height=42,
            frame_width=42,
            frame_stack=2,
            vector_dim=2,
        )
        env = AtariEnv(env_cfg)
        env.reset()  # populate frame stack
        obs = env.get_observation()
        assert obs["vector"].shape == (2,)
        # Should be all zeros since vector_dim < 4
        assert np.all(obs["vector"] == 0.0)
        env.close()

    def test_get_info(self):
        """Test get_info returns expected keys."""
        from demos.atari.atari_env import AtariEnv

        env_cfg = AtariEnvConfig(game_name="Fake", frame_height=42, frame_width=42, frame_stack=2)
        env = AtariEnv(env_cfg)
        info = env.get_info()
        assert "step_count" in info
        assert "episode_reward" in info
        assert "lives" in info
        env.close()

    def test_str_repr(self):
        """Test __str__ method."""
        from demos.atari.atari_env import AtariEnv

        env_cfg = AtariEnvConfig(game_name="TestGame", frame_height=42, frame_width=42, frame_stack=2)
        env = AtariEnv(env_cfg)
        s = str(env)
        assert "TestGame" in s
        env.close()


# ============================================================================
# Hierarchical - uncovered branches
# ============================================================================


class TestHierarchicalBranches:
    def test_defensive_play_strategy(self):
        """Test DEFENSIVE_PLAY triggered by negative reward."""
        from demos.atari.hierarchical import (
            AtariHierarchicalAdapter,
            AtariStrategicGoal,
        )

        cfg = AtariHierarchicalConfig(
            enabled=True, strategic_update_freq=1, tactical_update_freq=1
        )
        adapter = AtariHierarchicalAdapter(cfg, num_actions=4)
        dummy_obs = {
            "spatial": np.random.rand(4, 84, 84).astype(np.float32),
            "vector": np.zeros(32, dtype=np.float32),
            "action_mask": np.ones(4, dtype=np.float32),
        }

        # Simulate negative reward to trigger DEFENSIVE_PLAY
        adapter._episode_reward = -5.0
        adapter._step = cfg.strategic_update_freq + 1
        adapter._strategic.last_update_step = 0
        adapter.decide(dummy_obs)

        goal_types = [g.goal_type for g in adapter._strategic.active_goals]
        assert AtariStrategicGoal.DEFENSIVE_PLAY in goal_types

    def test_balanced_play_strategy(self):
        """Test BALANCED_PLAY triggered by zero reward."""
        from demos.atari.hierarchical import (
            AtariHierarchicalAdapter,
            AtariStrategicGoal,
        )

        cfg = AtariHierarchicalConfig(
            enabled=True, strategic_update_freq=1, tactical_update_freq=1
        )
        adapter = AtariHierarchicalAdapter(cfg, num_actions=4)
        dummy_obs = {
            "spatial": np.random.rand(4, 84, 84).astype(np.float32),
            "vector": np.zeros(32, dtype=np.float32),
            "action_mask": np.ones(4, dtype=np.float32),
        }

        adapter._episode_reward = 0.0
        adapter._step = cfg.strategic_update_freq + 1
        adapter._strategic.last_update_step = 0
        adapter.decide(dummy_obs)

        goal_types = [g.goal_type for g in adapter._strategic.active_goals]
        assert AtariStrategicGoal.BALANCED_PLAY in goal_types

    def test_tactical_decomposition_defensive(self):
        """Test tactical decomposition from DEFENSIVE_PLAY."""
        from demos.atari.hierarchical import (
            AtariGoal,
            AtariHierarchicalAdapter,
            AtariStrategicGoal,
            AtariTacticalGoal,
        )

        cfg = AtariHierarchicalConfig(
            enabled=True, strategic_update_freq=10000, tactical_update_freq=1,
            max_active_goals_per_level=10,
        )
        adapter = AtariHierarchicalAdapter(cfg, num_actions=4)

        # Set only DEFENSIVE_PLAY (not MAXIMIZE_SCORE) to test that branch
        # Set strategic last_update_step high to prevent strategic re-planning
        adapter._strategic.active_goals = [
            AtariGoal("strategic", AtariStrategicGoal.DEFENSIVE_PLAY)
        ]
        adapter._strategic.last_update_step = 99
        adapter._tactical.last_update_step = -999
        adapter._step = 100

        dummy_obs = {
            "spatial": np.random.rand(4, 84, 84).astype(np.float32),
            "vector": np.zeros(32, dtype=np.float32),
            "action_mask": np.ones(4, dtype=np.float32),
        }
        adapter.decide(dummy_obs)

        tactic_types = [g.goal_type for g in adapter._tactical.active_goals]
        assert AtariTacticalGoal.AVOID_HAZARD in tactic_types
        assert AtariTacticalGoal.HOLD_POSITION in tactic_types

    def test_strategic_progress_update(self):
        """Test that strategic goal progress increments on positive reward."""
        from demos.atari.hierarchical import (
            AtariGoal,
            AtariHierarchicalAdapter,
            AtariStrategicGoal,
        )

        cfg = AtariHierarchicalConfig(enabled=True)
        adapter = AtariHierarchicalAdapter(cfg, num_actions=4)
        adapter._strategic.active_goals = [
            AtariGoal("strategic", AtariStrategicGoal.MAXIMIZE_SCORE, progress=0.0, active=True)
        ]
        adapter.update(1.0, False)
        assert adapter._strategic.active_goals[0].progress > 0.0


# ============================================================================
# Trainer - evaluate function
# ============================================================================


class TestEvaluateFunction:
    def test_evaluate_with_simple_mcts(self):
        """Evaluate works with SimpleMCTS (no add_noise support)."""
        from alpha_zero.core.civ6_mcts import SimpleMCTS
        from demos.atari.atari_env import AtariEnv
        from demos.atari.atari_network import AtariNetwork
        from demos.atari.trainer import evaluate

        env_cfg = AtariEnvConfig(
            game_name="Fake", frame_height=42, frame_width=42, frame_stack=2, vector_dim=8
        )
        env = AtariEnv(env_cfg)
        net = AtariNetwork(
            config=AtariNetworkConfig(num_res_blocks=1, num_filters=8, fc_hidden_dim=16, value_head_hidden=8),
            num_actions=env.action_space.n,
            spatial_size=env_cfg.spatial_size,
            spatial_channels=env_cfg.spatial_channels,
            vector_dim=env_cfg.vector_dim,
        )
        mcts = SimpleMCTS(num_simulations=3, c_puct=1.5)

        metrics = evaluate(env, net, mcts, num_episodes=2)
        assert "mean_reward" in metrics
        assert "max_reward" in metrics
        assert "min_reward" in metrics
        assert metrics["max_reward"] >= metrics["min_reward"]
        env.close()

    def test_evaluate_with_civ6_mcts(self):
        """Evaluate works with Civ6MCTS (supports add_noise)."""
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig
        from demos.atari.atari_env import AtariEnv
        from demos.atari.atari_network import AtariNetwork
        from demos.atari.trainer import evaluate

        env_cfg = AtariEnvConfig(
            game_name="Fake", frame_height=42, frame_width=42, frame_stack=2, vector_dim=8
        )
        env = AtariEnv(env_cfg)
        net = AtariNetwork(
            config=AtariNetworkConfig(num_res_blocks=1, num_filters=8, fc_hidden_dim=16, value_head_hidden=8),
            num_actions=env.action_space.n,
            spatial_size=env_cfg.spatial_size,
            spatial_channels=env_cfg.spatial_channels,
            vector_dim=env_cfg.vector_dim,
        )
        mcts_cfg = MCTSConfig(num_simulations=3, c_puct=1.5)
        mcts = Civ6MCTS(net, mcts_cfg, device="cpu")

        metrics = evaluate(env, net, mcts, num_episodes=1)
        assert "mean_reward" in metrics
        env.close()


# ============================================================================
# Config computed properties
# ============================================================================


class TestConfigProperties:
    def test_spatial_channels_grayscale(self):
        cfg = AtariEnvConfig(grayscale=True, frame_stack=4)
        assert cfg.spatial_channels == 4

    def test_spatial_channels_color(self):
        cfg = AtariEnvConfig(grayscale=False, frame_stack=4)
        assert cfg.spatial_channels == 12  # 4 * 3

    def test_spatial_size(self):
        cfg = AtariEnvConfig(frame_height=84, frame_width=84)
        assert cfg.spatial_size == (84, 84)

    def test_gymnasium_id(self):
        cfg = AtariEnvConfig(game_name="Pong")
        assert cfg.gymnasium_id == "ALE/Pong-v5"

    def test_gymnasium_id_custom_game(self):
        cfg = AtariEnvConfig(game_name="MyGame")
        assert cfg.gymnasium_id == "ALE/MyGame-v5"
