"""Unit tests for Atari demo configuration system."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from demos.atari.config import (
    AtariConfig,
    AtariEnvConfig,
    AtariHierarchicalConfig,
    AtariMCTSConfig,
    AtariNetworkConfig,
    AtariTrainingConfig,
)


# ============================================================================
# AtariEnvConfig
# ============================================================================


class TestAtariEnvConfig:
    def test_defaults(self):
        cfg = AtariEnvConfig()
        assert cfg.game_name == "Pong"
        assert cfg.frame_height == 84
        assert cfg.frame_width == 84
        assert cfg.frame_stack == 4
        assert cfg.grayscale is True

    def test_spatial_channels_grayscale(self):
        cfg = AtariEnvConfig(frame_stack=4, grayscale=True)
        assert cfg.spatial_channels == 4

    def test_spatial_channels_rgb(self):
        cfg = AtariEnvConfig(frame_stack=4, grayscale=False)
        assert cfg.spatial_channels == 12  # 4 * 3

    def test_spatial_size(self):
        cfg = AtariEnvConfig(frame_height=42, frame_width=42)
        assert cfg.spatial_size == (42, 42)

    def test_gymnasium_id(self):
        cfg = AtariEnvConfig(game_name="Breakout")
        assert cfg.gymnasium_id == "ALE/Breakout-v5"

    def test_env_overrides(self):
        cfg = AtariEnvConfig()
        with mock.patch.dict(os.environ, {"ATARI_DEMO_GAME_NAME": "SpaceInvaders"}):
            cfg.apply_env_overrides()
        assert cfg.game_name == "SpaceInvaders"


# ============================================================================
# AtariNetworkConfig
# ============================================================================


class TestAtariNetworkConfig:
    def test_defaults(self):
        cfg = AtariNetworkConfig()
        assert cfg.num_res_blocks == 6
        assert cfg.num_filters == 64
        assert cfg.fc_hidden_dim == 256
        assert cfg.value_head_hidden == 128

    def test_env_overrides(self):
        cfg = AtariNetworkConfig()
        with mock.patch.dict(os.environ, {"ATARI_DEMO_NUM_FILTERS": "128"}):
            cfg.apply_env_overrides()
        assert cfg.num_filters == 128


# ============================================================================
# AtariMCTSConfig
# ============================================================================


class TestAtariMCTSConfig:
    def test_defaults(self):
        cfg = AtariMCTSConfig()
        assert cfg.num_simulations == 50
        assert cfg.c_puct == 1.5

    def test_env_overrides(self):
        cfg = AtariMCTSConfig()
        with mock.patch.dict(os.environ, {"ATARI_DEMO_NUM_SIMULATIONS": "100"}):
            cfg.apply_env_overrides()
        assert cfg.num_simulations == 100


# ============================================================================
# AtariTrainingConfig
# ============================================================================


class TestAtariTrainingConfig:
    def test_defaults(self):
        cfg = AtariTrainingConfig()
        assert cfg.num_iterations == 100
        assert cfg.batch_size == 128
        assert cfg.discount_factor == 0.99
        assert cfg.device in ("cpu", "cuda")

    def test_device_auto_detection(self):
        cfg = AtariTrainingConfig(device="")
        # Should resolve to cpu or cuda
        assert cfg.device in ("cpu", "cuda")

    def test_env_overrides(self):
        cfg = AtariTrainingConfig()
        with mock.patch.dict(os.environ, {"ATARI_DEMO_BATCH_SIZE": "256"}):
            cfg.apply_env_overrides()
        assert cfg.batch_size == 256


# ============================================================================
# AtariHierarchicalConfig
# ============================================================================


class TestAtariHierarchicalConfig:
    def test_defaults_disabled(self):
        cfg = AtariHierarchicalConfig()
        assert cfg.enabled is False
        assert cfg.strategic_update_freq == 500
        assert cfg.tactical_update_freq == 50

    def test_env_override_enable(self):
        cfg = AtariHierarchicalConfig()
        with mock.patch.dict(os.environ, {"ATARI_DEMO_HIERARCHICAL_ENABLED": "true"}):
            cfg.apply_env_overrides()
        assert cfg.enabled is True


# ============================================================================
# AtariConfig (master)
# ============================================================================


class TestAtariConfig:
    def test_defaults(self):
        cfg = AtariConfig()
        assert isinstance(cfg.environment, AtariEnvConfig)
        assert isinstance(cfg.network, AtariNetworkConfig)
        assert isinstance(cfg.mcts, AtariMCTSConfig)
        assert isinstance(cfg.training, AtariTrainingConfig)
        assert isinstance(cfg.hierarchical, AtariHierarchicalConfig)

    def test_for_game_known(self):
        cfg = AtariConfig.for_game("Breakout")
        assert cfg.environment.game_name == "Breakout"

    def test_for_game_unknown(self):
        cfg = AtariConfig.for_game("UnknownGame")
        assert cfg.environment.game_name == "UnknownGame"

    def test_from_dict(self):
        data = {
            "environment": {"game_name": "MsPacman", "frame_stack": 8},
            "network": {"num_filters": 128},
            "mcts": {"num_simulations": 100},
            "training": {"batch_size": 64},
            "hierarchical": {"enabled": True},
        }
        cfg = AtariConfig.from_dict(data)
        assert cfg.environment.game_name == "MsPacman"
        assert cfg.environment.frame_stack == 8
        assert cfg.network.num_filters == 128
        assert cfg.mcts.num_simulations == 100
        assert cfg.training.batch_size == 64
        assert cfg.hierarchical.enabled is True

    def test_from_yaml(self, tmp_path):
        yaml_content = """
environment:
  game_name: Pong
  frame_stack: 4
network:
  num_res_blocks: 3
training:
  num_iterations: 5
"""
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml_content)

        cfg = AtariConfig.from_yaml(str(cfg_path))
        assert cfg.environment.game_name == "Pong"
        assert cfg.network.num_res_blocks == 3
        assert cfg.training.num_iterations == 5

    def test_from_yaml_with_atari_key(self, tmp_path):
        yaml_content = """
atari:
  environment:
    game_name: Breakout
"""
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml_content)
        cfg = AtariConfig.from_yaml(str(cfg_path))
        assert cfg.environment.game_name == "Breakout"

    def test_from_yaml_not_found(self):
        with pytest.raises(FileNotFoundError):
            AtariConfig.from_yaml("/nonexistent/path.yaml")

    def test_apply_env_overrides(self):
        cfg = AtariConfig()
        with mock.patch.dict(
            os.environ,
            {
                "ATARI_DEMO_GAME_NAME": "SpaceInvaders",
                "ATARI_DEMO_BATCH_SIZE": "256",
            },
        ):
            cfg.apply_env_overrides()
        assert cfg.environment.game_name == "SpaceInvaders"
        assert cfg.training.batch_size == 256

    def test_game_presets_exist(self):
        assert "Pong" in AtariConfig.GAME_PRESETS
        assert "Breakout" in AtariConfig.GAME_PRESETS
        assert "SpaceInvaders" in AtariConfig.GAME_PRESETS
