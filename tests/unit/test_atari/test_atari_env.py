"""Unit tests for the Atari environment wrapper."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from demos.atari.atari_env import AtariEnv, AtariEnvConfig, _preprocess_frame, _resize_nearest
from demos.atari.config import AtariEnvConfig as EnvConfig


# ============================================================================
# Preprocessing helpers
# ============================================================================


class TestPreprocessFrame:
    def test_grayscale_conversion(self):
        frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        result = _preprocess_frame(frame, 84, 84, grayscale=True)
        assert result.shape == (84, 84)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() and result.max() <= 1.0

    def test_rgb_preserved(self):
        frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        result = _preprocess_frame(frame, 84, 84, grayscale=False)
        assert result.shape == (84, 84, 3)

    def test_already_correct_size(self):
        frame = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
        result = _preprocess_frame(frame, 84, 84, grayscale=False)
        assert result.shape == (84, 84, 3)

    def test_normalisation(self):
        # All-white frame
        frame = np.full((84, 84, 3), 255, dtype=np.uint8)
        result = _preprocess_frame(frame, 84, 84, grayscale=True)
        assert np.isclose(result.max(), 1.0, atol=0.01)

    def test_zero_frame(self):
        frame = np.zeros((84, 84, 3), dtype=np.uint8)
        result = _preprocess_frame(frame, 84, 84, grayscale=True)
        assert result.max() == 0.0


class TestResizeNearest:
    def test_downscale(self):
        img = np.random.rand(210, 160)
        resized = _resize_nearest(img, 84, 84)
        assert resized.shape == (84, 84)

    def test_upscale(self):
        img = np.random.rand(42, 42)
        resized = _resize_nearest(img, 84, 84)
        assert resized.shape == (84, 84)

    def test_identity(self):
        img = np.random.rand(84, 84)
        resized = _resize_nearest(img, 84, 84)
        assert resized.shape == (84, 84)
        np.testing.assert_array_equal(img, resized)

    def test_3d_image(self):
        img = np.random.rand(210, 160, 3)
        resized = _resize_nearest(img, 84, 84)
        assert resized.shape == (84, 84, 3)


# ============================================================================
# AtariEnv (mocked — no actual Atari ROM needed)
# ============================================================================


class _FakeAtariEnv:
    """Minimal mock of a Gymnasium Atari environment."""

    class _Space:
        n = 6

    action_space = _Space()
    observation_space = None

    def reset(self, seed=None):
        obs = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        return obs, {"lives": 3}

    def step(self, action):
        obs = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        reward = 1.0
        terminated = False
        truncated = False
        info = {"lives": 3}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


@pytest.fixture
def mock_env(monkeypatch):
    """Patch gymnasium.make to return a fake Atari env."""
    import gymnasium as gym

    monkeypatch.setattr(gym, "make", lambda *a, **kw: _FakeAtariEnv())
    return EnvConfig(game_name="FakeGame")


class TestAtariEnv:
    def test_creation(self, mock_env):
        env = AtariEnv(mock_env)
        assert env.action_space.n == 6
        assert env.current_player == 0

    def test_reset_returns_observation(self, mock_env):
        env = AtariEnv(mock_env)
        obs, info = env.reset()
        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs

    def test_observation_shapes(self, mock_env):
        cfg = mock_env
        env = AtariEnv(cfg)
        obs, _ = env.reset()
        assert obs["spatial"].shape == (cfg.frame_stack, cfg.frame_height, cfg.frame_width)
        assert obs["vector"].shape == (cfg.vector_dim,)
        assert obs["action_mask"].shape == (6,)

    def test_step_returns_tuple(self, mock_env):
        env = AtariEnv(mock_env)
        env.reset()
        result = env.step(0)
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_get_valid_actions(self, mock_env):
        env = AtariEnv(mock_env)
        env.reset()
        mask = env.get_valid_actions()
        assert mask.shape == (6,)
        assert np.all(mask == 1.0)

    def test_get_info(self, mock_env):
        env = AtariEnv(mock_env)
        env.reset()
        info = env.get_info()
        assert "episode_reward" in info
        assert "step_count" in info

    def test_clone(self, mock_env):
        env = AtariEnv(mock_env)
        env.reset()
        cloned = env.clone()
        # Must be independent
        assert cloned is not env
        assert cloned._env is not env._env
        obs_orig = env.get_observation()
        obs_clone = cloned.get_observation()
        np.testing.assert_array_equal(obs_orig["spatial"], obs_clone["spatial"])

    def test_reward_clipping(self, mock_env):
        mock_env.clip_rewards = True
        env = AtariEnv(mock_env)
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward in (-1.0, 0.0, 1.0)

    def test_frame_stack_fills_on_reset(self, mock_env):
        env = AtariEnv(mock_env)
        env.reset()
        assert len(env._frame_stack) == mock_env.frame_stack

    def test_repr(self, mock_env):
        env = AtariEnv(mock_env)
        r = repr(env)
        assert "AtariEnv" in r
        assert mock_env.game_name in r

    def test_close(self, mock_env):
        env = AtariEnv(mock_env)
        env.reset()
        env.close()  # should not raise
