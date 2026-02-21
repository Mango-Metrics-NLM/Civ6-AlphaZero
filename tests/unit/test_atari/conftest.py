"""Shared test fixtures for Atari demo unit tests.

Provides reusable fake environments and factory helpers to eliminate
duplication across test modules.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from demos.atari.config import AtariEnvConfig, AtariNetworkConfig
from demos.atari.trainer import TrainingExample

# ============================================================================
# Fake Gymnasium environment (no ROM required)
# ============================================================================

# Standard Atari frame dimensions from real ROMs
_ATARI_FRAME_SHAPE = (210, 160, 3)
_DEFAULT_LIVES = 3


class FakeAtariGym:
    """Minimal Atari-like gymnasium environment for testing.

    Terminates after ``max_steps`` steps.  Reward behaviour is configurable
    via ``reward_fn``.

    Args:
        max_steps: Steps before the episode terminates.
        reward_fn: Callable ``(step) -> float`` for per-step rewards.
            Defaults to a constant ``1.0``.
        num_actions: Size of the discrete action space.
    """

    class _Space:
        def __init__(self, n: int = 4) -> None:
            self.n = n

    def __init__(
        self,
        max_steps: int = 10,
        reward_fn: object = None,
        num_actions: int = 4,
    ) -> None:
        self.action_space = self._Space(num_actions)
        self.observation_space = None
        self._step_count = 0
        self._max_steps = max_steps
        self._reward_fn = reward_fn or (lambda _step: 1.0)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._step_count = 0
        obs = np.random.randint(0, 256, _ATARI_FRAME_SHAPE, dtype=np.uint8)
        return obs, {"lives": _DEFAULT_LIVES}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        obs = np.random.randint(0, 256, _ATARI_FRAME_SHAPE, dtype=np.uint8)
        reward = float(self._reward_fn(self._step_count))
        terminated = self._step_count >= self._max_steps
        return obs, reward, terminated, False, {"lives": _DEFAULT_LIVES}

    def close(self) -> None:
        pass


# ============================================================================
# Pytest fixtures
# ============================================================================


@pytest.fixture()
def fake_atari_gym():
    """Return a ``FakeAtariGym`` factory.

    Usage::

        env = fake_atari_gym()                     # defaults
        env = fake_atari_gym(max_steps=5)          # custom termination
        env = fake_atari_gym(reward_fn=lambda s: float(np.random.choice([-1, 0, 1])))
    """
    return FakeAtariGym


@pytest.fixture(autouse=True)
def mock_gymnasium(monkeypatch):
    """Globally replace ``gymnasium.make`` with :class:`FakeAtariGym`."""
    monkeypatch.setattr(gym, "make", lambda *_a, **_kw: FakeAtariGym())


@pytest.fixture()
def dummy_obs():
    """Standard dummy observation dict for tests."""
    return {
        "spatial": np.random.rand(4, 84, 84).astype(np.float32),
        "vector": np.zeros(32, dtype=np.float32),
        "action_mask": np.ones(4, dtype=np.float32),
    }


@pytest.fixture()
def fast_env_config():
    """Minimal environment config for fast tests."""
    return AtariEnvConfig(
        game_name="Fake",
        frame_height=42,
        frame_width=42,
        frame_stack=2,
        vector_dim=8,
    )


@pytest.fixture()
def fast_network_config():
    """Minimal network config for fast tests."""
    return AtariNetworkConfig(
        num_res_blocks=1,
        num_filters=8,
        fc_hidden_dim=16,
        value_head_hidden=8,
    )


def make_training_example(value: float = 0.5, num_actions: int = 6) -> TrainingExample:
    """Factory for creating ``TrainingExample`` objects in tests.

    Args:
        value: Target value for the example.
        num_actions: Number of actions in the action space.

    Returns:
        A ``TrainingExample`` with random spatial data.
    """
    policy = np.random.dirichlet(np.ones(num_actions)).astype(np.float32)
    return TrainingExample(
        spatial=np.random.rand(4, 84, 84).astype(np.float32),
        vector=np.zeros(32, dtype=np.float32),
        action_mask=np.ones(num_actions, dtype=np.float32),
        policy=policy,
        value=value,
    )
