"""Shared test fixtures for Atari demo integration tests.

Reuses ``FakeAtariGym`` from the unit test conftest and adds
integration-specific fixtures (e.g. ``fast_config``).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from demos.atari.config import (
    AtariConfig,
    AtariEnvConfig,
    AtariNetworkConfig,
    AtariTrainingConfig,
)

# Re-use the canonical FakeAtariGym from unit tests
from tests.unit.test_atari.conftest import FakeAtariGym


@pytest.fixture(autouse=True)
def mock_gymnasium(monkeypatch):
    """Globally replace ``gymnasium.make`` with :class:`FakeAtariGym`.

    Uses random rewards ([-1, 0, 1]) for more realistic integration testing.
    """

    def _make(*_args, **_kwargs):
        return FakeAtariGym(
            reward_fn=lambda _s: float(np.random.choice([-1, 0, 1])),
        )

    monkeypatch.setattr(gym, "make", _make)


@pytest.fixture()
def fast_config():
    """Create a minimal configuration for fast integration tests."""
    return AtariConfig(
        environment=AtariEnvConfig(
            game_name="FakeGame",
            frame_height=42,
            frame_width=42,
            frame_stack=2,
            vector_dim=8,
        ),
        network=AtariNetworkConfig(
            num_res_blocks=1,
            num_filters=8,
            fc_hidden_dim=16,
            value_head_hidden=8,
        ),
        training=AtariTrainingConfig(
            num_iterations=2,
            episodes_per_iteration=1,
            batch_size=4,
            num_epochs=1,
            replay_buffer_size=200,
            min_replay_size=5,
            save_interval=1,
            eval_episodes=1,
            eval_interval=1,
            device="cpu",
        ),
    )
