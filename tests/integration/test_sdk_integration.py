"""Integration tests for the Strategic Planning SDK pipeline.

Tests the full data flow: config -> environment -> MCTS -> self-play -> replay buffer.
All tests are pure numpy (no torch dependency).
"""
from __future__ import annotations

import numpy as np

from src.adapters.civ6 import MicroCivAdapter
from src.core.config import MCTSConfig, load_config
from src.core.mcts import MCTSEngine
from src.core.protocols import (
    StrategicEnvironment,
    assert_valid_environment,
    check_protocol_compliance,
)
from src.core.training import ReplayBuffer, SelfPlayCollector
from src.core.types import GameOutcome, GameTrajectory
from src.environments import EnvironmentRegistry
from src.environments.mini_games import MiniResourceEnv


class TestConfigToEnvironmentPipeline:
    """Config loading -> environment creation pipeline."""

    def test_load_default_config_and_create_env(self):
        config = load_config()
        env = MiniResourceEnv(board_size=8, max_steps=config.training.max_steps_per_game)
        assert_valid_environment(env)

    def test_config_overrides_propagate_to_env(self):
        config = load_config(overrides={"training": {"max_steps_per_game": 10}})
        assert config.training.max_steps_per_game == 10

    def test_registry_creates_valid_environment(self):
        registry = EnvironmentRegistry()
        registry.register("mini_resource", MiniResourceEnv)
        env = registry.create("mini_resource", board_size=6, max_steps=20)
        assert_valid_environment(env)


class TestMCTSWithEnvironment:
    """MCTS search with a real environment."""

    def test_mcts_search_on_mini_resource(self):
        env = MiniResourceEnv(board_size=4, max_steps=20, num_resources=2, seed=42)
        env.reset(seed=42)

        def eval_fn(spatial, vector, mask):
            policy = np.ones(env.action_space_size, dtype=np.float32)
            policy = policy * mask
            total = policy.sum()
            if total > 0:
                policy /= total
            return policy, 0.0

        config = MCTSConfig(num_simulations=10, c_puct=1.5)
        engine = MCTSEngine(config, eval_fn)
        probs, value = engine.search(env, add_noise=False)

        assert probs.shape == (env.action_space_size,)
        assert abs(probs.sum() - 1.0) < 1e-4
        assert isinstance(value, float)

    def test_mcts_does_not_mutate_env(self):
        env = MiniResourceEnv(board_size=4, max_steps=20, seed=42)
        env.reset(seed=42)
        obs_before = env.get_observation()

        def eval_fn(spatial, vector, mask):
            return np.ones(5, dtype=np.float32) / 5, 0.0

        config = MCTSConfig(num_simulations=5)
        engine = MCTSEngine(config, eval_fn)
        engine.search(env, add_noise=False)

        obs_after = env.get_observation()
        np.testing.assert_array_equal(obs_before["spatial"], obs_after["spatial"])


class TestSelfPlayPipeline:
    """Full self-play -> replay buffer pipeline."""

    def test_self_play_generates_valid_trajectory(self):
        def eval_fn(spatial, vector, mask):
            return np.ones(5, dtype=np.float32) / 5, 0.0

        config = MCTSConfig(num_simulations=5)
        collector = SelfPlayCollector(eval_fn, config, max_steps=10)

        env = MiniResourceEnv(board_size=4, max_steps=10, num_resources=2)
        trajectory = collector.play_game(env, add_noise=False, seed=42)

        assert isinstance(trajectory, GameTrajectory)
        assert len(trajectory.examples) > 0
        assert trajectory.outcome in list(GameOutcome)

    def test_self_play_to_replay_buffer(self):
        def eval_fn(spatial, vector, mask):
            return np.ones(5, dtype=np.float32) / 5, 0.0

        config = MCTSConfig(num_simulations=3)
        collector = SelfPlayCollector(eval_fn, config, max_steps=8)

        buffer = ReplayBuffer(capacity=1000)

        # Play 3 games and add to buffer
        trajectories = collector.play_games(
            env_factory=lambda: MiniResourceEnv(board_size=4, max_steps=8, num_resources=2),
            num_games=3,
        )

        for traj in trajectories:
            buffer.add_trajectory(traj)

        assert buffer.size > 0

        # Sample a batch
        batch_size = min(4, buffer.size)
        batch = buffer.sample_batch(batch_size)

        assert batch["spatial"].shape[0] == batch_size
        assert batch["vector"].shape[0] == batch_size
        assert batch["policy"].shape[0] == batch_size
        assert batch["value"].shape == (batch_size,)


class TestAdapterIntegration:
    """Test Civ6 adapter integration."""

    def test_adapter_stub_satisfies_protocol(self):
        adapter = MicroCivAdapter(board_size=6, use_stub=True)
        assert_valid_environment(adapter)

    def test_adapter_stub_works_with_mcts(self):
        adapter = MicroCivAdapter(board_size=4, max_steps=10, use_stub=True)
        adapter.reset(seed=0)

        action_size = adapter.action_space_size

        def eval_fn(spatial, vector, mask):
            return np.ones(action_size, dtype=np.float32) / action_size, 0.0

        config = MCTSConfig(num_simulations=3)
        engine = MCTSEngine(config, eval_fn)
        probs, _value = engine.search(adapter, add_noise=False)

        assert probs.shape == (action_size,)

    def test_adapter_in_registry(self):
        registry = EnvironmentRegistry()
        registry.register("microciv_stub", lambda **kw: MicroCivAdapter(use_stub=True, **kw))
        env = registry.create("microciv_stub", board_size=4)
        assert_valid_environment(env)


class TestProtocolCompliance:
    """Cross-cutting protocol compliance checks."""

    def test_mini_resource_full_compliance(self):
        env = MiniResourceEnv()
        missing = check_protocol_compliance(env, StrategicEnvironment)
        assert missing == []

    def test_adapter_stub_full_compliance(self):
        env = MicroCivAdapter(board_size=4, use_stub=True)
        missing = check_protocol_compliance(env, StrategicEnvironment)
        assert missing == []
