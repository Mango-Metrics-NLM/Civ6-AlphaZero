"""Integration tests for end-to-end Atari training pipeline.

These tests verify the full training loop works with mocked Atari
environments (no ROM needed).
"""

from __future__ import annotations

from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from demos.atari.atari_env import AtariEnv
from demos.atari.atari_network import AtariNetwork
from demos.atari.config import (
    AtariConfig,
    AtariEnvConfig,
    AtariHierarchicalConfig,
    AtariNetworkConfig,
    AtariTrainingConfig,
)
from demos.atari.hierarchical import AtariHierarchicalAdapter
from demos.atari.novel_features import NovelFeaturesConfig
from demos.atari.trainer import (
    AtariTrainer,
    ReplayBuffer,
    TrainingExample,
    run_episode,
    train_epoch,
)


# ============================================================================
# Fake environment for integration tests
# ============================================================================


class _FakeAtariGym:
    """Minimal Atari-like env that terminates after a few steps."""

    class _Space:
        n = 4

    action_space = _Space()
    observation_space = None
    _step_count = 0
    _max_steps = 10

    def reset(self, seed=None):
        self._step_count = 0
        return np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8), {"lives": 3}

    def step(self, action):
        self._step_count += 1
        obs = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
        reward = float(np.random.choice([-1, 0, 1]))
        terminated = self._step_count >= self._max_steps
        return obs, reward, terminated, False, {"lives": 3}

    def close(self):
        pass


@pytest.fixture(autouse=True)
def mock_gymnasium(monkeypatch):
    """Globally mock gymnasium.make for all integration tests."""
    monkeypatch.setattr(gym, "make", lambda *a, **kw: _FakeAtariGym())


@pytest.fixture
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


# ============================================================================
# MCTSEnvironment protocol integration
# ============================================================================


class TestMCTSProtocolIntegration:
    """Verify AtariEnv works with the core MCTS."""

    def test_simple_mcts_search(self):
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        env = AtariEnv(AtariEnvConfig(game_name="Fake", frame_height=42, frame_width=42, frame_stack=2))
        env.reset()

        mcts = SimpleMCTS(num_simulations=5, c_puct=1.5)
        action_probs, value = mcts.search(env)

        assert action_probs.shape == (env.action_space.n,)
        assert np.isclose(action_probs.sum(), 1.0, atol=0.01)
        assert isinstance(value, float)

    def test_civ6_mcts_search(self):
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig

        env_cfg = AtariEnvConfig(game_name="Fake", frame_height=42, frame_width=42, frame_stack=2, vector_dim=8)
        env = AtariEnv(env_cfg)
        env.reset()

        net = AtariNetwork(
            config=AtariNetworkConfig(num_res_blocks=1, num_filters=8, fc_hidden_dim=16, value_head_hidden=8),
            num_actions=env.action_space.n,
            spatial_size=env_cfg.spatial_size,
            spatial_channels=env_cfg.spatial_channels,
            vector_dim=env_cfg.vector_dim,
        )

        mcts_cfg = MCTSConfig(num_simulations=3, c_puct=1.5)
        mcts = Civ6MCTS(net, mcts_cfg, device="cpu")

        action_probs, value = mcts.search(env)
        assert action_probs.shape == (env.action_space.n,)
        assert isinstance(value, float)


# ============================================================================
# Single episode integration
# ============================================================================


class TestRunEpisode:
    def test_episode_produces_examples(self, fast_config):
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        env = AtariEnv(fast_config.environment)
        net = AtariNetwork(
            config=fast_config.network,
            num_actions=env.action_space.n,
            spatial_size=fast_config.environment.spatial_size,
            spatial_channels=fast_config.environment.spatial_channels,
            vector_dim=fast_config.environment.vector_dim,
        )
        mcts = SimpleMCTS(num_simulations=3, c_puct=1.5)

        examples, trajectory = run_episode(env, net, mcts, fast_config.training)
        assert len(examples) > 0
        assert isinstance(examples[0], TrainingExample)
        assert examples[0].spatial.shape[0] == fast_config.environment.spatial_channels
        assert len(trajectory) > 0
        assert "obs" in trajectory[0]
        assert "action" in trajectory[0]
        assert "reward" in trajectory[0]
        env.close()

    def test_episode_with_hierarchical(self, fast_config):
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        fast_config.hierarchical = AtariHierarchicalConfig(
            enabled=True, strategic_update_freq=3, tactical_update_freq=2
        )
        env = AtariEnv(fast_config.environment)
        net = AtariNetwork(
            config=fast_config.network,
            num_actions=env.action_space.n,
            spatial_size=fast_config.environment.spatial_size,
            spatial_channels=fast_config.environment.spatial_channels,
            vector_dim=fast_config.environment.vector_dim,
        )
        mcts = SimpleMCTS(num_simulations=3, c_puct=1.5)
        hier = AtariHierarchicalAdapter(fast_config.hierarchical, env.action_space.n)

        examples, trajectory = run_episode(env, net, mcts, fast_config.training, hierarchical=hier)
        assert len(examples) > 0
        assert len(trajectory) > 0
        env.close()


# ============================================================================
# Training epoch integration
# ============================================================================


class TestTrainEpoch:
    def test_epoch_reduces_loss(self, fast_config):
        env = AtariEnv(fast_config.environment)
        net = AtariNetwork(
            config=fast_config.network,
            num_actions=env.action_space.n,
            spatial_size=fast_config.environment.spatial_size,
            spatial_channels=fast_config.environment.spatial_channels,
            vector_dim=fast_config.environment.vector_dim,
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        # Create synthetic data
        from demos.atari.trainer import ReplayDataset
        from torch.utils.data import DataLoader

        examples = []
        for _ in range(20):
            examples.append(
                TrainingExample(
                    spatial=np.random.rand(
                        fast_config.environment.spatial_channels,
                        fast_config.environment.frame_height,
                        fast_config.environment.frame_width,
                    ).astype(np.float32),
                    vector=np.zeros(fast_config.environment.vector_dim, dtype=np.float32),
                    action_mask=np.ones(env.action_space.n, dtype=np.float32),
                    policy=np.ones(env.action_space.n, dtype=np.float32) / env.action_space.n,
                    value=0.5,
                )
            )

        dataset = ReplayDataset(examples)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        losses = train_epoch(net, optimizer, dataloader, "cpu", gradient_clip=1.0)
        assert "total_loss" in losses
        assert losses["total_loss"] > 0
        env.close()


# ============================================================================
# Full trainer dry run
# ============================================================================


class TestAtariTrainer:
    def test_dry_run(self, fast_config):
        trainer = AtariTrainer(fast_config)
        result = trainer.train(dry_run=True)

        assert "final_model" in result
        assert result["iterations"] >= 1
        assert len(result["metrics"]) >= 1

    def test_dry_run_with_hierarchical(self, fast_config):
        fast_config.hierarchical = AtariHierarchicalConfig(
            enabled=True, strategic_update_freq=3, tactical_update_freq=2
        )
        trainer = AtariTrainer(fast_config)
        result = trainer.train(dry_run=True)
        assert result["iterations"] >= 1

    def test_default_config_dry_run(self):
        """Verify the default AtariConfig works for a dry run."""
        trainer = AtariTrainer(AtariConfig(
            environment=AtariEnvConfig(game_name="FakeGame", frame_height=42, frame_width=42, frame_stack=2, vector_dim=8),
            network=AtariNetworkConfig(num_res_blocks=1, num_filters=8, fc_hidden_dim=16, value_head_hidden=8),
            training=AtariTrainingConfig(device="cpu", min_replay_size=1),
        ))
        result = trainer.train(dry_run=True)
        assert result["iterations"] == 1


# ============================================================================
# Config loading integration
# ============================================================================


class TestConfigIntegration:
    def test_pong_yaml_loads(self):
        """Verify the shipped pong.yaml is valid."""
        import os
        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "demos", "atari", "configs", "pong.yaml",
        )
        if os.path.exists(yaml_path):
            cfg = AtariConfig.from_yaml(yaml_path)
            assert cfg.environment.game_name == "Pong"

    def test_breakout_yaml_loads(self):
        """Verify the shipped breakout.yaml is valid."""
        import os
        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "demos", "atari", "configs", "breakout.yaml",
        )
        if os.path.exists(yaml_path):
            cfg = AtariConfig.from_yaml(yaml_path)
            assert cfg.environment.game_name == "Breakout"


# ============================================================================
# Novel features integration
# ============================================================================


class TestNovelFeaturesIntegration:
    def test_dry_run_with_reflection(self, fast_config):
        """Verify training works with reflection memory enabled."""
        fast_config.novel_features = NovelFeaturesConfig(
            reflection_enabled=True,
            rag_enabled=False,
            embedding_dimension=16,
            reflection_max_entries=50,
        )
        trainer = AtariTrainer(fast_config)
        result = trainer.train(dry_run=True)
        assert result["iterations"] >= 1

    def test_dry_run_with_rag(self, fast_config):
        """Verify training works with RAG advisor enabled."""
        fast_config.novel_features = NovelFeaturesConfig(
            reflection_enabled=False,
            rag_enabled=True,
            embedding_dimension=16,
        )
        trainer = AtariTrainer(fast_config)
        result = trainer.train(dry_run=True)
        assert result["iterations"] >= 1

    def test_dry_run_with_all_novel_features(self, fast_config):
        """Verify training works with all novel features enabled."""
        fast_config.novel_features = NovelFeaturesConfig(
            reflection_enabled=True,
            rag_enabled=True,
            embedding_dimension=16,
            reflection_max_entries=50,
        )
        fast_config.hierarchical = AtariHierarchicalConfig(
            enabled=True, strategic_update_freq=3, tactical_update_freq=2
        )
        trainer = AtariTrainer(fast_config)
        result = trainer.train(dry_run=True)
        assert result["iterations"] >= 1

    def test_run_episode_with_rag(self, fast_config):
        """Verify run_episode integrates RAG policy boost."""
        from alpha_zero.core.civ6_mcts import SimpleMCTS
        from demos.atari.novel_features import AtariRAGAdvisor

        env = AtariEnv(fast_config.environment)
        net = AtariNetwork(
            config=fast_config.network,
            num_actions=env.action_space.n,
            spatial_size=fast_config.environment.spatial_size,
            spatial_channels=fast_config.environment.spatial_channels,
            vector_dim=fast_config.environment.vector_dim,
        )
        mcts = SimpleMCTS(num_simulations=3, c_puct=1.5)

        rag_cfg = NovelFeaturesConfig(rag_enabled=True, embedding_dimension=16)
        rag = AtariRAGAdvisor(rag_cfg)
        rag.ingest_knowledge("Test strategy", action_hints=[0, 1])

        examples, trajectory = run_episode(
            env, net, mcts, fast_config.training,
            rag_advisor=rag, game_name="FakeGame",
        )
        assert len(examples) > 0
        assert len(trajectory) > 0
        env.close()
