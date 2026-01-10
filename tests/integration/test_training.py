"""Integration tests for training pipeline."""

import numpy as np
import pytest
import torch

from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig, SimpleMCTS
from alpha_zero.core.civ6_network import MicroCivNetwork, compute_loss
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.training_civ6 import (
    ReplayBuffer,
    ReplayDataset,
    TrainingConfig,
    TrainingExample,
    self_play_game,
)


class TestSelfPlay:
    """Test suite for self-play functionality."""

    @pytest.fixture
    def env(self):
        return MicroCivEnv(board_size=8, max_turns=50, seed=42)

    @pytest.fixture
    def mcts(self):
        return SimpleMCTS(num_simulations=10, c_puct=1.5, rollout_depth=5)

    @pytest.fixture
    def config(self):
        return TrainingConfig(
            board_size=8,
            max_turns=50,
            num_simulations=10,
            temperature=1.0,
            temperature_threshold=10,
        )

    def test_self_play_generates_examples(self, env, mcts, config):
        """Self-play generates training examples."""
        examples = self_play_game(env, mcts, config)

        assert len(examples) > 0
        for ex in examples:
            assert ex.spatial.shape == (13, 8, 8)
            assert ex.vector.shape == (64,)
            assert ex.action_mask.shape == (env.action_space.n,)
            assert ex.policy.shape == (env.action_space.n,)
            assert -1 <= ex.value <= 1

    def test_examples_have_valid_values(self, env, mcts, config):
        """Training examples have correct value assignments."""
        examples = self_play_game(env, mcts, config)

        # Values should be -1, 0, or 1
        for ex in examples:
            assert ex.value in [-1.0, 0.0, 1.0]

    def test_examples_have_valid_policies(self, env, mcts, config):
        """Policies are valid probability distributions."""
        examples = self_play_game(env, mcts, config)

        for ex in examples:
            assert np.all(ex.policy >= 0)
            assert abs(ex.policy.sum() - 1.0) < 1e-5


class TestReplayBuffer:
    """Test suite for replay buffer."""

    @pytest.fixture
    def buffer(self):
        return ReplayBuffer(max_size=100)

    @pytest.fixture
    def sample_example(self):
        return TrainingExample(
            spatial=np.random.randn(13, 8, 8).astype(np.float32),
            vector=np.random.randn(64).astype(np.float32),
            action_mask=np.ones(100, dtype=np.float32),
            policy=np.ones(100, dtype=np.float32) / 100,
            value=0.5,
        )

    def test_add_and_sample(self, buffer, sample_example):
        """Buffer can add and sample examples."""
        for _ in range(50):
            buffer.add(sample_example)

        assert len(buffer) == 50

        samples = buffer.sample(10)
        assert len(samples) == 10

    def test_max_size_limit(self, buffer, sample_example):
        """Buffer respects max size."""
        for _ in range(200):
            buffer.add(sample_example)

        assert len(buffer) == 100

    def test_add_game(self, buffer, sample_example):
        """Can add multiple examples from a game."""
        examples = [sample_example for _ in range(10)]
        buffer.add_game(examples)

        assert len(buffer) == 10


class TestReplayDataset:
    """Test suite for replay dataset."""

    @pytest.fixture
    def examples(self):
        return [
            TrainingExample(
                spatial=np.random.randn(13, 8, 8).astype(np.float32),
                vector=np.random.randn(64).astype(np.float32),
                action_mask=np.ones(100, dtype=np.float32),
                policy=np.ones(100, dtype=np.float32) / 100,
                value=float(i) / 10,
            )
            for i in range(20)
        ]

    def test_dataset_length(self, examples):
        """Dataset has correct length."""
        dataset = ReplayDataset(examples)
        assert len(dataset) == 20

    def test_dataset_getitem(self, examples):
        """Dataset returns correct items."""
        dataset = ReplayDataset(examples)
        item = dataset[5]

        assert "spatial" in item
        assert "vector" in item
        assert "action_mask" in item
        assert "policy" in item
        assert "value" in item

        assert item["spatial"].shape == (13, 8, 8)
        assert item["vector"].shape == (64,)
        assert item["action_mask"].shape == (100,)
        assert item["policy"].shape == (100,)
        assert item["value"].shape == ()

    def test_dataloader_iteration(self, examples):
        """Dataset works with DataLoader."""
        from torch.utils.data import DataLoader

        dataset = ReplayDataset(examples)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        batch = next(iter(loader))
        assert batch["spatial"].shape == (4, 13, 8, 8)
        assert batch["value"].shape == (4,)


class TestEndToEndTraining:
    """End-to-end training tests."""

    def test_training_step(self):
        """Single training step works."""
        # Create network
        env = MicroCivEnv(board_size=8)
        network = MicroCivNetwork(
            board_size=8,
            action_space_size=env.action_space.n,
            num_res_blocks=2,
            num_filters=16,
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Create batch
        batch_size = 4
        spatial = torch.randn(batch_size, 13, 8, 8)
        vector = torch.randn(batch_size, 64)
        action_mask = torch.ones(batch_size, env.action_space.n)
        target_policy = torch.softmax(torch.randn(batch_size, env.action_space.n), dim=-1)
        target_value = torch.randn(batch_size)

        # Training step
        network.train()
        optimizer.zero_grad()

        policy_logits, value = network(spatial, vector, action_mask)
        loss, _, _ = compute_loss(
            policy_logits, value, target_policy, target_value, action_mask
        )

        loss.backward()
        optimizer.step()

        # Should not have NaN
        assert not torch.isnan(loss)
        for param in network.parameters():
            assert not torch.any(torch.isnan(param))

    def test_network_improves_with_training(self):
        """Network loss decreases with training on fixed data."""
        env = MicroCivEnv(board_size=8)
        network = MicroCivNetwork(
            board_size=8,
            action_space_size=env.action_space.n,
            num_res_blocks=2,
            num_filters=16,
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

        # Fixed training data
        batch_size = 8
        spatial = torch.randn(batch_size, 13, 8, 8)
        vector = torch.randn(batch_size, 64)
        action_mask = torch.ones(batch_size, env.action_space.n)
        target_policy = torch.softmax(torch.randn(batch_size, env.action_space.n), dim=-1)
        target_value = torch.zeros(batch_size)

        initial_loss = None
        final_loss = None

        for epoch in range(50):
            network.train()
            optimizer.zero_grad()

            policy_logits, value = network(spatial, vector, action_mask)
            loss, _, _ = compute_loss(
                policy_logits, value, target_policy, target_value, action_mask
            )

            if epoch == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_mcts_with_trained_network(self):
        """MCTS works with a network that has been trained."""
        env = MicroCivEnv(board_size=8, seed=42)
        network = MicroCivNetwork(
            board_size=8,
            action_space_size=env.action_space.n,
            num_res_blocks=2,
            num_filters=16,
        )

        # Train briefly
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        for _ in range(10):
            spatial = torch.randn(4, 13, 8, 8)
            vector = torch.randn(4, 64)
            action_mask = torch.ones(4, env.action_space.n)
            target_policy = torch.softmax(torch.randn(4, env.action_space.n), dim=-1)
            target_value = torch.zeros(4)

            network.train()
            optimizer.zero_grad()
            policy_logits, value = network(spatial, vector, action_mask)
            loss, _, _ = compute_loss(
                policy_logits, value, target_policy, target_value, action_mask
            )
            loss.backward()
            optimizer.step()

        # Now run MCTS
        env.reset()
        config = MCTSConfig(num_simulations=10, c_puct=1.5)
        mcts = Civ6MCTS(network, config, device="cpu")

        action_probs, root_value = mcts.search(env)

        # Should produce valid output
        assert np.all(action_probs >= 0)
        assert abs(action_probs.sum() - 1.0) < 1e-5
        assert -1 <= root_value <= 1
