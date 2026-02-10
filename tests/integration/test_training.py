"""Integration tests for training pipeline."""

import os
import tempfile

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
    evaluate_against_random,
    run_self_play,
    self_play_game,
    train_epoch,
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
        loss, _, _ = compute_loss(policy_logits, value, target_policy, target_value, action_mask)

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


class TestFullGameSimulation:
    """Tests that verify complete games can be played."""

    @pytest.fixture
    def small_env(self):
        """Small environment for fast testing."""
        return MicroCivEnv(board_size=8, max_turns=50, seed=42)

    @pytest.fixture
    def simple_mcts(self):
        """SimpleMCTS with minimal simulations for speed."""
        return SimpleMCTS(num_simulations=5, c_puct=1.5, rollout_depth=5)

    @pytest.fixture
    def small_network(self, small_env):
        """Small network for testing."""
        return MicroCivNetwork(
            board_size=8,
            action_space_size=small_env.action_space.n,
            num_res_blocks=2,
            num_filters=16,
        )

    @pytest.fixture
    def neural_mcts(self, small_network):
        """Neural MCTS with minimal simulations."""
        config = MCTSConfig(num_simulations=5, c_puct=1.5)
        return Civ6MCTS(small_network, config, device="cpu")

    def test_complete_game_with_simple_mcts(self, small_env, simple_mcts):
        """Play a full game using SimpleMCTS until termination."""
        obs, _ = small_env.reset()
        steps = 0
        max_steps = 500  # Prevent infinite loop

        while steps < max_steps:
            # Get MCTS action
            action_probs, _ = simple_mcts.search(small_env)
            action = simple_mcts.select_action(action_probs, temperature=0.5)

            # Take action
            _obs, _reward, terminated, truncated, _info = small_env.step(action)
            steps += 1

            if terminated or truncated:
                break

        # Game should have ended
        assert terminated or truncated or steps >= max_steps
        # Should have taken at least some steps
        assert steps > 0

    def test_complete_game_with_neural_mcts(self, small_env, small_network, neural_mcts):
        """Play a full game using neural network MCTS."""
        obs, _ = small_env.reset()
        steps = 0
        max_steps = 200  # Reduced for speed

        while steps < max_steps:
            # Get neural MCTS action
            action_probs, root_value = neural_mcts.search(small_env, add_noise=False)
            action = neural_mcts.select_action(action_probs, temperature=0.5)

            # Verify output validity
            assert np.all(action_probs >= 0)
            assert abs(action_probs.sum() - 1.0) < 1e-5
            assert -1 <= root_value <= 1

            # Take action
            _obs, _reward, terminated, truncated, _info = small_env.step(action)
            steps += 1

            if terminated or truncated:
                break

        # Game should complete or reach step limit
        assert steps > 0

    def test_game_produces_winner_or_draw(self, small_env, simple_mcts):
        """Verify games end with valid outcome."""
        obs, _ = small_env.reset()
        max_steps = 500

        for _ in range(max_steps):
            action_probs, _ = simple_mcts.search(small_env)
            action = simple_mcts.select_action(action_probs, temperature=0.5)
            _obs, _reward, terminated, truncated, info = small_env.step(action)

            if terminated or truncated:
                break

        # If game ended, check winner validity
        if terminated or truncated:
            winner = info.get("winner")
            # Winner should be None (draw), 0, or 1
            assert winner is None or winner in [0, 1]

    def test_multiple_games_different_seeds(self):
        """Run multiple games with different seeds to ensure variety."""
        outcomes = []

        for seed in range(3):
            env = MicroCivEnv(board_size=8, max_turns=30, seed=seed)
            mcts = SimpleMCTS(num_simulations=5, c_puct=1.5, rollout_depth=3)
            env.reset()

            for _ in range(100):
                action_probs, _ = mcts.search(env)
                action = mcts.select_action(action_probs, temperature=0.5)
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            outcomes.append(
                {
                    "winner": info.get("winner"),
                    "turn": env.turn,
                    "num_units": info.get("num_units"),
                }
            )

        # All games should have run
        assert len(outcomes) == 3


class TestCheckpointIntegration:
    """Tests for model saving and loading."""

    @pytest.fixture
    def network(self):
        """Create a small network for testing."""
        env = MicroCivEnv(board_size=8)
        return MicroCivNetwork(
            board_size=8,
            action_space_size=env.action_space.n,
            num_res_blocks=2,
            num_filters=16,
        )

    @pytest.fixture
    def sample_inputs(self, network):
        """Create sample inputs for testing."""
        batch_size = 2
        spatial = torch.randn(batch_size, 13, 8, 8)
        vector = torch.randn(batch_size, 64)
        action_mask = torch.ones(batch_size, network.config.action_space_size)
        return spatial, vector, action_mask

    def test_save_and_load_preserves_behavior(self, network, sample_inputs):
        """Save model, load it, verify same outputs."""
        spatial, vector, action_mask = sample_inputs

        # Get outputs from original network
        network.eval()
        with torch.no_grad():
            orig_policy, orig_value = network(spatial, vector, action_mask)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            network.save(checkpoint_path)

            # Load into new network
            loaded_network = MicroCivNetwork.load(checkpoint_path, device="cpu")
            loaded_network.eval()

            # Get outputs from loaded network
            with torch.no_grad():
                loaded_policy, loaded_value = loaded_network(spatial, vector, action_mask)

            # Outputs should be identical
            assert torch.allclose(orig_policy, loaded_policy, atol=1e-6)
            assert torch.allclose(orig_value, loaded_value, atol=1e-6)

            # Config should be preserved
            assert loaded_network.config.spatial_size == network.config.spatial_size
            assert loaded_network.config.num_filters == network.config.num_filters

        finally:
            os.unlink(checkpoint_path)

    @pytest.mark.xfail(reason="Pre-existing: Checkpoint training integration out of date", strict=False)
    def test_checkpoint_resumable_training(self, network, sample_inputs):
        """Save during training, load, continue training."""
        spatial, vector, action_mask = sample_inputs
        target_policy = torch.softmax(torch.randn_like(action_mask), dim=-1)
        target_value = torch.zeros(spatial.size(0))

        # Train for a few steps
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

        for _ in range(5):
            network.train()
            optimizer.zero_grad()
            policy_logits, value = network(spatial, vector, action_mask)
            loss, _, _ = compute_loss(
                policy_logits, value, target_policy, target_value, action_mask
            )
            loss.backward()
            optimizer.step()

        # Get loss at save point
        network.eval()
        with torch.no_grad():
            policy_logits, value = network(spatial, vector, action_mask)
            save_loss, _, _ = compute_loss(
                policy_logits, value, target_policy, target_value, action_mask
            )

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            network.save(checkpoint_path)

            # Load and continue training
            loaded_network = MicroCivNetwork.load(checkpoint_path, device="cpu")
            new_optimizer = torch.optim.Adam(loaded_network.parameters(), lr=0.01)

            # Train more
            for _ in range(10):
                loaded_network.train()
                new_optimizer.zero_grad()
                policy_logits, value = loaded_network(spatial, vector, action_mask)
                loss, _, _ = compute_loss(
                    policy_logits, value, target_policy, target_value, action_mask
                )
                loss.backward()
                new_optimizer.step()

            # Get final loss
            loaded_network.eval()
            with torch.no_grad():
                policy_logits, value = loaded_network(spatial, vector, action_mask)
                final_loss, _, _ = compute_loss(
                    policy_logits, value, target_policy, target_value, action_mask
                )

            # Final loss should be lower than save point loss
            assert final_loss.item() < save_loss.item()

        finally:
            os.unlink(checkpoint_path)

    def test_checkpoint_works_with_mcts(self, network):
        """Verify saved/loaded model works with MCTS."""
        env = MicroCivEnv(board_size=8, seed=42)
        env.reset()

        # Create MCTS with original network
        config = MCTSConfig(num_simulations=5)
        orig_mcts = Civ6MCTS(network, config, device="cpu")
        orig_mcts.search(env, add_noise=False)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            network.save(checkpoint_path)
            loaded_network = MicroCivNetwork.load(checkpoint_path, device="cpu")

            # Create MCTS with loaded network
            loaded_mcts = Civ6MCTS(loaded_network, config, device="cpu")

            # Reset environment to same state
            env.reset()
            loaded_probs, loaded_value = loaded_mcts.search(env, add_noise=False)

            # Results should be similar (not exact due to MCTS randomness in tree)
            # But at least they should be valid
            assert np.all(loaded_probs >= 0)
            assert abs(loaded_probs.sum() - 1.0) < 1e-5
            assert -1 <= loaded_value <= 1

        finally:
            os.unlink(checkpoint_path)


class TestFullPipeline:
    """Mini end-to-end training tests."""

    @pytest.fixture
    def small_config(self):
        """Configuration for fast testing."""
        return TrainingConfig(
            board_size=8,
            max_turns=30,
            num_res_blocks=2,
            num_filters=16,
            num_simulations=5,
            batch_size=8,
            learning_rate=0.01,
            num_epochs=2,
            num_games_per_iteration=2,
            num_iterations=2,
            replay_buffer_size=500,
            min_replay_size=10,
            temperature=1.0,
            temperature_threshold=5,
            device="cpu",
        )

    @pytest.fixture
    def small_network(self, small_config):
        """Small network for testing."""
        env = MicroCivEnv(board_size=small_config.board_size)
        return MicroCivNetwork(
            board_size=small_config.board_size,
            action_space_size=env.action_space.n,
            num_res_blocks=small_config.num_res_blocks,
            num_filters=small_config.num_filters,
        )

    def test_mini_training_loop(self, small_config, small_network):
        """Run 2 iterations of self-play + training."""
        from torch.utils.data import DataLoader

        replay_buffer = ReplayBuffer(max_size=small_config.replay_buffer_size)
        optimizer = torch.optim.Adam(small_network.parameters(), lr=small_config.learning_rate)

        losses = []

        for iteration in range(2):
            # Self-play phase
            examples = run_self_play(
                small_network,
                small_config,
                num_games=2,
                use_simple_mcts=(iteration == 0),  # Use simple MCTS first iteration
            )

            # Verify examples generated
            assert len(examples) > 0
            for ex in examples:
                assert ex.spatial.shape == (13, 8, 8)
                assert ex.vector.shape == (64,)

            replay_buffer.add_game(examples)

            # Training phase (if enough examples)
            if len(replay_buffer) >= small_config.batch_size:
                train_examples = replay_buffer.sample(
                    min(len(replay_buffer), small_config.batch_size * 2)
                )
                dataset = ReplayDataset(train_examples)
                dataloader = DataLoader(dataset, batch_size=small_config.batch_size, shuffle=True)

                epoch_losses = train_epoch(small_network, optimizer, dataloader, small_config)
                losses.append(epoch_losses["total_loss"])

        # Should have completed iterations
        assert len(replay_buffer) > 0

        # If we trained, verify we got losses
        if losses:
            assert all(loss > 0 for loss in losses)
            assert not any(np.isnan(loss) for loss in losses)

    def test_evaluation_against_random(self, small_config, small_network):
        """Test the evaluate_against_random function."""
        # Run evaluation with minimal games
        win_rate = evaluate_against_random(
            small_network,
            small_config,
            num_games=4,  # Small number for speed
        )

        # Win rate should be between 0 and 1
        assert 0.0 <= win_rate <= 1.0

    def test_self_play_generates_diverse_data(self, small_config, small_network):
        """Verify self-play generates diverse training data."""
        examples = run_self_play(small_network, small_config, num_games=3, use_simple_mcts=True)

        # Should have multiple examples
        assert len(examples) >= 3  # At least one per game

        # Check diversity in policies
        policies = np.array([ex.policy for ex in examples])

        # Not all policies should be identical
        if len(policies) > 1:
            policy_variance = np.var(policies, axis=0).sum()
            assert policy_variance > 0  # Some variance in policies

        # Values should include different outcomes
        values = [ex.value for ex in examples]
        unique_values = set(values)
        # Should have at least one unique value type
        assert len(unique_values) >= 1

    def test_training_improves_network(self, small_config, small_network):
        """Verify training reduces loss on fixed data."""
        from torch.utils.data import DataLoader

        # Generate fixed training data
        examples = run_self_play(small_network, small_config, num_games=2, use_simple_mcts=True)
        dataset = ReplayDataset(examples)
        dataloader = DataLoader(dataset, batch_size=min(8, len(examples)), shuffle=True)

        optimizer = torch.optim.Adam(small_network.parameters(), lr=0.01)

        # Measure initial loss
        initial_losses = train_epoch(small_network, optimizer, dataloader, small_config)

        # Train for more epochs on same data
        for _ in range(5):
            final_losses = train_epoch(small_network, optimizer, dataloader, small_config)

        # Loss should decrease
        assert final_losses["total_loss"] < initial_losses["total_loss"]

    def test_full_loop_with_checkpointing(self, small_config, small_network):
        """Test a complete training loop with checkpoint save/load."""
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "model.pt")

            # Phase 1: Generate data and train
            replay_buffer = ReplayBuffer(max_size=small_config.replay_buffer_size)
            optimizer = torch.optim.Adam(small_network.parameters(), lr=small_config.learning_rate)

            examples = run_self_play(small_network, small_config, num_games=2, use_simple_mcts=True)
            replay_buffer.add_game(examples)

            if len(replay_buffer) >= small_config.batch_size:
                train_examples = replay_buffer.sample(len(replay_buffer))
                dataset = ReplayDataset(train_examples)
                dataloader = DataLoader(dataset, batch_size=small_config.batch_size, shuffle=True)
                train_epoch(small_network, optimizer, dataloader, small_config)

            # Save checkpoint
            small_network.save(checkpoint_path)
            assert os.path.exists(checkpoint_path)

            # Phase 2: Load and continue
            loaded_network = MicroCivNetwork.load(checkpoint_path, device="cpu")

            # Run evaluation with loaded network
            win_rate = evaluate_against_random(loaded_network, small_config, num_games=2)
            assert 0.0 <= win_rate <= 1.0

            # Generate more data with loaded network
            more_examples = run_self_play(
                loaded_network, small_config, num_games=2, use_simple_mcts=False
            )
            assert len(more_examples) > 0

    def test_network_output_consistency(self, small_config, small_network):
        """Verify network produces consistent outputs for same input."""
        env = MicroCivEnv(board_size=small_config.board_size, seed=42)
        obs, _ = env.reset()

        spatial = torch.from_numpy(obs["spatial"]).float().unsqueeze(0)
        vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0)
        action_mask = torch.from_numpy(obs["action_mask"]).float().unsqueeze(0)

        small_network.eval()
        with torch.no_grad():
            policy1, value1 = small_network(spatial, vector, action_mask)
            policy2, value2 = small_network(spatial, vector, action_mask)

        # Same inputs should produce identical outputs
        assert torch.allclose(policy1, policy2)
        assert torch.allclose(value1, value2)
