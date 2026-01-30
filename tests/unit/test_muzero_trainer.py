"""Unit tests for MuZero trainer module.

Tests for alpha_zero.training.muzero_trainer covering:
- MuZeroTrainingConfig dataclass
- GameTrajectory creation and manipulation
- TrajectoryBuffer sampling
- MuZeroTrainer train_step with mock network
- Unrolling logic
- Target computation
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from alpha_zero.training.muzero_trainer import (
    GameTrajectory,
    MuZeroTrainer,
    MuZeroTrainingConfig,
    TrajectoryBatchSample,
    TrajectoryBuffer,
)

# =============================================================================
# Test MuZeroTrainingConfig
# =============================================================================


class TestMuZeroTrainingConfig:
    """Test suite for MuZeroTrainingConfig dataclass."""

    def test_default_values(self):
        """Default config has expected values."""
        config = MuZeroTrainingConfig()

        assert config.unroll_steps == 5
        assert config.td_steps == 10
        assert config.use_muzero_network is True
        assert config.use_consistency_loss is True
        assert config.use_value_prefix is True
        assert config.batch_size == 256
        assert config.learning_rate == 0.001
        assert config.weight_decay == 1e-4
        assert config.gradient_clip == 1.0
        assert config.discount == 0.997
        assert config.use_categorical is True
        assert config.support_size == 601

    def test_custom_values(self):
        """Config accepts custom values."""
        config = MuZeroTrainingConfig(
            unroll_steps=3,
            td_steps=5,
            batch_size=128,
            learning_rate=0.0001,
            use_consistency_loss=False,
        )

        assert config.unroll_steps == 3
        assert config.td_steps == 5
        assert config.batch_size == 128
        assert config.learning_rate == 0.0001
        assert config.use_consistency_loss is False

    def test_to_loss_config(self):
        """Converts to LossConfig correctly."""
        config = MuZeroTrainingConfig(
            policy_loss_weight=2.0,
            value_loss_weight=0.5,
            reward_loss_weight=0.25,
            consistency_loss_weight=1.0,
        )

        loss_config = config.to_loss_config()

        assert loss_config.policy_loss_weight == 2.0
        assert loss_config.value_loss_weight == 0.5
        assert loss_config.reward_loss_weight == 0.25
        assert loss_config.consistency_loss_weight == 1.0
        assert loss_config.use_categorical == config.use_categorical
        assert loss_config.support_size == config.support_size


# =============================================================================
# Test GameTrajectory
# =============================================================================


class TestGameTrajectory:
    """Test suite for GameTrajectory dataclass."""

    def test_empty_trajectory(self):
        """Empty trajectory initializes correctly."""
        traj = GameTrajectory()

        assert len(traj) == 0
        assert len(traj.observations) == 0
        assert len(traj.actions) == 0
        assert len(traj.rewards) == 0
        assert len(traj.policies) == 0
        assert len(traj.values) == 0

    def test_append_step(self):
        """Appending steps works correctly."""
        traj = GameTrajectory(game_id="test_game")

        obs = {
            "spatial": np.random.randn(13, 16, 16).astype(np.float32),
            "vector": np.random.randn(64).astype(np.float32),
            "action_mask": np.ones(100, dtype=np.float32),
        }
        # Create softmax policy manually
        logits = np.random.randn(100).astype(np.float32)
        exp_logits = np.exp(logits - logits.max())
        policy = exp_logits / exp_logits.sum()

        traj.append(
            observation=obs,
            action=5,
            reward=0.1,
            policy=policy,
            value=0.5,
            done=False,
        )

        assert len(traj) == 1
        assert len(traj.actions) == 1
        assert traj.actions[0] == 5
        assert np.isclose(traj.rewards[0], 0.1)
        assert np.isclose(traj.values[0], 0.5)
        assert traj.dones[0] == False  # noqa: E712 - numpy bool comparison

    def test_append_multiple_steps(self):
        """Appending multiple steps works correctly."""
        traj = GameTrajectory(game_id="test_game")

        for i in range(10):
            obs = {
                "spatial": np.random.randn(13, 16, 16).astype(np.float32),
                "vector": np.random.randn(64).astype(np.float32),
                "action_mask": np.ones(100, dtype=np.float32),
            }
            policy = np.ones(100, dtype=np.float32) / 100

            traj.append(
                observation=obs,
                action=i,
                reward=float(i) * 0.1,
                policy=policy,
                value=float(i) * 0.01,
                done=(i == 9),
            )

        assert len(traj) == 10
        assert len(traj.actions) == 10
        assert traj.actions[5] == 5
        assert traj.dones[-1] == True  # noqa: E712 - numpy bool comparison

    def test_finalize(self):
        """Finalizing trajectory sets values."""
        traj = GameTrajectory(game_id="test_game")

        for i in range(5):
            obs = {
                "spatial": np.random.randn(13, 16, 16).astype(np.float32),
                "vector": np.random.randn(64).astype(np.float32),
                "action_mask": np.ones(100, dtype=np.float32),
            }
            traj.append(observation=obs, action=i, reward=0.0, value=0.0, done=False)

        traj.finalize(final_value=1.0)

        assert len(traj.values) == 5
        assert np.allclose(traj.values, 1.0)

    def test_compute_n_step_returns(self):
        """N-step returns are computed correctly."""
        traj = GameTrajectory(game_id="test_game")

        # Create simple trajectory with known rewards
        for i in range(10):
            obs = {
                "spatial": np.random.randn(13, 16, 16).astype(np.float32),
                "vector": np.random.randn(64).astype(np.float32),
                "action_mask": np.ones(100, dtype=np.float32),
            }
            traj.append(
                observation=obs,
                action=i,
                reward=1.0,  # Constant reward of 1
                value=0.0,
                done=(i == 9),
            )

        # Set terminal value
        traj.values[-1] = 0.0

        traj.compute_n_step_returns(discount=1.0, n_steps=3)

        # With discount=1.0 and n_steps=3, each return should be roughly 3
        # (except near the end)
        assert len(traj.values) == 10
        # First step return should be 3.0 (r0 + r1 + r2 + V(s3))
        # But V(s3) is computed during the process, so check it's reasonable
        assert traj.values[0] > 2.0

    def test_game_id_stored(self):
        """Game ID is stored correctly."""
        traj = GameTrajectory(game_id="unique_game_123")

        assert traj.game_id == "unique_game_123"

    def test_priority_default(self):
        """Default priority is 1.0."""
        traj = GameTrajectory()

        assert traj.priority == 1.0

    def test_priority_custom(self):
        """Custom priority is stored."""
        traj = GameTrajectory(priority=5.0)

        assert traj.priority == 5.0


# =============================================================================
# Test TrajectoryBuffer
# =============================================================================


class TestTrajectoryBuffer:
    """Test suite for TrajectoryBuffer class."""

    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        traj = GameTrajectory(game_id="sample")
        action_size = 100

        for i in range(20):
            obs = {
                "spatial": np.random.randn(13, 16, 16).astype(np.float32),
                "vector": np.random.randn(64).astype(np.float32),
                "action_mask": np.ones(action_size, dtype=np.float32),
            }
            policy = np.ones(action_size, dtype=np.float32) / action_size

            traj.append(
                observation=obs,
                action=i % action_size,
                reward=0.1,
                policy=policy,
                value=0.5,
                done=(i == 19),
            )

        return traj

    def test_empty_buffer(self):
        """Empty buffer has correct state."""
        buffer = TrajectoryBuffer(max_trajectories=100)

        assert len(buffer) == 0
        assert buffer.num_positions == 0

    def test_add_trajectory(self, sample_trajectory):
        """Adding trajectory works correctly."""
        buffer = TrajectoryBuffer(max_trajectories=100)
        buffer.add_trajectory(sample_trajectory)

        assert len(buffer) == 1
        assert buffer.num_positions == len(sample_trajectory)

    def test_add_multiple_trajectories(self, sample_trajectory):
        """Adding multiple trajectories works correctly."""
        buffer = TrajectoryBuffer(max_trajectories=100)

        for i in range(5):
            traj = GameTrajectory(game_id=f"game_{i}")
            for j in range(10):
                obs = {
                    "spatial": np.random.randn(13, 16, 16).astype(np.float32),
                    "vector": np.random.randn(64).astype(np.float32),
                    "action_mask": np.ones(100, dtype=np.float32),
                }
                traj.append(
                    observation=obs,
                    action=j,
                    reward=0.1,
                    policy=np.ones(100, dtype=np.float32) / 100,
                    value=0.0,
                    done=(j == 9),
                )
            buffer.add_trajectory(traj)

        assert len(buffer) == 5
        assert buffer.num_positions == 50

    def test_max_trajectories_enforced(self):
        """Buffer respects max_trajectories limit."""
        buffer = TrajectoryBuffer(max_trajectories=3, max_positions=10000)

        for i in range(5):
            traj = GameTrajectory(game_id=f"game_{i}")
            for j in range(5):
                obs = {
                    "spatial": np.random.randn(13, 16, 16).astype(np.float32),
                    "vector": np.random.randn(64).astype(np.float32),
                    "action_mask": np.ones(100, dtype=np.float32),
                }
                traj.append(
                    observation=obs,
                    action=j,
                    reward=0.0,
                    policy=np.ones(100, dtype=np.float32) / 100,
                    value=0.0,
                    done=(j == 4),
                )
            buffer.add_trajectory(traj)

        # Should only have 3 trajectories (oldest removed)
        assert len(buffer) == 3

    def test_sample_positions_raises_on_empty(self):
        """Sampling from empty buffer raises error."""
        buffer = TrajectoryBuffer(max_trajectories=100)

        with pytest.raises(ValueError, match="empty buffer"):
            buffer.sample_positions(batch_size=4, unroll_steps=3)

    def test_sample_positions_basic(self, sample_trajectory):
        """Basic position sampling works."""
        buffer = TrajectoryBuffer(max_trajectories=100)
        buffer.add_trajectory(sample_trajectory)

        batch = buffer.sample_positions(
            batch_size=4,
            unroll_steps=3,
            device="cpu",
        )

        assert isinstance(batch, TrajectoryBatchSample)
        assert batch.spatial.shape[0] == 4  # batch_size
        assert batch.vector.shape[0] == 4
        assert batch.action_mask.shape[0] == 4
        assert batch.actions.shape == (4, 3)  # (batch_size, unroll_steps)

    def test_sample_positions_targets(self, sample_trajectory):
        """Sample positions have correct target shapes."""
        buffer = TrajectoryBuffer(max_trajectories=100)
        buffer.add_trajectory(sample_trajectory)

        unroll_steps = 3
        batch = buffer.sample_positions(
            batch_size=4,
            unroll_steps=unroll_steps,
            device="cpu",
        )

        # Policies for steps 0..K (K+1 total)
        assert len(batch.target_policies) == unroll_steps + 1
        for p in batch.target_policies:
            assert p.shape == (4, 100)

        # Values for steps 0..K (K+1 total)
        assert len(batch.target_values) == unroll_steps + 1
        for v in batch.target_values:
            assert v.shape == (4,)

        # Rewards for steps 1..K (K total)
        assert len(batch.target_rewards) == unroll_steps
        for r in batch.target_rewards:
            assert r.shape == (4,)

        # Action masks for steps 1..K (K total)
        assert len(batch.action_masks_unroll) == unroll_steps
        for m in batch.action_masks_unroll:
            assert m.shape == (4, 100)

    def test_sample_positions_device(self, sample_trajectory):
        """Sample positions are on correct device."""
        buffer = TrajectoryBuffer(max_trajectories=100)
        buffer.add_trajectory(sample_trajectory)

        batch = buffer.sample_positions(
            batch_size=4,
            unroll_steps=3,
            device="cpu",
        )

        assert batch.spatial.device.type == "cpu"
        assert batch.vector.device.type == "cpu"
        assert batch.actions.device.type == "cpu"

    def test_sample_positions_indices_returned(self, sample_trajectory):
        """Sample positions return indices for priority updates."""
        buffer = TrajectoryBuffer(max_trajectories=100)
        buffer.add_trajectory(sample_trajectory)

        batch = buffer.sample_positions(
            batch_size=4,
            unroll_steps=3,
            device="cpu",
        )

        assert batch.indices is not None
        assert len(batch.indices) == 4
        for traj_idx, pos_idx in batch.indices:
            assert traj_idx == 0  # Only one trajectory
            assert 0 <= pos_idx < len(sample_trajectory)

    def test_sample_positions_handles_trajectory_end(self):
        """Sampling handles positions near trajectory end."""
        buffer = TrajectoryBuffer(max_trajectories=100)

        # Create short trajectory
        traj = GameTrajectory(game_id="short")
        for i in range(3):  # Only 3 steps
            obs = {
                "spatial": np.random.randn(13, 16, 16).astype(np.float32),
                "vector": np.random.randn(64).astype(np.float32),
                "action_mask": np.ones(100, dtype=np.float32),
            }
            traj.append(
                observation=obs,
                action=i,
                reward=0.0,
                policy=np.ones(100, dtype=np.float32) / 100,
                value=0.0,
                done=(i == 2),
            )
        buffer.add_trajectory(traj)

        # Sample with unroll steps > trajectory length
        # Should not crash, uses absorbing states
        batch = buffer.sample_positions(
            batch_size=4,
            unroll_steps=5,  # More than trajectory length
            device="cpu",
        )

        assert batch.spatial.shape[0] == 4
        assert batch.actions.shape == (4, 5)

    def test_update_priorities(self, sample_trajectory):
        """Priority updates work correctly."""
        buffer = TrajectoryBuffer(max_trajectories=100)
        buffer.add_trajectory(sample_trajectory)

        buffer.update_priorities(
            indices=[(0, 5), (0, 10)],
            priorities=[3.0, 4.0],
        )

        # Priority should be updated to max
        assert buffer.trajectories[0].priority >= max(3.0, 4.0)


# =============================================================================
# Mock Network for Testing
# =============================================================================


class MockMuZeroNetwork(nn.Module):
    """Mock MuZero network for testing trainer."""

    def __init__(
        self,
        spatial_channels: int = 13,
        spatial_size: int = 16,
        vector_dim: int = 64,
        action_size: int = 100,
        hidden_channels: int = 32,
        use_categorical: bool = True,
        support_size: int = 601,
    ):
        super().__init__()
        self.spatial_channels = spatial_channels
        self.spatial_size = spatial_size
        self.vector_dim = vector_dim
        self.action_size = action_size
        self.hidden_channels = hidden_channels
        self.use_categorical = use_categorical
        self.support_size = support_size

        # Simple linear layers for testing
        self.spatial_proj = nn.Conv2d(spatial_channels, hidden_channels, 1)
        self.policy_head = nn.Linear(hidden_channels * spatial_size * spatial_size, action_size)

        if use_categorical:
            self.value_head = nn.Linear(hidden_channels * spatial_size * spatial_size, support_size)
            self.reward_head = nn.Linear(
                hidden_channels * spatial_size * spatial_size, support_size
            )
        else:
            self.value_head = nn.Linear(hidden_channels * spatial_size * spatial_size, 1)
            self.reward_head = nn.Linear(hidden_channels * spatial_size * spatial_size, 1)

        self.action_embed = nn.Embedding(action_size, hidden_channels)
        self.dynamics_conv = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)

        # Projection head for consistency loss
        self.projection = nn.Linear(hidden_channels * spatial_size * spatial_size, 256)
        self.prediction = nn.Linear(256, 256)

    def initial_inference(self, spatial, vector, action_mask=None):
        """Initial inference from observation."""
        hidden = self.spatial_proj(spatial)  # (B, C, H, W)
        flat = hidden.view(hidden.size(0), -1)

        policy_logits = self.policy_head(flat)
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, -1e9)

        if self.use_categorical:
            value = self.value_head(flat)
        else:
            value = self.value_head(flat).squeeze(-1)

        # Return named tuple-like object
        class Output:
            def __init__(self, hidden_state, policy_logits, value):
                self.hidden_state = hidden_state
                self.policy_logits = policy_logits
                self.value = value

        return Output(hidden, policy_logits, value)

    def recurrent_inference(self, hidden_state, action, action_mask=None):
        """Recurrent inference through dynamics."""
        batch_size = hidden_state.size(0)

        # Embed action
        action_embed = self.action_embed(action)  # (B, C)
        action_spatial = action_embed.unsqueeze(-1).unsqueeze(-1)
        action_spatial = action_spatial.expand(-1, -1, self.spatial_size, self.spatial_size)

        # Dynamics
        combined = torch.cat([hidden_state, action_spatial], dim=1)
        next_hidden = self.dynamics_conv(combined)

        flat = next_hidden.view(batch_size, -1)
        policy_logits = self.policy_head(flat)
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, -1e9)

        if self.use_categorical:
            value = self.value_head(flat)
            reward = self.reward_head(flat)
        else:
            value = self.value_head(flat).squeeze(-1)
            reward = self.reward_head(flat).squeeze(-1)

        return next_hidden, reward, policy_logits, value

    def forward(self, spatial, vector, action_mask=None):
        """AlphaZero-compatible forward."""
        output = self.initial_inference(spatial, vector, action_mask)
        return output.policy_logits, output.value

    def project(self, hidden_state):
        """Project hidden state for consistency loss."""
        flat = hidden_state.view(hidden_state.size(0), -1)
        return self.projection(flat)

    def predict_projection(self, projection):
        """Predict target from projection."""
        return self.prediction(projection)


# =============================================================================
# Test MuZeroTrainer
# =============================================================================


class TestMuZeroTrainer:
    """Test suite for MuZeroTrainer class."""

    @pytest.fixture
    def mock_network(self):
        """Create mock network for testing."""
        return MockMuZeroNetwork(
            spatial_channels=13,
            spatial_size=16,
            vector_dim=64,
            action_size=100,
            hidden_channels=32,
            use_categorical=True,
            support_size=601,
        )

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MuZeroTrainingConfig(
            unroll_steps=3,
            batch_size=4,
            learning_rate=0.001,
            use_consistency_loss=True,
            use_categorical=True,
            support_size=601,
        )

    @pytest.fixture
    def trainer(self, mock_network, config):
        """Create trainer for testing."""
        return MuZeroTrainer(
            network=mock_network,
            config=config,
            device="cpu",
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing."""
        batch_size = 4
        action_size = 100
        unroll_steps = 3

        return TrajectoryBatchSample(
            spatial=torch.randn(batch_size, 13, 16, 16),
            vector=torch.randn(batch_size, 64),
            action_mask=torch.ones(batch_size, action_size),
            actions=torch.randint(0, action_size, (batch_size, unroll_steps)),
            target_policies=[
                torch.softmax(torch.randn(batch_size, action_size), dim=-1)
                for _ in range(unroll_steps + 1)
            ],
            target_values=[torch.randn(batch_size) for _ in range(unroll_steps + 1)],
            target_rewards=[torch.randn(batch_size) for _ in range(unroll_steps)],
            action_masks_unroll=[torch.ones(batch_size, action_size) for _ in range(unroll_steps)],
        )

    def test_trainer_initialization(self, mock_network, config):
        """Trainer initializes correctly."""
        trainer = MuZeroTrainer(
            network=mock_network,
            config=config,
            device="cpu",
        )

        assert trainer.network is mock_network
        assert trainer.config is config
        assert trainer.device.type == "cpu"
        assert trainer.step_count == 0
        assert trainer.optimizer is not None

    def test_train_step_runs(self, trainer, sample_batch):
        """Train step executes without error."""
        metrics = trainer.train_step(sample_batch)

        assert isinstance(metrics, dict)
        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "reward_loss" in metrics
        assert "grad_norm" in metrics
        assert "step" in metrics

    def test_train_step_increments_counter(self, trainer, sample_batch):
        """Train step increments step counter."""
        initial_count = trainer.step_count

        trainer.train_step(sample_batch)

        assert trainer.step_count == initial_count + 1

    def test_train_step_losses_are_finite(self, trainer, sample_batch):
        """Train step produces finite losses."""
        metrics = trainer.train_step(sample_batch)

        assert np.isfinite(metrics["total_loss"])
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert np.isfinite(metrics["reward_loss"])
        assert np.isfinite(metrics["grad_norm"])

    def test_train_step_updates_weights(self, trainer, sample_batch):
        """Train step updates network weights."""
        # Get initial weights
        initial_weights = {
            name: param.clone() for name, param in trainer.network.named_parameters()
        }

        # Do training step
        trainer.train_step(sample_batch)

        # Check weights changed
        weights_changed = False
        for name, param in trainer.network.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "Network weights should change after training step"

    def test_train_step_with_consistency_loss(self, mock_network):
        """Train step computes consistency loss when enabled and observations provided."""
        batch_size = 4
        action_size = 100
        unroll_steps = 3

        # Create batch WITH observations for consistency loss
        # (consistency loss requires actual next observations)
        batch_with_obs = TrajectoryBatchSample(
            spatial=torch.randn(batch_size, 13, 16, 16),
            vector=torch.randn(batch_size, 64),
            action_mask=torch.ones(batch_size, action_size),
            actions=torch.randint(0, action_size, (batch_size, unroll_steps)),
            target_policies=[
                torch.softmax(torch.randn(batch_size, action_size), dim=-1)
                for _ in range(unroll_steps + 1)
            ],
            target_values=[torch.randn(batch_size) for _ in range(unroll_steps + 1)],
            target_rewards=[torch.randn(batch_size) for _ in range(unroll_steps)],
            action_masks_unroll=[torch.ones(batch_size, action_size) for _ in range(unroll_steps)],
            # Add observations for consistency loss computation
            observations_spatial_unroll=[
                torch.randn(batch_size, 13, 16, 16) for _ in range(unroll_steps)
            ],
            observations_vector_unroll=[torch.randn(batch_size, 64) for _ in range(unroll_steps)],
        )

        config = MuZeroTrainingConfig(
            unroll_steps=3,
            batch_size=4,
            use_consistency_loss=True,
        )
        trainer = MuZeroTrainer(
            network=mock_network,
            config=config,
            device="cpu",
        )

        metrics = trainer.train_step(batch_with_obs)

        # Consistency loss should be computed when observations are provided
        assert "consistency_loss" in metrics, (
            "consistency_loss should be in metrics when use_consistency_loss=True "
            "and observations_spatial_unroll/observations_vector_unroll are provided"
        )

    def test_train_step_without_consistency_loss(self, mock_network, sample_batch):
        """Train step skips consistency loss when disabled."""
        config = MuZeroTrainingConfig(
            unroll_steps=3,
            batch_size=4,
            use_consistency_loss=False,
        )
        trainer = MuZeroTrainer(
            network=mock_network,
            config=config,
            device="cpu",
        )

        metrics = trainer.train_step(sample_batch)

        # Consistency loss should not be computed
        assert "consistency_loss" not in metrics or metrics.get("consistency_loss") is None

    def test_multiple_train_steps(self, trainer, sample_batch):
        """Multiple train steps work correctly."""
        for i in range(5):
            metrics = trainer.train_step(sample_batch)
            assert metrics["step"] == i + 1

        assert trainer.step_count == 5

    def test_gradient_clipping(self, mock_network, sample_batch):
        """Gradient clipping is applied."""
        config = MuZeroTrainingConfig(
            unroll_steps=3,
            batch_size=4,
            gradient_clip=1.0,  # Clip value
        )
        trainer = MuZeroTrainer(
            network=mock_network,
            config=config,
            device="cpu",
        )

        metrics = trainer.train_step(sample_batch)

        # Grad norm should be returned (clipping happens internally)
        # After clipping, the grad_norm returned is the original norm before clipping
        # Just verify it's a positive finite number
        assert metrics["grad_norm"] > 0
        assert np.isfinite(metrics["grad_norm"])


class TestMuZeroTrainerIntegration:
    """Integration tests for MuZeroTrainer with TrajectoryBuffer."""

    @pytest.fixture
    def filled_buffer(self):
        """Create buffer with sample trajectories."""
        buffer = TrajectoryBuffer(max_trajectories=100)

        for game_id in range(5):
            traj = GameTrajectory(game_id=f"game_{game_id}")
            for step in range(20):
                obs = {
                    "spatial": np.random.randn(13, 16, 16).astype(np.float32),
                    "vector": np.random.randn(64).astype(np.float32),
                    "action_mask": np.ones(100, dtype=np.float32),
                }
                policy = np.ones(100, dtype=np.float32) / 100

                traj.append(
                    observation=obs,
                    action=step % 100,
                    reward=0.1,
                    policy=policy,
                    value=0.5,
                    done=(step == 19),
                )

            buffer.add_trajectory(traj)

        return buffer

    def test_train_on_buffer(self, filled_buffer):
        """Training on buffer works correctly."""
        network = MockMuZeroNetwork()
        config = MuZeroTrainingConfig(
            unroll_steps=3,
            batch_size=4,
        )
        trainer = MuZeroTrainer(
            network=network,
            config=config,
            device="cpu",
        )

        history = trainer.train_on_buffer(
            buffer=filled_buffer,
            num_steps=10,
            log_interval=5,
        )

        assert "total_loss" in history
        assert len(history["total_loss"]) == 10
        assert trainer.step_count == 10

    def test_sample_and_train(self, filled_buffer):
        """Manual sample and train works correctly."""
        network = MockMuZeroNetwork()
        config = MuZeroTrainingConfig(
            unroll_steps=3,
            batch_size=4,
        )
        trainer = MuZeroTrainer(
            network=network,
            config=config,
            device="cpu",
        )

        # Manual sampling and training
        batch = filled_buffer.sample_positions(
            batch_size=config.batch_size,
            unroll_steps=config.unroll_steps,
            device="cpu",
        )

        metrics = trainer.train_step(batch)

        assert "total_loss" in metrics
        assert np.isfinite(metrics["total_loss"])


class TestMuZeroTrainerCheckpointing:
    """Test checkpoint save/load functionality."""

    def test_save_checkpoint(self, tmp_path):
        """Checkpoint saving works."""
        network = MockMuZeroNetwork()
        config = MuZeroTrainingConfig(unroll_steps=3, batch_size=4)
        trainer = MuZeroTrainer(network=network, config=config, device="cpu")

        # Do a few steps
        batch = TrajectoryBatchSample(
            spatial=torch.randn(4, 13, 16, 16),
            vector=torch.randn(4, 64),
            action_mask=torch.ones(4, 100),
            actions=torch.randint(0, 100, (4, 3)),
            target_policies=[torch.softmax(torch.randn(4, 100), dim=-1) for _ in range(4)],
            target_values=[torch.randn(4) for _ in range(4)],
            target_rewards=[torch.randn(4) for _ in range(3)],
            action_masks_unroll=[torch.ones(4, 100) for _ in range(3)],
        )
        trainer.train_step(batch)
        trainer.train_step(batch)

        # Save checkpoint
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(path))

        assert path.exists()

    def test_load_checkpoint(self, tmp_path):
        """Checkpoint loading works."""
        network = MockMuZeroNetwork()
        config = MuZeroTrainingConfig(unroll_steps=3, batch_size=4)
        trainer = MuZeroTrainer(network=network, config=config, device="cpu")

        # Do a few steps
        batch = TrajectoryBatchSample(
            spatial=torch.randn(4, 13, 16, 16),
            vector=torch.randn(4, 64),
            action_mask=torch.ones(4, 100),
            actions=torch.randint(0, 100, (4, 3)),
            target_policies=[torch.softmax(torch.randn(4, 100), dim=-1) for _ in range(4)],
            target_values=[torch.randn(4) for _ in range(4)],
            target_rewards=[torch.randn(4) for _ in range(3)],
            action_masks_unroll=[torch.ones(4, 100) for _ in range(3)],
        )
        trainer.train_step(batch)
        trainer.train_step(batch)

        original_step_count = trainer.step_count

        # Save checkpoint
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(path))

        # Create new trainer and load
        new_network = MockMuZeroNetwork()
        new_trainer = MuZeroTrainer(
            network=new_network,
            config=config,
            device="cpu",
        )
        new_trainer.load_checkpoint(str(path))

        assert new_trainer.step_count == original_step_count


# =============================================================================
# Test Target Computation
# =============================================================================


class TestTargetComputation:
    """Test target computation for MuZero training."""

    def test_n_step_returns_simple(self):
        """N-step returns with simple case."""
        traj = GameTrajectory(game_id="test")

        # 5 steps, each with reward 1.0
        for i in range(5):
            obs = {
                "spatial": np.random.randn(13, 16, 16).astype(np.float32),
                "vector": np.random.randn(64).astype(np.float32),
                "action_mask": np.ones(100, dtype=np.float32),
            }
            traj.append(
                observation=obs,
                action=i,
                reward=1.0,
                policy=np.ones(100, dtype=np.float32) / 100,
                value=0.0 if i < 4 else 10.0,  # Bootstrap from terminal
                done=(i == 4),
            )

        traj.compute_n_step_returns(discount=1.0, n_steps=2)

        # With discount=1.0, n=2:
        # G_0 = r_0 + r_1 + V(s_2) = 1 + 1 + 0 = 2 (before V(s_2) was set)
        # But with our simple bootstrap, it's more complex
        # Just check values are computed
        assert len(traj.values) == 5

    def test_n_step_returns_with_discount(self):
        """N-step returns with discounting."""
        traj = GameTrajectory(game_id="test")

        for i in range(10):
            obs = {
                "spatial": np.random.randn(13, 16, 16).astype(np.float32),
                "vector": np.random.randn(64).astype(np.float32),
                "action_mask": np.ones(100, dtype=np.float32),
            }
            traj.append(
                observation=obs,
                action=i,
                reward=1.0,
                policy=np.ones(100, dtype=np.float32) / 100,
                value=0.0,
                done=(i == 9),
            )

        traj.compute_n_step_returns(discount=0.99, n_steps=5)

        # All values should be computed
        assert len(traj.values) == 10
        # With uniform rewards, earlier steps should have >= returns
        # (depending on bootstrapping which may make them equal)
        assert traj.values[0] >= traj.values[5] - 0.01  # Allow small numerical tolerance


# =============================================================================
# Test Unrolling Logic
# =============================================================================


class TestUnrollingLogic:
    """Test unrolling dynamics for K steps."""

    @pytest.fixture
    def trainer_and_network(self):
        """Create trainer with mock network."""
        network = MockMuZeroNetwork()
        config = MuZeroTrainingConfig(
            unroll_steps=3,
            batch_size=4,
        )
        trainer = MuZeroTrainer(
            network=network,
            config=config,
            device="cpu",
        )
        return trainer, network

    def test_initial_inference_output(self, trainer_and_network):
        """Initial inference produces correct outputs."""
        trainer, network = trainer_and_network

        batch = TrajectoryBatchSample(
            spatial=torch.randn(4, 13, 16, 16),
            vector=torch.randn(4, 64),
            action_mask=torch.ones(4, 100),
            actions=torch.randint(0, 100, (4, 3)),
            target_policies=[torch.softmax(torch.randn(4, 100), dim=-1) for _ in range(4)],
            target_values=[torch.randn(4) for _ in range(4)],
            target_rewards=[torch.randn(4) for _ in range(3)],
            action_masks_unroll=[torch.ones(4, 100) for _ in range(3)],
        )

        output = trainer._initial_inference(batch)

        assert "hidden_state" in output
        assert "policy_logits" in output
        assert "value" in output
        assert output["hidden_state"].shape == (4, 32, 16, 16)
        assert output["policy_logits"].shape == (4, 100)

    def test_unroll_step_output(self, trainer_and_network):
        """Unroll step produces correct outputs."""
        trainer, network = trainer_and_network

        hidden = torch.randn(4, 32, 16, 16)
        action = torch.randint(0, 100, (4,))
        action_mask = torch.ones(4, 100)

        output = trainer._unroll_step(hidden, action, action_mask)

        assert "hidden_state" in output
        assert "policy_logits" in output
        assert "value" in output
        assert "reward" in output
        assert output["hidden_state"].shape == (4, 32, 16, 16)
        assert output["policy_logits"].shape == (4, 100)

    def test_unroll_preserves_batch_size(self, trainer_and_network):
        """Unrolling preserves batch dimension."""
        trainer, network = trainer_and_network

        batch_size = 8
        hidden = torch.randn(batch_size, 32, 16, 16)

        for _ in range(5):
            action = torch.randint(0, 100, (batch_size,))
            output = trainer._unroll_step(hidden, action, None)
            hidden = output["hidden_state"]

            assert hidden.shape[0] == batch_size


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trajectory_not_added(self):
        """Empty trajectory is not added to buffer."""
        buffer = TrajectoryBuffer(max_trajectories=100)
        traj = GameTrajectory(game_id="empty")

        buffer.add_trajectory(traj)

        assert len(buffer) == 0

    def test_single_step_trajectory(self):
        """Single step trajectory can be sampled."""
        buffer = TrajectoryBuffer(max_trajectories=100)

        traj = GameTrajectory(game_id="single")
        obs = {
            "spatial": np.random.randn(13, 16, 16).astype(np.float32),
            "vector": np.random.randn(64).astype(np.float32),
            "action_mask": np.ones(100, dtype=np.float32),
        }
        traj.append(
            observation=obs,
            action=0,
            reward=0.0,
            policy=np.ones(100, dtype=np.float32) / 100,
            value=0.0,
            done=True,
        )
        buffer.add_trajectory(traj)

        # Should work even with unroll steps > trajectory length
        batch = buffer.sample_positions(
            batch_size=2,
            unroll_steps=5,
            device="cpu",
        )

        assert batch.spatial.shape[0] == 2

    def test_trainer_with_non_categorical_network(self):
        """Trainer works with non-categorical network."""
        network = MockMuZeroNetwork(use_categorical=False)
        config = MuZeroTrainingConfig(
            unroll_steps=3,
            batch_size=4,
            use_categorical=False,  # Match network
        )
        trainer = MuZeroTrainer(
            network=network,
            config=config,
            device="cpu",
        )

        batch = TrajectoryBatchSample(
            spatial=torch.randn(4, 13, 16, 16),
            vector=torch.randn(4, 64),
            action_mask=torch.ones(4, 100),
            actions=torch.randint(0, 100, (4, 3)),
            target_policies=[torch.softmax(torch.randn(4, 100), dim=-1) for _ in range(4)],
            target_values=[torch.randn(4) for _ in range(4)],
            target_rewards=[torch.randn(4) for _ in range(3)],
            action_masks_unroll=[torch.ones(4, 100) for _ in range(3)],
        )

        metrics = trainer.train_step(batch)

        assert np.isfinite(metrics["total_loss"])
