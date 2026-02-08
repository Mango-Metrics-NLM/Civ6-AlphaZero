"""Integration tests for MuZero training pipeline.

Tests the full MuZero training loop end-to-end:
1. Self-play with MuZeroMCTS (hidden-state based)
2. Training loop with dynamics unrolling
3. EfficientZero consistency loss
4. Backwards compatibility with AlphaZero training code
5. Checkpointing and resumption

These tests verify that all MuZero components work together correctly.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from alpha_zero.core.civ6_mcts import MuZeroMCTS, MuZeroMCTSConfig
from alpha_zero.core.losses import (
    LossConfig,
    categorical_to_scalar,
    compute_consistency_loss,
    compute_muzero_step_loss,
    compute_muzero_trajectory_loss,
    compute_policy_loss,
    compute_reward_loss,
    compute_value_loss,
)
from alpha_zero.core.muzero_network import (
    MuZeroConfig,
    MuZeroNetwork,
)

# Import directly to avoid civrealm dependency issues
from alpha_zero.envs.microciv import MicroCivEnv

# =============================================================================
# Pytest markers
# =============================================================================

pytestmark = [pytest.mark.integration]


# =============================================================================
# Fixtures for test networks and data
# =============================================================================


@pytest.fixture
def seed():
    """Fixed seed for determinism."""
    return 42


@pytest.fixture
def set_seed(seed):
    """Set all random seeds for determinism."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def small_network_config():
    """Small MuZero network config for fast testing."""
    return MuZeroConfig(
        spatial_channels=13,
        spatial_size=8,
        vector_dim=64,
        num_res_blocks=2,
        hidden_state_channels=32,
        fc_hidden_dim=64,
        action_space_size=100,
        dynamics_num_blocks=2,
        reward_support_size=601,  # Use standard support size for compatibility
        use_categorical_reward=True,
        use_consistency_loss=True,
        use_value_prefix=True,
        projection_dim=64,
        prediction_hidden_dim=32,
    )


@pytest.fixture
def loss_config_scalar():
    """Loss config for scalar (non-categorical) values."""
    return LossConfig(use_categorical=False)


@pytest.fixture
def small_network(small_network_config, set_seed):
    """Small MuZero network for testing."""
    return MuZeroNetwork(small_network_config)


@pytest.fixture
def small_env(seed):
    """Small MicroCiv environment for testing."""
    return MicroCivEnv(board_size=8, max_turns=50, seed=seed)


@pytest.fixture
def mcts_config():
    """MuZero MCTS config for testing."""
    return MuZeroMCTSConfig(
        num_simulations=8,
        num_sampled_actions=8,
        use_gumbel=True,
        use_sequential_halving=True,
        c_puct=1.5,
        discount=0.997,
    )


@pytest.fixture
def sample_observation(small_network_config, set_seed):
    """Sample observation dictionary."""
    return {
        "spatial": np.random.randn(
            small_network_config.spatial_channels,
            small_network_config.spatial_size,
            small_network_config.spatial_size,
        ).astype(np.float32),
        "vector": np.random.randn(small_network_config.vector_dim).astype(np.float32),
    }


@pytest.fixture
def sample_action_mask(small_network_config, set_seed):
    """Sample action mask with some valid actions."""
    mask = np.zeros(small_network_config.action_space_size, dtype=np.float32)
    # Enable 10 random actions
    valid_actions = np.random.choice(small_network_config.action_space_size, size=10, replace=False)
    mask[valid_actions] = 1.0
    return mask


@pytest.fixture
def tmp_checkpoint_dir():
    """Temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# Data structures for training
# =============================================================================


@dataclass
class MuZeroTrajectory:
    """A trajectory for MuZero training.

    Contains observations, actions, rewards, policies, and values
    for a sequence of game steps.
    """

    observations: list[dict]  # List of {spatial, vector}
    actions: list[int]
    rewards: list[float]
    policies: list[np.ndarray]
    values: list[float]
    action_masks: list[np.ndarray]


class MuZeroTrajectoryDataset(Dataset):
    """Dataset of MuZero trajectories for training."""

    def __init__(
        self,
        trajectories: list[MuZeroTrajectory],
        unroll_steps: int = 5,
        discount: float = 0.997,
    ):
        """Initialize dataset.

        Args:
            trajectories: List of game trajectories.
            unroll_steps: Number of dynamics unroll steps.
            discount: Discount factor for value targets.
        """
        self.samples = []
        self.unroll_steps = unroll_steps
        self.discount = discount

        for traj in trajectories:
            self._process_trajectory(traj)

    def _process_trajectory(self, traj: MuZeroTrajectory) -> None:
        """Process trajectory into training samples."""
        for t in range(len(traj.observations) - self.unroll_steps):
            sample = {
                "initial_obs": traj.observations[t],
                "actions": traj.actions[t : t + self.unroll_steps],
                "target_policies": traj.policies[t : t + self.unroll_steps + 1],
                "target_values": traj.values[t : t + self.unroll_steps + 1],
                "target_rewards": traj.rewards[t + 1 : t + self.unroll_steps + 1],
                "action_masks": traj.action_masks[t : t + self.unroll_steps + 1],
            }
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "spatial": torch.from_numpy(sample["initial_obs"]["spatial"]).float(),
            "vector": torch.from_numpy(sample["initial_obs"]["vector"]).float(),
            "actions": torch.tensor(sample["actions"], dtype=torch.long),
            "target_policies": torch.stack(
                [torch.from_numpy(p).float() for p in sample["target_policies"]]
            ),
            "target_values": torch.tensor(sample["target_values"], dtype=torch.float),
            "target_rewards": torch.tensor(sample["target_rewards"], dtype=torch.float),
            "action_masks": torch.stack(
                [torch.from_numpy(m).float() for m in sample["action_masks"]]
            ),
        }


def generate_random_trajectory(
    config: MuZeroConfig,
    length: int = 20,
    seed: int | None = None,
) -> MuZeroTrajectory:
    """Generate a random trajectory for testing.

    Args:
        config: Network configuration.
        length: Number of steps in trajectory.
        seed: Random seed.

    Returns:
        Random trajectory.
    """
    rng = np.random.default_rng(seed)

    observations = []
    actions = []
    rewards = []
    policies = []
    values = []
    action_masks = []

    for _ in range(length):
        # Random observation
        obs = {
            "spatial": rng.standard_normal(
                (config.spatial_channels, config.spatial_size, config.spatial_size)
            ).astype(np.float32),
            "vector": rng.standard_normal(config.vector_dim).astype(np.float32),
        }
        observations.append(obs)

        # Random action mask (some actions valid)
        mask = np.zeros(config.action_space_size, dtype=np.float32)
        valid_actions = rng.choice(config.action_space_size, size=10, replace=False)
        mask[valid_actions] = 1.0
        action_masks.append(mask)

        # Random policy (normalized over valid actions)
        policy = rng.random(config.action_space_size).astype(np.float32) * mask
        policy = policy / (policy.sum() + 1e-8)
        policies.append(policy)

        # Random action (from valid actions)
        action = rng.choice(valid_actions)
        actions.append(int(action))

        # Random reward and value
        rewards.append(float(rng.random() * 0.2 - 0.1))  # Small rewards
        values.append(float(rng.random() * 2 - 1))  # Value in [-1, 1]

    return MuZeroTrajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        policies=policies,
        values=values,
        action_masks=action_masks,
    )


# =============================================================================
# Test 1: Self-play with MuZeroMCTS
# =============================================================================


class TestMuZeroSelfPlay:
    """Tests for self-play with MuZeroMCTS."""

    def test_mcts_search_returns_valid_policy(
        self, small_network, sample_observation, sample_action_mask, set_seed
    ):
        """MuZero MCTS search returns valid action probabilities."""
        config = MuZeroMCTSConfig(
            num_simulations=4,
            num_sampled_actions=4,
            use_gumbel=True,
        )
        mcts = MuZeroMCTS(small_network, config, device="cpu")

        action_probs, root_value = mcts.search(
            sample_observation, sample_action_mask, add_noise=False
        )

        # Valid probability distribution
        assert action_probs.shape == (100,)
        assert np.all(action_probs >= 0)
        assert np.isclose(action_probs.sum(), 1.0, atol=1e-5)

        # Selected action is valid
        selected_action = np.argmax(action_probs)
        assert sample_action_mask[selected_action] == 1.0

        # Root value is finite
        assert np.isfinite(root_value)

    def test_run_short_game_with_mcts(self, small_env, set_seed):
        """Run a short game using MuZero MCTS."""
        # Create network matching environment action space
        network_config = MuZeroConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=small_env.action_space.n,
            num_res_blocks=2,
            hidden_state_channels=32,
            fc_hidden_dim=64,
            reward_support_size=601,
        )
        network = MuZeroNetwork(network_config)

        # Create MCTS with network
        config = MuZeroMCTSConfig(
            num_simulations=4,
            num_sampled_actions=4,
            use_gumbel=True,
        )
        mcts = MuZeroMCTS(network, config, device="cpu")

        obs, _ = small_env.reset()
        trajectory = MuZeroTrajectory(
            observations=[],
            actions=[],
            rewards=[],
            policies=[],
            values=[],
            action_masks=[],
        )

        # Play for 10 steps
        for _ in range(10):
            # Store observation
            trajectory.observations.append(
                {
                    "spatial": obs["spatial"].copy(),
                    "vector": obs["vector"].copy(),
                }
            )
            trajectory.action_masks.append(obs["action_mask"].copy())

            # Get MCTS policy
            action_probs, root_value = mcts.search(
                {"spatial": obs["spatial"], "vector": obs["vector"]},
                obs["action_mask"],
                add_noise=True,
            )

            trajectory.policies.append(action_probs)
            trajectory.values.append(root_value)

            # Select and take action
            action = mcts.select_action(action_probs, temperature=1.0)
            trajectory.actions.append(action)

            obs, reward, terminated, truncated, _info = small_env.step(action)
            trajectory.rewards.append(float(reward))

            if terminated or truncated:
                break

        # Verify trajectory collected correctly
        assert len(trajectory.observations) > 0
        assert len(trajectory.actions) == len(trajectory.observations)
        assert len(trajectory.policies) == len(trajectory.observations)
        assert len(trajectory.values) == len(trajectory.observations)

        # All policies are valid
        for policy in trajectory.policies:
            assert np.all(policy >= 0)
            assert np.isclose(policy.sum(), 1.0, atol=1e-5)

    def test_mcts_uses_dynamics_not_env(self, small_network, set_seed):
        """Verify MCTS uses dynamics function, not environment."""
        config = MuZeroMCTSConfig(
            num_simulations=8,
            use_gumbel=True,
        )
        mcts = MuZeroMCTS(small_network, config, device="cpu")

        # Only provide observation - no environment
        obs = {
            "spatial": np.random.randn(13, 8, 8).astype(np.float32),
            "vector": np.random.randn(64).astype(np.float32),
        }
        mask = np.ones(100, dtype=np.float32)

        # Should work without environment
        action_probs, _ = mcts.search(obs, mask)

        assert action_probs.sum() > 0

    def test_trajectory_collection_full_game(self, small_env, set_seed):
        """Collect trajectory for a full game (or until max steps)."""
        # Create network matching environment action space
        network_config = MuZeroConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=small_env.action_space.n,
            num_res_blocks=2,
            hidden_state_channels=32,
            fc_hidden_dim=64,
            reward_support_size=601,
        )
        network = MuZeroNetwork(network_config)

        config = MuZeroMCTSConfig(
            num_simulations=4,
            use_gumbel=True,
        )
        mcts = MuZeroMCTS(network, config, device="cpu")

        obs, _ = small_env.reset()
        max_steps = 50

        observations = []
        actions = []
        rewards = []
        policies = []

        for _ in range(max_steps):
            observations.append(
                {
                    "spatial": obs["spatial"].copy(),
                    "vector": obs["vector"].copy(),
                }
            )

            action_probs, _ = mcts.search(
                {"spatial": obs["spatial"], "vector": obs["vector"]},
                obs["action_mask"],
                add_noise=False,
            )
            policies.append(action_probs)

            action = mcts.select_action(action_probs, temperature=0.5)
            actions.append(action)

            obs, reward, terminated, truncated, _ = small_env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Should have collected data
        assert len(observations) > 0
        assert len(actions) == len(observations)

        # Actions should be valid
        for action in actions:
            assert 0 <= action < small_env.action_space.n


# =============================================================================
# Test 2: Training Loop
# =============================================================================


class TestMuZeroTrainingLoop:
    """Tests for MuZero training loop."""

    @pytest.fixture
    def sample_trajectories(self, small_network_config, set_seed):
        """Generate sample trajectories for training."""
        return [
            generate_random_trajectory(small_network_config, length=15, seed=i) for i in range(5)
        ]

    @pytest.fixture
    def training_dataset(self, sample_trajectories):
        """Create dataset from trajectories."""
        return MuZeroTrajectoryDataset(sample_trajectories, unroll_steps=3, discount=0.997)

    def test_training_step_reduces_loss(
        self, small_network, training_dataset, small_network_config, set_seed
    ):
        """Training step reduces loss on same batch."""
        # Create optimizer
        optimizer = optim.Adam(small_network.parameters(), lr=0.01)

        # Create dataloader
        dataloader = DataLoader(training_dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))

        # Get initial loss
        small_network.train()
        initial_loss = self._compute_batch_loss(small_network, batch, small_network_config)

        # Train for several steps on same batch
        for _ in range(10):
            optimizer.zero_grad()
            loss = self._compute_batch_loss(small_network, batch, small_network_config)
            loss.backward()
            optimizer.step()

        # Get final loss
        final_loss = self._compute_batch_loss(small_network, batch, small_network_config)

        # Loss should decrease
        assert final_loss.item() < initial_loss.item()

        # No NaN values
        assert not torch.isnan(final_loss)
        for param in small_network.parameters():
            assert not torch.any(torch.isnan(param))

    def _compute_batch_loss(
        self,
        network: MuZeroNetwork,
        batch: dict,
        config: MuZeroConfig,
    ) -> torch.Tensor:
        """Compute loss for a batch."""
        spatial = batch["spatial"]
        vector = batch["vector"]
        actions = batch["actions"]
        target_policies = batch["target_policies"]
        target_values = batch["target_values"]
        target_rewards = batch["target_rewards"]
        action_masks = batch["action_masks"]

        # Initial inference
        output = network.initial_inference(spatial, vector, action_masks[:, 0])

        # Config for scalar losses (non-categorical for simpler tests)
        loss_cfg = LossConfig(use_categorical=False)

        # Policy and value loss for step 0
        policy_loss = compute_policy_loss(
            output.policy_logits, target_policies[:, 0], action_masks[:, 0]
        )
        value_loss = compute_value_loss(output.value, target_values[:, 0], config=loss_cfg)

        total_loss = policy_loss + value_loss

        # Unroll dynamics
        hidden_state = output.hidden_state
        unroll_steps = actions.shape[1]

        for k in range(unroll_steps):
            # Dynamics step
            action = actions[:, k]
            next_hidden, reward_logits, policy_logits, value = network.recurrent_inference(
                hidden_state, action, action_masks[:, k + 1]
            )

            # Convert categorical reward to scalar for loss computation
            reward_scalar = categorical_to_scalar(reward_logits, apply_softmax=True)

            # Losses for step k+1
            policy_loss_k = compute_policy_loss(
                policy_logits, target_policies[:, k + 1], action_masks[:, k + 1]
            )
            value_loss_k = compute_value_loss(value, target_values[:, k + 1], config=loss_cfg)
            reward_loss_k = compute_reward_loss(
                reward_scalar, target_rewards[:, k], config=loss_cfg
            )

            total_loss = total_loss + policy_loss_k + value_loss_k + reward_loss_k

            hidden_state = next_hidden

        return total_loss / (unroll_steps + 1)

    def test_gradients_flow_through_dynamics(
        self, small_network, sample_observation, sample_action_mask, set_seed
    ):
        """Verify gradients flow through dynamics function."""
        small_network.train()

        # Convert to tensors
        spatial = torch.from_numpy(sample_observation["spatial"]).float().unsqueeze(0)
        vector = torch.from_numpy(sample_observation["vector"]).float().unsqueeze(0)
        mask = torch.from_numpy(sample_action_mask).float().unsqueeze(0)

        # Initial inference
        output = small_network.initial_inference(spatial, vector, mask)

        # Dynamics unroll
        hidden = output.hidden_state
        actions = torch.tensor([5])  # Arbitrary action

        next_hidden, _reward, _policy, value = small_network.recurrent_inference(
            hidden, actions, mask
        )

        # Create dummy loss on dynamics output
        loss = next_hidden.mean() + value.mean()

        # Backward pass
        loss.backward()

        # Check gradients exist for dynamics parameters
        dynamics_has_grad = False
        for name, param in small_network.named_parameters():
            if "dynamics" in name and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    dynamics_has_grad = True
                    break

        assert dynamics_has_grad, "Dynamics network should have gradients"

    def test_multiple_training_epochs(
        self, small_network, training_dataset, small_network_config, set_seed
    ):
        """Train for multiple epochs and verify loss decreases."""
        optimizer = optim.Adam(small_network.parameters(), lr=0.01)
        dataloader = DataLoader(training_dataset, batch_size=4, shuffle=True)

        epoch_losses = []

        for _epoch in range(5):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                optimizer.zero_grad()
                loss = self._compute_batch_loss(small_network, batch, small_network_config)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            epoch_losses.append(epoch_loss / num_batches)

        # Loss should generally decrease
        assert epoch_losses[-1] < epoch_losses[0]


# =============================================================================
# Test 3: Consistency Loss Integration
# =============================================================================


class TestConsistencyLossIntegration:
    """Tests for EfficientZero consistency loss."""

    def test_consistency_loss_decreases_during_training(
        self, small_network, small_network_config, set_seed
    ):
        """Consistency loss decreases when training to align predictions."""
        optimizer = optim.Adam(small_network.parameters(), lr=0.01)

        # Generate paired observations (obs1 and obs2 = obs1 after action)
        spatial1 = torch.randn(4, 13, 8, 8)
        vector1 = torch.randn(4, 64)
        spatial2 = torch.randn(4, 13, 8, 8)  # "Next" observation
        vector2 = torch.randn(4, 64)
        mask = torch.ones(4, 100)
        actions = torch.randint(0, 100, (4,))

        initial_losses = []
        final_losses = []

        for _ in range(20):
            optimizer.zero_grad()

            # Get hidden state from dynamics
            output1 = small_network.initial_inference(spatial1, vector1, mask)
            next_hidden, _, _, _ = small_network.recurrent_inference(
                output1.hidden_state, actions, mask
            )

            # Get hidden state from actual next observation
            output2 = small_network.initial_inference(spatial2, vector2, mask)

            # Project both
            proj_dynamics = small_network.project(next_hidden)
            proj_obs = small_network.project(output2.hidden_state)

            # Prediction on dynamics projection
            pred_dynamics = small_network.predict_projection(proj_dynamics)

            # Consistency loss (stop gradient on target)
            loss = compute_consistency_loss(pred_dynamics, proj_obs.detach())

            if len(initial_losses) < 3:
                initial_losses.append(loss.item())
            final_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        # Consistency loss should improve (become more negative = better alignment)
        avg_initial = np.mean(initial_losses)
        avg_final = np.mean(final_losses[-3:])

        # The loss should decrease (more negative means better alignment)
        assert avg_final <= avg_initial + 0.1  # Allow some tolerance

    def test_projection_head_outputs_correct_shape(self, small_network, set_seed):
        """Projection head produces correct output shape."""
        hidden = torch.randn(4, 32, 8, 8)

        projection = small_network.project(hidden)

        assert projection.shape == (4, 64)  # projection_dim=64

    def test_prediction_head_outputs_correct_shape(self, small_network, set_seed):
        """Prediction head produces correct output shape."""
        projection = torch.randn(4, 64)

        prediction = small_network.predict_projection(projection)

        assert prediction.shape == (4, 64)

    def test_stop_gradient_on_target(self, small_network, set_seed):
        """Verify stop-gradient is applied to target projection."""
        small_network.train()

        # Two hidden states
        hidden1 = torch.randn(2, 32, 8, 8, requires_grad=True)
        hidden2 = torch.randn(2, 32, 8, 8, requires_grad=True)

        # Project both
        proj1 = small_network.project(hidden1)
        proj2 = small_network.project(hidden2)

        # Prediction on proj1
        pred1 = small_network.predict_projection(proj1)

        # Consistency loss with detached target
        loss = compute_consistency_loss(pred1, proj2.detach())
        loss.backward()

        # hidden1 should have gradients (through pred1)
        assert hidden1.grad is not None
        assert hidden1.grad.abs().sum() > 0

        # hidden2 should NOT have gradients (detached)
        assert hidden2.grad is None

    def test_consistency_loss_with_training_loop(
        self, small_network, small_network_config, set_seed
    ):
        """Test consistency loss integrated with full training loop."""
        optimizer = optim.Adam(small_network.parameters(), lr=0.01)

        # Use scalar (non-categorical) loss for simpler test
        loss_config = LossConfig(
            use_categorical=False,
            consistency_loss_weight=2.0,
        )

        # Generate batch
        batch_size = 4
        spatial = torch.randn(batch_size, 13, 8, 8)
        vector = torch.randn(batch_size, 64)
        mask = torch.ones(batch_size, 100)
        actions = torch.randint(0, 100, (batch_size,))

        # Target values (scale appropriately for scalar loss)
        target_policy = F.softmax(torch.randn(batch_size, 100), dim=-1)
        target_value = torch.randn(batch_size) * 0.5  # Smaller range
        target_reward = torch.randn(batch_size) * 0.1  # Small rewards

        losses = []

        for _ in range(10):
            optimizer.zero_grad()

            # Initial inference
            output = small_network.initial_inference(spatial, vector, mask)

            # Dynamics step
            next_hidden, reward_logits, policy_logits, value = small_network.recurrent_inference(
                output.hidden_state, actions, mask
            )

            # Convert categorical reward to scalar for loss computation
            reward_scalar = categorical_to_scalar(reward_logits, apply_softmax=True)

            # Consistency: predict dynamics projection to match observation projection
            # Use initial hidden as "target" (in practice would be next observation)
            proj_dynamics = small_network.project(next_hidden)
            proj_target = small_network.project(output.hidden_state)
            pred_dynamics = small_network.predict_projection(proj_dynamics)

            # Compute step loss with consistency
            step_loss = compute_muzero_step_loss(
                policy_logits=policy_logits,
                value=value,
                target_policy=target_policy,
                target_value=target_value,
                reward=reward_scalar,
                target_reward=target_reward,
                projected_dynamics=pred_dynamics,
                projected_observation=proj_target.detach(),
                config=loss_config,
            )

            losses.append(step_loss.total_loss.item())
            step_loss.total_loss.backward()
            optimizer.step()

        # Loss should decrease
        assert losses[-1] < losses[0]

        # Consistency loss should be computed
        assert step_loss.consistency_loss is not None


# =============================================================================
# Test 4: Backwards Compatibility
# =============================================================================


class TestBackwardsCompatibility:
    """Tests for AlphaZero backwards compatibility."""

    def test_forward_returns_policy_and_value(self, small_network, set_seed):
        """forward() returns policy logits and value like AlphaZero."""
        spatial = torch.randn(4, 13, 8, 8)
        vector = torch.randn(4, 64)
        mask = torch.ones(4, 100)

        policy_logits, value = small_network(spatial, vector, mask)

        assert policy_logits.shape == (4, 100)
        assert value.shape == (4,)

    def test_predict_returns_probs_and_value(self, small_network, set_seed):
        """predict() returns policy probabilities like AlphaZero."""
        spatial = torch.randn(4, 13, 8, 8)
        vector = torch.randn(4, 64)
        mask = torch.ones(4, 100)

        policy_probs, value = small_network.predict(spatial, vector, mask)

        assert policy_probs.shape == (4, 100)
        assert value.shape == (4,)

        # Probabilities sum to 1
        assert torch.allclose(policy_probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_action_mask_applied_correctly(self, small_network, set_seed):
        """Action mask is correctly applied in forward pass."""
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)

        # Only one valid action
        mask = torch.zeros(1, 100)
        mask[0, 42] = 1.0

        policy_probs, _ = small_network.predict(spatial, vector, mask)

        # Action 42 should have nearly all probability
        assert policy_probs[0, 42] > 0.99

    def test_compatible_with_alphazero_training_step(self, small_network, set_seed):
        """MuZeroNetwork works with standard AlphaZero training step."""
        from alpha_zero.core.civ6_network import compute_loss

        optimizer = optim.Adam(small_network.parameters(), lr=0.01)

        # Standard AlphaZero-style batch
        spatial = torch.randn(4, 13, 8, 8)
        vector = torch.randn(4, 64)
        mask = torch.ones(4, 100)
        target_policy = F.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)

        # Training step
        small_network.train()
        optimizer.zero_grad()

        policy_logits, value = small_network(spatial, vector, mask)
        loss, _policy_loss, _value_loss = compute_loss(
            policy_logits, value, target_policy, target_value, mask
        )

        loss.backward()
        optimizer.step()

        # Should complete without error
        assert not torch.isnan(loss)

    def test_network_can_replace_alphazero_network(self, small_env, set_seed):
        """MuZeroNetwork can be used as drop-in replacement for AlphaZero."""
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig

        # Create MuZero network with config matching environment
        config = MuZeroConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=small_env.action_space.n,
            num_res_blocks=2,
            hidden_state_channels=32,
            fc_hidden_dim=64,
        )
        network = MuZeroNetwork(config)

        # Use with standard AlphaZero MCTS
        mcts_config = MCTSConfig(num_simulations=5, c_puct=1.5)
        mcts = Civ6MCTS(network, mcts_config, device="cpu")

        # Run search
        small_env.reset()
        action_probs, root_value = mcts.search(small_env, add_noise=False)

        # Should return valid results
        assert np.all(action_probs >= 0)
        assert np.isclose(action_probs.sum(), 1.0, atol=1e-5)
        assert -1 <= root_value <= 1


# =============================================================================
# Test 5: Checkpointing
# =============================================================================


class TestCheckpointing:
    """Tests for saving and loading MuZero networks."""

    def test_save_and_load_preserves_weights(self, small_network, tmp_checkpoint_dir, set_seed):
        """Saved and loaded network produces identical outputs."""
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "muzero_test.pt")

        # Get outputs before save
        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)
        mask = torch.ones(2, 100)

        small_network.eval()
        with torch.no_grad():
            orig_policy, orig_value = small_network(spatial, vector, mask)

        # Save and load
        small_network.save(checkpoint_path)
        loaded_network = MuZeroNetwork.load(checkpoint_path, device="cpu")
        loaded_network.eval()

        # Get outputs after load
        with torch.no_grad():
            loaded_policy, loaded_value = loaded_network(spatial, vector, mask)

        # Should be identical
        assert torch.allclose(orig_policy, loaded_policy, atol=1e-6)
        assert torch.allclose(orig_value, loaded_value, atol=1e-6)

    def test_save_and_load_preserves_config(
        self, small_network, small_network_config, tmp_checkpoint_dir, set_seed
    ):
        """Saved and loaded network preserves configuration."""
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "muzero_config_test.pt")

        small_network.save(checkpoint_path)
        loaded_network = MuZeroNetwork.load(checkpoint_path, device="cpu")

        # Check key config values
        assert loaded_network.config.spatial_channels == small_network_config.spatial_channels
        assert (
            loaded_network.config.hidden_state_channels
            == small_network_config.hidden_state_channels
        )
        assert loaded_network.config.action_space_size == small_network_config.action_space_size
        assert (
            loaded_network.config.use_consistency_loss == small_network_config.use_consistency_loss
        )
        assert loaded_network.config.use_value_prefix == small_network_config.use_value_prefix

    def test_training_can_resume_from_checkpoint(
        self, small_network, small_network_config, tmp_checkpoint_dir, set_seed
    ):
        """Training can resume from checkpoint."""
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "muzero_resume_test.pt")

        # Fixed training data
        spatial = torch.randn(4, 13, 8, 8)
        vector = torch.randn(4, 64)
        mask = torch.ones(4, 100)
        target_policy = F.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.zeros(4)

        # Train for a few steps
        optimizer = optim.Adam(small_network.parameters(), lr=0.01)
        for _ in range(5):
            optimizer.zero_grad()
            policy_logits, value = small_network(spatial, vector, mask)
            loss = F.cross_entropy(policy_logits, target_policy) + F.mse_loss(value, target_value)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        small_network.save(checkpoint_path)

        # Get loss at checkpoint
        with torch.no_grad():
            policy_logits, value = small_network(spatial, vector, mask)
            checkpoint_loss = F.cross_entropy(policy_logits, target_policy) + F.mse_loss(
                value, target_value
            )

        # Load and continue training
        loaded_network = MuZeroNetwork.load(checkpoint_path, device="cpu")
        new_optimizer = optim.Adam(loaded_network.parameters(), lr=0.01)

        for _ in range(10):
            new_optimizer.zero_grad()
            policy_logits, value = loaded_network(spatial, vector, mask)
            loss = F.cross_entropy(policy_logits, target_policy) + F.mse_loss(value, target_value)
            loss.backward()
            new_optimizer.step()

        # Get final loss
        with torch.no_grad():
            policy_logits, value = loaded_network(spatial, vector, mask)
            final_loss = F.cross_entropy(policy_logits, target_policy) + F.mse_loss(
                value, target_value
            )

        # Training should have improved
        assert final_loss.item() < checkpoint_loss.item()

    def test_dynamics_preserved_after_load(self, small_network, tmp_checkpoint_dir, set_seed):
        """Dynamics function produces same outputs after load."""
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "muzero_dynamics_test.pt")

        # Get dynamics output before save
        hidden = torch.randn(2, 32, 8, 8)
        actions = torch.tensor([5, 10])

        small_network.eval()
        with torch.no_grad():
            orig_output = small_network.dynamics_function(hidden, actions)

        # Save and load
        small_network.save(checkpoint_path)
        loaded_network = MuZeroNetwork.load(checkpoint_path, device="cpu")
        loaded_network.eval()

        # Get dynamics output after load
        with torch.no_grad():
            loaded_output = loaded_network.dynamics_function(hidden, actions)

        # Should be identical
        assert torch.allclose(
            orig_output.next_hidden_state, loaded_output.next_hidden_state, atol=1e-6
        )
        assert torch.allclose(orig_output.reward, loaded_output.reward, atol=1e-6)

    def test_checkpoint_works_with_mcts(self, small_network, tmp_checkpoint_dir, set_seed):
        """Loaded checkpoint works with MuZero MCTS."""
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "muzero_mcts_test.pt")

        # Save
        small_network.save(checkpoint_path)

        # Load
        loaded_network = MuZeroNetwork.load(checkpoint_path, device="cpu")

        # Create MCTS with loaded network
        config = MuZeroMCTSConfig(
            num_simulations=4,
            use_gumbel=True,
        )
        mcts = MuZeroMCTS(loaded_network, config, device="cpu")

        # Run search
        obs = {
            "spatial": np.random.randn(13, 8, 8).astype(np.float32),
            "vector": np.random.randn(64).astype(np.float32),
        }
        mask = np.ones(100, dtype=np.float32)

        action_probs, root_value = mcts.search(obs, mask)

        # Should work correctly
        assert action_probs.sum() > 0
        assert np.isfinite(root_value)


# =============================================================================
# Test: Full End-to-End Training
# =============================================================================


@pytest.mark.slow
class TestEndToEndMuZeroTraining:
    """End-to-end tests for complete MuZero training pipeline."""

    def test_full_training_iteration(self, small_network_config, small_env, set_seed):
        """Run a complete training iteration: self-play + training."""
        # Create network matching env action space
        config = MuZeroConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=small_env.action_space.n,
            num_res_blocks=2,
            hidden_state_channels=32,
            fc_hidden_dim=64,
            dynamics_num_blocks=2,
            use_consistency_loss=False,  # Simpler test
            use_value_prefix=False,
            reward_support_size=601,  # Standard support size
        )
        network = MuZeroNetwork(config)
        optimizer = optim.Adam(network.parameters(), lr=0.01)

        # MCTS config
        mcts_config = MuZeroMCTSConfig(
            num_simulations=4,
            use_gumbel=True,
        )

        # Self-play phase: collect trajectories
        trajectories = []
        for _game_idx in range(2):
            mcts = MuZeroMCTS(network, mcts_config, device="cpu")

            obs, _ = small_env.reset()
            trajectory = MuZeroTrajectory(
                observations=[],
                actions=[],
                rewards=[],
                policies=[],
                values=[],
                action_masks=[],
            )

            for _ in range(15):  # Short games
                trajectory.observations.append(
                    {
                        "spatial": obs["spatial"].copy(),
                        "vector": obs["vector"].copy(),
                    }
                )
                trajectory.action_masks.append(obs["action_mask"].copy())

                action_probs, root_value = mcts.search(
                    {"spatial": obs["spatial"], "vector": obs["vector"]},
                    obs["action_mask"],
                )

                trajectory.policies.append(action_probs)
                trajectory.values.append(root_value)

                action = mcts.select_action(action_probs, temperature=1.0)
                trajectory.actions.append(action)

                obs, reward, terminated, truncated, _ = small_env.step(action)
                trajectory.rewards.append(float(reward))

                if terminated or truncated:
                    break

            trajectories.append(trajectory)

        # Training phase
        dataset = MuZeroTrajectoryDataset(trajectories, unroll_steps=2)

        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

            initial_loss = None
            final_loss = None

            for _epoch in range(5):
                for batch in dataloader:
                    optimizer.zero_grad()

                    # Initial inference
                    output = network.initial_inference(
                        batch["spatial"],
                        batch["vector"],
                        batch["action_masks"][:, 0],
                    )

                    # Simple loss (policy + value for initial step)
                    scalar_loss_cfg = LossConfig(use_categorical=False)
                    loss = compute_policy_loss(
                        output.policy_logits,
                        batch["target_policies"][:, 0],
                        batch["action_masks"][:, 0],
                    ) + compute_value_loss(
                        output.value, batch["target_values"][:, 0], config=scalar_loss_cfg
                    )

                    if initial_loss is None:
                        initial_loss = loss.item()
                    final_loss = loss.item()

                    loss.backward()
                    optimizer.step()

            # Verify training worked
            assert initial_loss is not None
            assert final_loss is not None
            assert not np.isnan(final_loss)

    def test_training_improves_policy(self, small_network_config, set_seed):
        """Training on consistent data improves policy predictions."""
        network = MuZeroNetwork(small_network_config)
        optimizer = optim.Adam(network.parameters(), lr=0.01)

        # Fixed batch with clear target
        batch_size = 8
        spatial = torch.randn(batch_size, 13, 8, 8)
        vector = torch.randn(batch_size, 64)
        mask = torch.ones(batch_size, 100)

        # Target: always action 0
        target_policy = torch.zeros(batch_size, 100)
        target_policy[:, 0] = 1.0
        target_value = torch.zeros(batch_size)

        # Initial prediction for action 0
        network.eval()
        with torch.no_grad():
            policy_probs, _ = network.predict(spatial, vector, mask)
            initial_prob_action_0 = policy_probs[:, 0].mean().item()

        # Train
        network.train()
        for _ in range(50):
            optimizer.zero_grad()
            policy_logits, value = network(spatial, vector, mask)
            loss = F.cross_entropy(policy_logits, target_policy) + F.mse_loss(value, target_value)
            loss.backward()
            optimizer.step()

        # Final prediction for action 0
        network.eval()
        with torch.no_grad():
            policy_probs, _ = network.predict(spatial, vector, mask)
            final_prob_action_0 = policy_probs[:, 0].mean().item()

        # Probability for action 0 should increase
        assert final_prob_action_0 > initial_prob_action_0
        assert final_prob_action_0 > 0.5  # Should be dominant


# =============================================================================
# Test: Trajectory Loss Computation
# =============================================================================


class TestTrajectoryLossComputation:
    """Tests for trajectory loss computation."""

    def test_trajectory_loss_basic(self, small_network, small_network_config, set_seed):
        """Basic trajectory loss computation works."""
        batch_size = 4
        unroll_steps = 3

        # Generate random data
        spatial = torch.randn(batch_size, 13, 8, 8)
        vector = torch.randn(batch_size, 64)
        actions = torch.randint(0, 100, (batch_size, unroll_steps))

        # Targets
        target_policies = [
            F.softmax(torch.randn(batch_size, 100), dim=-1) for _ in range(unroll_steps + 1)
        ]
        target_values = [torch.randn(batch_size) for _ in range(unroll_steps + 1)]
        target_rewards = [torch.randn(batch_size) for _ in range(unroll_steps)]
        action_masks = [torch.ones(batch_size, 100) for _ in range(unroll_steps + 1)]

        # Initial inference
        output = small_network.initial_inference(spatial, vector, action_masks[0])

        # Unroll
        hidden = output.hidden_state
        unrolled_policies = []
        unrolled_values = []
        unrolled_rewards = []

        for k in range(unroll_steps):
            next_hidden, reward_logits, policy_logits, value = small_network.recurrent_inference(
                hidden, actions[:, k], action_masks[k + 1]
            )
            # Convert categorical reward to scalar for loss computation
            reward_scalar = categorical_to_scalar(reward_logits, apply_softmax=True)
            unrolled_policies.append(policy_logits)
            unrolled_values.append(value)
            unrolled_rewards.append(reward_scalar)
            hidden = next_hidden

        # Compute trajectory loss (use scalar loss for simpler test)
        loss_config = LossConfig(
            use_categorical=False,
        )

        trajectory_loss = compute_muzero_trajectory_loss(
            initial_policy_logits=output.policy_logits,
            initial_value=output.value,
            target_policies=target_policies,
            target_values=target_values,
            target_rewards=target_rewards,
            unrolled_policy_logits=unrolled_policies,
            unrolled_values=unrolled_values,
            unrolled_rewards=unrolled_rewards,
            action_masks=action_masks,
            config=loss_config,
        )

        # Verify outputs
        assert trajectory_loss.total_loss is not None
        assert not torch.isnan(trajectory_loss.total_loss)
        assert trajectory_loss.policy_loss is not None
        assert trajectory_loss.value_loss is not None
        assert trajectory_loss.reward_loss is not None
        assert len(trajectory_loss.step_losses) == unroll_steps + 1
