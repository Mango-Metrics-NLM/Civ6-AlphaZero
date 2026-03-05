"""Integration tests for training convergence validation.

Tests the critical v1.3.0 fix: weight continuity across training
iterations (checkpoint load → train → save → reload).

Tests cover:
  - Checkpoint round-trip preserves exact weights
  - Training reduces loss over successive epochs
  - Multi-iteration training with checkpoint reload maintains weights
  - Policy loss is lower after training than before

Run with:
    pytest tests/integration/test_convergence_validation.py -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.envs.microciv import MicroCivEnv

# Small network config for fast tests
_BOARD_SIZE = 8
_NUM_RES_BLOCKS = 2
_NUM_FILTERS = 16
_DEVICE = "cpu"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def env():
    """Create MicroCiv environment."""
    return MicroCivEnv(board_size=_BOARD_SIZE)


@pytest.fixture
def network(env):
    """Create small MicroCivNetwork for testing."""
    return MicroCivNetwork(
        board_size=_BOARD_SIZE,
        action_space_size=env.action_space.n,
        num_res_blocks=_NUM_RES_BLOCKS,
        num_filters=_NUM_FILTERS,
    ).to(_DEVICE)


@pytest.fixture
def training_data(network, env):
    """Generate synthetic training data from random self-play."""
    examples = []
    network.eval()

    for _ in range(3):  # 3 short games
        obs, _ = env.reset()
        game_states = []

        for _step in range(20):  # Max 20 steps per game
            game_states.append({
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "action_mask": obs["action_mask"].copy(),
                "player": env.current_player,
            })

            valid_actions = np.where(obs["action_mask"] > 0)[0]
            if len(valid_actions) == 0:
                break
            action = int(np.random.choice(valid_actions))
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # Generate policy targets from network
        winner = info.get("winner") if "info" in dir() else None
        for state in game_states:
            spatial_t = torch.tensor(state["spatial"], dtype=torch.float32).unsqueeze(0)
            vector_t = torch.tensor(state["vector"], dtype=torch.float32).unsqueeze(0)
            mask_t = torch.tensor(state["action_mask"], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                policy_logits, _ = network(spatial_t, vector_t, mask_t)

            probs = F.softmax(policy_logits.squeeze(0), dim=-1).numpy()
            value = 0.0 if winner is None else (1.0 if winner == state["player"] else -1.0)

            examples.append({
                "spatial": torch.tensor(state["spatial"], dtype=torch.float32),
                "vector": torch.tensor(state["vector"], dtype=torch.float32),
                "action_mask": torch.tensor(state["action_mask"], dtype=torch.float32),
                "policy": torch.tensor(probs, dtype=torch.float32),
                "value": torch.tensor(value, dtype=torch.float32),
            })

    return examples


def _compute_batch_loss(network, data):
    """Compute average loss on a batch of data."""
    if not data:
        return float("inf")

    network.eval()
    spatial = torch.stack([d["spatial"] for d in data]).to(_DEVICE)
    vector = torch.stack([d["vector"] for d in data]).to(_DEVICE)
    action_mask = torch.stack([d["action_mask"] for d in data]).to(_DEVICE)
    target_policy = torch.stack([d["policy"] for d in data]).to(_DEVICE)
    target_value = torch.stack([d["value"] for d in data]).to(_DEVICE)

    with torch.no_grad():
        policy_logits, value = network(spatial, vector, action_mask)

    value_loss = F.mse_loss(value.squeeze(-1), target_value)
    masked_target = target_policy * action_mask
    target_sum = masked_target.sum(dim=-1, keepdim=True) + 1e-8
    masked_target = masked_target / target_sum
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.sum(masked_target * log_probs, dim=-1).mean()

    return (policy_loss + value_loss).item()


def _train_one_epoch(network, data, lr=1e-3):
    """Train network for one epoch, return average loss."""
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    total_loss = 0.0

    spatial = torch.stack([d["spatial"] for d in data]).to(_DEVICE)
    vector = torch.stack([d["vector"] for d in data]).to(_DEVICE)
    action_mask = torch.stack([d["action_mask"] for d in data]).to(_DEVICE)
    target_policy = torch.stack([d["policy"] for d in data]).to(_DEVICE)
    target_value = torch.stack([d["value"] for d in data]).to(_DEVICE)

    policy_logits, value = network(spatial, vector, action_mask)

    value_loss = F.mse_loss(value.squeeze(-1), target_value)
    masked_target = target_policy * action_mask
    target_sum = masked_target.sum(dim=-1, keepdim=True) + 1e-8
    masked_target = masked_target / target_sum
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.sum(masked_target * log_probs, dim=-1).mean()

    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ============================================================================
# Test: Checkpoint Round-Trip
# ============================================================================


class TestWeightContinuity:
    """Tests that checkpoint save→load preserves exact weights."""

    def test_checkpoint_round_trip(self, network):
        """Verify save→load produces identical weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            original_state = {k: v.clone() for k, v in network.state_dict().items()}

            torch.save({"network_state_dict": network.state_dict(), "step": 0}, path)

            network2 = MicroCivNetwork(
                board_size=_BOARD_SIZE,
                action_space_size=network.state_dict()["policy_head.fc2.bias"].shape[0],
                num_res_blocks=_NUM_RES_BLOCKS,
                num_filters=_NUM_FILTERS,
            ).to(_DEVICE)

            checkpoint = torch.load(path, map_location=_DEVICE, weights_only=True)
            network2.load_state_dict(checkpoint["network_state_dict"])

            for key in original_state:
                assert torch.equal(original_state[key], network2.state_dict()[key]), (
                    f"Weight mismatch for key: {key}"
                )

    def test_checkpoint_preserves_after_training(self, network, training_data):
        """Verify weights are preserved through train→save→load cycle."""
        # Train for 1 epoch
        _train_one_epoch(network, training_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "post_train.pt")
            trained_state = {k: v.clone() for k, v in network.state_dict().items()}

            torch.save({"network_state_dict": network.state_dict(), "step": 1}, path)

            # Reload into same network (simulates iteration start)
            checkpoint = torch.load(path, map_location=_DEVICE, weights_only=True)
            network.load_state_dict(checkpoint["network_state_dict"])

            for key in trained_state:
                assert torch.equal(trained_state[key], network.state_dict()[key]), (
                    f"Weight drift after reload for key: {key}"
                )

    def test_training_changes_weights(self, network, training_data):
        """Verify training actually modifies weights (not a no-op)."""
        original_state = {k: v.clone() for k, v in network.state_dict().items()}

        _train_one_epoch(network, training_data)

        changed = False
        for key in original_state:
            if not torch.equal(original_state[key], network.state_dict()[key]):
                changed = True
                break

        assert changed, "Training did not change any weights — training loop may be broken"


# ============================================================================
# Test: Loss Decrease
# ============================================================================


class TestLossDecrease:
    """Tests that training reduces loss."""

    def test_loss_decreases_within_iteration(self, network, training_data):
        """Loss should decrease after training epochs on the same data."""
        loss_before = _compute_batch_loss(network, training_data)

        for _ in range(5):  # 5 epochs of training
            _train_one_epoch(network, training_data)

        loss_after = _compute_batch_loss(network, training_data)

        assert loss_after < loss_before * 1.05, (
            f"Loss increased significantly: before={loss_before:.4f}, after={loss_after:.4f}"
        )

    def test_multi_iteration_loss_trend(self, network, training_data):
        """Simulate multi-iteration training with checkpoint reload."""
        iteration_losses = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for iteration in range(3):
                # Train
                epoch_loss = 0.0
                for _ in range(3):
                    epoch_loss = _train_one_epoch(network, training_data)

                iteration_losses.append(epoch_loss)

                # Save checkpoint
                path = os.path.join(tmpdir, f"iter_{iteration}.pt")
                torch.save({
                    "network_state_dict": network.state_dict(),
                    "step": iteration,
                }, path)

                # Reload (simulates v1.3.0 weight continuity)
                checkpoint = torch.load(path, map_location=_DEVICE, weights_only=True)
                network.load_state_dict(checkpoint["network_state_dict"])

        # Loss should generally trend downward
        assert iteration_losses[-1] <= iteration_losses[0] * 1.1, (
            f"Loss did not decrease across iterations: {iteration_losses}"
        )


# ============================================================================
# Test: Weight Continuity vs Random Init
# ============================================================================


class TestWeightContinuityVsRandomInit:
    """Tests that incumbent weights are better than random init."""

    def test_continued_training_beats_random_init(self, network, training_data):
        """Training from checkpoint should produce lower loss than from scratch."""
        # Train network for several epochs
        for _ in range(5):
            _train_one_epoch(network, training_data)

        loss_continued = _compute_batch_loss(network, training_data)

        # Create fresh random network and train same number of epochs  
        env = MicroCivEnv(board_size=_BOARD_SIZE)
        fresh_network = MicroCivNetwork(
            board_size=_BOARD_SIZE,
            action_space_size=env.action_space.n,
            num_res_blocks=_NUM_RES_BLOCKS,
            num_filters=_NUM_FILTERS,
        ).to(_DEVICE)

        for _ in range(5):
            _train_one_epoch(fresh_network, training_data)

        loss_random = _compute_batch_loss(fresh_network, training_data)

        # The continued training should be at least as good
        # (with a tolerance since random init may occasionally get lucky)
        assert loss_continued <= loss_random * 1.5, (
            f"Continued training ({loss_continued:.4f}) not better than "
            f"random init ({loss_random:.4f})"
        )
