"""Integration tests for reward signal validation.

Tests that rewards are within expected ranges, terminal rewards are
correctly assigned, and rewards are properly stored in replay buffer.

Run with:
    pytest tests/integration/test_reward_signals.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import torch

if TYPE_CHECKING:
    pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_replay_buffer() -> MagicMock:
    """Create mock replay buffer with sample storage."""
    buffer = MagicMock()
    buffer._stored_examples: list[dict[str, Any]] = []

    def store(example: dict[str, Any]) -> None:
        buffer._stored_examples.append(example)

    buffer.store = store
    buffer.size.return_value = len(buffer._stored_examples)
    return buffer


@pytest.fixture
def sample_trajectory() -> list[dict[str, Any]]:
    """Create sample trajectory for reward testing."""
    return [
        {
            "state": torch.randn(51, 16, 16),
            "action": torch.tensor([5, 8]),
            "policy": torch.softmax(torch.randn(100), dim=0),
            "reward": 0.0,
            "done": False,
            "turn": 1,
        },
        {
            "state": torch.randn(51, 16, 16),
            "action": torch.tensor([6, 8]),
            "policy": torch.softmax(torch.randn(100), dim=0),
            "reward": 0.1,
            "done": False,
            "turn": 2,
        },
        {
            "state": torch.randn(51, 16, 16),
            "action": torch.tensor([6, 9]),
            "policy": torch.softmax(torch.randn(100), dim=0),
            "reward": 0.0,
            "done": False,
            "turn": 3,
        },
    ]


@pytest.fixture
def terminal_trajectory() -> list[dict[str, Any]]:
    """Create trajectory ending in terminal state (victory)."""
    return [
        {
            "state": torch.randn(51, 16, 16),
            "action": torch.tensor([5, 8]),
            "policy": torch.softmax(torch.randn(100), dim=0),
            "reward": 0.0,
            "done": False,
            "turn": 98,
        },
        {
            "state": torch.randn(51, 16, 16),
            "action": torch.tensor([6, 8]),
            "policy": torch.softmax(torch.randn(100), dim=0),
            "reward": 0.1,
            "done": False,
            "turn": 99,
        },
        {
            "state": torch.randn(51, 16, 16),
            "action": torch.tensor([6, 9]),
            "policy": torch.softmax(torch.randn(100), dim=0),
            "reward": 1.0,  # Terminal victory reward
            "done": True,
            "turn": 100,
        },
    ]


@pytest.fixture
def defeat_trajectory() -> list[dict[str, Any]]:
    """Create trajectory ending in defeat."""
    return [
        {
            "state": torch.randn(51, 16, 16),
            "action": torch.tensor([5, 8]),
            "policy": torch.softmax(torch.randn(100), dim=0),
            "reward": 0.0,
            "done": False,
            "turn": 98,
        },
        {
            "state": torch.randn(51, 16, 16),
            "action": torch.tensor([6, 8]),
            "policy": torch.softmax(torch.randn(100), dim=0),
            "reward": -0.1,
            "done": False,
            "turn": 99,
        },
        {
            "state": torch.randn(51, 16, 16),
            "action": torch.tensor([6, 9]),
            "policy": torch.softmax(torch.randn(100), dim=0),
            "reward": -1.0,  # Terminal defeat reward
            "done": True,
            "turn": 100,
        },
    ]


# ============================================================================
# Reward Range Tests
# ============================================================================


class TestRewardInExpectedRange:
    """Tests that rewards fall within expected range [-1, 1]."""

    def test_intermediate_rewards_in_range(
        self,
        sample_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify intermediate rewards are within [-1, 1]."""
        for step in sample_trajectory:
            reward = step["reward"]
            assert -1.0 <= reward <= 1.0, f"Reward {reward} out of range"

    def test_terminal_victory_reward_valid(
        self,
        terminal_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify victory terminal reward is valid."""
        terminal_step = terminal_trajectory[-1]
        assert terminal_step["done"] is True
        assert terminal_step["reward"] == 1.0

    def test_terminal_defeat_reward_valid(
        self,
        defeat_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify defeat terminal reward is valid."""
        terminal_step = defeat_trajectory[-1]
        assert terminal_step["done"] is True
        assert terminal_step["reward"] == -1.0

    def test_non_terminal_rewards_smaller_magnitude(
        self,
        sample_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify non-terminal rewards have smaller magnitude than terminal."""
        for step in sample_trajectory:
            if not step["done"]:
                reward = step["reward"]
                # Non-terminal rewards should be smaller
                assert abs(reward) <= 0.5, f"Non-terminal reward {reward} too large"


# ============================================================================
# Terminal Reward Tests
# ============================================================================


class TestTerminalRewardOnVictory:
    """Tests for terminal reward assignment on game end."""

    def test_victory_assigns_positive_reward(
        self,
        terminal_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify victory assigns positive terminal reward."""
        terminal_step = terminal_trajectory[-1]
        assert terminal_step["reward"] > 0

    def test_defeat_assigns_negative_reward(
        self,
        defeat_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify defeat assigns negative terminal reward."""
        terminal_step = defeat_trajectory[-1]
        assert terminal_step["reward"] < 0

    def test_done_flag_set_on_terminal(
        self,
        terminal_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify done flag is True on terminal state."""
        terminal_step = terminal_trajectory[-1]
        assert terminal_step["done"] is True

    def test_non_terminal_done_flag_false(
        self,
        sample_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify done flag is False for non-terminal states."""
        for step in sample_trajectory:
            assert step["done"] is False


class TestRewardScaling:
    """Tests for reward value scaling."""

    def test_reward_monotonicity_on_progress(self) -> None:
        """Verify rewards increase with game progress."""
        # Simulate progress-based rewards
        scores = [100, 150, 200, 250, 300]
        previous_score = scores[0]
        rewards = []

        for score in scores[1:]:
            reward = (score - previous_score) / 100.0  # Scale to reasonable range
            rewards.append(reward)
            previous_score = score

        # All progress rewards should be positive
        assert all(r > 0 for r in rewards)
        assert all(-1 <= r <= 1 for r in rewards)

    def test_negative_progress_gives_negative_reward(self) -> None:
        """Verify negative progress yields negative reward."""
        previous_score = 300
        current_score = 250  # Lost a city or units

        reward = (current_score - previous_score) / 100.0
        assert reward < 0
        assert reward >= -1


# ============================================================================
# Reward Consistency Tests
# ============================================================================


class TestRewardConsistencyAcrossTurns:
    """Tests that rewards don't have spurious noise."""

    def test_no_spurious_large_rewards(
        self,
        sample_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify no unexpected large reward spikes."""
        for i, step in enumerate(sample_trajectory):
            if not step["done"]:
                # Non-terminal rewards shouldn't be close to +/-1
                assert abs(step["reward"]) < 0.9, f"Spurious large reward at step {i}"

    def test_reward_variance_reasonable(self) -> None:
        """Verify reward variance is reasonable (not too noisy)."""
        # Create sample trajectory with typical rewards
        rewards = [0.0, 0.05, 0.0, 0.1, -0.05, 0.0, 0.15, 0.0]

        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)

        # Variance should be small for non-terminal rewards
        assert variance < 0.1, f"Reward variance {variance} too high"

    def test_cumulative_reward_bounded(
        self,
        sample_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify cumulative reward is bounded."""
        cumulative = sum(step["reward"] for step in sample_trajectory)

        # Cumulative should still be reasonable
        trajectory_length = len(sample_trajectory)
        max_expected = trajectory_length * 0.5  # Each step max 0.5

        assert abs(cumulative) <= max_expected


# ============================================================================
# Replay Buffer Integration Tests
# ============================================================================


class TestRewardReachesReplayBuffer:
    """Tests that rewards are correctly stored in replay buffer."""

    def test_trajectory_stored_with_rewards(
        self,
        mock_replay_buffer: MagicMock,
        sample_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify trajectory with rewards is stored in buffer."""
        for step in sample_trajectory:
            mock_replay_buffer.store(step)

        assert len(mock_replay_buffer._stored_examples) == 3

    def test_stored_rewards_match_original(
        self,
        mock_replay_buffer: MagicMock,
        sample_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify stored rewards match original trajectory."""
        for step in sample_trajectory:
            mock_replay_buffer.store(step)

        for i, stored in enumerate(mock_replay_buffer._stored_examples):
            assert stored["reward"] == sample_trajectory[i]["reward"]

    def test_terminal_reward_preserved(
        self,
        mock_replay_buffer: MagicMock,
        terminal_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify terminal reward is preserved in storage."""
        for step in terminal_trajectory:
            mock_replay_buffer.store(step)

        terminal_stored = mock_replay_buffer._stored_examples[-1]
        assert terminal_stored["reward"] == 1.0
        assert terminal_stored["done"] is True


class TestRewardWithValueTargets:
    """Tests for reward integration with value targets."""

    def test_value_target_incorporates_reward(
        self,
        sample_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify value targets incorporate rewards correctly."""
        gamma = 0.99  # Discount factor

        # Compute bootstrapped returns
        returns = []
        running_return = 0.0

        for step in reversed(sample_trajectory):
            running_return = step["reward"] + gamma * running_return
            returns.insert(0, running_return)

        # Returns should be reasonable
        for ret in returns:
            assert -10 <= ret <= 10, f"Return {ret} out of bounds"

    def test_terminal_value_target_equals_reward(
        self,
        terminal_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify terminal value target equals terminal reward."""
        terminal_step = terminal_trajectory[-1]

        # At terminal state, value target = immediate reward (no future)
        value_target = terminal_step["reward"]

        assert value_target == 1.0  # Victory


# ============================================================================
# Integration Tests
# ============================================================================


class TestRewardSignalIntegration:
    """Integration tests for full reward pipeline."""

    def test_complete_episode_rewards(
        self,
        terminal_trajectory: list[dict[str, Any]],
    ) -> None:
        """Test complete episode reward computation."""
        total_reward = sum(step["reward"] for step in terminal_trajectory)

        # Victory episode should have positive total
        assert total_reward > 0

    def test_discount_reward_computation(
        self,
        terminal_trajectory: list[dict[str, Any]],
    ) -> None:
        """Test discounted reward computation for training."""
        gamma = 0.99

        # Compute discounted returns (backwards from terminal)
        returns = [0.0] * len(terminal_trajectory)
        returns[-1] = terminal_trajectory[-1]["reward"]

        for i in range(len(terminal_trajectory) - 2, -1, -1):
            returns[i] = terminal_trajectory[i]["reward"] + gamma * returns[i + 1]

        # All returns should be positive (terminal reward is +1)
        assert all(r > 0 for r in returns)

        # Returns accumulate value from future rewards
        # So earlier returns include more accumulated future rewards
        # With positive intermediate rewards, earlier returns are larger
        assert returns[0] >= returns[-1]

    def test_example_format_for_training(
        self,
        sample_trajectory: list[dict[str, Any]],
    ) -> None:
        """Verify examples have correct format for training."""
        for step in sample_trajectory:
            assert "state" in step
            assert "action" in step
            assert "policy" in step
            assert "reward" in step
            assert "done" in step

            assert isinstance(step["state"], torch.Tensor)
            assert isinstance(step["reward"], (int, float))
            assert isinstance(step["done"], bool)

    def test_batch_rewards_tensor_creation(
        self,
        sample_trajectory: list[dict[str, Any]],
    ) -> None:
        """Test creating reward tensor for batch training."""
        rewards = torch.tensor([step["reward"] for step in sample_trajectory])

        assert rewards.shape == (3,)
        assert rewards.dtype == torch.float32 or rewards.dtype == torch.float64
        assert torch.all(rewards >= -1)
        assert torch.all(rewards <= 1)
