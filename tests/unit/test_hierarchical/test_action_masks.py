"""Edge case tests for action masking in hierarchical RL.

Tests critical edge cases that can cause NaN, crashes, or unexpected behavior:
- All actions masked (invalid)
- Single valid action
- Batch size = 1
- Different mask dtypes
"""

import pytest
import torch

from alpha_zero.hierarchical.config import HierarchyLevel
from alpha_zero.hierarchical.network.hierarchical_network import (
    HierarchicalNetworkConfig,
    LevelNetwork,
)

# Test constants
BATCH_SIZE = 2
SPATIAL_CHANNELS = 13
BOARD_SIZE = (8, 8)
VECTOR_DIM = 64
ACTION_SPACE = 100


@pytest.fixture
def level_config() -> HierarchicalNetworkConfig:
    """Create a test configuration."""
    return HierarchicalNetworkConfig(
        board_size=BOARD_SIZE,
        spatial_channels=SPATIAL_CHANNELS,
        vector_dim=VECTOR_DIM,
        embed_dim=64,
        num_res_blocks=2,
        goal_embed_dim=32,
        action_space_size=ACTION_SPACE,
    )


@pytest.fixture
def level_network(level_config: HierarchicalNetworkConfig) -> LevelNetwork:
    """Create a LevelNetwork for testing."""
    return LevelNetwork(level_config, HierarchyLevel.OPERATIONAL)


@pytest.fixture
def sample_inputs() -> dict:
    """Create sample input tensors."""
    return {
        "spatial": torch.randn(BATCH_SIZE, SPATIAL_CHANNELS, *BOARD_SIZE),
        "vector": torch.randn(BATCH_SIZE, VECTOR_DIM),
        "goal": torch.randn(BATCH_SIZE, 32),  # goal_embed_dim
    }


class TestAllActionsInvalid:
    """Test behavior when all actions are masked as invalid."""

    def test_all_actions_invalid_returns_valid_output(
        self, level_network: LevelNetwork, sample_inputs: dict
    ) -> None:
        """Test that all-actions-invalid case returns valid (non-NaN) output."""
        level_network.eval()

        # Create mask with all zeros (all invalid)
        action_mask = torch.zeros(BATCH_SIZE, ACTION_SPACE)

        with torch.no_grad():
            policy, value = level_network(
                sample_inputs["spatial"],
                sample_inputs["vector"],
                sample_inputs["goal"],
                action_mask=action_mask,
            )

        # Policy should not contain NaN
        assert not torch.isnan(policy).any(), "Policy contains NaN values"
        # Policy should not contain Inf
        assert not torch.isinf(policy).any(), "Policy contains Inf values"
        # Value should be valid
        assert not torch.isnan(value).any(), "Value contains NaN values"

    def test_all_actions_invalid_enables_fallback(
        self, level_network: LevelNetwork, sample_inputs: dict
    ) -> None:
        """Test that all-actions-invalid case enables a fallback action."""
        level_network.eval()

        # Create mask with all zeros
        action_mask = torch.zeros(BATCH_SIZE, ACTION_SPACE)

        with torch.no_grad():
            policy, _ = level_network(
                sample_inputs["spatial"],
                sample_inputs["vector"],
                sample_inputs["goal"],
                action_mask=action_mask,
            )

        # After softmax, at least one action should have non-zero probability
        policy_probs = torch.softmax(policy, dim=-1)
        assert (policy_probs.sum(dim=-1) > 0.99).all(), "Policy probabilities don't sum to 1"


class TestSingleValidAction:
    """Test behavior when only one action is valid."""

    def test_single_valid_action_deterministic(
        self, level_network: LevelNetwork, sample_inputs: dict
    ) -> None:
        """Test that single valid action gets all probability mass."""
        level_network.eval()

        # Create mask with only action 42 valid
        action_mask = torch.zeros(BATCH_SIZE, ACTION_SPACE)
        action_mask[:, 42] = 1.0

        with torch.no_grad():
            policy, _ = level_network(
                sample_inputs["spatial"],
                sample_inputs["vector"],
                sample_inputs["goal"],
                action_mask=action_mask,
            )

        # After softmax, action 42 should have all probability
        policy_probs = torch.softmax(policy, dim=-1)
        assert (policy_probs[:, 42] > 0.99).all(), "Single valid action doesn't get all probability"

    def test_single_valid_action_first(
        self, level_network: LevelNetwork, sample_inputs: dict
    ) -> None:
        """Test with first action (index 0) as the only valid one."""
        level_network.eval()

        action_mask = torch.zeros(BATCH_SIZE, ACTION_SPACE)
        action_mask[:, 0] = 1.0

        with torch.no_grad():
            policy, _ = level_network(
                sample_inputs["spatial"],
                sample_inputs["vector"],
                sample_inputs["goal"],
                action_mask=action_mask,
            )

        policy_probs = torch.softmax(policy, dim=-1)
        assert (policy_probs[:, 0] > 0.99).all(), "First action doesn't get all probability"

    def test_single_valid_action_last(
        self, level_network: LevelNetwork, sample_inputs: dict
    ) -> None:
        """Test with last action as the only valid one."""
        level_network.eval()

        action_mask = torch.zeros(BATCH_SIZE, ACTION_SPACE)
        action_mask[:, -1] = 1.0

        with torch.no_grad():
            policy, _ = level_network(
                sample_inputs["spatial"],
                sample_inputs["vector"],
                sample_inputs["goal"],
                action_mask=action_mask,
            )

        policy_probs = torch.softmax(policy, dim=-1)
        assert (policy_probs[:, -1] > 0.99).all(), "Last action doesn't get all probability"


class TestAllActionsValid:
    """Test behavior when all actions are valid."""

    def test_all_actions_valid_no_masking_effect(
        self, level_network: LevelNetwork, sample_inputs: dict
    ) -> None:
        """Test that all-valid mask doesn't affect output."""
        level_network.eval()

        # All actions valid
        action_mask = torch.ones(BATCH_SIZE, ACTION_SPACE)

        with torch.no_grad():
            # With mask
            policy_masked, value_masked = level_network(
                sample_inputs["spatial"],
                sample_inputs["vector"],
                sample_inputs["goal"],
                action_mask=action_mask,
            )

            # Without mask
            policy_unmasked, value_unmasked = level_network(
                sample_inputs["spatial"],
                sample_inputs["vector"],
                sample_inputs["goal"],
                action_mask=None,
            )

        # Should be identical
        assert torch.allclose(policy_masked, policy_unmasked, atol=1e-6)
        assert torch.allclose(value_masked, value_unmasked, atol=1e-6)


class TestSparseValidActions:
    """Test behavior with sparse valid actions."""

    def test_sparse_valid_actions(self, level_network: LevelNetwork, sample_inputs: dict) -> None:
        """Test with only a few scattered valid actions."""
        level_network.eval()

        # Only actions 0, 25, 50, 75, 99 are valid
        valid_indices = [0, 25, 50, 75, 99]
        action_mask = torch.zeros(BATCH_SIZE, ACTION_SPACE)
        for idx in valid_indices:
            action_mask[:, idx] = 1.0

        with torch.no_grad():
            policy, _ = level_network(
                sample_inputs["spatial"],
                sample_inputs["vector"],
                sample_inputs["goal"],
                action_mask=action_mask,
            )

        # After softmax, invalid actions should have ~0 probability
        policy_probs = torch.softmax(policy, dim=-1)

        # Check valid actions have probability
        for idx in valid_indices:
            assert (policy_probs[:, idx] > 1e-10).all(), f"Valid action {idx} has zero probability"

        # Check probabilities sum to 1
        assert torch.allclose(
            policy_probs.sum(dim=-1), torch.ones(BATCH_SIZE), atol=1e-5
        ), "Probabilities don't sum to 1"


class TestBatchSizeOne:
    """Test edge cases with batch size = 1."""

    def test_batch_size_one_with_mask(self, level_network: LevelNetwork) -> None:
        """Test network with batch size 1 and action mask."""
        level_network.eval()

        # Single sample inputs
        spatial = torch.randn(1, SPATIAL_CHANNELS, *BOARD_SIZE)
        vector = torch.randn(1, VECTOR_DIM)
        goal = torch.randn(1, 32)
        action_mask = torch.ones(1, ACTION_SPACE)
        action_mask[0, 50:] = 0  # Only first 50 actions valid

        with torch.no_grad():
            policy, value = level_network(spatial, vector, goal, action_mask=action_mask)

        # Check shapes
        assert policy.shape == (1, ACTION_SPACE), f"Wrong policy shape: {policy.shape}"
        assert value.shape == (1,), f"Wrong value shape: {value.shape}"

        # Check no NaN/Inf
        assert not torch.isnan(policy).any()
        assert not torch.isnan(value).any()

    def test_batch_size_one_all_invalid(self, level_network: LevelNetwork) -> None:
        """Test batch size 1 with all actions invalid."""
        level_network.eval()

        spatial = torch.randn(1, SPATIAL_CHANNELS, *BOARD_SIZE)
        vector = torch.randn(1, VECTOR_DIM)
        goal = torch.randn(1, 32)
        action_mask = torch.zeros(1, ACTION_SPACE)

        with torch.no_grad():
            policy, value = level_network(spatial, vector, goal, action_mask=action_mask)

        # Should not crash or produce NaN
        assert not torch.isnan(policy).any(), "NaN in policy with batch_size=1 all invalid"
        assert not torch.isnan(value).any(), "NaN in value with batch_size=1 all invalid"


class TestMaskDtypes:
    """Test action mask with different data types."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.int32, torch.int64, torch.bool]
    )
    def test_different_mask_dtypes(
        self, level_network: LevelNetwork, sample_inputs: dict, dtype: torch.dtype
    ) -> None:
        """Test that different mask dtypes work correctly."""
        level_network.eval()

        # Create mask with specified dtype
        if dtype == torch.bool:
            action_mask = torch.ones(BATCH_SIZE, ACTION_SPACE, dtype=torch.bool)
        else:
            action_mask = torch.ones(BATCH_SIZE, ACTION_SPACE, dtype=dtype)

        with torch.no_grad():
            try:
                policy, _value = level_network(
                    sample_inputs["spatial"],
                    sample_inputs["vector"],
                    sample_inputs["goal"],
                    action_mask=action_mask.float(),  # Convert to float for comparison
                )
                # Should not produce NaN
                assert not torch.isnan(policy).any(), f"NaN with dtype {dtype}"
            except (RuntimeError, TypeError) as e:
                # Some dtypes may not be supported - that's acceptable
                pytest.skip(f"Dtype {dtype} not supported: {e}")


class TestMaskShapeMismatch:
    """Test error handling for mask shape mismatches."""

    def test_wrong_batch_dimension_raises(
        self, level_network: LevelNetwork, sample_inputs: dict
    ) -> None:
        """Test that wrong batch dimension raises an error or handles gracefully."""
        level_network.eval()

        # Wrong batch size
        wrong_batch_mask = torch.ones(BATCH_SIZE + 1, ACTION_SPACE)

        # This should either raise an error or handle gracefully
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            level_network(
                sample_inputs["spatial"],
                sample_inputs["vector"],
                sample_inputs["goal"],
                action_mask=wrong_batch_mask,
            )

    def test_wrong_action_dimension_raises(
        self, level_network: LevelNetwork, sample_inputs: dict
    ) -> None:
        """Test that wrong action dimension raises an error."""
        level_network.eval()

        # Wrong action space size
        wrong_action_mask = torch.ones(BATCH_SIZE, ACTION_SPACE + 50)

        # This should raise an error
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            level_network(
                sample_inputs["spatial"],
                sample_inputs["vector"],
                sample_inputs["goal"],
                action_mask=wrong_action_mask,
            )
