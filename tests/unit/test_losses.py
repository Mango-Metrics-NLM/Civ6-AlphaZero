"""Unit tests for centralized loss functions.

Tests for alpha_zero.core.losses module covering:
- Scalar-categorical conversion utilities
- MuZero core losses (policy, value, reward)
- EfficientZero auxiliary losses (consistency, value prefix)
- Combined loss functions (step and trajectory)
- MuZeroLossCalculator with mock networks
- Gradient flow verification
- Edge cases and numerical stability
"""

import logging
import pytest
import torch
import torch.nn as nn

from alpha_zero.core.losses import (
    LossConfig,
    MuZeroLossCalculator,
    StepLossOutput,
    TrajectoryBatch,
    TrajectoryLossOutput,
    categorical_to_scalar,
    compute_consistency_loss,
    compute_muzero_loss,
    compute_muzero_step_loss,
    compute_muzero_trajectory_loss,
    compute_policy_loss,
    compute_reward_loss,
    compute_value_loss,
    compute_value_prefix_loss,
    scalar_to_categorical,
)


# =============================================================================
# Test Utilities: Scalar-Categorical Conversion
# =============================================================================


class TestScalarToCategorical:
    """Test suite for scalar_to_categorical function."""

    def test_output_shape(self):
        """Output has correct shape."""
        scalars = torch.tensor([0.0, 10.0, -50.0])
        categorical = scalar_to_categorical(scalars, support_size=601)

        assert categorical.shape == (3, 601)

    def test_sums_to_one(self):
        """Categorical distribution sums to 1."""
        scalars = torch.tensor([5.0, -5.0, 0.0, 100.0, -100.0])
        categorical = scalar_to_categorical(scalars, support_size=601)

        assert torch.allclose(categorical.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_non_negative(self):
        """All values are non-negative (valid probability distribution)."""
        scalars = torch.tensor([-200.0, 0.0, 200.0])
        categorical = scalar_to_categorical(scalars, support_size=601)

        assert (categorical >= 0).all()

    def test_two_hot_encoding(self):
        """Result is approximately two-hot (at most 2 non-zero values per sample)."""
        scalars = torch.tensor([0.5])  # Non-integer value
        categorical = scalar_to_categorical(scalars, support_size=601)

        # Should have exactly 2 non-zero entries
        non_zero_count = (categorical > 1e-6).sum().item()
        assert non_zero_count == 2

    def test_integer_value_is_one_hot(self):
        """Integer values at support points are approximately one-hot."""
        # Support goes from -300 to 300 with 601 points, step = 1
        scalars = torch.tensor([0.0])  # Exactly at a support point
        categorical = scalar_to_categorical(
            scalars, support_size=601, min_value=-300, max_value=300
        )

        # Should have exactly 1 non-zero entry (or very close due to numerical issues)
        max_val = categorical.max().item()
        assert max_val > 0.99  # One value should be close to 1

    def test_extreme_values_clamped(self):
        """Extreme values outside support range are clamped."""
        scalars = torch.tensor([1000.0, -1000.0])
        categorical = scalar_to_categorical(
            scalars, support_size=601, min_value=-300, max_value=300
        )

        # Should not crash and still sum to 1
        assert categorical.shape == (2, 601)
        assert torch.allclose(categorical.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_handles_2d_input(self):
        """Handles (B, 1) input shape."""
        scalars = torch.tensor([[0.0], [10.0], [-50.0]])  # Shape (3, 1)
        categorical = scalar_to_categorical(scalars, support_size=601)

        assert categorical.shape == (3, 601)

    def test_custom_support_range(self):
        """Works with custom support range."""
        scalars = torch.tensor([0.0, 0.5, 1.0])
        categorical = scalar_to_categorical(
            scalars, support_size=11, min_value=0.0, max_value=1.0
        )

        assert categorical.shape == (3, 11)
        assert torch.allclose(categorical.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_batch_independence(self):
        """Each batch element is processed independently."""
        scalar1 = torch.tensor([10.0])
        scalar2 = torch.tensor([10.0, 20.0])

        cat1 = scalar_to_categorical(scalar1, support_size=601)
        cat2 = scalar_to_categorical(scalar2, support_size=601)

        # First element of cat2 should match cat1
        assert torch.allclose(cat1[0], cat2[0], atol=1e-6)


class TestCategoricalToScalar:
    """Test suite for categorical_to_scalar function."""

    def test_output_shape(self):
        """Output has correct shape."""
        categorical = torch.zeros(4, 601)
        categorical[:, 300] = 1.0  # All mass at center (value 0)

        scalar = categorical_to_scalar(categorical, support_size=601)

        assert scalar.shape == (4,)

    def test_peak_at_center(self):
        """Distribution peaked at center returns center value."""
        categorical = torch.zeros(1, 601)
        categorical[0, 300] = 1.0  # Center is value 0

        scalar = categorical_to_scalar(
            categorical, support_size=601, min_value=-300, max_value=300
        )

        assert abs(scalar.item()) < 0.1  # Should be close to 0

    def test_peak_at_max(self):
        """Distribution peaked at max returns max value."""
        categorical = torch.zeros(1, 601)
        categorical[0, 600] = 1.0  # Last index is max value

        scalar = categorical_to_scalar(
            categorical, support_size=601, min_value=-300, max_value=300
        )

        assert abs(scalar.item() - 300) < 0.1

    def test_peak_at_min(self):
        """Distribution peaked at min returns min value."""
        categorical = torch.zeros(1, 601)
        categorical[0, 0] = 1.0  # First index is min value

        scalar = categorical_to_scalar(
            categorical, support_size=601, min_value=-300, max_value=300
        )

        assert abs(scalar.item() + 300) < 0.1

    def test_uniform_distribution(self):
        """Uniform distribution returns mean of support."""
        categorical = torch.ones(1, 601) / 601

        scalar = categorical_to_scalar(
            categorical, support_size=601, min_value=-300, max_value=300
        )

        # Mean of symmetric support is 0
        assert abs(scalar.item()) < 0.1

    def test_with_softmax(self):
        """Apply softmax to logits before conversion."""
        logits = torch.zeros(1, 601)
        logits[0, 300] = 10.0  # Strong preference for center

        scalar = categorical_to_scalar(
            logits, support_size=601, min_value=-300, max_value=300, apply_softmax=True
        )

        # Should be close to 0 (center value)
        assert abs(scalar.item()) < 1.0


class TestScalarCategoricalRoundtrip:
    """Test round-trip conversion scalar -> categorical -> scalar."""

    @pytest.mark.parametrize(
        "value",
        [0.0, 10.0, -50.0, 100.0, -100.0, 299.0, -299.0],
    )
    def test_roundtrip_single_value(self, value):
        """Round-trip preserves single scalar value."""
        scalar = torch.tensor([value])
        categorical = scalar_to_categorical(scalar, support_size=601)
        recovered = categorical_to_scalar(categorical, support_size=601)

        # Allow small tolerance due to discretization
        assert torch.allclose(scalar, recovered, atol=2.0)

    def test_roundtrip_batch(self):
        """Round-trip preserves batch of values."""
        scalars = torch.tensor([0.0, 10.0, -50.0, 100.0, -100.0])
        categorical = scalar_to_categorical(scalars, support_size=601)
        recovered = categorical_to_scalar(categorical, support_size=601)

        assert torch.allclose(scalars, recovered, atol=2.0)

    def test_roundtrip_random_values(self):
        """Round-trip works for random values in range."""
        scalars = torch.rand(100) * 600 - 300  # Random in [-300, 300]
        categorical = scalar_to_categorical(scalars, support_size=601)
        recovered = categorical_to_scalar(categorical, support_size=601)

        assert torch.allclose(scalars, recovered, atol=2.0)


# =============================================================================
# Test MuZero Core Losses
# =============================================================================


class TestComputePolicyLoss:
    """Test suite for compute_policy_loss function."""

    def test_output_is_scalar(self):
        """Policy loss is a scalar tensor."""
        policy_logits = torch.randn(4, 100)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)

        loss = compute_policy_loss(policy_logits, target_policy)

        assert loss.dim() == 0
        assert loss.dtype == torch.float32

    def test_identical_distributions_low_loss(self):
        """Loss is low when predicted matches target."""
        # Create target with high probability on one action
        target_policy = torch.zeros(1, 100)
        target_policy[0, 50] = 1.0

        # Make logits match (high logit for action 50)
        policy_logits = torch.full((1, 100), -10.0)
        policy_logits[0, 50] = 10.0

        loss = compute_policy_loss(policy_logits, target_policy)

        # Loss should be very low (close to 0)
        assert loss.item() < 0.01

    def test_different_distributions_high_loss(self):
        """Loss is high when predicted differs from target."""
        # Target: all mass on action 0
        target_policy = torch.zeros(1, 100)
        target_policy[0, 0] = 1.0

        # Prediction: all mass on action 99
        policy_logits = torch.full((1, 100), -10.0)
        policy_logits[0, 99] = 10.0

        loss = compute_policy_loss(policy_logits, target_policy)

        # Loss should be high
        assert loss.item() > 10.0

    def test_with_action_mask(self):
        """Action mask renormalizes target distribution."""
        policy_logits = torch.randn(1, 100)

        # Target has mass on actions 0 and 50
        target_policy = torch.zeros(1, 100)
        target_policy[0, 0] = 0.5
        target_policy[0, 50] = 0.5

        # Mask only allows action 0
        action_mask = torch.zeros(1, 100)
        action_mask[0, 0] = 1.0

        loss = compute_policy_loss(policy_logits, target_policy, action_mask)

        # Should compute without error
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_gradient_flows(self):
        """Gradients flow through policy loss."""
        policy_logits = torch.randn(4, 100, requires_grad=True)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)

        loss = compute_policy_loss(policy_logits, target_policy)
        loss.backward()

        assert policy_logits.grad is not None
        assert policy_logits.grad.shape == policy_logits.shape

    def test_reduction_none(self):
        """Reduction='none' returns per-sample loss."""
        policy_logits = torch.randn(4, 100)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)

        loss = compute_policy_loss(policy_logits, target_policy, reduction="none")

        assert loss.shape == (4,)

    def test_reduction_sum(self):
        """Reduction='sum' returns sum of losses."""
        policy_logits = torch.randn(4, 100)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)

        loss_none = compute_policy_loss(policy_logits, target_policy, reduction="none")
        loss_sum = compute_policy_loss(policy_logits, target_policy, reduction="sum")

        assert torch.allclose(loss_sum, loss_none.sum())


class TestComputeValueLoss:
    """Test suite for compute_value_loss function."""

    def test_categorical_output_is_scalar(self):
        """Categorical value loss is a scalar tensor."""
        value = torch.randn(4, 601)  # Logits
        target_value = torch.randn(4)

        loss = compute_value_loss(value, target_value, config=LossConfig(use_categorical=True))

        assert loss.dim() == 0

    def test_mse_output_is_scalar(self):
        """MSE value loss is a scalar tensor."""
        value = torch.randn(4)
        target_value = torch.randn(4)

        loss = compute_value_loss(value, target_value, config=LossConfig(use_categorical=False))

        assert loss.dim() == 0

    def test_categorical_identical_low_loss(self):
        """Categorical loss is low when prediction matches target."""
        target_value = torch.tensor([0.0])
        target_categorical = scalar_to_categorical(target_value, support_size=601)

        # Make prediction match target
        value = target_categorical.clone() + 0.01  # Add small noise to make logits

        loss = compute_value_loss(
            value * 100,  # Scale to get sharper distribution
            target_value,
            config=LossConfig(use_categorical=True, support_size=601),
        )

        # Loss should be relatively low
        assert loss.item() < 5.0

    def test_mse_identical_zero_loss(self):
        """MSE loss is zero when prediction equals target."""
        value = torch.tensor([1.0, 2.0, 3.0])
        target_value = value.clone()

        loss = compute_value_loss(value, target_value, config=LossConfig(use_categorical=False))

        assert loss.item() < 1e-6

    def test_mse_different_positive_loss(self):
        """MSE loss is positive when prediction differs from target."""
        value = torch.tensor([1.0, 2.0, 3.0])
        target_value = torch.tensor([4.0, 5.0, 6.0])

        loss = compute_value_loss(value, target_value, config=LossConfig(use_categorical=False))

        # MSE of diff = [3,3,3] is 9
        assert abs(loss.item() - 9.0) < 1e-5

    def test_gradient_flows_categorical(self):
        """Gradients flow through categorical value loss."""
        value = torch.randn(4, 601, requires_grad=True)
        target_value = torch.randn(4)

        loss = compute_value_loss(value, target_value, config=LossConfig(use_categorical=True))
        loss.backward()

        assert value.grad is not None

    def test_gradient_flows_mse(self):
        """Gradients flow through MSE value loss."""
        value = torch.randn(4, requires_grad=True)
        target_value = torch.randn(4)

        loss = compute_value_loss(value, target_value, config=LossConfig(use_categorical=False))
        loss.backward()

        assert value.grad is not None

    def test_with_config_parameters(self):
        """Config parameters are respected."""
        value = torch.randn(4, 101)  # Different support size
        target_value = torch.randn(4) * 50 - 25  # In range [-50, 50]

        config = LossConfig(
            use_categorical=True,
            support_size=101,
            min_value=-50.0,
            max_value=50.0,
        )

        loss = compute_value_loss(value, target_value, config=config)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestComputeRewardLoss:
    """Test suite for compute_reward_loss function."""

    def test_categorical_output_is_scalar(self):
        """Categorical reward loss is a scalar tensor."""
        reward = torch.randn(4, 601)
        target_reward = torch.randn(4)

        loss = compute_reward_loss(reward, target_reward, config=LossConfig(use_categorical=True))

        assert loss.dim() == 0

    def test_mse_output_is_scalar(self):
        """MSE reward loss is a scalar tensor."""
        reward = torch.randn(4)
        target_reward = torch.randn(4)

        loss = compute_reward_loss(reward, target_reward, config=LossConfig(use_categorical=False))

        assert loss.dim() == 0

    def test_identical_values_low_loss(self):
        """Loss is low when prediction matches target."""
        target_reward = torch.tensor([1.0])

        # Make prediction match
        reward = torch.tensor([1.0])

        loss = compute_reward_loss(reward, target_reward, config=LossConfig(use_categorical=False))

        assert loss.item() < 1e-6

    def test_gradient_flows(self):
        """Gradients flow through reward loss."""
        reward = torch.randn(4, 601, requires_grad=True)
        target_reward = torch.randn(4)

        loss = compute_reward_loss(reward, target_reward, config=LossConfig(use_categorical=True))
        loss.backward()

        assert reward.grad is not None


# =============================================================================
# Test EfficientZero Auxiliary Losses
# =============================================================================


class TestComputeConsistencyLoss:
    """Test suite for compute_consistency_loss function."""

    def test_output_is_scalar(self):
        """Consistency loss is a scalar tensor."""
        projected_dynamics = torch.randn(4, 256)
        projected_observation = torch.randn(4, 256)

        loss = compute_consistency_loss(projected_dynamics, projected_observation)

        assert loss.dim() == 0

    def test_identical_inputs_minimum_loss(self):
        """Loss is minimal (-1) for identical inputs."""
        projection = torch.randn(4, 256)
        target = projection.clone()

        loss = compute_consistency_loss(projection, target)

        # Negative cosine similarity of identical vectors is -1
        assert loss.item() < -0.99

    def test_orthogonal_inputs_zero_loss(self):
        """Loss is ~0 for orthogonal inputs."""
        # Create orthogonal vectors
        proj1 = torch.zeros(1, 4)
        proj1[0, 0] = 1.0
        proj2 = torch.zeros(1, 4)
        proj2[0, 1] = 1.0

        loss = compute_consistency_loss(proj1, proj2)

        # Cosine similarity of orthogonal vectors is 0
        assert abs(loss.item()) < 1e-5

    def test_opposite_inputs_maximum_loss(self):
        """Loss is maximal (+1) for opposite inputs."""
        projection = torch.randn(4, 256)
        target = -projection  # Opposite direction

        loss = compute_consistency_loss(projection, target)

        # Negative cosine similarity of opposite vectors is +1
        assert loss.item() > 0.99

    def test_loss_range(self):
        """Loss is in range [-1, 1]."""
        for _ in range(10):
            proj1 = torch.randn(8, 256)
            proj2 = torch.randn(8, 256)
            loss = compute_consistency_loss(proj1, proj2)

            assert -1.0 <= loss.item() <= 1.0

    def test_gradient_flows(self):
        """Gradients flow through consistency loss."""
        proj_dyn = torch.randn(4, 256, requires_grad=True)
        proj_obs = torch.randn(4, 256)  # Typically detached

        loss = compute_consistency_loss(proj_dyn, proj_obs)
        loss.backward()

        assert proj_dyn.grad is not None

    def test_invariant_to_scale(self):
        """Loss is invariant to input scale (due to normalization)."""
        proj1 = torch.randn(4, 256)
        proj2 = torch.randn(4, 256)

        loss1 = compute_consistency_loss(proj1, proj2)
        loss2 = compute_consistency_loss(proj1 * 10, proj2 * 0.1)

        assert torch.allclose(loss1, loss2, atol=1e-5)

    def test_reduction_none(self):
        """Reduction='none' returns per-sample loss."""
        proj1 = torch.randn(4, 256)
        proj2 = torch.randn(4, 256)

        loss = compute_consistency_loss(proj1, proj2, reduction="none")

        assert loss.shape == (4,)

    def test_reduction_sum(self):
        """Reduction='sum' returns sum of losses."""
        proj1 = torch.randn(4, 256)
        proj2 = torch.randn(4, 256)

        loss_none = compute_consistency_loss(proj1, proj2, reduction="none")
        loss_sum = compute_consistency_loss(proj1, proj2, reduction="sum")

        assert torch.allclose(loss_sum, loss_none.sum())


class TestComputeValuePrefixLoss:
    """Test suite for compute_value_prefix_loss function."""

    def test_categorical_output_is_scalar(self):
        """Categorical value prefix loss is a scalar tensor."""
        predicted = torch.randn(4, 601)
        target = torch.randn(4)

        loss = compute_value_prefix_loss(predicted, target, config=LossConfig(use_categorical=True))

        assert loss.dim() == 0

    def test_mse_output_is_scalar(self):
        """MSE value prefix loss is a scalar tensor."""
        predicted = torch.randn(4)
        target = torch.randn(4)

        loss = compute_value_prefix_loss(predicted, target, config=LossConfig(use_categorical=False))

        assert loss.dim() == 0

    def test_identical_values_low_loss(self):
        """Loss is low when prediction matches target."""
        target = torch.tensor([10.0])
        predicted = torch.tensor([10.0])

        loss = compute_value_prefix_loss(predicted, target, config=LossConfig(use_categorical=False))

        assert loss.item() < 1e-6

    def test_gradient_flows(self):
        """Gradients flow through value prefix loss."""
        predicted = torch.randn(4, 601, requires_grad=True)
        target = torch.randn(4)

        loss = compute_value_prefix_loss(predicted, target, config=LossConfig(use_categorical=True))
        loss.backward()

        assert predicted.grad is not None


# =============================================================================
# Test Combined Loss Functions
# =============================================================================


class TestComputeMuzeroStepLoss:
    """Test suite for compute_muzero_step_loss function."""

    def test_output_type(self):
        """Output is StepLossOutput named tuple."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert isinstance(result, StepLossOutput)

    def test_always_has_policy_and_value_loss(self):
        """Policy and value loss are always present."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert result.policy_loss is not None
        assert result.value_loss is not None
        assert result.policy_loss.dim() == 0
        assert result.value_loss.dim() == 0

    def test_reward_loss_optional(self):
        """Reward loss is None when reward not provided."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            reward=None,
            target_reward=None,
        )

        assert result.reward_loss is None

    def test_reward_loss_computed_when_provided(self):
        """Reward loss is computed when reward provided."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        reward = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)
        target_reward = torch.randn(4)

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            reward=reward,
            target_reward=target_reward,
        )

        assert result.reward_loss is not None
        assert result.reward_loss.dim() == 0

    def test_consistency_loss_optional(self):
        """Consistency loss is None when projections not provided."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert result.consistency_loss is None

    def test_consistency_loss_computed_when_provided(self):
        """Consistency loss is computed when projections provided."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)
        proj_dyn = torch.randn(4, 256)
        proj_obs = torch.randn(4, 256)

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            projected_dynamics=proj_dyn,
            projected_observation=proj_obs,
        )

        assert result.consistency_loss is not None
        assert result.consistency_loss.dim() == 0

    def test_value_prefix_loss_optional(self):
        """Value prefix loss is None when not provided."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert result.value_prefix_loss is None

    def test_value_prefix_loss_computed_when_provided(self):
        """Value prefix loss is computed when provided."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)
        pred_prefix = torch.randn(4, 601)
        target_prefix = torch.randn(4)

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            predicted_value_prefix=pred_prefix,
            target_value_prefix=target_prefix,
        )

        assert result.value_prefix_loss is not None
        assert result.value_prefix_loss.dim() == 0

    def test_total_loss_is_weighted_sum(self):
        """Total loss is weighted sum of individual losses."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        reward = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)
        target_reward = torch.randn(4)

        config = LossConfig(
            policy_loss_weight=1.0,
            value_loss_weight=0.5,
            reward_loss_weight=0.25,
        )

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            reward=reward,
            target_reward=target_reward,
            config=config,
        )

        expected_total = (
            1.0 * result.policy_loss
            + 0.5 * result.value_loss
            + 0.25 * result.reward_loss
        )

        assert torch.allclose(result.total_loss, expected_total, atol=1e-5)

    def test_gradient_flows_through_total_loss(self):
        """Gradients flow through total loss to all inputs."""
        policy_logits = torch.randn(4, 100, requires_grad=True)
        value = torch.randn(4, 601, requires_grad=True)
        reward = torch.randn(4, 601, requires_grad=True)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)
        target_reward = torch.randn(4)

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            reward=reward,
            target_reward=target_reward,
        )

        result.total_loss.backward()

        assert policy_logits.grad is not None
        assert value.grad is not None
        assert reward.grad is not None

    def test_with_action_mask(self):
        """Works correctly with action mask."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)
        action_mask = torch.ones(4, 100)
        action_mask[:, 50:] = 0  # Only first 50 actions valid

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            action_mask=action_mask,
        )

        assert not torch.isnan(result.total_loss)


class TestComputeMuzeroTrajectoryLoss:
    """Test suite for compute_muzero_trajectory_loss function."""

    @pytest.fixture
    def trajectory_data(self):
        """Create sample trajectory data for testing."""
        batch_size = 4
        action_size = 100
        support_size = 601
        num_unroll = 3  # K=3 unroll steps

        return {
            "initial_policy_logits": torch.randn(batch_size, action_size),
            "initial_value": torch.randn(batch_size, support_size),
            "target_policies": [
                torch.softmax(torch.randn(batch_size, action_size), dim=-1)
                for _ in range(num_unroll + 1)
            ],
            "target_values": [torch.randn(batch_size) for _ in range(num_unroll + 1)],
            "target_rewards": [torch.randn(batch_size) for _ in range(num_unroll)],
            "unrolled_policy_logits": [
                torch.randn(batch_size, action_size) for _ in range(num_unroll)
            ],
            "unrolled_values": [
                torch.randn(batch_size, support_size) for _ in range(num_unroll)
            ],
            "unrolled_rewards": [
                torch.randn(batch_size, support_size) for _ in range(num_unroll)
            ],
        }

    def test_output_type(self, trajectory_data):
        """Output is TrajectoryLossOutput named tuple."""
        result = compute_muzero_trajectory_loss(**trajectory_data)

        assert isinstance(result, TrajectoryLossOutput)

    def test_step_losses_count(self, trajectory_data):
        """Number of step losses equals num_unroll + 1."""
        result = compute_muzero_trajectory_loss(**trajectory_data)

        # 1 initial + 3 unroll = 4 steps
        assert len(result.step_losses) == 4

    def test_all_losses_are_scalars(self, trajectory_data):
        """All aggregated losses are scalar tensors."""
        result = compute_muzero_trajectory_loss(**trajectory_data)

        assert result.total_loss.dim() == 0
        assert result.policy_loss.dim() == 0
        assert result.value_loss.dim() == 0
        assert result.reward_loss.dim() == 0

    def test_initial_step_has_no_reward_loss(self, trajectory_data):
        """First step (initial inference) has no reward loss."""
        result = compute_muzero_trajectory_loss(**trajectory_data)

        assert result.step_losses[0].reward_loss is None

    def test_unroll_steps_have_reward_loss(self, trajectory_data):
        """Unroll steps (k >= 1) have reward loss."""
        result = compute_muzero_trajectory_loss(**trajectory_data)

        for i in range(1, len(result.step_losses)):
            assert result.step_losses[i].reward_loss is not None

    def test_total_loss_is_sum_of_step_losses(self, trajectory_data):
        """Total loss equals sum of step total losses."""
        result = compute_muzero_trajectory_loss(**trajectory_data)

        expected_total = sum(s.total_loss for s in result.step_losses)

        assert torch.allclose(result.total_loss, expected_total, atol=1e-5)

    def test_policy_loss_is_sum_across_steps(self, trajectory_data):
        """Policy loss equals sum of step policy losses."""
        result = compute_muzero_trajectory_loss(**trajectory_data)

        expected_policy = sum(s.policy_loss for s in result.step_losses)

        assert torch.allclose(result.policy_loss, expected_policy, atol=1e-5)

    def test_with_consistency_loss(self, trajectory_data):
        """Consistency loss is computed when projections provided."""
        num_unroll = len(trajectory_data["target_rewards"])
        batch_size = 4

        trajectory_data["projected_dynamics_list"] = [
            torch.randn(batch_size, 256) for _ in range(num_unroll)
        ]
        trajectory_data["projected_observation_list"] = [
            torch.randn(batch_size, 256) for _ in range(num_unroll)
        ]

        result = compute_muzero_trajectory_loss(**trajectory_data)

        assert result.consistency_loss is not None
        assert result.consistency_loss.dim() == 0

    def test_with_value_prefix_loss(self, trajectory_data):
        """Value prefix loss is computed when prefixes provided."""
        num_unroll = len(trajectory_data["target_rewards"])
        batch_size = 4

        trajectory_data["predicted_value_prefixes"] = [
            torch.randn(batch_size, 601) for _ in range(num_unroll + 1)
        ]
        trajectory_data["target_value_prefixes"] = [
            torch.randn(batch_size) for _ in range(num_unroll + 1)
        ]

        result = compute_muzero_trajectory_loss(**trajectory_data)

        assert result.value_prefix_loss is not None
        assert result.value_prefix_loss.dim() == 0

    def test_with_action_masks(self, trajectory_data):
        """Works correctly with action masks for all steps."""
        num_unroll = len(trajectory_data["target_rewards"])
        batch_size = 4
        action_size = 100

        trajectory_data["action_masks"] = [
            torch.ones(batch_size, action_size) for _ in range(num_unroll + 1)
        ]

        result = compute_muzero_trajectory_loss(**trajectory_data)

        assert not torch.isnan(result.total_loss)

    def test_gradient_flows(self, trajectory_data):
        """Gradients flow through trajectory loss."""
        # Make tensors require grad
        trajectory_data["initial_policy_logits"] = trajectory_data[
            "initial_policy_logits"
        ].requires_grad_(True)
        trajectory_data["initial_value"] = trajectory_data["initial_value"].requires_grad_(
            True
        )

        for i in range(len(trajectory_data["unrolled_policy_logits"])):
            trajectory_data["unrolled_policy_logits"][i] = trajectory_data[
                "unrolled_policy_logits"
            ][i].requires_grad_(True)
            trajectory_data["unrolled_values"][i] = trajectory_data["unrolled_values"][
                i
            ].requires_grad_(True)
            trajectory_data["unrolled_rewards"][i] = trajectory_data["unrolled_rewards"][
                i
            ].requires_grad_(True)

        result = compute_muzero_trajectory_loss(**trajectory_data)
        result.total_loss.backward()

        assert trajectory_data["initial_policy_logits"].grad is not None
        assert trajectory_data["initial_value"].grad is not None


# =============================================================================
# Test MuZeroLossCalculator Class
# =============================================================================


class MockMuZeroNetwork(nn.Module):
    """Mock MuZero network for testing the loss calculator."""

    def __init__(
        self,
        spatial_channels: int = 13,
        spatial_size: int = 16,
        vector_dim: int = 64,
        action_size: int = 100,
        hidden_channels: int = 64,
        support_size: int = 601,
        projection_dim: int = 256,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.hidden_channels = hidden_channels
        self.support_size = support_size
        self.action_size = action_size
        self.projection_dim = projection_dim

        # Simple representation
        self.repr_conv = nn.Conv2d(spatial_channels, hidden_channels, 3, padding=1)

        # Simple dynamics
        self.dyn_conv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.reward_fc = nn.Linear(hidden_channels * spatial_size * spatial_size, support_size)

        # Simple prediction
        self.policy_fc = nn.Linear(hidden_channels * spatial_size * spatial_size, action_size)
        self.value_fc = nn.Linear(hidden_channels * spatial_size * spatial_size, support_size)

        # Projection for consistency loss
        self.proj_fc = nn.Linear(hidden_channels * spatial_size * spatial_size, projection_dim)

        # Value prefix prediction
        self.prefix_fc = nn.Linear(hidden_channels * spatial_size * spatial_size, support_size)

    def representation(self, spatial, vector):
        return self.repr_conv(spatial)

    def initial_inference(self, spatial, vector, action_mask=None):
        hidden = self.representation(spatial, vector)
        policy_logits, value = self._predict(hidden)
        return hidden, policy_logits, value

    def recurrent_inference(self, hidden_state, action, action_mask=None):
        next_hidden = self.dyn_conv(hidden_state)
        flat = next_hidden.view(next_hidden.size(0), -1)
        reward = self.reward_fc(flat)
        policy_logits, value = self._predict(next_hidden)
        return next_hidden, reward, policy_logits, value

    def _predict(self, hidden):
        flat = hidden.view(hidden.size(0), -1)
        policy = self.policy_fc(flat)
        value = self.value_fc(flat)
        return policy, value

    def project(self, hidden_state):
        flat = hidden_state.view(hidden_state.size(0), -1)
        return self.proj_fc(flat)

    def predict_value_prefix(self, hidden_state):
        flat = hidden_state.view(hidden_state.size(0), -1)
        return self.prefix_fc(flat)


class MockAlphaZeroNetwork(nn.Module):
    """Mock AlphaZero network for testing backward compatibility."""

    def __init__(
        self,
        spatial_channels: int = 13,
        spatial_size: int = 16,
        vector_dim: int = 64,
        action_size: int = 100,
    ):
        super().__init__()
        self.conv = nn.Conv2d(spatial_channels, 64, 3, padding=1)
        flat_dim = 64 * spatial_size * spatial_size
        self.policy_fc = nn.Linear(flat_dim, action_size)
        self.value_fc = nn.Linear(flat_dim, 1)

    def forward(self, spatial, vector, action_mask=None):
        x = self.conv(spatial)
        x = x.view(x.size(0), -1)
        policy = self.policy_fc(x)
        value = self.value_fc(x).squeeze(-1)
        return policy, value


class TestMuZeroLossCalculator:
    """Test suite for MuZeroLossCalculator class."""

    @pytest.fixture
    def loss_calculator(self):
        """Create a loss calculator for testing."""
        config = LossConfig(
            use_categorical=True,
            support_size=601,
        )
        return MuZeroLossCalculator(
            config=config,
            num_unroll_steps=3,
            use_consistency_loss=True,
            use_value_prefix=True,
        )

    @pytest.fixture
    def muzero_network(self):
        """Create a mock MuZero network."""
        return MockMuZeroNetwork()

    @pytest.fixture
    def alphazero_network(self):
        """Create a mock AlphaZero network."""
        return MockAlphaZeroNetwork()

    @pytest.fixture
    def trajectory_batch(self):
        """Create a sample trajectory batch."""
        batch_size = 4
        num_unroll = 3
        spatial_channels = 13
        spatial_size = 16
        vector_dim = 64
        action_size = 100

        return TrajectoryBatch(
            observations_spatial=[
                torch.randn(batch_size, spatial_channels, spatial_size, spatial_size)
                for _ in range(num_unroll + 1)
            ],
            observations_vector=[
                torch.randn(batch_size, vector_dim)
                for _ in range(num_unroll + 1)
            ],
            actions=[
                torch.randint(0, action_size, (batch_size,))
                for _ in range(num_unroll)
            ],
            target_policies=[
                torch.softmax(torch.randn(batch_size, action_size), dim=-1)
                for _ in range(num_unroll + 1)
            ],
            target_values=[
                torch.randn(batch_size) for _ in range(num_unroll + 1)
            ],
            target_rewards=[
                torch.randn(batch_size) for _ in range(num_unroll)
            ],
        )

    def test_initialization(self, loss_calculator):
        """Calculator initializes with correct parameters."""
        assert loss_calculator.num_unroll_steps == 3
        assert loss_calculator.use_consistency_loss is True
        assert loss_calculator.use_value_prefix is True
        assert loss_calculator.config.use_categorical is True

    def test_compute_loss_muzero_network(self, loss_calculator, muzero_network, trajectory_batch):
        """Loss computation works with MuZero network."""
        result = loss_calculator.compute_loss(muzero_network, trajectory_batch)

        assert isinstance(result, TrajectoryLossOutput)
        assert result.total_loss.dim() == 0
        assert not torch.isnan(result.total_loss)

    def test_compute_loss_alphazero_network(self, alphazero_network, trajectory_batch):
        """Loss computation works with AlphaZero network."""
        # AlphaZero doesn't use consistency or value prefix
        calculator = MuZeroLossCalculator(
            config=LossConfig(use_categorical=False),
            use_consistency_loss=False,
            use_value_prefix=False,
        )

        result = calculator.compute_loss(alphazero_network, trajectory_batch)

        assert isinstance(result, TrajectoryLossOutput)
        assert result.total_loss.dim() == 0
        assert not torch.isnan(result.total_loss)

    def test_gradient_flows_muzero(self, loss_calculator, muzero_network, trajectory_batch):
        """Gradients flow through MuZero loss computation."""
        muzero_network.train()

        result = loss_calculator.compute_loss(muzero_network, trajectory_batch)
        result.total_loss.backward()

        # Check that some parameters have gradients
        has_grad = False
        for param in muzero_network.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_gradient_flows_alphazero(self, alphazero_network, trajectory_batch):
        """Gradients flow through AlphaZero loss computation."""
        calculator = MuZeroLossCalculator(
            config=LossConfig(use_categorical=False),
            use_consistency_loss=False,
            use_value_prefix=False,
        )

        alphazero_network.train()
        result = calculator.compute_loss(alphazero_network, trajectory_batch)
        result.total_loss.backward()

        # Check that some parameters have gradients
        has_grad = False
        for param in alphazero_network.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_consistency_loss_computed(self, loss_calculator, muzero_network, trajectory_batch):
        """Consistency loss is computed when enabled."""
        result = loss_calculator.compute_loss(muzero_network, trajectory_batch)

        assert result.consistency_loss is not None

    def test_no_consistency_loss_when_disabled(self, muzero_network, trajectory_batch):
        """Consistency loss is None when disabled."""
        calculator = MuZeroLossCalculator(
            config=LossConfig(),
            use_consistency_loss=False,
            use_value_prefix=False,
        )

        result = calculator.compute_loss(muzero_network, trajectory_batch)

        assert result.consistency_loss is None

    def test_step_losses_match_unroll(self, loss_calculator, muzero_network, trajectory_batch):
        """Number of step losses matches unroll + 1."""
        result = loss_calculator.compute_loss(muzero_network, trajectory_batch)

        # 1 initial + 3 unroll = 4 steps
        assert len(result.step_losses) == 4

    def test_with_action_masks(self, loss_calculator, muzero_network):
        """Works correctly with action masks in batch."""
        batch_size = 4
        num_unroll = 3
        spatial_channels = 13
        spatial_size = 16
        vector_dim = 64
        action_size = 100

        batch = TrajectoryBatch(
            observations_spatial=[
                torch.randn(batch_size, spatial_channels, spatial_size, spatial_size)
                for _ in range(num_unroll + 1)
            ],
            observations_vector=[
                torch.randn(batch_size, vector_dim)
                for _ in range(num_unroll + 1)
            ],
            actions=[
                torch.randint(0, action_size, (batch_size,))
                for _ in range(num_unroll)
            ],
            target_policies=[
                torch.softmax(torch.randn(batch_size, action_size), dim=-1)
                for _ in range(num_unroll + 1)
            ],
            target_values=[
                torch.randn(batch_size) for _ in range(num_unroll + 1)
            ],
            target_rewards=[
                torch.randn(batch_size) for _ in range(num_unroll)
            ],
            action_masks=[
                torch.ones(batch_size, action_size)
                for _ in range(num_unroll + 1)
            ],
        )

        result = loss_calculator.compute_loss(muzero_network, batch)

        assert not torch.isnan(result.total_loss)


# =============================================================================
# Test Backward Compatibility
# =============================================================================


class TestComputeMuzeroLossBackwardCompat:
    """Test compute_muzero_loss for backward compatibility."""

    def test_returns_dict(self):
        """Returns dictionary of losses."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)

        losses = compute_muzero_loss(
            policy_logits=policy_logits,
            value=value,
            reward=None,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert isinstance(losses, dict)

    def test_has_policy_and_value_loss(self):
        """Dictionary contains policy_loss and value_loss."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)

        losses = compute_muzero_loss(
            policy_logits=policy_logits,
            value=value,
            reward=None,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert "policy_loss" in losses
        assert "value_loss" in losses

    def test_reward_loss_when_provided(self):
        """Dictionary contains reward_loss when reward provided."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        reward = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)
        target_reward = torch.randn(4)

        losses = compute_muzero_loss(
            policy_logits=policy_logits,
            value=value,
            reward=reward,
            target_policy=target_policy,
            target_value=target_value,
            target_reward=target_reward,
        )

        assert "reward_loss" in losses

    def test_no_reward_loss_when_none(self):
        """Dictionary does not contain reward_loss when reward is None."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)

        losses = compute_muzero_loss(
            policy_logits=policy_logits,
            value=value,
            reward=None,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert "reward_loss" not in losses

    def test_with_action_mask(self):
        """Works with action mask."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)
        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)
        action_mask = torch.ones(4, 100)

        losses = compute_muzero_loss(
            policy_logits=policy_logits,
            value=value,
            reward=None,
            target_policy=target_policy,
            target_value=target_value,
            action_mask=action_mask,
        )

        assert "policy_loss" in losses


# =============================================================================
# Test LossConfig
# =============================================================================


class TestLossConfig:
    """Test suite for LossConfig dataclass."""

    def test_default_values(self):
        """Default config has expected values."""
        config = LossConfig()

        assert config.use_categorical is True
        assert config.support_size == 601
        assert config.min_value == -300.0
        assert config.max_value == 300.0
        assert config.policy_loss_weight == 1.0
        assert config.value_loss_weight == 1.0
        assert config.reward_loss_weight == 1.0
        assert config.consistency_loss_weight == 2.0
        assert config.value_prefix_loss_weight == 1.0

    def test_custom_values(self):
        """Config accepts custom values."""
        config = LossConfig(
            use_categorical=False,
            support_size=301,
            policy_loss_weight=2.0,
            consistency_loss_weight=0.5,
        )

        assert config.use_categorical is False
        assert config.support_size == 301
        assert config.policy_loss_weight == 2.0
        assert config.consistency_loss_weight == 0.5


# =============================================================================
# Test Edge Cases and Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Test numerical stability of loss functions."""

    def test_policy_loss_with_zero_target(self):
        """Policy loss handles zero target probabilities."""
        policy_logits = torch.randn(4, 100)
        target_policy = torch.zeros(4, 100)
        target_policy[:, 0] = 1.0  # All mass on first action

        loss = compute_policy_loss(policy_logits, target_policy)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_consistency_loss_with_zero_vectors(self):
        """Consistency loss handles near-zero vectors."""
        # Vectors that are almost zero (will be normalized)
        proj1 = torch.ones(4, 256) * 1e-10
        proj2 = torch.ones(4, 256) * 1e-10

        loss = compute_consistency_loss(proj1, proj2)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_value_loss_with_extreme_targets(self):
        """Value loss handles extreme target values."""
        value = torch.randn(4, 601)
        target_value = torch.tensor([300.0, -300.0, 0.0, 150.0])  # Boundary values

        loss = compute_value_loss(value, target_value, config=LossConfig(use_categorical=True))

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_large_batch_size(self):
        """Loss functions handle large batch sizes."""
        batch_size = 1024
        policy_logits = torch.randn(batch_size, 100)
        value = torch.randn(batch_size, 601)
        target_policy = torch.softmax(torch.randn(batch_size, 100), dim=-1)
        target_value = torch.randn(batch_size)

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert not torch.isnan(result.total_loss)
        assert not torch.isinf(result.total_loss)

    def test_empty_mask_handling(self):
        """Policy loss handles edge case where mask zeros all actions."""
        policy_logits = torch.randn(1, 100)
        target_policy = torch.softmax(torch.randn(1, 100), dim=-1)
        # All zeros mask - edge case
        action_mask = torch.zeros(1, 100)

        # Should not crash (renormalization handles this)
        loss = compute_policy_loss(policy_logits, target_policy, action_mask)

        # Loss might be nan/inf due to edge case, but should not crash
        assert loss is not None

    def test_very_small_support_size(self):
        """Loss works with small support sizes."""
        value = torch.randn(4, 5)  # Small support
        target_value = torch.randn(4) * 2 - 1  # In range [-1, 1]

        config = LossConfig(
            use_categorical=True,
            support_size=5,
            min_value=-2.0,
            max_value=2.0,
        )

        loss = compute_value_loss(value, target_value, config=config)

        assert not torch.isnan(loss)


# =============================================================================
# Test Logging
# =============================================================================


class TestLogging:
    """Test that logging is properly configured."""

    def test_logging_debug_messages(self, caplog):
        """Debug logging works without errors."""
        with caplog.at_level(logging.DEBUG, logger="alpha_zero.core.losses"):
            scalars = torch.tensor([0.0])
            scalar_to_categorical(scalars)

        # Check that some log messages were generated
        # (Exact messages depend on log level and handlers)

    def test_logging_info_messages(self, caplog):
        """Info logging for trajectory loss computation."""
        with caplog.at_level(logging.INFO, logger="alpha_zero.core.losses"):
            batch_size = 2
            action_size = 10
            support_size = 11

            trajectory_data = {
                "initial_policy_logits": torch.randn(batch_size, action_size),
                "initial_value": torch.randn(batch_size, support_size),
                "target_policies": [
                    torch.softmax(torch.randn(batch_size, action_size), dim=-1)
                    for _ in range(2)
                ],
                "target_values": [torch.randn(batch_size) for _ in range(2)],
                "target_rewards": [torch.randn(batch_size)],
                "unrolled_policy_logits": [torch.randn(batch_size, action_size)],
                "unrolled_values": [torch.randn(batch_size, support_size)],
                "unrolled_rewards": [torch.randn(batch_size, support_size)],
            }

            config = LossConfig(support_size=support_size, min_value=-5, max_value=5)
            compute_muzero_trajectory_loss(**trajectory_data, config=config)


# =============================================================================
# Test Device Compatibility
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDACompatibility:
    """Test that loss functions work on CUDA devices."""

    def test_scalar_to_categorical_cuda(self):
        """scalar_to_categorical works on CUDA."""
        scalars = torch.tensor([0.0, 10.0], device="cuda")
        categorical = scalar_to_categorical(scalars, support_size=601)

        assert categorical.device.type == "cuda"

    def test_policy_loss_cuda(self):
        """Policy loss works on CUDA."""
        policy_logits = torch.randn(4, 100, device="cuda")
        target_policy = torch.softmax(torch.randn(4, 100, device="cuda"), dim=-1)

        loss = compute_policy_loss(policy_logits, target_policy)

        assert loss.device.type == "cuda"

    def test_step_loss_cuda(self):
        """Step loss works on CUDA."""
        policy_logits = torch.randn(4, 100, device="cuda")
        value = torch.randn(4, 601, device="cuda")
        target_policy = torch.softmax(torch.randn(4, 100, device="cuda"), dim=-1)
        target_value = torch.randn(4, device="cuda")

        result = compute_muzero_step_loss(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert result.total_loss.device.type == "cuda"

    def test_muzero_loss_calculator_cuda(self):
        """MuZeroLossCalculator works on CUDA."""
        device = "cuda"
        batch_size = 4
        num_unroll = 2
        spatial_channels = 13
        spatial_size = 16
        vector_dim = 64
        action_size = 100

        calculator = MuZeroLossCalculator(
            config=LossConfig(),
            num_unroll_steps=num_unroll,
            use_consistency_loss=False,
            use_value_prefix=False,
        )

        network = MockMuZeroNetwork().to(device)

        batch = TrajectoryBatch(
            observations_spatial=[
                torch.randn(batch_size, spatial_channels, spatial_size, spatial_size, device=device)
                for _ in range(num_unroll + 1)
            ],
            observations_vector=[
                torch.randn(batch_size, vector_dim, device=device)
                for _ in range(num_unroll + 1)
            ],
            actions=[
                torch.randint(0, action_size, (batch_size,), device=device)
                for _ in range(num_unroll)
            ],
            target_policies=[
                torch.softmax(torch.randn(batch_size, action_size, device=device), dim=-1)
                for _ in range(num_unroll + 1)
            ],
            target_values=[
                torch.randn(batch_size, device=device) for _ in range(num_unroll + 1)
            ],
            target_rewards=[
                torch.randn(batch_size, device=device) for _ in range(num_unroll)
            ],
        )

        result = calculator.compute_loss(network, batch)

        assert result.total_loss.device.type == "cuda"


# =============================================================================
# Test TrajectoryBatch
# =============================================================================


class TestTrajectoryBatch:
    """Test suite for TrajectoryBatch dataclass."""

    def test_creation(self):
        """TrajectoryBatch can be created with required fields."""
        batch = TrajectoryBatch(
            observations_spatial=[torch.randn(4, 3, 8, 8)],
            observations_vector=[torch.randn(4, 16)],
            actions=[],
            target_policies=[torch.randn(4, 10)],
            target_values=[torch.randn(4)],
            target_rewards=[],
        )

        assert len(batch.observations_spatial) == 1
        assert len(batch.observations_vector) == 1

    def test_optional_fields(self):
        """Optional fields default to None."""
        batch = TrajectoryBatch(
            observations_spatial=[torch.randn(4, 3, 8, 8)],
            observations_vector=[torch.randn(4, 16)],
            actions=[],
            target_policies=[torch.randn(4, 10)],
            target_values=[torch.randn(4)],
            target_rewards=[],
        )

        assert batch.action_masks is None
        assert batch.target_value_prefixes is None

    def test_with_optional_fields(self):
        """Optional fields can be set."""
        batch = TrajectoryBatch(
            observations_spatial=[torch.randn(4, 3, 8, 8)],
            observations_vector=[torch.randn(4, 16)],
            actions=[],
            target_policies=[torch.randn(4, 10)],
            target_values=[torch.randn(4)],
            target_rewards=[],
            action_masks=[torch.ones(4, 10)],
            target_value_prefixes=[torch.randn(4)],
        )

        assert batch.action_masks is not None
        assert batch.target_value_prefixes is not None
