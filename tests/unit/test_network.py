"""Unit tests for neural network."""

import pytest
import torch

from alpha_zero.core.civ6_network import (
    Civ6Network,
    MicroCivNetwork,
    NetworkConfig,
    ResidualBlock,
    compute_loss,
)


class TestResidualBlock:
    """Test suite for residual block."""

    def test_output_shape(self):
        """Residual block preserves shape."""
        block = ResidualBlock(num_filters=64)
        x = torch.randn(4, 64, 16, 16)
        y = block(x)

        assert y.shape == x.shape

    def test_residual_connection(self):
        """Output includes residual connection."""
        block = ResidualBlock(num_filters=64)

        # Zero out weights to test residual
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()

        x = torch.randn(4, 64, 16, 16)
        y = block(x)

        # Should be close to ReLU(x) due to residual
        assert torch.allclose(y, torch.relu(x), atol=1e-5)


class TestMicroCivNetwork:
    """Test suite for MicroCiv network."""

    @pytest.fixture
    def network(self):
        return MicroCivNetwork(
            board_size=16,
            action_space_size=381,
            num_res_blocks=2,
            num_filters=32,
        )

    def test_forward_shapes(self, network):
        """Forward pass produces correct output shapes."""
        batch_size = 4
        spatial = torch.randn(batch_size, 13, 16, 16)
        vector = torch.randn(batch_size, 64)

        policy_logits, value = network(spatial, vector)

        assert policy_logits.shape == (batch_size, 381)
        assert value.shape == (batch_size,)

    def test_value_range(self, network):
        """Value is in [-1, 1] range (tanh)."""
        spatial = torch.randn(10, 13, 16, 16)
        vector = torch.randn(10, 64)

        _, value = network(spatial, vector)

        assert torch.all(value >= -1)
        assert torch.all(value <= 1)

    def test_action_masking(self, network):
        """Action mask sets invalid actions to low probability."""
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)
        mask = torch.zeros(2, 381)
        mask[:, :10] = 1.0  # Only first 10 actions valid

        policy_logits, _ = network(spatial, vector, mask)

        # Invalid actions should have very negative logits
        assert torch.all(policy_logits[:, 10:] < -1e8)

    def test_predict_returns_probs(self, network):
        """Predict method returns probabilities."""
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)

        probs, _ = network.predict(spatial, vector)

        # Should be valid probability distribution
        assert torch.all(probs >= 0)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_predict_single(self, network):
        """predict_single works without batch dimension."""
        spatial = torch.randn(13, 16, 16)
        vector = torch.randn(64)
        mask = torch.ones(381)
        mask[100:] = 0

        probs, value = network.predict_single(spatial, vector, mask)

        assert probs.shape == (381,)
        assert isinstance(value, float)
        assert -1 <= value <= 1

    def test_gradient_flow(self, network):
        """Gradients flow through network."""
        spatial = torch.randn(2, 13, 16, 16, requires_grad=True)
        vector = torch.randn(2, 64, requires_grad=True)

        policy_logits, value = network(spatial, vector)
        loss = policy_logits.sum() + value.sum()
        loss.backward()

        assert spatial.grad is not None
        assert vector.grad is not None

    def test_save_load(self, network, tmp_path):
        """Network can be saved and loaded."""
        path = tmp_path / "test_model.pt"
        network.save(str(path))

        loaded = MicroCivNetwork.load(str(path))

        # Check weights match
        for (n1, p1), (n2, p2) in zip(
            network.named_parameters(), loaded.named_parameters(), strict=True
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)


class TestCiv6Network:
    """Test suite for full Civ6 network."""

    @pytest.fixture
    def network(self):
        config = NetworkConfig(
            spatial_channels=84,
            spatial_size=44,
            vector_dim=512,
            num_res_blocks=4,
            num_filters=64,
            action_space_size=10000,
        )
        return Civ6Network(config)

    def test_large_action_space(self, network):
        """Network handles large action space."""
        spatial = torch.randn(2, 84, 44, 44)
        vector = torch.randn(2, 512)

        policy_logits, value = network(spatial, vector)

        assert policy_logits.shape == (2, 10000)

    def test_config_override(self):
        """Kwargs can override config."""
        network = Civ6Network(num_filters=128, spatial_channels=13, spatial_size=16)

        assert network.config.num_filters == 128
        assert network.config.spatial_channels == 13


class TestComputeLoss:
    """Test suite for loss computation."""

    def test_loss_shapes(self):
        """Loss computation produces scalar outputs."""
        policy_logits = torch.randn(8, 100)
        value = torch.randn(8)
        target_policy = torch.softmax(torch.randn(8, 100), dim=-1)
        target_value = torch.randn(8)

        total, policy, value_loss = compute_loss(policy_logits, value, target_policy, target_value)

        assert total.shape == ()
        assert policy.shape == ()
        assert value_loss.shape == ()

    def test_loss_positive(self):
        """Losses are non-negative."""
        policy_logits = torch.randn(8, 100)
        value = torch.randn(8)
        target_policy = torch.softmax(torch.randn(8, 100), dim=-1)
        target_value = torch.randn(8)

        total, policy, value_loss = compute_loss(policy_logits, value, target_policy, target_value)

        assert total >= 0
        assert value_loss >= 0
        # Policy loss can be negative (cross-entropy)

    def test_loss_with_mask(self):
        """Loss computation respects action mask."""
        policy_logits = torch.randn(4, 50)
        value = torch.randn(4)
        target_policy = torch.zeros(4, 50)
        target_policy[:, :10] = 0.1  # Only first 10 valid
        target_value = torch.randn(4)
        mask = torch.zeros(4, 50)
        mask[:, :10] = 1.0

        total, _, _ = compute_loss(policy_logits, value, target_policy, target_value, mask)

        assert not torch.isnan(total)

    def test_perfect_prediction_low_loss(self):
        """Perfect predictions have low loss."""
        # Create "perfect" prediction
        target_policy = torch.zeros(4, 20)
        target_policy[:, 5] = 1.0  # All mass on action 5

        policy_logits = torch.zeros(4, 20)
        policy_logits[:, 5] = 100.0  # High logit for action 5

        target_value = torch.tensor([0.5, -0.5, 0.0, 1.0])
        value = target_value.clone()

        total, policy, value_loss = compute_loss(policy_logits, value, target_policy, target_value)

        assert value_loss < 1e-6
        assert total < 1.0  # Should be small
