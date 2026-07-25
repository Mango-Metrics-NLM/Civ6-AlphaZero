"""Tests for N-player neural network heads."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from alpha_zero.multiplayer.network_heads import (
    NPlayerNetworkAdapter,
    NPlayerNetworkConfig,
    NPlayerValueHead,
    compute_nplayer_value_loss,
)


class TestNPlayerValueHead:
    """Tests for NPlayerValueHead."""

    @pytest.fixture
    def head_2p(self):
        return NPlayerValueHead(spatial_filters=64, spatial_size=16, vector_dim=256, num_players=2)

    @pytest.fixture
    def head_4p(self):
        return NPlayerValueHead(spatial_filters=64, spatial_size=16, vector_dim=256, num_players=4)

    @pytest.fixture
    def spatial_features(self):
        return torch.randn(4, 64, 16, 16)

    @pytest.fixture
    def vector_features(self):
        return torch.randn(4, 256)

    def test_output_shape_2_players(self, head_2p, spatial_features, vector_features):
        """2-player head outputs (B, 2)."""
        output = head_2p(spatial_features, vector_features)
        assert output.shape == (4, 2)

    def test_output_shape_4_players(self, head_4p, spatial_features, vector_features):
        """4-player head outputs (B, 4)."""
        output = head_4p(spatial_features, vector_features)
        assert output.shape == (4, 4)

    def test_output_range(self, head_4p, spatial_features, vector_features):
        """All values are in [-1, 1] due to tanh."""
        output = head_4p(spatial_features, vector_features)
        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)

    def test_num_players_attribute(self, head_4p):
        """num_players is stored correctly."""
        assert head_4p.num_players == 4

    def test_batch_size_1(self):
        """Works with batch size 1."""
        head = NPlayerValueHead(spatial_filters=32, spatial_size=8, vector_dim=64, num_players=3)
        spatial = torch.randn(1, 32, 8, 8)
        vector = torch.randn(1, 64)
        output = head(spatial, vector)
        assert output.shape == (1, 3)

    def test_different_spatial_sizes(self):
        """Works with various board sizes."""
        for size in [4, 8, 16, 32]:
            head = NPlayerValueHead(
                spatial_filters=32, spatial_size=size, vector_dim=64, num_players=2
            )
            spatial = torch.randn(2, 32, size, size)
            vector = torch.randn(2, 64)
            output = head(spatial, vector)
            assert output.shape == (2, 2)

    def test_gradient_flow(self, head_4p, spatial_features, vector_features):
        """Gradients flow through the value head."""
        output = head_4p(spatial_features, vector_features)
        loss = output.sum()
        loss.backward()

        for param in head_4p.parameters():
            assert param.grad is not None

    def test_single_player_degenerate_case(self):
        """num_players=1 works (single-player evaluation)."""
        head = NPlayerValueHead(spatial_filters=32, spatial_size=8, vector_dim=64, num_players=1)
        spatial = torch.randn(2, 32, 8, 8)
        vector = torch.randn(2, 64)
        output = head(spatial, vector)
        assert output.shape == (2, 1)

    def test_large_player_count(self):
        """Works with 8 players (Civ6 max)."""
        head = NPlayerValueHead(spatial_filters=32, spatial_size=8, vector_dim=64, num_players=8)
        spatial = torch.randn(2, 32, 8, 8)
        vector = torch.randn(2, 64)
        output = head(spatial, vector)
        assert output.shape == (2, 8)


class TestComputeNPlayerValueLoss:
    """Tests for compute_nplayer_value_loss."""

    def test_scalar_inputs(self):
        """Handles legacy scalar (B,) inputs."""
        pred = torch.tensor([0.5, -0.5])
        target = torch.tensor([1.0, -1.0])
        loss = compute_nplayer_value_loss(pred, target, num_players=2)
        assert loss.item() > 0

    def test_vector_inputs(self):
        """Handles (B, N) inputs."""
        pred = torch.tensor([[0.5, -0.5], [0.0, 0.0]])
        target = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
        loss = compute_nplayer_value_loss(pred, target, num_players=2)
        assert loss.item() > 0

    def test_perfect_prediction_zero_loss(self):
        """Perfect predictions produce zero loss."""
        pred = torch.tensor([[1.0, -1.0, 0.0]])
        target = torch.tensor([[1.0, -1.0, 0.0]])
        loss = compute_nplayer_value_loss(pred, target, num_players=3)
        assert loss.item() == pytest.approx(0.0)

    def test_loss_increases_with_error(self):
        """Larger errors produce larger losses."""
        target = torch.tensor([[1.0, -1.0]])
        small_error = torch.tensor([[0.9, -0.9]])
        large_error = torch.tensor([[0.0, 0.0]])

        loss_small = compute_nplayer_value_loss(small_error, target, num_players=2)
        loss_large = compute_nplayer_value_loss(large_error, target, num_players=2)
        assert loss_large.item() > loss_small.item()

    def test_reduction_none(self):
        """reduction='none' returns per-element losses."""
        pred = torch.tensor([[0.5, -0.5]])
        target = torch.tensor([[1.0, -1.0]])
        loss = compute_nplayer_value_loss(pred, target, num_players=2, reduction="none")
        assert loss.shape == (1, 2)


class TestNPlayerNetworkAdapter:
    """Tests for NPlayerNetworkAdapter."""

    @pytest.fixture
    def base_network_with_config(self):
        """Create a mock base network with config attribute."""

        class MockConfig:
            num_filters = 32
            spatial_size = 8
            fc_hidden_dim = 64

        class MockNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.linear = nn.Linear(10, 10)  # Dummy param

            def forward(self, spatial, vector, action_mask=None):
                b = spatial.shape[0]
                return torch.randn(b, 10), torch.randn(b, 1)

        return MockNetwork()

    @pytest.fixture
    def base_network_no_config(self):
        """Create a mock base network without config attribute."""

        class MockNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, spatial, vector, action_mask=None):
                b = spatial.shape[0]
                return torch.randn(b, 10), torch.randn(b, 1)

        return MockNetwork()

    def test_adapter_creation_with_config(self, base_network_with_config):
        """Adapter created from network with config attribute."""
        adapter = NPlayerNetworkAdapter(base_network_with_config, num_players=4)
        assert adapter.num_players == 4
        assert hasattr(adapter, "nplayer_value_head")

    def test_adapter_creation_no_config(self, base_network_no_config):
        """Adapter falls back to defaults when no config attribute."""
        adapter = NPlayerNetworkAdapter(base_network_no_config, num_players=3)
        assert adapter.num_players == 3

    def test_adapter_freeze_encoder(self, base_network_with_config):
        """Freeze encoder freezes non-value-head parameters."""
        adapter = NPlayerNetworkAdapter(
            base_network_with_config, num_players=2, freeze_encoder=True
        )
        # Base network params should be frozen
        for name, param in adapter.base_network.named_parameters():
            if "value_head" not in name:
                assert not param.requires_grad

    def test_adapter_forward_fallback(self, base_network_no_config):
        """Forward with fallback path (no spatial_encoder/vector_encoder)."""
        adapter = NPlayerNetworkAdapter(base_network_no_config, num_players=3)
        # Default spatial_size=16, vector_dim=256 from fallback defaults
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 256)
        _policy, values = adapter(spatial, vector)
        assert values.shape[0] == 2
        assert values.shape[1] == 3

    def test_adapter_forward_with_action_mask(self, base_network_no_config):
        """Forward with action mask."""
        adapter = NPlayerNetworkAdapter(base_network_no_config, num_players=2)
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 256)
        mask = torch.ones(2, 10)
        _policy, values = adapter(spatial, vector, action_mask=mask)
        assert values.shape == (2, 2)

    def test_adapter_forward_with_encoders(self):
        """Forward with spatial_encoder and vector_encoder attributes."""

        class MockNetworkWithEncoders(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type(
                    "C", (), {"num_filters": 32, "spatial_size": 8, "fc_hidden_dim": 64}
                )()
                self.spatial_encoder = nn.Conv2d(13, 32, 3, padding=1)
                self.vector_encoder = nn.Linear(64, 64)
                self.policy_head = self._make_policy_head()

            def _make_policy_head(self):
                class PolicyHead(nn.Module):
                    def forward(self, spatial_feat, vector_feat):
                        b = spatial_feat.shape[0]
                        return torch.randn(b, 10)

                return PolicyHead()

            def forward(self, spatial, vector, action_mask=None):
                sf = self.spatial_encoder(spatial)
                vf = self.vector_encoder(vector)
                return self.policy_head(sf, vf), torch.zeros(spatial.shape[0])

        base = MockNetworkWithEncoders()
        adapter = NPlayerNetworkAdapter(base, num_players=4)
        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)
        mask = torch.ones(2, 10)
        policy, values = adapter(spatial, vector, action_mask=mask)
        assert values.shape == (2, 4)
        assert policy.shape == (2, 10)

    def test_adapter_predict(self, base_network_no_config):
        """predict() returns probabilities and values."""
        adapter = NPlayerNetworkAdapter(base_network_no_config, num_players=2)
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 256)
        mask = torch.ones(2, 10)
        probs, values = adapter.predict(spatial, vector, mask)
        # Policy probs should sum to ~1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
        assert values.shape == (2, 2)

    def test_adapter_num_players_property(self, base_network_with_config):
        """num_players property returns correct value."""
        adapter = NPlayerNetworkAdapter(base_network_with_config, num_players=6)
        assert adapter.num_players == 6


class TestComputeNPlayerValueLossMixedDims:
    """Tests for mixed-dimension edge cases in value loss."""

    def test_scalar_pred_vector_target(self):
        """Scalar predicted expanded to match vector target."""
        pred = torch.tensor([0.5, -0.5])  # (B,)
        target = torch.tensor([[1.0, -1.0], [0.5, -0.5]])  # (B, N)
        loss = compute_nplayer_value_loss(pred, target, num_players=2)
        assert loss.item() > 0

    def test_vector_pred_scalar_target(self):
        """Scalar target expanded to match vector predicted."""
        pred = torch.tensor([[0.5, -0.5], [0.0, 0.0]])  # (B, N)
        target = torch.tensor([1.0, 0.5])  # (B,)
        loss = compute_nplayer_value_loss(pred, target, num_players=2)
        assert loss.item() > 0

    def test_reduction_sum(self):
        """reduction='sum' produces a scalar."""
        pred = torch.tensor([[0.5, -0.5]])
        target = torch.tensor([[1.0, -1.0]])
        loss = compute_nplayer_value_loss(pred, target, num_players=2, reduction="sum")
        assert loss.dim() == 0


class TestNPlayerNetworkConfig:
    """Tests for NPlayerNetworkConfig."""

    def test_default_values(self):
        config = NPlayerNetworkConfig()
        assert config.num_players == 2
        assert config.spatial_channels == 13
        assert config.action_space_size == 381

    def test_custom_values(self):
        config = NPlayerNetworkConfig(
            num_players=4,
            spatial_channels=19,
            spatial_size=32,
        )
        assert config.num_players == 4
        assert config.spatial_channels == 19
        assert config.spatial_size == 32
