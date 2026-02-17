"""Tests for N-player neural network heads."""

from __future__ import annotations

import pytest
import torch

from alpha_zero.multiplayer.network_heads import (
    NPlayerNetworkConfig,
    NPlayerValueHead,
    compute_nplayer_value_loss,
)


class TestNPlayerValueHead:
    """Tests for NPlayerValueHead."""

    @pytest.fixture
    def head_2p(self):
        return NPlayerValueHead(
            spatial_filters=64, spatial_size=16, vector_dim=256, num_players=2
        )

    @pytest.fixture
    def head_4p(self):
        return NPlayerValueHead(
            spatial_filters=64, spatial_size=16, vector_dim=256, num_players=4
        )

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
        head = NPlayerValueHead(
            spatial_filters=32, spatial_size=8, vector_dim=64, num_players=3
        )
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
        head = NPlayerValueHead(
            spatial_filters=32, spatial_size=8, vector_dim=64, num_players=1
        )
        spatial = torch.randn(2, 32, 8, 8)
        vector = torch.randn(2, 64)
        output = head(spatial, vector)
        assert output.shape == (2, 1)

    def test_large_player_count(self):
        """Works with 8 players (Civ6 max)."""
        head = NPlayerValueHead(
            spatial_filters=32, spatial_size=8, vector_dim=64, num_players=8
        )
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
