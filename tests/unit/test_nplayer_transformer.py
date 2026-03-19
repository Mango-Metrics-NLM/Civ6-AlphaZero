"""Tests for N-player support in Transformer network.

Sprint 6: N-Player Transformer Architecture
Tests the new value_output_dim parameter for multi-player games.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from alpha_zero.core.network_heads import HeadFactory
from alpha_zero.core.transformer_network import (
    Civ6TransformerNetwork,
    TransformerConfig,
    TransformerValueHead,
)


class TestTransformerConfigNPlayer:
    """Tests for TransformerConfig N-player parameters."""

    def test_default_value_output_dim(self) -> None:
        """Test default value_output_dim is 6 for N-player."""
        config = TransformerConfig()
        assert config.value_output_dim == 6

    def test_custom_value_output_dim_1(self) -> None:
        """Test value_output_dim=1 for backward compatibility."""
        config = TransformerConfig(value_output_dim=1)
        assert config.value_output_dim == 1

    def test_custom_value_output_dim_6(self) -> None:
        """Test value_output_dim=6 for N-player."""
        config = TransformerConfig(value_output_dim=6)
        assert config.value_output_dim == 6

    def test_from_network_config_with_value_output_dim(self) -> None:
        """Test from_network_config preserves value_output_dim."""
        from types import SimpleNamespace

        net_config = SimpleNamespace(
            spatial_channels=28,
            board_size=16,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=6,
            num_heads=8,
            policy_hidden_dim=128,
            value_head_dim=64,
            value_output_dim=6,
            encoder_type="patch_embed",
        )

        config = TransformerConfig.from_network_config(net_config)
        assert config.value_output_dim == 6

    def test_from_network_config_defaults_value_output_dim(self) -> None:
        """Test from_network_config uses default when field missing."""
        from types import SimpleNamespace

        net_config = SimpleNamespace(
            spatial_channels=13,
            board_size=8,
        )

        config = TransformerConfig.from_network_config(net_config)
        assert config.value_output_dim == 6  # Default


class TestTransformerValueHeadNPlayer:
    """Tests for TransformerValueHead N-player support."""

    def test_default_output_dim_is_1(self) -> None:
        """Test default output_dim=1 for backward compatibility."""
        head = TransformerValueHead(embed_dim=128, hidden_dim=64)
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 1)

    def test_output_dim_1(self) -> None:
        """Test value head with output_dim=1."""
        head = TransformerValueHead(embed_dim=128, hidden_dim=64, output_dim=1)
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 1)

    def test_output_dim_6(self) -> None:
        """Test value head with output_dim=6 for N-player."""
        head = TransformerValueHead(embed_dim=128, hidden_dim=64, output_dim=6)
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 6)

    def test_output_range_tanh(self) -> None:
        """Test value output is in [-1, 1] range (tanh)."""
        head = TransformerValueHead(embed_dim=128, hidden_dim=64, output_dim=6)
        x = torch.randn(10, 128)
        out = head(x)

        assert torch.all(out >= -1.0)
        assert torch.all(out <= 1.0)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through value head."""
        head = TransformerValueHead(embed_dim=128, hidden_dim=64, output_dim=6)
        x = torch.randn(2, 128, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestCiv6TransformerNetworkNPlayer:
    """Tests for Civ6TransformerNetwork N-player support."""

    def test_network_with_value_output_dim_6(self) -> None:
        """Test network with 6-dimensional value output (N-player)."""
        config = TransformerConfig(
            spatial_channels=28,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            patch_size=4,
            value_output_dim=6,
        )
        network = Civ6TransformerNetwork(config=config)

        spatial = torch.randn(2, 28, 8, 8)
        vector = torch.randn(2, 64)

        policy_logits, value = network(spatial, vector)

        assert policy_logits.shape == (2, 376)
        assert value.shape == (2, 6), f"Expected (2, 6), got {value.shape}"

    def test_network_with_value_output_dim_1_backward_compat(self) -> None:
        """Test network with 1-dimensional value (2-player backward compat)."""
        config = TransformerConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            patch_size=4,
            value_output_dim=1,
        )
        network = Civ6TransformerNetwork(config=config)

        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)

        policy_logits, value = network(spatial, vector)

        assert policy_logits.shape == (2, 376)
        # value_output_dim=1 gets squeezed to (B,)
        assert value.shape == (2,), f"Expected (2,), got {value.shape}"

    def test_value_range_6dim(self) -> None:
        """Test 6-dim value output is in [-1, 1] range."""
        config = TransformerConfig(
            spatial_channels=28,
            spatial_size=8,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            patch_size=4,
            value_output_dim=6,
        )
        network = Civ6TransformerNetwork(config=config)

        spatial = torch.randn(10, 28, 8, 8)
        vector = torch.randn(10, 64)

        _, value = network(spatial, vector)

        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_save_load_with_value_output_dim_6(self) -> None:
        """Test save/load with 6-dimensional value."""
        config = TransformerConfig(
            spatial_channels=28,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            patch_size=4,
            value_output_dim=6,
        )
        network = Civ6TransformerNetwork(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model_6dim.pt")
            network.save(path)

            loaded = Civ6TransformerNetwork.load(path)

        # Check config
        assert loaded.config.value_output_dim == 6
        assert loaded.config.spatial_channels == 28

        # Check same output
        spatial = torch.randn(2, 28, 8, 8)
        vector = torch.randn(2, 64)

        network.eval()
        loaded.eval()

        with torch.no_grad():
            out1 = network(spatial, vector)
            out2 = loaded(spatial, vector)

        assert torch.allclose(out1[0], out2[0], atol=1e-5)
        assert torch.allclose(out1[1], out2[1], atol=1e-5)

    def test_save_load_with_value_output_dim_1(self) -> None:
        """Test save/load preserves value_output_dim=1 backward compat."""
        config = TransformerConfig(
            spatial_channels=13,
            spatial_size=8,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            patch_size=4,
            value_output_dim=1,
        )
        network = Civ6TransformerNetwork(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model_1dim.pt")
            network.save(path)

            loaded = Civ6TransformerNetwork.load(path)

        assert loaded.config.value_output_dim == 1

        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)

        loaded.eval()
        _, value = loaded(spatial, vector)

        # Should be squeezed to scalar
        assert value.shape == (2,)


class TestMigrationCompatibility:
    """Tests for migration utility compatibility."""

    def test_simulated_migration_1_to_6(self) -> None:
        """Test that migrated weights (1→6) load correctly.

        Simulates the migration scenario:
        1. Old model has output_dim=1
        2. Migration script replicates weights to output_dim=6
        3. New code loads with value_output_dim=6
        """
        # Create old network (output_dim=1)
        old_config = TransformerConfig(
            spatial_channels=13,
            spatial_size=8,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            patch_size=4,
            value_output_dim=1,
        )
        old_network = Civ6TransformerNetwork(config=old_config)

        # Simulate migration: replicate value head output layer weights
        old_weight = old_network.value_head.mlp[-2].weight  # (1, hidden_dim)
        old_bias = old_network.value_head.mlp[-2].bias  # (1,)

        new_weight = old_weight.repeat(6, 1)  # (6, hidden_dim)
        new_bias = old_bias.repeat(6)  # (6,)

        # Create new network and inject migrated weights
        new_config = TransformerConfig(
            spatial_channels=13,  # Same encoder for this test
            spatial_size=8,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            patch_size=4,
            value_output_dim=6,
        )
        new_network = Civ6TransformerNetwork(config=new_config)

        # Copy all encoder weights (unchanged)
        new_network.encoder.load_state_dict(old_network.encoder.state_dict())
        new_network.policy_head.load_state_dict(old_network.policy_head.state_dict())

        # Inject migrated value head weights
        new_network.value_head.mlp[-2].weight.data = new_weight
        new_network.value_head.mlp[-2].bias.data = new_bias

        # Copy value head layer 1 weights
        new_network.value_head.mlp[0].weight.data = old_network.value_head.mlp[
            0
        ].weight.data.clone()
        new_network.value_head.mlp[0].bias.data = old_network.value_head.mlp[
            0
        ].bias.data.clone()
        new_network.value_head.mlp[3].weight.data = old_network.value_head.mlp[
            3
        ].weight.data.clone()
        new_network.value_head.mlp[3].bias.data = old_network.value_head.mlp[
            3
        ].bias.data.clone()

        # Verify output shape
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)

        old_network.eval()
        new_network.eval()

        with torch.no_grad():
            _, old_value = old_network(spatial, vector)
            _, new_value = new_network(spatial, vector)

        # Old value is scalar
        assert old_value.shape == (1,)
        # New value is 6-dim
        assert new_value.shape == (1, 6)

        # All 6 dimensions should be approximately equal (replicated weights)
        assert torch.allclose(new_value[0, 0], new_value[0, 1], atol=1e-5)
        assert torch.allclose(new_value[0, 0], new_value[0, 2], atol=1e-5)


class TestHeadFactoryNPlayer:
    """Tests for HeadFactory N-player support."""

    def test_create_transformer_value_head_default_output_dim_1(self) -> None:
        """Test default output_dim=1 for backward compatibility."""
        head = HeadFactory.create_transformer_value_head(
            embed_dim=128,
            hidden_dim=64,
        )
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 1)

    def test_create_transformer_value_head_output_dim_6(self) -> None:
        """Test custom output_dim=6 for N-player."""
        head = HeadFactory.create_transformer_value_head(
            embed_dim=128,
            hidden_dim=64,
            output_dim=6,
        )
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 6)

    def test_create_transformer_value_head_output_dim_1_explicit(self) -> None:
        """Test explicit output_dim=1."""
        head = HeadFactory.create_transformer_value_head(
            embed_dim=128,
            hidden_dim=64,
            output_dim=1,
        )
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 1)
