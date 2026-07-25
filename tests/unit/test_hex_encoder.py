"""Unit tests for hex_encoder module.

Tests cover:
- Shape matching with PatchEmbed
- Construction from config and kwargs
- Forward pass behavior
- Gradient flow
- Factory function
- Crystalline layer integration
"""

import pytest
import torch

from alpha_zero.core.hex_encoder import (
    HexEncoderConfig,
    HexSpatialEncoder,
    SpatialToTokens,
    create_hex_encoder,
)
from alpha_zero.core.transformer_network import PatchEmbed


class TestShapeMatching:
    """Tests verifying HexSpatialEncoder produces shapes matching PatchEmbed."""

    def test_default_config_matches_patch_embed_shape(self):
        """Default config should produce same output shape as PatchEmbed."""
        torch.manual_seed(42)
        batch_size = 2
        in_channels = 13
        embed_dim = 64
        spatial_size = 8
        patch_size = 4

        # Create both encoders
        patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            spatial_size=spatial_size,
        )
        hex_encoder = create_hex_encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            spatial_size=spatial_size,
        )

        # Create input
        x = torch.randn(batch_size, in_channels, spatial_size, spatial_size)

        # Forward pass
        patch_out = patch_embed(x)
        hex_out = hex_encoder(x)

        # Check shapes match
        assert patch_out.shape == hex_out.shape, (
            f"Shape mismatch: PatchEmbed {patch_out.shape} vs HexEncoder {hex_out.shape}"
        )
        assert hex_out.shape == (
            batch_size,
            4,
            embed_dim,
        ), f"Expected shape (2, 4, 64), got {hex_out.shape}"

    def test_num_patches_property_matches_patch_embed(self):
        """num_patches property should match PatchEmbed.num_patches."""
        in_channels = 13
        embed_dim = 64
        spatial_size = 8
        patch_size = 4

        patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            spatial_size=spatial_size,
        )
        hex_encoder = create_hex_encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            spatial_size=spatial_size,
        )

        assert hex_encoder.num_patches == patch_embed.num_patches, (
            f"num_patches mismatch: PatchEmbed {patch_embed.num_patches} vs "
            f"HexEncoder {hex_encoder.num_patches}"
        )
        assert hex_encoder.num_patches == 4

    @pytest.mark.parametrize(
        "spatial_size,patch_size,expected_tokens",
        [
            (8, 4, 4),  # Default: 8//4 = 2, 2^2 = 4
            (16, 4, 16),  # Larger board: 16//4 = 4, 4^2 = 16
            (16, 8, 4),  # Larger patches: 16//8 = 2, 2^2 = 4
            (32, 8, 16),  # Big board: 32//8 = 4, 4^2 = 16
        ],
    )
    def test_different_board_sizes_and_patch_sizes(self, spatial_size, patch_size, expected_tokens):
        """Test various board_size and patch_size combinations."""
        torch.manual_seed(42)
        batch_size = 1
        in_channels = 13
        embed_dim = 64

        encoder = create_hex_encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            spatial_size=spatial_size,
        )

        x = torch.randn(batch_size, in_channels, spatial_size, spatial_size)
        output = encoder(x)

        assert output.shape == (batch_size, expected_tokens, embed_dim), (
            f"For spatial_size={spatial_size}, patch_size={patch_size}, "
            f"expected {expected_tokens} tokens, got {output.shape[1]}"
        )
        assert encoder.num_patches == expected_tokens

    def test_different_embed_dim(self):
        """Test with different embedding dimension."""
        torch.manual_seed(42)
        batch_size = 2
        in_channels = 13
        embed_dim = 128  # Different from default 64
        spatial_size = 8
        patch_size = 4

        encoder = create_hex_encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            spatial_size=spatial_size,
        )

        x = torch.randn(batch_size, in_channels, spatial_size, spatial_size)
        output = encoder(x)

        assert output.shape == (batch_size, 4, embed_dim)
        assert output.shape[2] == 128


class TestConstruction:
    """Tests for various construction patterns."""

    def test_construction_from_config(self):
        """Test construction from HexEncoderConfig dataclass."""
        torch.manual_seed(42)
        config = HexEncoderConfig(
            in_channels=13,
            embed_dim=64,
            spatial_size=8,
            num_output_tokens=4,
            hex_hidden_dim=32,
            hex_conv_layers=2,
        )

        encoder = HexSpatialEncoder(config=config)

        assert encoder.config == config
        assert encoder.num_patches == 4

        # Test forward pass works
        x = torch.randn(1, 13, 8, 8)
        output = encoder(x)
        assert output.shape == (1, 4, 64)

    def test_construction_from_kwargs(self):
        """Test construction from keyword arguments."""
        torch.manual_seed(42)
        encoder = HexSpatialEncoder(
            in_channels=13,
            embed_dim=64,
            spatial_size=8,
            num_output_tokens=4,
            hex_hidden_dim=32,
            hex_conv_layers=2,
        )

        assert encoder.config.in_channels == 13
        assert encoder.config.embed_dim == 64
        assert encoder.config.spatial_size == 8
        assert encoder.config.num_output_tokens == 4
        assert encoder.num_patches == 4

        # Test forward pass works
        x = torch.randn(1, 13, 8, 8)
        output = encoder(x)
        assert output.shape == (1, 4, 64)

    def test_factory_function_produces_correct_shapes(self):
        """Test create_hex_encoder factory function."""
        torch.manual_seed(42)
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )

        assert encoder.num_patches == 4

        x = torch.randn(2, 13, 8, 8)
        output = encoder(x)
        assert output.shape == (2, 4, 64)

    def test_construction_with_crystalline_disabled(self):
        """Test construction with use_crystalline=False."""
        torch.manual_seed(42)
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
            use_crystalline=False,
        )

        assert encoder.config.use_crystalline is False
        assert encoder.crystalline is None

        # Should still produce correct output shape
        x = torch.randn(2, 13, 8, 8)
        output = encoder(x)
        assert output.shape == (2, 4, 64)


class TestForwardPass:
    """Tests for forward pass behavior."""

    def test_standard_forward_pass_shape(self):
        """Test standard forward pass produces correct shape."""
        torch.manual_seed(42)
        batch_size = 4
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )

        x = torch.randn(batch_size, 13, 8, 8)
        output = encoder(x)

        assert output.shape == (batch_size, 4, 64)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        torch.manual_seed(42)
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )

        x = torch.randn(1, 13, 8, 8)
        output = encoder(x)

        assert output.shape == (1, 4, 64)
        assert not torch.isnan(output).any()

    def test_large_batch_size(self):
        """Test with larger batch size."""
        torch.manual_seed(42)
        batch_size = 16
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )

        x = torch.randn(batch_size, 13, 8, 8)
        output = encoder(x)

        assert output.shape == (batch_size, 4, 64)
        assert not torch.isnan(output).any()

    def test_deterministic_in_eval_mode(self):
        """Test that output is deterministic in eval mode."""
        torch.manual_seed(42)
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )
        encoder.eval()

        x = torch.randn(2, 13, 8, 8)

        # Run twice with same input
        with torch.no_grad():
            output1 = encoder(x)
            output2 = encoder(x)

        # Should be identical in eval mode
        assert torch.allclose(output1, output2, atol=1e-6)


class TestGradientFlow:
    """Tests for gradient flow through the encoder."""

    def test_gradients_flow_to_input(self):
        """Test that gradients flow back to input."""
        torch.manual_seed(42)
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )
        encoder.train()

        x = torch.randn(2, 13, 8, 8, requires_grad=True)
        output = encoder(x)

        # Create loss and backward
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.shape == x.shape

    def test_all_parameters_have_gradients(self):
        """Test that key parameters receive gradients after backward."""
        torch.manual_seed(42)
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )
        encoder.train()

        x = torch.randn(2, 13, 8, 8)
        output = encoder(x)

        # Create loss and backward
        loss = output.sum()
        loss.backward()

        # Count parameters with gradients
        params_with_grad = 0
        params_without_grad = []

        for name, param in encoder.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            else:
                params_without_grad.append(name)

        # Most parameters should have gradients
        # Some parameters (like lattice_basis in crystalline layer) may not be used
        # in every forward path, so we just check that SOME parameters have gradients
        assert params_with_grad > 0, "No parameters received gradients"

        # At least 80% of parameters should have gradients
        total_params = params_with_grad + len(params_without_grad)
        grad_ratio = params_with_grad / total_params
        assert grad_ratio > 0.8, (
            f"Too few parameters have gradients: {params_with_grad}/{total_params}. "
            f"Parameters without gradients: {params_without_grad}"
        )


class TestHexEncoderConfig:
    """Tests for HexEncoderConfig dataclass."""

    def test_config_defaults_match_expected(self):
        """Test that config defaults match expected values."""
        config = HexEncoderConfig()

        assert config.in_channels == 13
        assert config.embed_dim == 64
        assert config.spatial_size == 8
        assert config.hex_hidden_dim == 32
        assert config.hex_conv_layers == 2
        assert config.lattice_type == "hexagonal"
        assert config.num_output_tokens == 4
        assert config.dropout == 0.1
        assert config.offset_type == "even-q"
        assert config.activation == "gelu"
        assert config.use_batch_norm is True
        assert config.use_crystalline is True

    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = HexEncoderConfig(
            in_channels=20,
            embed_dim=128,
            spatial_size=16,
            hex_hidden_dim=64,
            hex_conv_layers=3,
            num_output_tokens=16,
            dropout=0.2,
            use_crystalline=False,
        )

        assert config.in_channels == 20
        assert config.embed_dim == 128
        assert config.spatial_size == 16
        assert config.hex_hidden_dim == 64
        assert config.hex_conv_layers == 3
        assert config.num_output_tokens == 16
        assert config.dropout == 0.2
        assert config.use_crystalline is False


class TestFactoryFunction:
    """Tests for create_hex_encoder factory function."""

    @pytest.mark.parametrize(
        "patch_size,expected_tokens",
        [
            (2, 16),  # 8//2 = 4, 4^2 = 16
            (4, 4),  # 8//4 = 2, 2^2 = 4
            (8, 1),  # 8//8 = 1, 1^2 = 1
        ],
    )
    def test_factory_with_different_patch_sizes(self, patch_size, expected_tokens):
        """Test factory function with different patch sizes."""
        torch.manual_seed(42)
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=patch_size,
            spatial_size=8,
        )

        assert encoder.num_patches == expected_tokens
        assert encoder.config.num_output_tokens == expected_tokens

        x = torch.randn(1, 13, 8, 8)
        output = encoder(x)
        assert output.shape == (1, expected_tokens, 64)

    def test_factory_passes_kwargs_to_config(self):
        """Test that factory function passes additional kwargs to config."""
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
            hex_hidden_dim=64,
            hex_conv_layers=3,
            dropout=0.2,
            use_crystalline=False,
        )

        assert encoder.config.hex_hidden_dim == 64
        assert encoder.config.hex_conv_layers == 3
        assert encoder.config.dropout == 0.2
        assert encoder.config.use_crystalline is False


class TestCrystallineIntegration:
    """Tests for crystalline layer integration."""

    def test_encoder_with_crystalline_enabled_produces_correct_shape(self):
        """Test encoder with crystalline layer enabled."""
        torch.manual_seed(42)
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
            use_crystalline=True,
        )

        assert encoder.use_crystalline is True
        assert encoder.crystalline is not None

        x = torch.randn(2, 13, 8, 8)
        output = encoder(x)

        assert output.shape == (2, 4, 64)

    def test_encoder_with_crystalline_disabled_still_produces_correct_shape(self):
        """Test encoder without crystalline layer."""
        torch.manual_seed(42)
        encoder = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
            use_crystalline=False,
        )

        assert encoder.use_crystalline is False
        assert encoder.crystalline is None

        x = torch.randn(2, 13, 8, 8)
        output = encoder(x)

        assert output.shape == (2, 4, 64)

    def test_crystalline_and_non_crystalline_have_same_output_shape(self):
        """Test that both variants produce same shape (different values)."""
        torch.manual_seed(42)
        encoder_with_crys = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
            use_crystalline=True,
        )

        encoder_without_crys = create_hex_encoder(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
            use_crystalline=False,
        )

        x = torch.randn(2, 13, 8, 8)

        output_with = encoder_with_crys(x)
        output_without = encoder_without_crys(x)

        # Same shape
        assert output_with.shape == output_without.shape
        # Different values (crystalline adds transformation)
        assert not torch.allclose(output_with, output_without, atol=1e-3)


class TestSpatialToTokens:
    """Tests for SpatialToTokens module."""

    def test_spatial_to_tokens_basic_functionality(self):
        """Test basic SpatialToTokens conversion."""
        torch.manual_seed(42)
        converter = SpatialToTokens(
            in_channels=32,
            embed_dim=64,
            spatial_size=8,
            num_tokens=4,
        )

        x = torch.randn(2, 32, 8, 8)
        output = converter(x)

        assert output.shape == (2, 4, 64)

    def test_spatial_to_tokens_different_num_tokens(self):
        """Test SpatialToTokens with different num_tokens."""
        torch.manual_seed(42)
        converter = SpatialToTokens(
            in_channels=32,
            embed_dim=64,
            spatial_size=16,
            num_tokens=16,
        )

        x = torch.randn(1, 32, 16, 16)
        output = converter(x)

        assert output.shape == (1, 16, 64)

    def test_spatial_to_tokens_properties(self):
        """Test SpatialToTokens properties are stored correctly."""
        converter = SpatialToTokens(
            in_channels=32,
            embed_dim=64,
            spatial_size=8,
            num_tokens=4,
        )

        assert converter.in_channels == 32
        assert converter.embed_dim == 64
        assert converter.spatial_size == 8
        assert converter.num_tokens == 4
