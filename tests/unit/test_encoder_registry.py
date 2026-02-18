"""Unit tests for encoder registry module.

Tests the encoder registry pattern for selecting and creating different
spatial encoder implementations (PatchEmbed, HexSpatialEncoder, etc.).
"""

import dataclasses

import pytest
import torch

from alpha_zero.core.encoder_registry import (
    EncoderType,
    create_spatial_encoder,
    get_registered_encoders,
    register_encoder,
)
from alpha_zero.core.hex_encoder import HexSpatialEncoder
from alpha_zero.core.transformer_network import PatchEmbed, TransformerConfig


class TestEncoderRegistry:
    """Test cases for encoder registration and retrieval."""

    def test_both_encoders_registered(self):
        """Both patch_embed and hex_crystalline types should be registered."""
        registered = get_registered_encoders()
        assert isinstance(registered, list)
        assert "patch_embed" in registered
        assert "hex_crystalline" in registered
        assert len(registered) >= 2

    def test_unknown_encoder_raises_error(self):
        """Creating an unknown encoder type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown encoder type"):
            create_spatial_encoder(
                encoder_type="nonexistent_encoder",
                in_channels=13,
                embed_dim=64,
                patch_size=4,
                spatial_size=8,
            )

    def test_enum_values_work(self):
        """EncoderType enum values should work with create_spatial_encoder."""
        torch.manual_seed(42)

        # Should not raise, should return a valid encoder
        encoder = create_spatial_encoder(
            encoder_type=EncoderType.PATCH_EMBED,
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )

        assert encoder is not None
        assert isinstance(encoder, PatchEmbed)

    def test_register_custom_encoder(self):
        """Custom encoder factories can be registered."""

        # Create a simple custom encoder
        class CustomEncoder(torch.nn.Module):
            def __init__(self, in_channels, embed_dim, patch_size, spatial_size):
                super().__init__()
                self.num_patches = (spatial_size // patch_size) ** 2
                self.embed_dim = embed_dim

            def forward(self, x):
                batch = x.shape[0]
                return torch.zeros(batch, self.num_patches, self.embed_dim)

        # Register it
        @register_encoder("custom_test")
        def _create_custom(in_channels, embed_dim, patch_size, spatial_size, **kwargs):
            return CustomEncoder(in_channels, embed_dim, patch_size, spatial_size)

        # Verify it's registered
        assert "custom_test" in get_registered_encoders()

        # Create an instance
        encoder = create_spatial_encoder(
            encoder_type="custom_test",
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )

        assert isinstance(encoder, CustomEncoder)
        assert encoder.num_patches == 4


class TestPatchEmbedCreation:
    """Test cases for PatchEmbed encoder creation."""

    def test_create_patch_embed_returns_correct_type(self):
        """create_spatial_encoder with patch_embed should return PatchEmbed."""
        torch.manual_seed(42)

        encoder = create_spatial_encoder(
            encoder_type="patch_embed",
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )

        assert isinstance(encoder, PatchEmbed)

    def test_patch_embed_output_shape(self):
        """PatchEmbed should produce correct output shape."""
        torch.manual_seed(42)

        encoder = create_spatial_encoder(
            encoder_type="patch_embed",
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )

        # Create dummy input: (B, C, H, W)
        batch_size = 2
        x = torch.randn(batch_size, 13, 8, 8)

        # Forward pass
        output = encoder(x)

        # Check output shape: (B, num_patches, embed_dim)
        # num_patches = (8 // 4)^2 = 4
        assert output.shape == (batch_size, 4, 64)


class TestHexCrystallineCreation:
    """Test cases for HexSpatialEncoder creation."""

    def test_create_hex_encoder_returns_correct_type(self):
        """create_spatial_encoder with hex_crystalline should return HexSpatialEncoder."""
        torch.manual_seed(42)

        encoder = create_spatial_encoder(
            encoder_type="hex_crystalline",
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )

        assert isinstance(encoder, HexSpatialEncoder)

    def test_hex_encoder_output_shape(self):
        """HexSpatialEncoder should produce correct output shape."""
        torch.manual_seed(42)

        encoder = create_spatial_encoder(
            encoder_type="hex_crystalline",
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
        )

        # Create dummy input: (B, C, H, W)
        batch_size = 2
        x = torch.randn(batch_size, 13, 8, 8)

        # Forward pass
        output = encoder(x)

        # Check output shape: (B, num_patches, embed_dim)
        # num_patches = (8 // 4)^2 = 4
        assert output.shape == (batch_size, 4, 64)


class TestShapeMatching:
    """Test cases verifying both encoders produce matching shapes."""

    def test_both_encoders_produce_same_shape(self):
        """Both encoders should produce identical output shapes."""
        torch.manual_seed(42)

        # Configuration
        config = {
            "in_channels": 13,
            "embed_dim": 64,
            "patch_size": 4,
            "spatial_size": 8,
        }

        # Create both encoders
        patch_encoder = create_spatial_encoder(encoder_type="patch_embed", **config)
        hex_encoder = create_spatial_encoder(encoder_type="hex_crystalline", **config)

        # Create dummy input
        batch_size = 3
        x = torch.randn(batch_size, 13, 8, 8)

        # Forward pass through both
        patch_output = patch_encoder(x)
        hex_output = hex_encoder(x)

        # Shapes should match
        assert patch_output.shape == hex_output.shape
        assert patch_output.shape == (batch_size, 4, 64)

    def test_both_have_matching_num_patches(self):
        """Both encoders should report the same num_patches property."""
        torch.manual_seed(42)

        config = {
            "in_channels": 13,
            "embed_dim": 64,
            "patch_size": 4,
            "spatial_size": 8,
        }

        patch_encoder = create_spatial_encoder(encoder_type="patch_embed", **config)
        hex_encoder = create_spatial_encoder(encoder_type="hex_crystalline", **config)

        # Both should have num_patches property
        assert hasattr(patch_encoder, "num_patches")
        assert hasattr(hex_encoder, "num_patches")

        # Values should match
        assert patch_encoder.num_patches == hex_encoder.num_patches
        assert patch_encoder.num_patches == 4

    def test_different_spatial_sizes(self):
        """Both encoders should handle different spatial sizes consistently."""
        torch.manual_seed(42)

        # Test with 16x16 spatial size
        config = {
            "in_channels": 13,
            "embed_dim": 64,
            "patch_size": 4,
            "spatial_size": 16,
        }

        patch_encoder = create_spatial_encoder(encoder_type="patch_embed", **config)
        hex_encoder = create_spatial_encoder(encoder_type="hex_crystalline", **config)

        # num_patches = (16 // 4)^2 = 16
        assert patch_encoder.num_patches == 16
        assert hex_encoder.num_patches == 16

        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 13, 16, 16)

        patch_output = patch_encoder(x)
        hex_output = hex_encoder(x)

        assert patch_output.shape == (batch_size, 16, 64)
        assert hex_output.shape == (batch_size, 16, 64)


class TestConfigIntegration:
    """Test cases for TransformerConfig integration."""

    def test_transformer_config_has_encoder_type_field(self):
        """TransformerConfig should have encoder_type field with default."""
        config = TransformerConfig()

        # Should have encoder_type attribute
        assert hasattr(config, "encoder_type")

        # Default should be patch_embed
        assert config.encoder_type == "patch_embed"

    def test_transformer_config_dataclass_includes_encoder_type(self):
        """dataclasses.asdict should include encoder_type field."""
        config = TransformerConfig()
        config_dict = dataclasses.asdict(config)

        assert "encoder_type" in config_dict
        assert config_dict["encoder_type"] == "patch_embed"

    def test_transformer_config_with_custom_encoder_type(self):
        """TransformerConfig should allow setting custom encoder_type."""
        config = TransformerConfig(encoder_type="hex_crystalline")

        assert config.encoder_type == "hex_crystalline"

        config_dict = dataclasses.asdict(config)
        assert config_dict["encoder_type"] == "hex_crystalline"


class TestEncoderTypeEnum:
    """Test cases for EncoderType enum."""

    def test_enum_has_patch_embed(self):
        """EncoderType enum should have PATCH_EMBED member."""
        assert hasattr(EncoderType, "PATCH_EMBED")
        assert EncoderType.PATCH_EMBED.value == "patch_embed"

    def test_enum_has_hex_crystalline(self):
        """EncoderType enum should have HEX_CRYSTALLINE member."""
        assert hasattr(EncoderType, "HEX_CRYSTALLINE")
        assert EncoderType.HEX_CRYSTALLINE.value == "hex_crystalline"

    def test_enum_is_string_enum(self):
        """EncoderType should be a string enum."""
        # Can compare with strings directly
        assert EncoderType.PATCH_EMBED == "patch_embed"
        assert EncoderType.HEX_CRYSTALLINE == "hex_crystalline"


class TestEncoderKwargs:
    """Test cases for passing additional kwargs to encoders."""

    def test_hex_encoder_accepts_custom_kwargs(self):
        """HexSpatialEncoder should accept additional configuration kwargs."""
        torch.manual_seed(42)

        encoder = create_spatial_encoder(
            encoder_type="hex_crystalline",
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
            hex_hidden_dim=48,  # Custom value
            hex_conv_layers=3,  # Custom value
            dropout=0.2,  # Custom value
        )

        assert isinstance(encoder, HexSpatialEncoder)
        assert encoder.config.hex_hidden_dim == 48
        assert encoder.config.hex_conv_layers == 3
        assert encoder.config.dropout == 0.2

    def test_patch_embed_ignores_extra_kwargs(self):
        """PatchEmbed should ignore extra kwargs without error."""
        torch.manual_seed(42)

        # Should not raise even with extra kwargs
        encoder = create_spatial_encoder(
            encoder_type="patch_embed",
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=8,
            extra_param="ignored",
            another_param=123,
        )

        assert isinstance(encoder, PatchEmbed)
        # Should work normally
        x = torch.randn(2, 13, 8, 8)
        output = encoder(x)
        assert output.shape == (2, 4, 64)
