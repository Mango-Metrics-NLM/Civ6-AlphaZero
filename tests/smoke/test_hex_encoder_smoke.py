"""Smoke tests for the Crystalline Hex Spatial Encoder feature.

Fast sanity checks (<5s total) verifying that the hex encoder feature
is minimally functional. These tests are designed to run on every commit
and catch import errors, shape mismatches, and registry breakage early.

Smoke tests cover:
- Module imports succeed
- Registry contains expected encoders
- Basic construction of each component
- Forward pass produces valid output (no NaN/Inf, correct shape)
- Default config values are sane
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.smoke


# ============================================================================
# Import smoke tests — catch broken imports before anything else
# ============================================================================


class TestImportSmoke:
    """Verify all new modules import without error."""

    def test_import_crystalline_layers(self) -> None:
        from alpha_zero.core.crystalline_layers import (
            CrystallineLatticeConfig,
            CrystallineLatticeLayer,
            SymmetryType,
        )

        assert CrystallineLatticeLayer is not None
        assert SymmetryType is not None
        assert CrystallineLatticeConfig is not None

    def test_import_hex_conv(self) -> None:
        from alpha_zero.core.hex_conv import (
            HexAdjacency,
            HexConvBlock,
            HexConvConfig,
            HexConvolution,
        )

        assert HexAdjacency is not None
        assert HexConvolution is not None
        assert HexConvBlock is not None
        assert HexConvConfig is not None

    def test_import_hex_encoder(self) -> None:
        from alpha_zero.core.hex_encoder import (
            HexEncoderConfig,
            HexSpatialEncoder,
            SpatialToTokens,
            create_hex_encoder,
        )

        assert HexSpatialEncoder is not None
        assert HexEncoderConfig is not None
        assert SpatialToTokens is not None
        assert create_hex_encoder is not None

    def test_import_encoder_registry(self) -> None:
        from alpha_zero.core.encoder_registry import (
            EncoderType,
            create_spatial_encoder,
            get_registered_encoders,
            register_encoder,
        )

        assert EncoderType is not None
        assert create_spatial_encoder is not None
        assert get_registered_encoders is not None
        assert register_encoder is not None


# ============================================================================
# Registry smoke tests
# ============================================================================


class TestRegistrySmoke:
    """Verify encoder registry is functional."""

    def test_both_encoders_registered(self) -> None:
        from alpha_zero.core.encoder_registry import get_registered_encoders

        registered = get_registered_encoders()
        assert "patch_embed" in registered
        assert "hex_crystalline" in registered

    def test_enum_values_match_strings(self) -> None:
        from alpha_zero.core.encoder_registry import EncoderType

        assert EncoderType.PATCH_EMBED.value == "patch_embed"
        assert EncoderType.HEX_CRYSTALLINE.value == "hex_crystalline"

    def test_unknown_encoder_raises(self) -> None:
        from alpha_zero.core.encoder_registry import create_spatial_encoder

        with pytest.raises(ValueError, match="Unknown encoder type"):
            create_spatial_encoder(
                encoder_type="nonexistent",
                in_channels=13,
                embed_dim=64,
                patch_size=4,
                spatial_size=8,
            )


# ============================================================================
# Component construction smoke tests
# ============================================================================


class TestConstructionSmoke:
    """Verify each component constructs without error using defaults."""

    def test_crystalline_layer_default(self) -> None:
        from alpha_zero.core.crystalline_layers import CrystallineLatticeLayer

        layer = CrystallineLatticeLayer(dim=64)
        assert isinstance(layer, torch.nn.Module)

    def test_hex_adjacency_default(self) -> None:
        from alpha_zero.core.hex_conv import HexAdjacency

        adj = HexAdjacency(board_size=8)
        indices = adj.get_neighbor_indices()
        assert indices.shape == (64, 6)

    def test_hex_convolution_default(self) -> None:
        from alpha_zero.core.hex_conv import HexConvolution

        conv = HexConvolution(in_channels=13, out_channels=64)
        assert isinstance(conv, torch.nn.Module)

    def test_hex_conv_block_default(self) -> None:
        from alpha_zero.core.hex_conv import HexConvBlock

        block = HexConvBlock(in_channels=13, out_channels=64)
        assert isinstance(block, torch.nn.Module)

    def test_hex_spatial_encoder_default(self) -> None:
        from alpha_zero.core.hex_encoder import HexSpatialEncoder

        encoder = HexSpatialEncoder()
        assert isinstance(encoder, torch.nn.Module)
        assert encoder.num_patches == 4

    def test_spatial_to_tokens_default(self) -> None:
        from alpha_zero.core.hex_encoder import SpatialToTokens

        s2t = SpatialToTokens(in_channels=32, embed_dim=64, spatial_size=8, num_tokens=4)
        assert isinstance(s2t, torch.nn.Module)


# ============================================================================
# Forward pass smoke tests — shapes + numerical validity
# ============================================================================


class TestForwardSmoke:
    """Verify forward pass of each component produces valid output."""

    def test_crystalline_forward_produces_finite_output(self) -> None:
        from alpha_zero.core.crystalline_layers import CrystallineLatticeLayer

        torch.manual_seed(42)
        layer = CrystallineLatticeLayer(dim=64)
        x = torch.randn(2, 4, 64)
        out = layer(x)

        assert out.shape == (2, 4, 64)
        assert torch.isfinite(out).all(), "Crystalline output contains NaN/Inf"

    def test_hex_convolution_forward_produces_finite_output(self) -> None:
        from alpha_zero.core.hex_conv import HexConvolution

        torch.manual_seed(42)
        conv = HexConvolution(in_channels=13, out_channels=64, board_size=8)
        x = torch.randn(2, 13, 8, 8)
        out = conv(x)

        assert out.shape == (2, 64, 8, 8)
        assert torch.isfinite(out).all(), "HexConvolution output contains NaN/Inf"

    def test_hex_encoder_forward_matches_patch_embed_shape(self) -> None:
        """The core contract: hex encoder output shape matches PatchEmbed."""
        from alpha_zero.core.encoder_registry import create_spatial_encoder

        torch.manual_seed(42)
        x = torch.randn(2, 13, 8, 8)

        patch_encoder = create_spatial_encoder("patch_embed", 13, 64, 4, 8)
        hex_encoder = create_spatial_encoder("hex_crystalline", 13, 64, 4, 8)

        out_patch = patch_encoder(x)
        out_hex = hex_encoder(x)

        assert out_patch.shape == out_hex.shape, (
            f"Shape mismatch: patch={out_patch.shape}, hex={out_hex.shape}"
        )
        assert out_patch.shape == (2, 4, 64)
        assert torch.isfinite(out_hex).all()

    def test_full_network_forward_with_hex_encoder(self) -> None:
        """Smoke test: full Civ6TransformerNetwork with hex encoder."""
        from alpha_zero.core.transformer_network import (
            Civ6TransformerNetwork,
            TransformerConfig,
        )

        torch.manual_seed(42)
        config = TransformerConfig(encoder_type="hex_crystalline")
        network = Civ6TransformerNetwork(config=config)

        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(spatial, vector)

        assert policy_logits.shape == (2, 376)
        assert value.shape == (2,)
        assert torch.isfinite(policy_logits).all(), "Policy logits contain NaN/Inf"
        assert torch.isfinite(value).all(), "Value contains NaN/Inf"

    def test_full_network_forward_with_patch_embed_still_works(self) -> None:
        """Smoke test: default PatchEmbed network is not broken."""
        from alpha_zero.core.transformer_network import (
            Civ6TransformerNetwork,
            TransformerConfig,
        )

        torch.manual_seed(42)
        config = TransformerConfig(encoder_type="patch_embed")
        network = Civ6TransformerNetwork(config=config)

        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)

        network.eval()
        with torch.no_grad():
            policy_logits, value = network(spatial, vector)

        assert policy_logits.shape == (2, 376)
        assert value.shape == (2,)
        assert torch.isfinite(policy_logits).all()


# ============================================================================
# Config default smoke tests
# ============================================================================


class TestConfigSmoke:
    """Verify config defaults are sane and consistent."""

    def test_transformer_config_has_encoder_type(self) -> None:
        from alpha_zero.core.transformer_network import TransformerConfig

        config = TransformerConfig()
        assert hasattr(config, "encoder_type")
        assert config.encoder_type == "patch_embed"

    def test_network_architecture_config_has_encoder_type(self) -> None:
        from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig

        config = NetworkArchitectureConfig()
        assert hasattr(config, "encoder_type")
        assert config.encoder_type == "patch_embed"

    def test_hex_encoder_config_defaults(self) -> None:
        from alpha_zero.core.hex_encoder import HexEncoderConfig

        config = HexEncoderConfig()
        assert config.in_channels == 13
        assert config.embed_dim == 64
        assert config.spatial_size == 8
        assert config.num_output_tokens == 4
        assert config.use_crystalline is True

    def test_encoder_type_propagates_through_from_network_config(self) -> None:
        from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig
        from alpha_zero.core.transformer_network import TransformerConfig

        net_config = NetworkArchitectureConfig(encoder_type="hex_crystalline")
        tc = TransformerConfig.from_network_config(net_config)
        assert tc.encoder_type == "hex_crystalline"

    def test_network_architecture_config_to_dict_includes_encoder_type(self) -> None:
        from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig

        config = NetworkArchitectureConfig(encoder_type="hex_crystalline")
        d = config.to_dict()
        assert "encoder_type" in d
        assert d["encoder_type"] == "hex_crystalline"
