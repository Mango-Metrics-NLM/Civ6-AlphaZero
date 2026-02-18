"""
Tests for Phase 14: Verify TransformerConfig defaults match NetworkArchitectureConfig.

This test suite ensures that the TransformerConfig dataclass uses the same default
values as NetworkArchitectureConfig, which serves as the single source of truth for
network architecture parameters across selfplay, training, and evaluation workers.

Critical for:
- Preventing config drift between transformer_network.py and cloud/gcp/config.py
- Ensuring legacy model compatibility (v0-v2 use NetworkArchitectureConfig fallback)
- Catching stale defaults (e.g., action_space_size=381 instead of 376)
"""

from __future__ import annotations

import warnings

import pytest

from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig
from alpha_zero.core.transformer_network import TransformerConfig


class TestTransformerConfigDefaultsMatchNetworkArchitecture:
    """Test that TransformerConfig defaults align with NetworkArchitectureConfig."""

    def test_spatial_size_matches(self) -> None:
        """Verify TransformerConfig.spatial_size matches NetworkArchitectureConfig.board_size."""
        transformer_config = TransformerConfig()
        network_config = NetworkArchitectureConfig()

        assert transformer_config.spatial_size == network_config.board_size, (
            f"TransformerConfig.spatial_size ({transformer_config.spatial_size}) "
            f"!= NetworkArchitectureConfig.board_size ({network_config.board_size}). "
            "These must match for consistent spatial encoding."
        )

        # Also verify the expected value (8x8 board)
        assert transformer_config.spatial_size == 8

    def test_action_space_matches(self) -> None:
        """Verify TransformerConfig.action_space_size matches NetworkArchitectureConfig."""
        transformer_config = TransformerConfig()
        network_config = NetworkArchitectureConfig()

        assert transformer_config.action_space_size == network_config.action_space_size, (
            f"TransformerConfig.action_space_size ({transformer_config.action_space_size}) "
            f"!= NetworkArchitectureConfig.action_space_size ({network_config.action_space_size}). "
            "These must match for correct policy head output."
        )

        # Verify the correct value (376 = 1 + 20*8 + 20*8 + 20 + 20 + 5*3)
        assert transformer_config.action_space_size == 376

    def test_embed_dim_matches(self) -> None:
        """Verify TransformerConfig.embed_dim matches NetworkArchitectureConfig."""
        transformer_config = TransformerConfig()
        network_config = NetworkArchitectureConfig()

        assert transformer_config.embed_dim == network_config.embed_dim, (
            f"TransformerConfig.embed_dim ({transformer_config.embed_dim}) "
            f"!= NetworkArchitectureConfig.embed_dim ({network_config.embed_dim}). "
            "These must match for consistent embedding dimensions."
        )

        # Verify the expected value (64 for budget training)
        assert transformer_config.embed_dim == 64

    def test_num_layers_matches(self) -> None:
        """Verify TransformerConfig.num_layers matches NetworkArchitectureConfig."""
        transformer_config = TransformerConfig()
        network_config = NetworkArchitectureConfig()

        assert transformer_config.num_layers == network_config.num_layers, (
            f"TransformerConfig.num_layers ({transformer_config.num_layers}) "
            f"!= NetworkArchitectureConfig.num_layers ({network_config.num_layers}). "
            "These must match for consistent network depth."
        )

        # Verify the expected value (6 layers)
        assert transformer_config.num_layers == 6


class TestTransformerConfigHeadDimensions:
    """Test that head dimensions are correctly scaled relative to embed_dim."""

    def test_value_head_dim_scaled(self) -> None:
        """Verify value_head_dim equals embed_dim (1x scaling)."""
        config = TransformerConfig()

        assert config.value_head_dim == config.embed_dim, (
            f"TransformerConfig.value_head_dim ({config.value_head_dim}) "
            f"!= embed_dim ({config.embed_dim}). "
            "Value head should use 1x embed_dim for consistent capacity."
        )

        # With default embed_dim=64, value_head_dim should be 64
        assert config.value_head_dim == 64

    def test_policy_hidden_dim_scaled(self) -> None:
        """Verify policy_hidden_dim equals 2 * embed_dim (2x scaling)."""
        config = TransformerConfig()

        assert config.policy_hidden_dim == 2 * config.embed_dim, (
            f"TransformerConfig.policy_hidden_dim ({config.policy_hidden_dim}) "
            f"!= 2 * embed_dim ({2 * config.embed_dim}). "
            "Policy head should use 2x embed_dim for larger action space."
        )

        # With default embed_dim=64, policy_hidden_dim should be 128
        assert config.policy_hidden_dim == 128


class TestTransformerConfigWarnings:
    """Test that TransformerConfig emits warnings for known incorrect values."""

    def test_action_space_381_warns(self) -> None:
        """Verify TransformerConfig(action_space_size=381) emits UserWarning."""
        # 381 is the OLD incorrect value, should trigger a warning
        with pytest.warns(
            UserWarning,
            match=r"TransformerConfig\.action_space_size=381 is a known incorrect value.*376",
        ):
            TransformerConfig(action_space_size=381)

    def test_action_space_376_no_warn(self) -> None:
        """Verify TransformerConfig(action_space_size=376) does NOT warn."""
        # 376 is correct, should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Convert warnings to errors
            # This should succeed without raising
            TransformerConfig(action_space_size=376)


class TestTransformerConfigBackwardsCompatibility:
    """Test that custom values override defaults correctly."""

    def test_custom_values_override_defaults(self) -> None:
        """Verify custom values can override defaults without errors."""
        # Create config with custom values (e.g., for larger models)
        custom_config = TransformerConfig(
            spatial_size=16,  # Larger board
            embed_dim=256,  # Larger embedding
            num_layers=12,  # Deeper network
            action_space_size=512,  # Different action space
        )

        # Verify custom values are used
        assert custom_config.spatial_size == 16
        assert custom_config.embed_dim == 256
        assert custom_config.num_layers == 12
        assert custom_config.action_space_size == 512

        # Verify head dimensions scale correctly with new embed_dim
        assert custom_config.value_head_dim == 64  # Still default
        assert custom_config.policy_hidden_dim == 128  # Still default

    def test_partial_override(self) -> None:
        """Verify partial overrides work correctly."""
        config = TransformerConfig(
            spatial_size=12,  # Custom board size
            # Other params use defaults
        )

        # Custom value
        assert config.spatial_size == 12

        # Defaults unchanged
        assert config.action_space_size == 376
        assert config.embed_dim == 64
        assert config.num_layers == 6

    def test_head_dims_can_be_overridden(self) -> None:
        """Verify head dimensions can be independently overridden."""
        config = TransformerConfig(
            embed_dim=128,
            value_head_dim=256,  # Override default 1x scaling
            policy_hidden_dim=512,  # Override default 2x scaling
        )

        assert config.embed_dim == 128
        assert config.value_head_dim == 256  # Not 1x embed_dim
        assert config.policy_hidden_dim == 512  # Not 2x embed_dim


class TestNetworkArchitectureConfigDefaults:
    """Additional tests for NetworkArchitectureConfig default values."""

    def test_all_expected_defaults(self) -> None:
        """Verify all NetworkArchitectureConfig defaults match documentation."""
        config = NetworkArchitectureConfig()

        # Architecture dimensions
        assert config.board_size == 8
        assert config.action_space_size == 376
        assert config.spatial_channels == 13  # MicroCivEncoder.TOTAL_CHANNELS
        assert config.vector_dim == 64

        # Network parameters
        assert config.num_layers == 6
        assert config.embed_dim == 64

    def test_from_env_with_no_env_vars(self) -> None:
        """Verify from_env() uses defaults when no env vars are set."""
        # This should behave identically to default constructor
        config = NetworkArchitectureConfig.from_env()

        assert config.board_size == 8
        assert config.action_space_size == 376
        assert config.spatial_channels == 13
        assert config.vector_dim == 64
        assert config.num_layers == 6
        assert config.embed_dim == 64

    def test_to_dict_roundtrip(self) -> None:
        """Verify to_dict() includes all fields."""
        config = NetworkArchitectureConfig()
        config_dict = config.to_dict()

        # All fields present
        assert set(config_dict.keys()) == {
            "board_size",
            "action_space_size",
            "spatial_channels",
            "vector_dim",
            "num_layers",
            "embed_dim",
            "num_heads",
            "policy_hidden_dim",
            "value_head_dim",
            "encoder_type",
        }

        # Values match
        assert config_dict["board_size"] == 8
        assert config_dict["action_space_size"] == 376
        assert config_dict["spatial_channels"] == 13
        assert config_dict["vector_dim"] == 64
        assert config_dict["num_layers"] == 6
        assert config_dict["embed_dim"] == 64


class TestTransformerConfigDerivedProperties:
    """Test derived properties computed in __post_init__."""

    def test_num_patches_computed(self) -> None:
        """Verify _num_patches is computed correctly."""
        config = TransformerConfig(spatial_size=8, patch_size=4)
        # (8 // 4) ** 2 = 2 ** 2 = 4 patches
        assert config._num_patches == 4

        config = TransformerConfig(spatial_size=16, patch_size=4)
        # (16 // 4) ** 2 = 4 ** 2 = 16 patches
        assert config._num_patches == 16

    def test_head_dim_computed(self) -> None:
        """Verify _head_dim is computed correctly."""
        config = TransformerConfig(embed_dim=64, num_heads=8)
        # 64 // 8 = 8
        assert config._head_dim == 8

        config = TransformerConfig(embed_dim=256, num_heads=8)
        # 256 // 8 = 32
        assert config._head_dim == 32

    def test_embed_dim_divisible_by_num_heads(self) -> None:
        """Verify assertion when embed_dim is not divisible by num_heads."""
        with pytest.raises(AssertionError, match=r"embed_dim.*must be divisible by num_heads"):
            TransformerConfig(embed_dim=65, num_heads=8)  # 65 % 8 != 0
