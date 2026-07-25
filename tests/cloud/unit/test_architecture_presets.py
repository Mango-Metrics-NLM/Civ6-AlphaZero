"""Tests for architecture presets and TransformerConfig bridge.

Covers:
- NetworkArchitectureConfig.from_preset() for small/medium/large
- New fields (num_heads, policy_hidden_dim, value_head_dim)
- to_dict() / from_env() with new fields
- TransformerConfig.from_network_config() bridge
- ARCHITECTURE_PRESETS dict validation
"""

from __future__ import annotations

import pytest

from alpha_zero.cloud.gcp.config import (
    ARCHITECTURE_PRESETS,
    NetworkArchitectureConfig,
)
from alpha_zero.cloud.gcp.exceptions import ConfigurationError


class TestArchitecturePresets:
    """Tests for ARCHITECTURE_PRESETS dict."""

    def test_presets_have_required_keys(self) -> None:
        """Every preset must define the core architecture fields."""
        required_keys = {
            "embed_dim",
            "num_layers",
            "num_heads",
            "policy_hidden_dim",
            "value_head_dim",
        }
        for name, preset in ARCHITECTURE_PRESETS.items():
            assert required_keys <= set(preset.keys()), (
                f"Preset '{name}' missing: {required_keys - set(preset.keys())}"
            )

    def test_presets_values_positive(self) -> None:
        """All preset values must be positive integers."""
        for name, preset in ARCHITECTURE_PRESETS.items():
            for key, value in preset.items():
                assert isinstance(value, int) and value > 0, f"Preset '{name}'.{key} = {value}"

    def test_presets_embed_dim_divisible_by_heads(self) -> None:
        """embed_dim must be divisible by num_heads for attention."""
        for name, preset in ARCHITECTURE_PRESETS.items():
            assert preset["embed_dim"] % preset["num_heads"] == 0, (
                f"Preset '{name}': embed_dim={preset['embed_dim']} not divisible by num_heads={preset['num_heads']}"
            )


class TestNetworkArchitectureConfigFromPreset:
    """Tests for NetworkArchitectureConfig.from_preset()."""

    def test_preset_small_matches_defaults(self) -> None:
        """Small preset should produce identical values to default constructor."""
        default = NetworkArchitectureConfig()
        small = NetworkArchitectureConfig.from_preset("small")

        assert small.embed_dim == default.embed_dim == 64
        assert small.num_layers == default.num_layers == 6
        assert small.num_heads == default.num_heads == 8
        assert small.policy_hidden_dim == default.policy_hidden_dim == 128
        assert small.value_head_dim == default.value_head_dim == 64
        # Environment-independent fields should remain at defaults
        assert small.board_size == 8
        assert small.action_space_size == 376
        assert small.spatial_channels == 13
        assert small.vector_dim == 64

    def test_preset_medium_values(self) -> None:
        """Medium preset matches v000's proven architecture."""
        medium = NetworkArchitectureConfig.from_preset("medium")

        assert medium.embed_dim == 256
        assert medium.num_layers == 8
        assert medium.num_heads == 8
        assert medium.policy_hidden_dim == 256
        assert medium.value_head_dim == 128
        # Unchanged fields
        assert medium.board_size == 8
        assert medium.action_space_size == 376

    def test_preset_large_values(self) -> None:
        """Large preset for future scaling."""
        large = NetworkArchitectureConfig.from_preset("large")

        assert large.embed_dim == 512
        assert large.num_layers == 12
        assert large.num_heads == 16
        assert large.policy_hidden_dim == 512
        assert large.value_head_dim == 256

    def test_invalid_preset_raises(self) -> None:
        """Unknown preset name should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Unknown architecture preset"):
            NetworkArchitectureConfig.from_preset("nonexistent")

    def test_invalid_preset_error_lists_valid(self) -> None:
        """Error message should list valid preset names."""
        with pytest.raises(ConfigurationError, match="large") as exc_info:
            NetworkArchitectureConfig.from_preset("xxl")
        msg = str(exc_info.value)
        assert "small" in msg
        assert "medium" in msg
        assert "large" in msg


class TestNetworkArchitectureConfigNewFields:
    """Tests for new fields on NetworkArchitectureConfig."""

    def test_new_fields_have_defaults(self) -> None:
        """New fields should have backwards-compatible defaults."""
        config = NetworkArchitectureConfig()
        assert config.num_heads == 8
        assert config.policy_hidden_dim == 128
        assert config.value_head_dim == 64

    def test_new_fields_in_to_dict(self) -> None:
        """to_dict() should include all new fields."""
        config = NetworkArchitectureConfig.from_preset("medium")
        d = config.to_dict()

        assert d["num_heads"] == 8
        assert d["policy_hidden_dim"] == 256
        assert d["value_head_dim"] == 128
        assert d["embed_dim"] == 256
        assert d["num_layers"] == 8
        # Also verify original fields still present
        assert d["board_size"] == 8
        assert d["action_space_size"] == 376

    def test_from_env_new_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_env() should read new fields from env vars."""
        monkeypatch.setenv("CIV6AZ_NUM_HEADS", "16")
        monkeypatch.setenv("CIV6AZ_POLICY_HIDDEN_DIM", "512")
        monkeypatch.setenv("CIV6AZ_VALUE_HEAD_DIM", "256")

        config = NetworkArchitectureConfig.from_env()
        assert config.num_heads == 16
        assert config.policy_hidden_dim == 512
        assert config.value_head_dim == 256

    def test_from_env_defaults_without_vars(self) -> None:
        """from_env() without env vars should use defaults."""
        config = NetworkArchitectureConfig.from_env()
        assert config.num_heads == 8
        assert config.policy_hidden_dim == 128
        assert config.value_head_dim == 64

    def test_to_dict_roundtrip(self) -> None:
        """to_dict() then construct from dict should reproduce config."""
        original = NetworkArchitectureConfig.from_preset("medium")
        d = original.to_dict()
        restored = NetworkArchitectureConfig(**d)
        assert restored == original


class TestTransformerConfigBridge:
    """Tests for TransformerConfig.from_network_config()."""

    def test_from_network_config_small(self) -> None:
        """Bridge from small preset should match TransformerConfig defaults."""
        from alpha_zero.core.transformer_network import TransformerConfig

        net_config = NetworkArchitectureConfig.from_preset("small")
        tc = TransformerConfig.from_network_config(net_config)

        assert tc.embed_dim == 64
        assert tc.num_layers == 6
        assert tc.num_heads == 8
        assert tc.spatial_size == 8
        assert tc.spatial_channels == 13
        assert tc.vector_dim == 64
        assert tc.action_space_size == 376
        assert tc.policy_hidden_dim == 128
        assert tc.value_head_dim == 64

    def test_from_network_config_medium(self) -> None:
        """Bridge from medium preset should map all fields."""
        from alpha_zero.core.transformer_network import TransformerConfig

        net_config = NetworkArchitectureConfig.from_preset("medium")
        tc = TransformerConfig.from_network_config(net_config)

        assert tc.embed_dim == 256
        assert tc.num_layers == 8
        assert tc.num_heads == 8
        assert tc.policy_hidden_dim == 256
        assert tc.value_head_dim == 128

    def test_from_network_config_large(self) -> None:
        """Bridge from large preset should map all fields."""
        from alpha_zero.core.transformer_network import TransformerConfig

        net_config = NetworkArchitectureConfig.from_preset("large")
        tc = TransformerConfig.from_network_config(net_config)

        assert tc.embed_dim == 512
        assert tc.num_layers == 12
        assert tc.num_heads == 16
        assert tc.policy_hidden_dim == 512
        assert tc.value_head_dim == 256

    def test_from_network_config_preserves_non_mapped_defaults(self) -> None:
        """Non-mapped fields should retain TransformerConfig defaults."""
        from alpha_zero.core.transformer_network import TransformerConfig

        net_config = NetworkArchitectureConfig.from_preset("medium")
        tc = TransformerConfig.from_network_config(net_config)

        # These should be TransformerConfig defaults, not touched by bridge
        assert tc.mlp_ratio == 4.0
        assert tc.dropout == 0.1
        assert tc.patch_size == 4
        assert tc.cls_token is True

    def test_from_network_config_backwards_compat(self) -> None:
        """Bridge should work with objects missing new fields (getattr fallback)."""
        from alpha_zero.core.transformer_network import TransformerConfig

        # Simulate old-style config without new fields
        class OldConfig:
            board_size = 8
            action_space_size = 376
            spatial_channels = 13
            vector_dim = 64
            num_layers = 6
            embed_dim = 64
            # Missing: num_heads, policy_hidden_dim, value_head_dim

        tc = TransformerConfig.from_network_config(OldConfig())
        assert tc.num_heads == 8  # default fallback
        assert tc.policy_hidden_dim == 128  # default fallback
        assert tc.value_head_dim == 64  # default fallback
