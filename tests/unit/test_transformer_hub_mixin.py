"""Unit tests for alpha_zero.core.transformer_hub_mixin.

Tests cover:
- Construction with default and custom kwargs
- save_pretrained / from_pretrained round-trip
- Weight parity with plain Civ6TransformerNetwork
- from_network weight transfer bridge
- Config serialisation to config.json
"""

import json

import pytest
import torch

# Guard: skip if hub extra is missing.
hf_hub = pytest.importorskip("huggingface_hub", reason="requires hub extra")

from alpha_zero.core.transformer_hub_mixin import (  # noqa: E402
    TRANSFORMER_HUB_TAGS,
    Civ6TransformerNetworkForHub,
    _config_to_flat_kwargs,
)
from alpha_zero.core.transformer_network import (  # noqa: E402
    Civ6TransformerNetwork,
    TransformerConfig,
)


class TestConstruction:
    """Tests for Civ6TransformerNetworkForHub construction."""

    def test_default_construction(self):
        model = Civ6TransformerNetworkForHub()
        assert isinstance(model, Civ6TransformerNetwork)
        assert hasattr(model, "config")

    def test_custom_kwargs(self):
        model = Civ6TransformerNetworkForHub(
            spatial_channels=7,
            embed_dim=32,
            num_layers=2,
            num_heads=4,
        )
        assert model.config.spatial_channels == 7
        assert model.config.embed_dim == 32

    def test_parameter_count_matches_parent(self):
        hub = Civ6TransformerNetworkForHub(num_layers=2, num_heads=4, embed_dim=32)
        plain = Civ6TransformerNetwork(
            TransformerConfig(
                num_layers=2,
                num_heads=4,
                embed_dim=32,
            )
        )
        hub_params = sum(p.numel() for p in hub.parameters())
        plain_params = sum(p.numel() for p in plain.parameters())
        assert hub_params == plain_params


class TestSaveLoadRoundTrip:
    """Tests for save_pretrained / from_pretrained."""

    def test_round_trip_preserves_weights(self, tmp_path):
        torch.manual_seed(42)
        original = Civ6TransformerNetworkForHub(
            spatial_channels=7,
            embed_dim=32,
            num_layers=2,
            num_heads=4,
        )
        save_dir = tmp_path / "saved"
        original.save_pretrained(str(save_dir))

        # Verify files created.
        assert (save_dir / "model.safetensors").exists()
        assert (save_dir / "config.json").exists()

        # Reload.
        loaded = Civ6TransformerNetworkForHub.from_pretrained(str(save_dir))

        # Verify weight parity.
        for (name_a, param_a), (name_b, param_b) in zip(
            original.named_parameters(),
            loaded.named_parameters(),
            strict=True,
        ):
            assert name_a == name_b
            assert torch.allclose(param_a, param_b, atol=1e-6), f"Weight mismatch for {name_a}"

    def test_config_json_is_valid(self, tmp_path):
        model = Civ6TransformerNetworkForHub(embed_dim=32, num_layers=2, num_heads=4)
        save_dir = tmp_path / "saved"
        model.save_pretrained(str(save_dir))

        config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
        assert config["embed_dim"] == 32
        assert config["num_layers"] == 2

    def test_config_json_has_no_private_fields(self, tmp_path):
        model = Civ6TransformerNetworkForHub()
        save_dir = tmp_path / "saved"
        model.save_pretrained(str(save_dir))

        config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
        # Private/computed fields should not appear.
        assert "_num_patches" not in config
        assert "_head_dim" not in config


class TestFromNetwork:
    """Tests for the weight transfer bridge."""

    def test_from_network_preserves_weights(self):
        torch.manual_seed(42)
        config = TransformerConfig(
            spatial_channels=7,
            embed_dim=32,
            num_layers=2,
            num_heads=4,
        )
        plain = Civ6TransformerNetwork(config)
        hub = Civ6TransformerNetworkForHub.from_network(plain)

        for (n_p, p_p), (n_h, p_h) in zip(
            plain.named_parameters(),
            hub.named_parameters(),
            strict=True,
        ):
            assert n_p == n_h
            assert torch.equal(p_p, p_h), f"Weight mismatch for {n_p}"

    def test_from_transformer_config(self):
        config = TransformerConfig(embed_dim=32, num_layers=2, num_heads=4)
        hub = Civ6TransformerNetworkForHub.from_transformer_config(config)
        assert hub.config.embed_dim == 32


class TestConfigToFlatKwargs:
    """Tests for the _config_to_flat_kwargs helper."""

    def test_excludes_non_init_fields(self):
        config = TransformerConfig()
        kwargs = _config_to_flat_kwargs(config)
        assert "_num_patches" not in kwargs
        assert "_head_dim" not in kwargs

    def test_enums_converted_to_strings(self):
        config = TransformerConfig()
        kwargs = _config_to_flat_kwargs(config)
        # norm_type and attention_type should be string values.
        assert isinstance(kwargs["norm_type"], str)
        assert isinstance(kwargs["attention_type"], str)


class TestHubTags:
    """Tests for TRANSFORMER_HUB_TAGS."""

    def test_contains_expected_tags(self):
        assert "transformer" in TRANSFORMER_HUB_TAGS
        assert "alphazero" in TRANSFORMER_HUB_TAGS
        assert "pytorch" in TRANSFORMER_HUB_TAGS

    def test_is_tuple_of_strings(self):
        assert isinstance(TRANSFORMER_HUB_TAGS, tuple)
        assert all(isinstance(t, str) for t in TRANSFORMER_HUB_TAGS)
