"""Unit tests for alpha_zero.core.muzero_hub_mixin.

Tests cover:
- Construction with default and custom kwargs
- save_pretrained / from_pretrained round-trip
- Weight parity with plain MuZeroNetwork
- from_network weight transfer bridge
"""

import json

import pytest
import torch

# Guard: skip if hub extra is missing.
hf_hub = pytest.importorskip("huggingface_hub", reason="requires hub extra")

from alpha_zero.core.muzero_hub_mixin import (  # noqa: E402
    MUZERO_HUB_TAGS,
    MuZeroNetworkForHub,
)
from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork  # noqa: E402


class TestConstruction:
    """Tests for MuZeroNetworkForHub construction."""

    def test_default_construction(self):
        model = MuZeroNetworkForHub()
        assert isinstance(model, MuZeroNetwork)
        assert hasattr(model, "config")

    def test_custom_kwargs(self):
        model = MuZeroNetworkForHub(
            spatial_channels=7,
            spatial_size=8,
            num_res_blocks=2,
        )
        assert model.config.spatial_channels == 7
        assert model.config.spatial_size == 8

    def test_parameter_count_matches_parent(self):
        hub = MuZeroNetworkForHub(num_res_blocks=2, num_filters=32, spatial_size=8)
        plain = MuZeroNetwork(
            MuZeroConfig(
                num_res_blocks=2,
                num_filters=32,
                spatial_size=8,
            )
        )
        hub_params = sum(p.numel() for p in hub.parameters())
        plain_params = sum(p.numel() for p in plain.parameters())
        assert hub_params == plain_params


class TestSaveLoadRoundTrip:
    """Tests for save_pretrained / from_pretrained."""

    def test_round_trip_preserves_weights(self, tmp_path):
        torch.manual_seed(42)
        original = MuZeroNetworkForHub(
            spatial_channels=7,
            spatial_size=8,
            num_res_blocks=2,
            num_filters=32,
        )
        save_dir = tmp_path / "saved"
        original.save_pretrained(str(save_dir))

        assert (save_dir / "model.safetensors").exists()
        assert (save_dir / "config.json").exists()

        loaded = MuZeroNetworkForHub.from_pretrained(str(save_dir))

        for (name_a, param_a), (name_b, param_b) in zip(
            original.named_parameters(),
            loaded.named_parameters(),
            strict=True,
        ):
            assert name_a == name_b
            assert torch.allclose(param_a, param_b, atol=1e-6), f"Weight mismatch for {name_a}"

    def test_config_json_is_valid(self, tmp_path):
        model = MuZeroNetworkForHub(num_res_blocks=2, num_filters=32, spatial_size=8)
        save_dir = tmp_path / "saved"
        model.save_pretrained(str(save_dir))

        config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
        assert config["num_res_blocks"] == 2
        assert config["num_filters"] == 32


class TestFromNetwork:
    """Tests for the weight transfer bridge."""

    def test_from_network_preserves_weights(self):
        torch.manual_seed(42)
        config = MuZeroConfig(
            spatial_channels=7,
            spatial_size=8,
            num_res_blocks=2,
            num_filters=32,
        )
        plain = MuZeroNetwork(config)
        hub = MuZeroNetworkForHub.from_network(plain)

        for (n_p, p_p), (n_h, p_h) in zip(
            plain.named_parameters(),
            hub.named_parameters(),
            strict=True,
        ):
            assert n_p == n_h
            assert torch.equal(p_p, p_h), f"Weight mismatch for {n_p}"

    def test_from_muzero_config(self):
        config = MuZeroConfig(num_res_blocks=2, num_filters=32, spatial_size=8)
        hub = MuZeroNetworkForHub.from_muzero_config(config)
        assert hub.config.num_res_blocks == 2


class TestHubTags:
    """Tests for MUZERO_HUB_TAGS."""

    def test_contains_expected_tags(self):
        assert "muzero" in MUZERO_HUB_TAGS
        assert "world-model" in MUZERO_HUB_TAGS
        assert "pytorch" in MUZERO_HUB_TAGS

    def test_is_tuple_of_strings(self):
        assert isinstance(MUZERO_HUB_TAGS, tuple)
        assert all(isinstance(t, str) for t in MUZERO_HUB_TAGS)
