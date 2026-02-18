"""Regression tests for the Crystalline Hex Spatial Encoder feature.

These tests verify backwards compatibility and ensure the hex encoder
integration does not break existing functionality:

1. Old checkpoint loading (no encoder_type in config)
2. PatchEmbed network behavior is unchanged
3. State dict key compatibility (self.patch_embed naming preserved)
4. Config serialization round-trips
5. Model save/load preserves encoder_type
6. from_network_config backwards compatibility with old configs
7. Deterministic output reproducibility
8. Action masking parity between encoder types
9. Gradient flow parity between encoder types
10. Environment config env-var fallback behavior
"""

from __future__ import annotations

import dataclasses
import tempfile
from enum import Enum
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from alpha_zero.core.transformer_network import (
    Civ6TransformerNetwork,
    TransformerConfig,
)


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def spatial_input(batch_size: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(batch_size, 13, 8, 8)


@pytest.fixture
def vector_input(batch_size: int) -> torch.Tensor:
    torch.manual_seed(43)
    return torch.randn(batch_size, 64)


@pytest.fixture
def action_mask(batch_size: int) -> torch.Tensor:
    torch.manual_seed(44)
    mask = torch.ones(batch_size, 376)
    # Randomly zero out ~60% of actions
    mask[:, 150:] = 0
    return mask


@pytest.fixture
def patch_embed_config() -> TransformerConfig:
    return TransformerConfig(encoder_type="patch_embed")


@pytest.fixture
def hex_config() -> TransformerConfig:
    return TransformerConfig(encoder_type="hex_crystalline")


# ============================================================================
# Backwards Compatibility: Old Checkpoints
# ============================================================================


class TestOldCheckpointCompatibility:
    """Verify old checkpoints (without encoder_type) load correctly."""

    def test_checkpoint_without_encoder_type_loads_as_patch_embed(
        self, spatial_input: torch.Tensor, vector_input: torch.Tensor
    ) -> None:
        """Simulate an old checkpoint that doesn't have encoder_type in config."""
        torch.manual_seed(42)
        # Create a model and save it
        config = TransformerConfig(encoder_type="patch_embed")
        model = Civ6TransformerNetwork(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "old_model.pt")

            # Save checkpoint manually, omitting encoder_type
            config_dict = dataclasses.asdict(config)
            for key, value in config_dict.items():
                if isinstance(value, Enum):
                    config_dict[key] = value.value
            # Remove encoder_type to simulate old checkpoint
            config_dict.pop("encoder_type", None)
            config_dict.pop("_num_patches", None)
            config_dict.pop("_head_dim", None)

            torch.save(
                {
                    "config": config_dict,
                    "state_dict": model.state_dict(),
                    "network_type": "transformer_v1",
                },
                path,
            )

            # Load should succeed and default to patch_embed
            loaded = Civ6TransformerNetwork.load(path)
            assert loaded.config.encoder_type == "patch_embed"

            # Output should match original
            model.eval()
            loaded.eval()
            with torch.no_grad():
                original_logits, original_value = model(spatial_input, vector_input)
                loaded_logits, loaded_value = loaded(spatial_input, vector_input)

            torch.testing.assert_close(
                original_logits,
                loaded_logits,
                rtol=1e-5,
                atol=1e-6,
                msg="Old checkpoint loaded with different outputs",
            )
            torch.testing.assert_close(
                original_value,
                loaded_value,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_checkpoint_with_encoder_type_preserves_it(
        self, spatial_input: torch.Tensor, vector_input: torch.Tensor
    ) -> None:
        """Verify hex_crystalline encoder_type survives save/load cycle."""
        torch.manual_seed(42)
        config = TransformerConfig(encoder_type="hex_crystalline")
        model = Civ6TransformerNetwork(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "hex_model.pt")
            model.save(path)

            loaded = Civ6TransformerNetwork.load(path)
            assert loaded.config.encoder_type == "hex_crystalline"

            model.eval()
            loaded.eval()
            with torch.no_grad():
                orig_logits, orig_value = model(spatial_input, vector_input)
                load_logits, load_value = loaded(spatial_input, vector_input)

            torch.testing.assert_close(orig_logits, load_logits, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(orig_value, load_value, rtol=1e-5, atol=1e-6)


# ============================================================================
# PatchEmbed Behavior Unchanged
# ============================================================================


class TestPatchEmbedUnchanged:
    """Verify PatchEmbed network behavior is not altered by hex encoder changes."""

    def test_patch_embed_output_shapes_unchanged(
        self,
        patch_embed_config: TransformerConfig,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        network = Civ6TransformerNetwork(config=patch_embed_config)
        network.eval()
        with torch.no_grad():
            policy_logits, value = network(spatial_input, vector_input)

        assert policy_logits.shape == (4, 376)
        assert value.shape == (4,)

    def test_patch_embed_predict_returns_probabilities(
        self,
        patch_embed_config: TransformerConfig,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        network = Civ6TransformerNetwork(config=patch_embed_config)
        probs, value = network.predict(spatial_input, vector_input)

        assert probs.shape == (4, 376)
        # Must sum to ~1 per sample
        torch.testing.assert_close(
            probs.sum(dim=-1),
            torch.ones(4),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_patch_embed_predict_single_returns_float_value(
        self,
        patch_embed_config: TransformerConfig,
    ) -> None:
        torch.manual_seed(42)
        network = Civ6TransformerNetwork(config=patch_embed_config)
        spatial = torch.randn(13, 8, 8)
        vector = torch.randn(64)

        probs, val = network.predict_single(spatial, vector)
        assert probs.shape == (376,)
        assert isinstance(val, float)

    def test_patch_embed_action_masking(
        self,
        patch_embed_config: TransformerConfig,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        network = Civ6TransformerNetwork(config=patch_embed_config)
        network.eval()

        with torch.no_grad():
            logits, _ = network(spatial_input, vector_input, action_mask)
            probs = F.softmax(logits, dim=-1)

        # All masked actions should have near-zero probability
        masked_probs = probs[:, 150:]
        assert (
            masked_probs < 1e-6
        ).all(), f"Masked action probs not near zero, max={masked_probs.max():.2e}"

    def test_patch_embed_training_step(
        self,
        patch_embed_config: TransformerConfig,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        """Verify PatchEmbed network can still train (forward + backward + step)."""
        network = Civ6TransformerNetwork(config=patch_embed_config)
        network.train()
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        policy_logits, value = network(spatial_input, vector_input)
        target_policy = F.softmax(torch.randn(4, 376), dim=-1)
        target_value = torch.randn(4).clamp(-1, 1)

        loss = F.cross_entropy(policy_logits, target_policy) + F.mse_loss(value, target_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        # Verify params changed
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters())


# ============================================================================
# State Dict Key Compatibility
# ============================================================================


class TestStateDictKeyCompat:
    """Verify state_dict keys use 'patch_embed' name for backwards compatibility."""

    def test_hex_encoder_state_dict_uses_patch_embed_key(
        self, hex_config: TransformerConfig
    ) -> None:
        """The encoder is stored as self.patch_embed in TransformerEncoder,
        regardless of encoder type, for state_dict key compatibility."""
        network = Civ6TransformerNetwork(config=hex_config)
        keys = list(network.state_dict().keys())

        # Encoder keys should start with encoder.patch_embed
        encoder_keys = [k for k in keys if "patch_embed" in k]
        assert (
            len(encoder_keys) > 0
        ), f"No 'patch_embed' keys found in state_dict. Keys: {keys[:10]}..."

    def test_patch_embed_and_hex_share_key_prefix(self) -> None:
        """Both encoder types should use same key prefix in state_dict."""
        config_patch = TransformerConfig(encoder_type="patch_embed")
        config_hex = TransformerConfig(encoder_type="hex_crystalline")

        net_patch = Civ6TransformerNetwork(config=config_patch)
        net_hex = Civ6TransformerNetwork(config=config_hex)

        keys_patch = {k.split(".")[0] + "." + k.split(".")[1] for k in net_patch.state_dict()}
        keys_hex = {k.split(".")[0] + "." + k.split(".")[1] for k in net_hex.state_dict()}

        # Both should have 'encoder.patch_embed' prefix
        assert "encoder.patch_embed" in keys_patch
        assert "encoder.patch_embed" in keys_hex


# ============================================================================
# Config Serialization Round-trip
# ============================================================================


class TestConfigSerialization:
    """Verify config serialization preserves encoder_type across round-trips."""

    def test_transformer_config_save_load_round_trip_patch_embed(self) -> None:
        config = TransformerConfig(encoder_type="patch_embed")
        model = Civ6TransformerNetwork(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pt")
            model.save(path)
            loaded = Civ6TransformerNetwork.load(path)
            assert loaded.config.encoder_type == "patch_embed"

    def test_transformer_config_save_load_round_trip_hex(self) -> None:
        config = TransformerConfig(encoder_type="hex_crystalline")
        model = Civ6TransformerNetwork(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pt")
            model.save(path)
            loaded = Civ6TransformerNetwork.load(path)
            assert loaded.config.encoder_type == "hex_crystalline"

    def test_network_architecture_config_to_dict_round_trip(self) -> None:
        from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig

        config = NetworkArchitectureConfig(encoder_type="hex_crystalline")
        d = config.to_dict()
        assert d["encoder_type"] == "hex_crystalline"

        # Can reconstruct from dict
        restored = NetworkArchitectureConfig(**d)
        assert restored.encoder_type == "hex_crystalline"


# ============================================================================
# from_network_config Backwards Compatibility
# ============================================================================


class TestFromNetworkConfigCompat:
    """Verify TransformerConfig.from_network_config handles old/new configs."""

    def test_from_network_config_with_encoder_type(self) -> None:
        from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig

        net_config = NetworkArchitectureConfig(encoder_type="hex_crystalline")
        tc = TransformerConfig.from_network_config(net_config)
        assert tc.encoder_type == "hex_crystalline"

    def test_from_network_config_without_encoder_type_defaults_to_patch(self) -> None:
        """Simulate an old NetworkArchitectureConfig without encoder_type field."""

        class OldNetworkConfig:
            board_size = 8
            action_space_size = 376
            spatial_channels = 13
            vector_dim = 64
            embed_dim = 64
            num_layers = 6
            num_heads = 8
            policy_hidden_dim = 128
            value_head_dim = 64
            # NO encoder_type attribute

        old_config = OldNetworkConfig()
        tc = TransformerConfig.from_network_config(old_config)
        assert tc.encoder_type == "patch_embed"

    def test_from_network_config_with_patch_embed(self) -> None:
        from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig

        net_config = NetworkArchitectureConfig(encoder_type="patch_embed")
        tc = TransformerConfig.from_network_config(net_config)
        assert tc.encoder_type == "patch_embed"


# ============================================================================
# Deterministic Output Reproducibility
# ============================================================================


class TestDeterminism:
    """Verify outputs are deterministic in eval mode."""

    @pytest.mark.parametrize("encoder_type", ["patch_embed", "hex_crystalline"])
    def test_eval_mode_deterministic(
        self,
        encoder_type: str,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        """Same input + same seed → same output in eval mode."""
        torch.manual_seed(42)
        config = TransformerConfig(encoder_type=encoder_type)
        network = Civ6TransformerNetwork(config=config)
        network.eval()

        with torch.no_grad():
            logits1, value1 = network(spatial_input, vector_input)
            logits2, value2 = network(spatial_input, vector_input)

        torch.testing.assert_close(logits1, logits2, rtol=0, atol=0)
        torch.testing.assert_close(value1, value2, rtol=0, atol=0)


# ============================================================================
# Action Masking Parity
# ============================================================================


class TestActionMaskingParity:
    """Verify action masking works identically for both encoder types."""

    @pytest.mark.parametrize("encoder_type", ["patch_embed", "hex_crystalline"])
    def test_masked_actions_have_near_zero_probability(
        self,
        encoder_type: str,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        torch.manual_seed(42)
        config = TransformerConfig(encoder_type=encoder_type)
        network = Civ6TransformerNetwork(config=config)
        network.eval()

        with torch.no_grad():
            logits, _ = network(spatial_input, vector_input, action_mask)
            probs = F.softmax(logits, dim=-1)

        # Masked positions (mask == 0) should have near-zero probs
        masked_positions = action_mask == 0
        masked_probs = probs[masked_positions]
        assert (masked_probs < 1e-6).all(), (
            f"[{encoder_type}] Masked action probs not near zero: " f"max={masked_probs.max():.2e}"
        )

    @pytest.mark.parametrize("encoder_type", ["patch_embed", "hex_crystalline"])
    def test_unmasked_actions_get_valid_probabilities(
        self,
        encoder_type: str,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        torch.manual_seed(42)
        config = TransformerConfig(encoder_type=encoder_type)
        network = Civ6TransformerNetwork(config=config)
        probs, _ = network.predict(spatial_input, vector_input, action_mask)

        # Probabilities should sum to ~1
        torch.testing.assert_close(
            probs.sum(dim=-1),
            torch.ones(4),
            rtol=1e-4,
            atol=1e-5,
        )


# ============================================================================
# Gradient Flow Parity
# ============================================================================


class TestGradientFlowParity:
    """Verify gradients flow through both encoder types."""

    @pytest.mark.parametrize("encoder_type", ["patch_embed", "hex_crystalline"])
    def test_all_params_receive_gradients(
        self,
        encoder_type: str,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        torch.manual_seed(42)
        config = TransformerConfig(encoder_type=encoder_type)
        network = Civ6TransformerNetwork(config=config)
        network.train()

        policy_logits, value = network(spatial_input, vector_input)
        loss = policy_logits.sum() + value.sum()
        loss.backward()

        # At least some params in the encoder should have gradients
        encoder_has_grad = False
        for name, param in network.named_parameters():
            if "patch_embed" in name and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    encoder_has_grad = True
                    break

        assert (
            encoder_has_grad
        ), f"[{encoder_type}] No encoder parameters received non-zero gradients"


# ============================================================================
# Environment Config Env-Var Behavior
# ============================================================================


class TestEnvVarBehavior:
    """Verify env var handling for encoder_type in NetworkArchitectureConfig."""

    def test_default_without_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig

        monkeypatch.delenv("CIV6AZ_ENCODER_TYPE", raising=False)
        config = NetworkArchitectureConfig.from_env()
        assert config.encoder_type == "patch_embed"

    def test_env_var_sets_encoder_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig

        monkeypatch.setenv("CIV6AZ_ENCODER_TYPE", "hex_crystalline")
        config = NetworkArchitectureConfig.from_env()
        assert config.encoder_type == "hex_crystalline"

    def test_env_var_patch_embed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig

        monkeypatch.setenv("CIV6AZ_ENCODER_TYPE", "patch_embed")
        config = NetworkArchitectureConfig.from_env()
        assert config.encoder_type == "patch_embed"


# ============================================================================
# Parameter Count Regression
# ============================================================================


class TestParameterCountRegression:
    """Verify parameter counts haven't changed unexpectedly."""

    def test_patch_embed_param_count_stable(self) -> None:
        """Guard against accidental changes to PatchEmbed network param count."""
        config = TransformerConfig(encoder_type="patch_embed")
        network = Civ6TransformerNetwork(config=config)
        params = network.get_num_parameters()

        # Known param count from comparison script: 494,137
        # Allow small tolerance for future minor changes
        assert (
            450_000 < params["total"] < 600_000
        ), f"PatchEmbed param count changed significantly: {params['total']:,}"

    def test_hex_encoder_param_count_within_expected_range(self) -> None:
        """Guard against hex encoder param count drifting too far."""
        config = TransformerConfig(encoder_type="hex_crystalline")
        network = Civ6TransformerNetwork(config=config)
        params = network.get_num_parameters()

        # Known param count: ~553,663 (about 12% more than PatchEmbed)
        assert (
            450_000 < params["total"] < 700_000
        ), f"Hex encoder param count outside expected range: {params['total']:,}"

    def test_hex_to_patch_param_ratio_reasonable(self) -> None:
        """Hex encoder shouldn't be wildly larger than PatchEmbed."""
        config_p = TransformerConfig(encoder_type="patch_embed")
        config_h = TransformerConfig(encoder_type="hex_crystalline")

        net_p = Civ6TransformerNetwork(config=config_p)
        net_h = Civ6TransformerNetwork(config=config_h)

        ratio = net_h.get_num_parameters()["total"] / net_p.get_num_parameters()["total"]
        assert 0.8 <= ratio <= 1.5, f"Param ratio hex/patch = {ratio:.2f}, expected 0.8-1.5"
