"""Integration tests for hex encoder pipeline in Civ6 AlphaZero.

This test suite verifies that HexSpatialEncoder works correctly as a drop-in
replacement for PatchEmbed inside Civ6TransformerNetwork, ensuring:
- Forward pass compatibility
- Model serialization/deserialization
- Training compatibility
- Action masking behavior
- Parameter count reasonableness

The tests validate the complete pipeline from spatial input through hex encoder
to policy/value outputs.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from alpha_zero.core.transformer_network import (
    Civ6TransformerNetwork,
    TransformerConfig,
)


@pytest.fixture
def batch_size() -> int:
    """Batch size for tests."""
    return 4


@pytest.fixture
def spatial_input(batch_size: int) -> torch.Tensor:
    """Create dummy spatial input (B, 13, 8, 8)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, 13, 8, 8)


@pytest.fixture
def vector_input(batch_size: int) -> torch.Tensor:
    """Create dummy vector input (B, 64)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, 64)


@pytest.fixture
def action_mask(batch_size: int) -> torch.Tensor:
    """Create dummy action mask (B, 376)."""
    torch.manual_seed(42)
    # First 100 actions valid, rest masked
    mask = torch.zeros(batch_size, 376)
    mask[:, :100] = 1.0
    return mask


class TestHexEncoderIntegration:
    """Integration tests for hex encoder in transformer network."""

    def test_full_forward_pass_with_hex_encoder(
        self,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        """Test that forward pass works with hex encoder.

        Verifies:
        - Network can be created with encoder_type='hex_crystalline'
        - Forward pass produces correct output shapes
        - Policy logits are shape (B, 376)
        - Value is shape (B,) after squeeze
        """
        torch.manual_seed(42)

        # Create network with hex encoder
        config = TransformerConfig(
            encoder_type="hex_crystalline",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,  # Small for speed
            num_heads=4,
            dropout=0.0,  # Deterministic
        )
        network = Civ6TransformerNetwork(config=config)
        network.eval()

        # Forward pass
        with torch.no_grad():
            policy_logits, value = network(spatial_input, vector_input)

        # Check shapes
        assert policy_logits.shape == (
            4,
            376,
        ), f"Expected policy shape (4, 376), got {policy_logits.shape}"
        assert value.shape == (4,), f"Expected value shape (4,), got {value.shape}"

        # Check outputs are finite
        assert torch.isfinite(policy_logits).all(), "Policy logits contain NaN/Inf"
        assert torch.isfinite(value).all(), "Value contains NaN/Inf"

        # Check value is in reasonable range (tanh output)
        assert (value >= -1.0).all() and (value <= 1.0).all(), (
            f"Value should be in [-1, 1], got range [{value.min()}, {value.max()}]"
        )

    def test_full_forward_pass_with_patch_embed_regression(
        self,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        """Test that PatchEmbed still works (regression test).

        Verifies:
        - Default encoder_type='patch_embed' still works
        - Forward pass produces correct shapes
        - Baseline functionality is preserved
        """
        torch.manual_seed(42)

        # Create network with default PatchEmbed
        config = TransformerConfig(
            encoder_type="patch_embed",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        network = Civ6TransformerNetwork(config=config)
        network.eval()

        # Forward pass
        with torch.no_grad():
            policy_logits, value = network(spatial_input, vector_input)

        # Check shapes match expected
        assert policy_logits.shape == (4, 376)
        assert value.shape == (4,)
        assert torch.isfinite(policy_logits).all()
        assert torch.isfinite(value).all()

    def test_save_hex_model_load_verify_output_matches(
        self,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        """Test save/load cycle preserves hex encoder functionality.

        Verifies:
        - Hex network can be saved to checkpoint
        - Checkpoint can be loaded into new network
        - Loaded network produces identical outputs
        - encoder_type is preserved in config
        """
        torch.manual_seed(42)

        # Create and run original network
        config = TransformerConfig(
            encoder_type="hex_crystalline",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        network1 = Civ6TransformerNetwork(config=config)
        network1.eval()

        with torch.no_grad():
            policy1, value1 = network1(spatial_input, vector_input)

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "hex_model.pt"
            network1.save(str(checkpoint_path))

            # Load into new network
            network2 = Civ6TransformerNetwork.load(str(checkpoint_path))
            network2.eval()

            # Verify config preserved
            assert network2.config.encoder_type == "hex_crystalline", (
                f"Expected encoder_type='hex_crystalline', got '{network2.config.encoder_type}'"
            )

            # Run loaded network
            with torch.no_grad():
                policy2, value2 = network2(spatial_input, vector_input)

        # Outputs should match exactly (same weights, deterministic)
        torch.testing.assert_close(
            policy1,
            policy2,
            rtol=1e-5,
            atol=1e-6,
            msg="Policy outputs don't match after save/load",
        )
        torch.testing.assert_close(
            value1,
            value2,
            rtol=1e-5,
            atol=1e-6,
            msg="Value outputs don't match after save/load",
        )

    def test_save_patch_embed_model_load_patch_embed_used(
        self,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        """Test that PatchEmbed models save/load correctly.

        Verifies:
        - Default PatchEmbed network saves correctly
        - Loaded network uses PatchEmbed (not hex)
        - Outputs match after save/load cycle
        """
        torch.manual_seed(42)

        # Create PatchEmbed network
        config = TransformerConfig(
            encoder_type="patch_embed",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        network1 = Civ6TransformerNetwork(config=config)
        network1.eval()

        with torch.no_grad():
            policy1, value1 = network1(spatial_input, vector_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "patch_model.pt"
            network1.save(str(checkpoint_path))
            network2 = Civ6TransformerNetwork.load(str(checkpoint_path))
            network2.eval()

            # Verify encoder type preserved
            assert network2.config.encoder_type == "patch_embed"

            with torch.no_grad():
                policy2, value2 = network2(spatial_input, vector_input)

        # Outputs should match
        torch.testing.assert_close(policy1, policy2, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(value1, value2, rtol=1e-5, atol=1e-6)

    def test_old_checkpoint_without_encoder_type_loads_with_patch_embed(self) -> None:
        """Test backwards compatibility with old checkpoints.

        Verifies:
        - Old checkpoints without encoder_type field load successfully
        - Default encoder_type='patch_embed' is used
        - Config reconstruction works with missing fields
        """
        # Create config dict simulating old checkpoint (no encoder_type)
        config_dict = {
            "spatial_channels": 13,
            "spatial_size": 8,
            "vector_dim": 64,
            "action_space_size": 376,
            "embed_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "patch_size": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "path_dropout": 0.1,
            "policy_hidden_dim": 128,
            "value_head_dim": 64,
            "num_policy_heads": 1,
            "use_moe": False,
            "num_experts": 8,
            "experts_per_token": 2,
            "cls_token": True,
            "pool_type": "cls",
            "use_pre_norm": True,
            "use_rope": True,
            "use_qkv_bias": False,
            # Note: encoder_type is missing (simulates old checkpoint)
        }

        # Reconstruct config - should use default encoder_type
        config = TransformerConfig(**config_dict)

        # Should default to patch_embed
        assert config.encoder_type == "patch_embed", (
            f"Expected default encoder_type='patch_embed', got '{config.encoder_type}'"
        )

        # Network should create successfully
        network = Civ6TransformerNetwork(config=config)
        assert network.config.encoder_type == "patch_embed"

    def test_training_step_succeeds_with_hex_encoder(
        self,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        """Test that training step works with hex encoder.

        Verifies:
        - Network can be put in training mode
        - Forward pass works
        - Loss can be computed
        - Backward pass succeeds
        - Optimizer step succeeds
        - Gradients are computed for parameters
        """
        torch.manual_seed(42)

        # Create network with hex encoder
        config = TransformerConfig(
            encoder_type="hex_crystalline",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,  # Use dropout in training
        )
        network = Civ6TransformerNetwork(config=config)
        network.train()

        # Create optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        # Forward pass
        policy_logits, value = network(spatial_input, vector_input)

        # Create dummy targets
        target_policy = F.softmax(torch.randn(4, 376), dim=-1)
        target_value = torch.randn(4).clamp(-1, 1)

        # Compute loss
        policy_loss = F.cross_entropy(
            policy_logits,
            target_policy,
        )
        value_loss = F.mse_loss(value, target_value)
        total_loss = policy_loss + value_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Check gradients exist
        has_grad = False
        for param in network.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed during backward pass"

        # Optimizer step should succeed
        optimizer.step()

        # Loss should be finite
        assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss}"

    def test_predict_works_with_hex_encoder(
        self,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
    ) -> None:
        """Test that predict() method works with hex encoder.

        Verifies:
        - predict() returns probabilities (not logits)
        - Policy probabilities sum to ~1.0
        - Value is in [-1, 1] range
        - Outputs are deterministic in eval mode
        """
        torch.manual_seed(42)

        # Create network with hex encoder
        config = TransformerConfig(
            encoder_type="hex_crystalline",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        network = Civ6TransformerNetwork(config=config)
        network.eval()

        # Call predict
        with torch.no_grad():
            policy_probs, value = network.predict(spatial_input, vector_input)

        # Check shapes
        assert policy_probs.shape == (4, 376)
        assert value.shape == (4,)

        # Policy should be valid probabilities
        prob_sums = policy_probs.sum(dim=-1)
        torch.testing.assert_close(
            prob_sums,
            torch.ones(4),
            rtol=1e-5,
            atol=1e-6,
            msg="Policy probabilities don't sum to 1",
        )

        # All probabilities should be non-negative
        assert (policy_probs >= 0).all(), "Found negative probabilities"

        # Value should be in [-1, 1]
        assert (value >= -1.0).all() and (value <= 1.0).all(), (
            f"Value should be in [-1, 1], got range [{value.min()}, {value.max()}]"
        )

    def test_action_masking_with_hex_encoder(
        self,
        spatial_input: torch.Tensor,
        vector_input: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        """Test that action masking works correctly with hex encoder.

        Verifies:
        - Masked actions have very negative logits
        - Valid actions have reasonable logits
        - Masking doesn't break forward pass
        - predict() also respects action masks
        """
        torch.manual_seed(42)

        # Create network
        config = TransformerConfig(
            encoder_type="hex_crystalline",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        network = Civ6TransformerNetwork(config=config)
        network.eval()

        # Forward with mask (first 100 actions valid, rest masked)
        with torch.no_grad():
            policy_logits, _value = network(spatial_input, vector_input, action_mask)

        # Check masked positions have very negative logits
        masked_logits = policy_logits[:, 100:]  # Actions 100-375 are masked
        assert (masked_logits < -1e8).all(), (
            f"Masked logits should be < -1e8, got max {masked_logits.max()}"
        )

        # Check valid positions have reasonable logits
        valid_logits = policy_logits[:, :100]  # Actions 0-99 are valid
        assert (valid_logits > -1e8).all(), (
            f"Valid logits should be > -1e8, got min {valid_logits.min()}"
        )

        # Test predict() with mask
        with torch.no_grad():
            policy_probs, _ = network.predict(spatial_input, vector_input, action_mask)

        # Masked positions should have near-zero probability
        masked_probs = policy_probs[:, 100:]
        assert (masked_probs < 1e-6).all(), (
            f"Masked probabilities should be near zero, got max {masked_probs.max()}"
        )

        # Valid positions should sum to ~1.0
        valid_prob_sums = policy_probs[:, :100].sum(dim=-1)
        torch.testing.assert_close(
            valid_prob_sums,
            torch.ones(4),
            rtol=1e-3,
            atol=1e-4,
            msg="Valid probabilities don't sum to 1",
        )

    def test_parameter_count_comparison(self) -> None:
        """Test parameter counts for both encoder types.

        Verifies:
        - Both encoder types have reasonable parameter counts (< 10M)
        - Hex encoder is within 2x of PatchEmbed
        - Parameter counts are logged for transparency
        """
        torch.manual_seed(42)

        # Create PatchEmbed network
        config_patch = TransformerConfig(
            encoder_type="patch_embed",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=6,
            num_heads=8,
        )
        network_patch = Civ6TransformerNetwork(config=config_patch)
        params_patch = network_patch.get_num_parameters()

        # Create hex network
        config_hex = TransformerConfig(
            encoder_type="hex_crystalline",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=6,
            num_heads=8,
        )
        network_hex = Civ6TransformerNetwork(config=config_hex)
        params_hex = network_hex.get_num_parameters()

        # Check both are under 10M params
        assert params_patch["total"] < 10_000_000, (
            f"PatchEmbed network has too many params: {params_patch['total']}"
        )
        assert params_hex["total"] < 10_000_000, (
            f"Hex network has too many params: {params_hex['total']}"
        )

        # Check hex is within 2x of PatchEmbed
        ratio = params_hex["total"] / params_patch["total"]
        assert 0.5 <= ratio <= 2.0, (
            f"Hex encoder param count should be within 2x of PatchEmbed. "
            f"Ratio: {ratio:.2f} (hex={params_hex['total']:,}, patch={params_patch['total']:,})"
        )

        # Log for transparency (useful for debugging)
        print("\nParameter counts:")
        print(f"  PatchEmbed: {params_patch['total']:,} total")
        print(f"  Hex encoder: {params_hex['total']:,} total")
        print(f"  Ratio (hex/patch): {ratio:.2f}")

    def test_config_serialization(self) -> None:
        """Test config serialization preserves encoder_type.

        Verifies:
        - dataclasses.asdict() includes encoder_type
        - Config can be reconstructed from dict
        - Enum values are handled correctly
        """
        # Create config with hex encoder
        config = TransformerConfig(
            encoder_type="hex_crystalline",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
        )

        # Serialize to dict
        config_dict = dataclasses.asdict(config)

        # Should include encoder_type
        assert "encoder_type" in config_dict, "encoder_type missing from serialized config"
        assert config_dict["encoder_type"] == "hex_crystalline"

        # Remove computed fields
        config_dict.pop("_num_patches", None)
        config_dict.pop("_head_dim", None)

        # Convert enum values to their string representations
        from alpha_zero.core.transformer_network import AttentionType, NormType

        if isinstance(config_dict.get("norm_type"), NormType):
            config_dict["norm_type"] = config_dict["norm_type"].value
        if isinstance(config_dict.get("attention_type"), AttentionType):
            config_dict["attention_type"] = config_dict["attention_type"].value

        # Should be able to reconstruct
        config2 = TransformerConfig(**config_dict)

        assert config2.encoder_type == "hex_crystalline"
        assert config2.spatial_channels == 13
        assert config2.action_space_size == 376


class TestHexEncoderEdgeCases:
    """Test edge cases and error conditions."""

    def test_hex_encoder_with_different_spatial_sizes(self) -> None:
        """Test hex encoder works with different spatial sizes.

        Verifies:
        - Hex encoder handles 8x8, 16x16 grids
        - Output shapes are correct for different inputs
        """
        for spatial_size in [8, 16]:
            torch.manual_seed(42)

            config = TransformerConfig(
                encoder_type="hex_crystalline",
                spatial_channels=13,
                spatial_size=spatial_size,
                vector_dim=64,
                action_space_size=376,
                embed_dim=64,
                num_layers=2,
                num_heads=4,
            )
            network = Civ6TransformerNetwork(config=config)
            network.eval()

            # Create input
            spatial = torch.randn(2, 13, spatial_size, spatial_size)
            vector = torch.randn(2, 64)

            # Forward pass
            with torch.no_grad():
                policy_logits, value = network(spatial, vector)

            # Check shapes
            assert policy_logits.shape == (2, 376)
            assert value.shape == (2,)

    def test_single_sample_inference(self) -> None:
        """Test predict_single() with hex encoder.

        Verifies:
        - Single sample inference works (no batch dimension)
        - Output shapes are correct (squeezed)
        - Returns float for value (not tensor)
        """
        torch.manual_seed(42)

        config = TransformerConfig(
            encoder_type="hex_crystalline",
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        network = Civ6TransformerNetwork(config=config)

        # Single sample (no batch dimension)
        spatial = torch.randn(13, 8, 8)
        vector = torch.randn(64)

        policy_probs, value = network.predict_single(spatial, vector)

        # Check shapes
        assert policy_probs.shape == (376,), f"Expected (376,), got {policy_probs.shape}"
        assert isinstance(value, float), f"Expected float, got {type(value)}"

        # Check value range
        assert -1.0 <= value <= 1.0, f"Value {value} out of range [-1, 1]"
