"""Unit tests for model migration utility (Sprint 6 Phase 2)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from scripts.migrate_model_to_nplayer import (
    find_conv_layer_key,
    find_value_head_keys,
    migrate_checkpoint,
    migrate_spatial_encoder,
    migrate_value_head,
    update_model_config,
)


class TestFindConvLayerKey:
    """Test suite for finding conv layer in state_dict."""

    def test_finds_standard_key(self):
        """Find conv layer with standard key."""
        state_dict = {
            "spatial_encoder.conv_in.weight": torch.randn(32, 13, 3, 3),
            "other_layer.weight": torch.randn(64, 32),
        }

        key = find_conv_layer_key(state_dict)
        assert key == "spatial_encoder.conv_in.weight"

    def test_finds_alternative_key(self):
        """Find conv layer with alternative key pattern."""
        state_dict = {
            "conv1.weight": torch.randn(32, 13, 3, 3),
            "other_layer.weight": torch.randn(64, 32),
        }

        key = find_conv_layer_key(state_dict)
        assert key == "conv1.weight"

    def test_finds_by_shape(self):
        """Find conv layer by searching for 13 input channels."""
        state_dict = {
            "custom_name.weight": torch.randn(32, 13, 3, 3),
            "other_layer.weight": torch.randn(64, 32),
        }

        key = find_conv_layer_key(state_dict)
        assert key == "custom_name.weight"

    def test_returns_none_when_not_found(self):
        """Return None when no conv layer found."""
        state_dict = {
            "linear.weight": torch.randn(64, 32),
        }

        key = find_conv_layer_key(state_dict)
        assert key is None


class TestFindValueHeadKeys:
    """Test suite for finding value head layers."""

    def test_finds_standard_keys(self):
        """Find value head with standard keys."""
        state_dict = {
            "value_head.fc2.weight": torch.randn(1, 64),
            "value_head.fc2.bias": torch.randn(1),
            "other_layer.weight": torch.randn(32, 64),
        }

        keys = find_value_head_keys(state_dict)
        assert keys["output_weight"] == "value_head.fc2.weight"
        assert keys["output_bias"] == "value_head.fc2.bias"

    def test_finds_alternative_keys(self):
        """Find value head with alternative key patterns."""
        state_dict = {
            "value_head.output.weight": torch.randn(1, 64),
            "value_head.output.bias": torch.randn(1),
        }

        keys = find_value_head_keys(state_dict)
        assert keys["output_weight"] == "value_head.output.weight"
        assert keys["output_bias"] == "value_head.output.bias"

    def test_returns_empty_when_not_found(self):
        """Return empty dict when no value head found."""
        state_dict = {
            "policy_head.weight": torch.randn(381, 64),
        }

        keys = find_value_head_keys(state_dict)
        assert keys == {}


class TestMigrateSpatialEncoder:
    """Test suite for spatial encoder migration."""

    def test_pads_channels_correctly(self):
        """Pad from 13 to 28 channels with zero-fill."""
        old_weight = torch.randn(32, 13, 3, 3)
        state_dict = {
            "spatial_encoder.conv_in.weight": old_weight.clone(),
        }

        migrated = migrate_spatial_encoder(
            state_dict, "spatial_encoder.conv_in.weight"
        )

        new_weight = migrated["spatial_encoder.conv_in.weight"]
        assert new_weight.shape == (32, 28, 3, 3)

        # Verify old channels preserved
        assert torch.allclose(new_weight[:, :13, :, :], old_weight)

        # Verify new channels are zeros
        assert torch.all(new_weight[:, 13:, :, :] == 0)

    def test_raises_on_wrong_input_channels(self):
        """Raise error if input channels != 13."""
        state_dict = {
            "conv.weight": torch.randn(32, 28, 3, 3),  # Already migrated
        }

        with pytest.raises(ValueError, match="Expected 13 input channels"):
            migrate_spatial_encoder(state_dict, "conv.weight")


class TestMigrateValueHead:
    """Test suite for value head migration."""

    def test_replicates_weight_to_6dim(self):
        """Replicate scalar weight to 6 dimensions."""
        state_dict = {
            "value_head.fc2.weight": torch.randn(1, 64),
        }
        value_keys = {"output_weight": "value_head.fc2.weight"}

        migrated = migrate_value_head(state_dict, value_keys)

        new_weight = migrated["value_head.fc2.weight"]
        assert new_weight.shape == (6, 64)

        # Verify all 6 rows are identical (replicated)
        old_weight = state_dict["value_head.fc2.weight"]
        for i in range(6):
            assert torch.allclose(new_weight[i:i+1, :], old_weight)

    def test_replicates_bias_to_6dim(self):
        """Replicate scalar bias to 6 dimensions."""
        state_dict = {
            "value_head.fc2.bias": torch.randn(1),
        }
        value_keys = {"output_bias": "value_head.fc2.bias"}

        migrated = migrate_value_head(state_dict, value_keys)

        new_bias = migrated["value_head.fc2.bias"]
        assert new_bias.shape == (6,)

        # Verify all 6 values are identical
        old_bias = state_dict["value_head.fc2.bias"]
        for i in range(6):
            assert torch.isclose(new_bias[i], old_bias[0])

    def test_raises_on_wrong_output_size(self):
        """Raise error if output size != 1."""
        state_dict = {
            "value_head.weight": torch.randn(6, 64),  # Already migrated
        }
        value_keys = {"output_weight": "value_head.weight"}

        with pytest.raises(ValueError, match="Expected 1 output neuron"):
            migrate_value_head(state_dict, value_keys)


class TestUpdateModelConfig:
    """Test suite for model_config metadata update."""

    def test_updates_spatial_channels(self):
        """Update spatial_channels in model_config."""
        checkpoint = {
            "model_state_dict": {},
            "model_config": {
                "spatial_channels": 13,
                "action_space_size": 376,
            },
        }

        updated = update_model_config(checkpoint)

        assert updated["model_config"]["spatial_channels"] == 28
        assert updated["model_config"]["action_space_size"] == 376  # Unchanged

    def test_handles_missing_model_config(self):
        """Handle checkpoint without model_config gracefully."""
        checkpoint = {
            "model_state_dict": {},
        }

        updated = update_model_config(checkpoint)

        # Should return unchanged
        assert "model_config" not in updated


class TestMigrateCheckpointEndToEnd:
    """End-to-end tests for checkpoint migration."""

    def test_migrates_checkpoint_successfully(self, tmp_path: Path):
        """Migrate a complete checkpoint."""
        # Create a fake 2-player checkpoint
        checkpoint = {
            "model_state_dict": {
                "spatial_encoder.conv_in.weight": torch.randn(32, 13, 3, 3),
                "value_head.fc2.weight": torch.randn(1, 64),
                "value_head.fc2.bias": torch.randn(1),
                "policy_head.weight": torch.randn(376, 128),
            },
            "model_config": {
                "spatial_channels": 13,
                "action_space_size": 376,
            },
        }

        input_path = tmp_path / "old_model.pt"
        output_path = tmp_path / "migrated_model.pt"

        torch.save(checkpoint, input_path)

        # Migrate
        migrate_checkpoint(str(input_path), str(output_path), dry_run=False)

        # Load migrated checkpoint
        migrated = torch.load(output_path)

        # Verify spatial encoder
        new_conv = migrated["model_state_dict"]["spatial_encoder.conv_in.weight"]
        assert new_conv.shape == (32, 28, 3, 3)

        # Verify value head
        new_value_weight = migrated["model_state_dict"]["value_head.fc2.weight"]
        assert new_value_weight.shape == (6, 64)

        new_value_bias = migrated["model_state_dict"]["value_head.fc2.bias"]
        assert new_value_bias.shape == (6,)

        # Verify policy head unchanged
        policy_weight = migrated["model_state_dict"]["policy_head.weight"]
        assert policy_weight.shape == (376, 128)

        # Verify metadata
        assert migrated["model_config"]["spatial_channels"] == 28
        assert migrated["migrated_to_nplayer"] is True
        assert migrated["migration_version"] == "1.5.0"

    def test_dry_run_does_not_save(self, tmp_path: Path):
        """Dry-run mode does not save output."""
        checkpoint = {
            "model_state_dict": {
                "spatial_encoder.conv_in.weight": torch.randn(32, 13, 3, 3),
                "value_head.fc2.weight": torch.randn(1, 64),
            },
        }

        input_path = tmp_path / "old_model.pt"
        output_path = tmp_path / "migrated_model.pt"

        torch.save(checkpoint, input_path)

        # Dry-run
        migrate_checkpoint(str(input_path), str(output_path), dry_run=True)

        # Verify output was NOT created
        assert not output_path.exists()

    def test_preserves_non_migrated_layers(self, tmp_path: Path):
        """Non-migrated layers remain unchanged."""
        # Create checkpoint with many layers
        state_dict = {
            "spatial_encoder.conv_in.weight": torch.randn(32, 13, 3, 3),
            "spatial_encoder.res1.conv1.weight": torch.randn(32, 32, 3, 3),
            "spatial_encoder.res1.conv2.weight": torch.randn(32, 32, 3, 3),
            "value_head.fc1.weight": torch.randn(64, 128),
            "value_head.fc2.weight": torch.randn(1, 64),
            "policy_head.fc1.weight": torch.randn(128, 256),
        }

        checkpoint = {"model_state_dict": state_dict}

        input_path = tmp_path / "model.pt"
        output_path = tmp_path / "migrated.pt"

        torch.save(checkpoint, input_path)
        migrate_checkpoint(str(input_path), str(output_path), dry_run=False)

        # Load and verify
        migrated = torch.load(output_path)
        migrated_state = migrated["model_state_dict"]

        # Verify unchanged layers
        assert torch.allclose(
            migrated_state["spatial_encoder.res1.conv1.weight"],
            state_dict["spatial_encoder.res1.conv1.weight"],
        )
        assert torch.allclose(
            migrated_state["value_head.fc1.weight"],
            state_dict["value_head.fc1.weight"],
        )
        assert torch.allclose(
            migrated_state["policy_head.fc1.weight"],
            state_dict["policy_head.fc1.weight"],
        )
