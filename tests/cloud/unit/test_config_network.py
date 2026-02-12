"""Unit tests for NetworkArchitectureConfig and GCPConfig network integration.

Tests the NetworkArchitectureConfig dataclass including:
- Default values
- Environment variable loading via from_env()
- Serialization via to_dict()
- Integration with GCPConfig

Location: tests/cloud/unit/test_config_network.py
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from alpha_zero.cloud.gcp.config import GCPConfig, NetworkArchitectureConfig


class TestNetworkArchitectureConfigDefaults(unittest.TestCase):
    """Test default values for NetworkArchitectureConfig."""

    def test_default_board_size(self):
        """Test default board_size is 8."""
        config = NetworkArchitectureConfig()
        self.assertEqual(config.board_size, 8)

    def test_default_action_space_size(self):
        """Test default action_space_size is 376."""
        config = NetworkArchitectureConfig()
        self.assertEqual(config.action_space_size, 376)

    def test_default_spatial_channels(self):
        """Test default spatial_channels is 13."""
        config = NetworkArchitectureConfig()
        self.assertEqual(config.spatial_channels, 13)

    def test_default_vector_dim(self):
        """Test default vector_dim is 64."""
        config = NetworkArchitectureConfig()
        self.assertEqual(config.vector_dim, 64)

    def test_default_num_layers(self):
        """Test default num_layers is 6."""
        config = NetworkArchitectureConfig()
        self.assertEqual(config.num_layers, 6)

    def test_default_embed_dim(self):
        """Test default embed_dim is 64."""
        config = NetworkArchitectureConfig()
        self.assertEqual(config.embed_dim, 64)


class TestNetworkArchitectureConfigFromEnv(unittest.TestCase):
    """Test NetworkArchitectureConfig.from_env() with environment variables."""

    def test_from_env_defaults(self):
        """Test from_env() returns defaults when no env vars are set."""
        with patch.dict("os.environ", {}, clear=True):
            config = NetworkArchitectureConfig.from_env()
            self.assertEqual(config.board_size, 8)
            self.assertEqual(config.action_space_size, 376)
            self.assertEqual(config.spatial_channels, 13)
            self.assertEqual(config.vector_dim, 64)
            self.assertEqual(config.num_layers, 6)
            self.assertEqual(config.embed_dim, 64)

    def test_from_env_board_size_override(self):
        """Test from_env() with CIV6AZ_BOARD_SIZE override."""
        with patch.dict("os.environ", {"CIV6AZ_BOARD_SIZE": "16"}, clear=True):
            config = NetworkArchitectureConfig.from_env()
            self.assertEqual(config.board_size, 16)
            # Other fields should use defaults
            self.assertEqual(config.action_space_size, 376)
            self.assertEqual(config.spatial_channels, 13)

    def test_from_env_action_space_override(self):
        """Test from_env() with CIV6AZ_ACTION_SPACE_SIZE override."""
        with patch.dict("os.environ", {"CIV6AZ_ACTION_SPACE_SIZE": "500"}, clear=True):
            config = NetworkArchitectureConfig.from_env()
            self.assertEqual(config.action_space_size, 500)
            # Other fields should use defaults
            self.assertEqual(config.board_size, 8)
            self.assertEqual(config.spatial_channels, 13)

    def test_from_env_all_overrides(self):
        """Test from_env() with all environment variables set."""
        env_vars = {
            "CIV6AZ_BOARD_SIZE": "16",
            "CIV6AZ_ACTION_SPACE_SIZE": "500",
            "CIV6AZ_SPATIAL_CHANNELS": "20",
            "CIV6AZ_VECTOR_DIM": "128",
            "CIV6AZ_NUM_LAYERS": "12",
            "CIV6AZ_EMBED_DIM": "256",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = NetworkArchitectureConfig.from_env()
            self.assertEqual(config.board_size, 16)
            self.assertEqual(config.action_space_size, 500)
            self.assertEqual(config.spatial_channels, 20)
            self.assertEqual(config.vector_dim, 128)
            self.assertEqual(config.num_layers, 12)
            self.assertEqual(config.embed_dim, 256)

    def test_from_env_partial_overrides(self):
        """Test from_env() with only some environment variables set."""
        env_vars = {
            "CIV6AZ_BOARD_SIZE": "16",
            "CIV6AZ_NUM_LAYERS": "8",
            "CIV6AZ_EMBED_DIM": "128",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = NetworkArchitectureConfig.from_env()
            # Overridden values
            self.assertEqual(config.board_size, 16)
            self.assertEqual(config.num_layers, 8)
            self.assertEqual(config.embed_dim, 128)
            # Default values
            self.assertEqual(config.action_space_size, 376)
            self.assertEqual(config.spatial_channels, 13)
            self.assertEqual(config.vector_dim, 64)


class TestNetworkArchitectureConfigSerialization(unittest.TestCase):
    """Test NetworkArchitectureConfig.to_dict() serialization."""

    def test_to_dict_keys(self):
        """Test to_dict() returns all 6 expected keys."""
        config = NetworkArchitectureConfig()
        result = config.to_dict()

        expected_keys = {
            "board_size",
            "action_space_size",
            "spatial_channels",
            "vector_dim",
            "num_layers",
            "embed_dim",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_to_dict_values_match(self):
        """Test to_dict() values match the config attributes."""
        config = NetworkArchitectureConfig(
            board_size=16,
            action_space_size=500,
            spatial_channels=20,
            vector_dim=128,
            num_layers=12,
            embed_dim=256,
        )
        result = config.to_dict()

        self.assertEqual(result["board_size"], 16)
        self.assertEqual(result["action_space_size"], 500)
        self.assertEqual(result["spatial_channels"], 20)
        self.assertEqual(result["vector_dim"], 128)
        self.assertEqual(result["num_layers"], 12)
        self.assertEqual(result["embed_dim"], 256)

    def test_round_trip(self):
        """Test serialization round trip: create -> to_dict -> create -> compare."""
        original = NetworkArchitectureConfig(
            board_size=16,
            action_space_size=500,
            spatial_channels=20,
            vector_dim=128,
            num_layers=12,
            embed_dim=256,
        )

        # Serialize to dict
        serialized = original.to_dict()

        # Deserialize back to config
        reconstructed = NetworkArchitectureConfig(**serialized)

        # Compare all fields
        self.assertEqual(reconstructed.board_size, original.board_size)
        self.assertEqual(reconstructed.action_space_size, original.action_space_size)
        self.assertEqual(reconstructed.spatial_channels, original.spatial_channels)
        self.assertEqual(reconstructed.vector_dim, original.vector_dim)
        self.assertEqual(reconstructed.num_layers, original.num_layers)
        self.assertEqual(reconstructed.embed_dim, original.embed_dim)


class TestGCPConfigNetworkIntegration(unittest.TestCase):
    """Test integration of NetworkArchitectureConfig with GCPConfig."""

    def test_gcp_config_has_network_field(self):
        """Test GCPConfig has a network attribute."""
        config = GCPConfig()
        self.assertTrue(hasattr(config, "network"))
        self.assertIsInstance(config.network, NetworkArchitectureConfig)

    def test_gcp_config_network_defaults(self):
        """Test GCPConfig.network has correct default values."""
        config = GCPConfig()

        # Verify network field has default NetworkArchitectureConfig values
        self.assertEqual(config.network.board_size, 8)
        self.assertEqual(config.network.action_space_size, 376)
        self.assertEqual(config.network.spatial_channels, 13)
        self.assertEqual(config.network.vector_dim, 64)
        self.assertEqual(config.network.num_layers, 6)
        self.assertEqual(config.network.embed_dim, 64)

    def test_gcp_config_custom_network(self):
        """Test GCPConfig with custom NetworkArchitectureConfig."""
        custom_network = NetworkArchitectureConfig(
            board_size=16,
            action_space_size=500,
            spatial_channels=20,
            vector_dim=128,
            num_layers=12,
            embed_dim=256,
        )

        config = GCPConfig(
            project_id="test-project",
            region="us-west1",
            bucket="test-bucket",
            network=custom_network,
        )

        # Verify custom network configuration is preserved
        self.assertEqual(config.network.board_size, 16)
        self.assertEqual(config.network.action_space_size, 500)
        self.assertEqual(config.network.spatial_channels, 20)
        self.assertEqual(config.network.vector_dim, 128)
        self.assertEqual(config.network.num_layers, 12)
        self.assertEqual(config.network.embed_dim, 256)

        # Verify other GCPConfig fields are not affected
        self.assertEqual(config.project_id, "test-project")
        self.assertEqual(config.region, "us-west1")
        self.assertEqual(config.bucket, "test-bucket")


if __name__ == "__main__":
    unittest.main()
