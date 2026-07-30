"""Security and input sanitization validation tests."""

from pathlib import Path

import pytest

from alpha_zero.bridge.config import BridgeConfig


@pytest.mark.security
class TestSecuritySanitization:
    """Security test suite for configuration and path validation."""

    def test_comm_dir_does_not_contain_path_traversal(self):
        """Verify comm_dir paths do not contain path traversal characters."""
        config = BridgeConfig()
        str_path = str(config.comm_dir)
        assert ".." not in str_path

    def test_custom_comm_dir_validation(self):
        """Verify custom comm_dir paths resolve cleanly as Path objects."""
        custom = "C:\\Games\\CustomCiv6\\Bridge"
        config = BridgeConfig(comm_dir=custom)
        assert isinstance(config.comm_dir, Path)
        assert str(config.comm_dir) == str(Path(custom))
