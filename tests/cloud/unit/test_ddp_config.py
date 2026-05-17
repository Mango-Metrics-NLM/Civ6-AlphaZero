"""Unit tests for DDP configuration module."""

from __future__ import annotations

import os

import pytest

from alpha_zero.distributed.ddp_config import (
    DDPBackend,
    DDPConfig,
    detect_ddp_environment,
    setup_ddp_environment,
)


class TestDDPBackend:
    """Tests for DDPBackend enum."""

    def test_backend_values(self) -> None:
        """Test backend enum values."""
        assert DDPBackend.NCCL.value == "nccl"
        assert DDPBackend.GLOO.value == "gloo"
        assert DDPBackend.MPI.value == "mpi"
        assert DDPBackend.AUTO.value == "auto"


class TestDDPConfig:
    """Tests for DDPConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = DDPConfig()

        assert config.enabled is False
        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.master_addr == "localhost"
        assert config.master_port == 29500
        assert config.backend == DDPBackend.AUTO.value

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "enabled": True,
            "world_size": 4,
            "rank": 2,
            "local_rank": 0,
            "master_addr": "10.0.0.1",
            "master_port": 12345,
            "backend": "nccl",
            "use_amp": True,
        }

        config = DDPConfig.from_dict(data)

        assert config.enabled is True
        assert config.world_size == 4
        assert config.rank == 2
        assert config.master_addr == "10.0.0.1"
        assert config.master_port == 12345
        assert config.backend == "nccl"
        assert config.use_amp is True

    def test_from_dict_nested(self) -> None:
        """Test creation from nested dictionary."""
        data = {
            "ddp": {
                "enabled": True,
                "world_size": 2,
            }
        }

        config = DDPConfig.from_dict(data)

        assert config.enabled is True
        assert config.world_size == 2

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creation from environment variables."""
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "192.168.1.1")
        monkeypatch.setenv("MASTER_PORT", "29600")

        config = DDPConfig.from_env()

        assert config.enabled is True  # world_size > 1
        assert config.world_size == 4
        assert config.rank == 1
        assert config.local_rank == 0
        assert config.master_addr == "192.168.1.1"
        assert config.master_port == 29600

    def test_from_env_single_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that single process is not DDP enabled."""
        # Clear any existing env vars
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("RANK", raising=False)

        config = DDPConfig.from_env()

        assert config.enabled is False
        assert config.world_size == 1

    def test_from_env_cuda_devices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test CUDA device detection from environment."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")

        config = DDPConfig.from_env()

        assert config.device_ids == [0, 1, 2]

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = DDPConfig(
            enabled=True,
            world_size=4,
            rank=2,
            backend="nccl",
        )

        data = config.to_dict()

        assert data["enabled"] is True
        assert data["world_size"] == 4
        assert data["rank"] == 2
        assert data["backend"] == "nccl"

    def test_get_device_cuda(self) -> None:
        """Test getting CUDA device string."""
        config = DDPConfig(
            use_cuda=True,
            local_rank=1,
        )

        assert config.get_device() == "cuda:1"

    def test_get_device_cpu(self) -> None:
        """Test getting CPU device string."""
        config = DDPConfig(use_cuda=False)

        assert config.get_device() == "cpu"

    def test_get_device_with_device_ids(self) -> None:
        """Test device selection with explicit device IDs."""
        config = DDPConfig(
            use_cuda=True,
            local_rank=1,
            device_ids=[2, 3, 4],
        )

        assert config.get_device() == "cuda:3"

    def test_get_backend_auto_cuda(self) -> None:
        """Test auto backend selection with CUDA."""
        config = DDPConfig(
            backend=DDPBackend.AUTO.value,
            use_cuda=True,
        )

        assert config.get_backend() == DDPBackend.NCCL.value

    def test_get_backend_auto_cpu(self) -> None:
        """Test auto backend selection without CUDA."""
        config = DDPConfig(
            backend=DDPBackend.AUTO.value,
            use_cuda=False,
        )

        assert config.get_backend() == DDPBackend.GLOO.value

    def test_get_backend_explicit(self) -> None:
        """Test explicit backend specification."""
        config = DDPConfig(backend="mpi")

        assert config.get_backend() == "mpi"

    def test_is_main_process_rank_zero(self) -> None:
        """Test main process detection for rank 0."""
        config = DDPConfig(enabled=True, rank=0)

        assert config.is_main_process() is True

    def test_is_main_process_other_rank(self) -> None:
        """Test main process detection for other ranks."""
        config = DDPConfig(enabled=True, rank=1)

        assert config.is_main_process() is False

    def test_is_main_process_ddp_disabled(self) -> None:
        """Test main process detection when DDP disabled."""
        config = DDPConfig(enabled=False, rank=1)

        assert config.is_main_process() is True

    def test_is_local_main(self) -> None:
        """Test local main process detection."""
        config = DDPConfig(enabled=True, local_rank=0)
        assert config.is_local_main() is True

        config = DDPConfig(enabled=True, local_rank=1)
        assert config.is_local_main() is False

    def test_validate_valid_config(self) -> None:
        """Test validation of valid configuration."""
        config = DDPConfig(
            enabled=True,
            world_size=4,
            rank=2,
            local_rank=0,
        )

        issues = config.validate()

        assert len(issues) == 0
        assert config.is_valid()

    def test_validate_invalid_world_size(self) -> None:
        """Test validation catches invalid world_size."""
        config = DDPConfig(
            enabled=True,
            world_size=1,  # Should be >= 2 when enabled
        )

        issues = config.validate()

        assert any("world_size" in issue for issue in issues)
        assert not config.is_valid()

    def test_validate_invalid_rank(self) -> None:
        """Test validation catches invalid rank."""
        config = DDPConfig(
            enabled=True,
            world_size=2,
            rank=5,  # Out of range
        )

        issues = config.validate()

        assert any("rank" in issue for issue in issues)

    def test_validate_invalid_port(self) -> None:
        """Test validation catches invalid port."""
        config = DDPConfig(
            enabled=True,
            world_size=2,
            master_port=80,  # Too low
        )

        issues = config.validate()

        assert any("master_port" in issue for issue in issues)

    def test_validate_timeout_warning(self) -> None:
        """Test validation warns about low timeout."""
        config = DDPConfig(timeout_seconds=30)

        issues = config.validate()

        assert any("timeout_seconds" in issue for issue in issues)


class TestDetectDDPEnvironment:
    """Tests for detect_ddp_environment function."""

    def test_detect_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test detection with no environment variables."""
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("RANK", raising=False)

        config = detect_ddp_environment()

        assert config.enabled is False

    def test_detect_with_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test detection with environment variables."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("MASTER_ADDR", "localhost")
        monkeypatch.setenv("MASTER_PORT", "29500")

        config = detect_ddp_environment()

        assert config.enabled is True
        assert config.world_size == 2


class TestSetupDDPEnvironment:
    """Tests for setup_ddp_environment function."""

    def test_setup_env_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that setup does nothing when DDP disabled."""
        config = DDPConfig(enabled=False)

        # Clear any existing env
        monkeypatch.delenv("WORLD_SIZE", raising=False)

        setup_ddp_environment(config)

        # Should not set env vars
        assert os.environ.get("WORLD_SIZE") is None

    def test_setup_env_enabled(self) -> None:
        """Test environment setup when DDP enabled.

        ``setup_ddp_environment`` writes WORLD_SIZE/RANK/LOCAL_RANK/
        MASTER_ADDR/MASTER_PORT directly to os.environ; without explicit
        cleanup these leak to later tests and any code that triggers
        torch.distributed initialization will block waiting for peer
        processes that don't exist. This was a reproducible test-suite
        hang in tests/cloud/unit/ where test_ddp_config preceded
        test_training_muzero_dispatch.
        """
        ddp_vars = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
        original = {v: os.environ.get(v) for v in ddp_vars}
        try:
            config = DDPConfig(
                enabled=True,
                world_size=4,
                rank=2,
                local_rank=1,
                master_addr="10.0.0.1",
                master_port=12345,
            )

            setup_ddp_environment(config)

            assert os.environ["WORLD_SIZE"] == "4"
            assert os.environ["RANK"] == "2"
            assert os.environ["LOCAL_RANK"] == "1"
            assert os.environ["MASTER_ADDR"] == "10.0.0.1"
            assert os.environ["MASTER_PORT"] == "12345"
        finally:
            for var, value in original.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
