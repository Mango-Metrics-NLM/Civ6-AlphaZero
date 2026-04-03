"""Coverage tests for alpha_zero.distributed.ddp_config module."""

import os
from unittest.mock import patch

from alpha_zero.distributed.ddp_config import (
    DDPBackend,
    DDPConfig,
    detect_ddp_environment,
)


class TestDDPBackend:
    def test_enum_values(self):
        assert DDPBackend.NCCL == "nccl"
        assert DDPBackend.GLOO == "gloo"
        assert DDPBackend.MPI == "mpi"
        assert DDPBackend.AUTO == "auto"


class TestDDPConfigDefaults:
    def test_defaults(self):
        config = DDPConfig()
        assert config.enabled is False
        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.master_addr == "localhost"
        assert config.master_port == 29500
        assert config.backend == "auto"
        assert config.use_cuda is True
        assert config.device_ids == []
        assert config.find_unused_parameters is False
        assert config.use_amp is False
        assert config.elastic_training is False
        assert config.max_restarts == 3


class TestDDPConfigFromEnv:
    def test_no_env_vars(self):
        with patch.dict(os.environ, {}, clear=True):
            config = DDPConfig.from_env()
            assert config.enabled is False
            assert config.world_size == 1

    def test_with_world_size(self):
        env = {"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "1"}
        with patch.dict(os.environ, env, clear=True):
            config = DDPConfig.from_env()
            assert config.enabled is True
            assert config.world_size == 4
            assert config.rank == 2
            assert config.local_rank == 1

    def test_single_process_not_enabled(self):
        env = {"WORLD_SIZE": "1"}
        with patch.dict(os.environ, env, clear=True):
            config = DDPConfig.from_env()
            assert config.enabled is False

    def test_master_addr_port(self):
        env = {"WORLD_SIZE": "2", "MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "12345"}
        with patch.dict(os.environ, env, clear=True):
            config = DDPConfig.from_env()
            assert config.master_addr == "10.0.0.1"
            assert config.master_port == 12345

    def test_cuda_visible_devices(self):
        env = {"CUDA_VISIBLE_DEVICES": "0,1,2"}
        with patch.dict(os.environ, env, clear=True):
            config = DDPConfig.from_env()
            assert config.device_ids == [0, 1, 2]

    def test_cuda_visible_empty(self):
        env = {"CUDA_VISIBLE_DEVICES": ""}
        with patch.dict(os.environ, env, clear=True):
            config = DDPConfig.from_env()
            assert config.device_ids == []


class TestDDPConfigFromDict:
    def test_empty_dict(self):
        config = DDPConfig.from_dict({})
        assert config.enabled is False

    def test_nested_ddp_key(self):
        data = {"ddp": {"enabled": True, "world_size": 4, "rank": 1}}
        config = DDPConfig.from_dict(data)
        assert config.enabled is True
        assert config.world_size == 4
        assert config.rank == 1

    def test_flat_dict(self):
        data = {
            "enabled": True,
            "world_size": 8,
            "rank": 3,
            "local_rank": 1,
            "master_addr": "10.0.0.5",
            "master_port": 30000,
            "backend": "nccl",
            "use_amp": True,
            "amp_dtype": "bfloat16",
            "elastic_training": True,
            "min_world_size": 2,
            "max_restarts": 5,
        }
        config = DDPConfig.from_dict(data)
        assert config.world_size == 8
        assert config.backend == "nccl"
        assert config.use_amp is True
        assert config.amp_dtype == "bfloat16"
        assert config.elastic_training is True
        assert config.max_restarts == 5

    def test_all_fields(self):
        data = {
            "enabled": True,
            "world_size": 2,
            "rank": 0,
            "local_rank": 0,
            "master_addr": "localhost",
            "master_port": 29500,
            "backend": "gloo",
            "init_method": "tcp://localhost:29500",
            "timeout_seconds": 3600,
            "use_cuda": False,
            "device_ids": [0],
            "find_unused_parameters": True,
            "broadcast_buffers": False,
            "gradient_as_bucket_view": False,
            "bucket_cap_mb": 50,
            "use_amp": False,
            "amp_dtype": "float16",
            "elastic_training": False,
            "min_world_size": 1,
            "max_restarts": 0,
        }
        config = DDPConfig.from_dict(data)
        assert config.init_method == "tcp://localhost:29500"
        assert config.timeout_seconds == 3600
        assert config.use_cuda is False
        assert config.find_unused_parameters is True
        assert config.broadcast_buffers is False
        assert config.gradient_as_bucket_view is False
        assert config.bucket_cap_mb == 50


class TestDDPConfigToDict:
    def test_roundtrip(self):
        config = DDPConfig(enabled=True, world_size=4, rank=2)
        d = config.to_dict()
        assert d["enabled"] is True
        assert d["world_size"] == 4
        assert d["rank"] == 2
        assert "master_addr" in d
        assert "bucket_cap_mb" in d

    def test_all_keys_present(self):
        d = DDPConfig().to_dict()
        expected_keys = {
            "enabled",
            "world_size",
            "rank",
            "local_rank",
            "master_addr",
            "master_port",
            "backend",
            "init_method",
            "timeout_seconds",
            "use_cuda",
            "device_ids",
            "find_unused_parameters",
            "broadcast_buffers",
            "gradient_as_bucket_view",
            "bucket_cap_mb",
            "use_amp",
            "amp_dtype",
            "elastic_training",
            "min_world_size",
            "max_restarts",
        }
        assert set(d.keys()) == expected_keys


class TestDDPConfigGetDevice:
    def test_cpu(self):
        config = DDPConfig(use_cuda=False)
        assert config.get_device() == "cpu"

    def test_cuda_default(self):
        config = DDPConfig(use_cuda=True, local_rank=0)
        assert config.get_device() == "cuda:0"

    def test_cuda_with_device_ids(self):
        config = DDPConfig(use_cuda=True, local_rank=1, device_ids=[2, 3, 4])
        assert config.get_device() == "cuda:3"

    def test_cuda_local_rank_exceeds_devices(self):
        config = DDPConfig(use_cuda=True, local_rank=5, device_ids=[0, 1])
        # min(5, 1) = 1, so device_ids[1] = 1
        assert config.get_device() == "cuda:1"


class TestDDPConfigGetBackend:
    def test_explicit_backend(self):
        config = DDPConfig(backend="gloo")
        assert config.get_backend() == "gloo"

    def test_auto_with_cuda(self):
        config = DDPConfig(backend="auto", use_cuda=True)
        assert config.get_backend() == "nccl"

    def test_auto_without_cuda(self):
        config = DDPConfig(backend="auto", use_cuda=False)
        assert config.get_backend() == "gloo"

    def test_mpi_backend(self):
        config = DDPConfig(backend="mpi")
        assert config.get_backend() == "mpi"


class TestDDPConfigProcessChecks:
    def test_is_main_process_rank0(self):
        config = DDPConfig(enabled=True, rank=0, world_size=2)
        assert config.is_main_process() is True

    def test_is_main_process_not_rank0(self):
        config = DDPConfig(enabled=True, rank=1, world_size=2)
        assert config.is_main_process() is False

    def test_is_main_process_disabled(self):
        config = DDPConfig(enabled=False, rank=1)
        assert config.is_main_process() is True

    def test_is_local_main_rank0(self):
        config = DDPConfig(enabled=True, local_rank=0, world_size=2)
        assert config.is_local_main() is True

    def test_is_local_main_not_rank0(self):
        config = DDPConfig(enabled=True, local_rank=1, world_size=2)
        assert config.is_local_main() is False

    def test_is_local_main_disabled(self):
        config = DDPConfig(enabled=False, local_rank=1)
        assert config.is_local_main() is True


class TestDDPConfigValidation:
    def test_valid_disabled(self):
        config = DDPConfig()
        assert config.validate() == []
        assert config.is_valid() is True

    def test_world_size_too_small(self):
        config = DDPConfig(enabled=True, world_size=1, rank=0)
        issues = config.validate()
        assert any("world_size" in i for i in issues)

    def test_rank_out_of_range(self):
        config = DDPConfig(enabled=True, world_size=2, rank=5)
        issues = config.validate()
        assert any("rank" in i for i in issues)

    def test_negative_local_rank(self):
        config = DDPConfig(enabled=True, world_size=2, rank=0, local_rank=-1)
        issues = config.validate()
        assert any("local_rank" in i for i in issues)

    def test_invalid_port(self):
        config = DDPConfig(enabled=True, world_size=2, rank=0, master_port=80)
        issues = config.validate()
        assert any("master_port" in i for i in issues)

    def test_high_port(self):
        config = DDPConfig(enabled=True, world_size=2, rank=0, master_port=70000)
        issues = config.validate()
        assert any("master_port" in i for i in issues)

    def test_low_timeout(self):
        config = DDPConfig(timeout_seconds=10)
        issues = config.validate()
        assert any("timeout" in i for i in issues)

    def test_low_bucket_cap(self):
        config = DDPConfig(bucket_cap_mb=0)
        issues = config.validate()
        assert any("bucket_cap" in i for i in issues)

    def test_is_valid_false(self):
        config = DDPConfig(enabled=True, world_size=1)
        assert config.is_valid() is False

    def test_valid_enabled(self):
        config = DDPConfig(enabled=True, world_size=4, rank=0)
        assert config.is_valid() is True


class TestDetectDDPEnvironment:
    def test_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            config = detect_ddp_environment()
            assert config.enabled is False

    def test_with_env(self):
        env = {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            config = detect_ddp_environment()
            assert config.enabled is True

    def test_with_invalid_env_logs_warning(self):
        env = {"WORLD_SIZE": "2", "RANK": "5", "LOCAL_RANK": "0"}
        with patch.dict(os.environ, env, clear=True):
            config = detect_ddp_environment()
            issues = config.validate()
            assert len(issues) > 0
