"""Comprehensive tests for alpha_zero.utils.memory_guard module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import yaml

from alpha_zero.utils.memory_guard import (
    MemoryGuard,
    MemoryGuardConfig,
    MemoryInfo,
    MemoryStatus,
)

# ---------------------------------------------------------------------------
# MemoryInfo
# ---------------------------------------------------------------------------


class TestMemoryInfo:
    def test_to_dict(self):
        info = MemoryInfo(
            total_mb=8192.0,
            used_mb=4096.0,
            free_mb=4096.0,
            usage_ratio=0.5,
            device="cpu",
            timestamp=1000.0,
        )
        d = info.to_dict()
        assert d["total_mb"] == 8192.0
        assert d["used_mb"] == 4096.0
        assert d["free_mb"] == 4096.0
        assert d["usage_ratio"] == 0.5
        assert d["device"] == "cpu"
        assert d["timestamp"] == 1000.0

    def test_default_timestamp(self):
        info = MemoryInfo(total_mb=1.0, used_mb=0.5, free_mb=0.5, usage_ratio=0.5, device="cpu")
        assert info.timestamp > 0


# ---------------------------------------------------------------------------
# MemoryStatus enum
# ---------------------------------------------------------------------------


class TestMemoryStatus:
    def test_values(self):
        assert MemoryStatus.NORMAL == "normal"
        assert MemoryStatus.WARNING == "warning"
        assert MemoryStatus.CRITICAL == "critical"
        assert MemoryStatus.OOM_RISK == "oom_risk"


# ---------------------------------------------------------------------------
# MemoryGuardConfig
# ---------------------------------------------------------------------------


class TestMemoryGuardConfig:
    def test_defaults(self):
        cfg = MemoryGuardConfig()
        assert cfg.warning_threshold == 0.7
        assert cfg.critical_threshold == 0.85
        assert cfg.oom_threshold == 0.95
        assert cfg.min_batch_size == 1
        assert 0 < cfg.batch_reduction_factor < 1
        assert cfg.monitor_interval_seconds > 0

    # --- validation errors -----------------------------------------------

    @pytest.mark.parametrize(
        "field, value, match",
        [
            ("warning_threshold", -0.1, "warning_threshold must be between"),
            ("warning_threshold", 1.1, "warning_threshold must be between"),
            ("critical_threshold", -0.1, "critical_threshold must be between"),
            ("critical_threshold", 1.1, "critical_threshold must be between"),
            ("oom_threshold", -0.1, "oom_threshold must be between"),
            ("oom_threshold", 1.1, "oom_threshold must be between"),
            ("min_batch_size", 0, "min_batch_size must be >= 1"),
            ("batch_reduction_factor", 0.0, "batch_reduction_factor must be between"),
            ("batch_reduction_factor", 1.0, "batch_reduction_factor must be between"),
            ("monitor_interval_seconds", -1.0, "monitor_interval_seconds must be positive"),
            ("memory_reserve_mb", -1.0, "memory_reserve_mb must be non-negative"),
        ],
    )
    def test_validation_errors(self, field, value, match):
        # Build kwargs that are valid except for the one we're testing
        kwargs = {
            "warning_threshold": 0.5,
            "critical_threshold": 0.7,
            "oom_threshold": 0.9,
            "min_batch_size": 1,
            "batch_reduction_factor": 0.5,
            "monitor_interval_seconds": 10.0,
            "memory_reserve_mb": 0.0,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=match):
            MemoryGuardConfig(**kwargs)

    def test_warning_ge_critical_raises(self):
        with pytest.raises(ValueError, match="warning_threshold must be less than"):
            MemoryGuardConfig(warning_threshold=0.9, critical_threshold=0.85, oom_threshold=0.95)

    def test_critical_ge_oom_raises(self):
        with pytest.raises(ValueError, match="critical_threshold must be less than"):
            MemoryGuardConfig(warning_threshold=0.5, critical_threshold=0.95, oom_threshold=0.95)

    # --- from_env --------------------------------------------------------

    def test_from_env_uses_defaults(self):
        cfg = MemoryGuardConfig.from_env()
        assert isinstance(cfg, MemoryGuardConfig)

    @patch.dict(
        "os.environ",
        {"CIV6AZ_MEMORY_WARNING_THRESHOLD": "0.6", "CIV6AZ_MEMORY_MIN_BATCH": "4"},
    )
    def test_from_env_reads_envvars(self):
        cfg = MemoryGuardConfig.from_env()
        assert cfg.warning_threshold == 0.6
        assert cfg.min_batch_size == 4

    # --- from_dict / to_dict roundtrip -----------------------------------

    def test_from_dict_and_to_dict_roundtrip(self):
        original = MemoryGuardConfig(
            warning_threshold=0.6,
            critical_threshold=0.8,
            oom_threshold=0.9,
            min_batch_size=2,
            batch_reduction_factor=0.4,
            enable_gradient_checkpointing=False,
            gc_on_warning=False,
            monitor_interval_seconds=5.0,
            memory_reserve_mb=256.0,
            enable_cuda_memory_management=False,
        )
        d = original.to_dict()
        restored = MemoryGuardConfig.from_dict(d)
        assert restored.to_dict() == d

    def test_from_dict_partial(self):
        cfg = MemoryGuardConfig.from_dict({"min_batch_size": 8})
        assert cfg.min_batch_size == 8
        # Other fields keep defaults
        assert cfg.warning_threshold == 0.7

    # --- from_yaml -------------------------------------------------------

    def test_from_yaml_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MemoryGuardConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml_with_memory_guard_key(self, tmp_path):
        data = {
            "memory_guard": {
                "warning_threshold": 0.6,
                "critical_threshold": 0.8,
                "oom_threshold": 0.9,
            }
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(data))
        cfg = MemoryGuardConfig.from_yaml(p)
        assert cfg.warning_threshold == 0.6

    def test_from_yaml_flat(self, tmp_path):
        data = {
            "warning_threshold": 0.65,
            "critical_threshold": 0.8,
            "oom_threshold": 0.92,
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(data))
        cfg = MemoryGuardConfig.from_yaml(p)
        assert cfg.warning_threshold == 0.65


# ---------------------------------------------------------------------------
# MemoryGuard
# ---------------------------------------------------------------------------


class TestMemoryGuard:
    @pytest.fixture()
    def guard(self):
        return MemoryGuard()

    # --- init / repr / reset ---------------------------------------------

    def test_default_init(self, guard):
        assert guard.config.warning_threshold == 0.7
        assert not guard._monitoring

    def test_repr(self, guard):
        r = repr(guard)
        assert "MemoryGuard" in r
        assert "warning=" in r

    def test_reset(self, guard):
        guard._gc_runs = 5
        guard._batch_reductions = 3
        guard._oom_prevented = 1
        guard._memory_history = [MagicMock()]
        guard.reset()
        stats = guard.get_statistics()
        assert stats["gc_runs"] == 0
        assert stats["batch_reductions"] == 0
        assert stats["oom_prevented"] == 0
        assert stats["history_size"] == 0

    # --- get_cpu_memory ---------------------------------------------------

    @patch("alpha_zero.utils.memory_guard.PSUTIL_AVAILABLE", True)
    @patch("alpha_zero.utils.memory_guard.psutil")
    def test_get_cpu_memory_success(self, mock_psutil, guard):
        vm = MagicMock()
        vm.total = 16 * 1024 * 1024 * 1024  # 16 GB
        vm.used = 8 * 1024 * 1024 * 1024
        vm.available = 8 * 1024 * 1024 * 1024
        vm.percent = 50.0
        mock_psutil.virtual_memory.return_value = vm
        info = guard.get_cpu_memory()
        assert info is not None
        assert info.device == "cpu"
        assert abs(info.usage_ratio - 0.5) < 0.01

    @patch("alpha_zero.utils.memory_guard.PSUTIL_AVAILABLE", False)
    def test_get_cpu_memory_not_available(self, guard):
        assert guard.get_cpu_memory() is None

    @patch("alpha_zero.utils.memory_guard.PSUTIL_AVAILABLE", True)
    @patch("alpha_zero.utils.memory_guard.psutil")
    def test_get_cpu_memory_exception(self, mock_psutil, guard):
        mock_psutil.virtual_memory.side_effect = RuntimeError("fail")
        assert guard.get_cpu_memory() is None

    # --- get_gpu_memory ---------------------------------------------------

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", False)
    def test_get_gpu_memory_no_torch(self, guard):
        assert guard.get_gpu_memory() is None

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", True)
    @patch("alpha_zero.utils.memory_guard.torch")
    def test_get_gpu_memory_no_cuda(self, mock_torch, guard):
        mock_torch.cuda.is_available.return_value = False
        assert guard.get_gpu_memory() is None

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", True)
    @patch("alpha_zero.utils.memory_guard.torch")
    def test_get_gpu_memory_success(self, mock_torch, guard):
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_memory = 8 * 1024 * 1024 * 1024
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_reserved.return_value = 4 * 1024 * 1024 * 1024
        info = guard.get_gpu_memory(0)
        assert info is not None
        assert info.device == "gpu:0"
        assert abs(info.usage_ratio - 0.5) < 0.01

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", True)
    @patch("alpha_zero.utils.memory_guard.torch")
    def test_get_gpu_memory_exception(self, mock_torch, guard):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.side_effect = RuntimeError("nope")
        assert guard.get_gpu_memory() is None

    # --- get_memory_status ------------------------------------------------

    def test_memory_status_normal(self, guard):
        guard.get_cpu_memory = MagicMock(
            return_value=MemoryInfo(
                total_mb=8192, used_mb=2048, free_mb=6144, usage_ratio=0.25, device="cpu"
            )
        )
        assert guard.get_memory_status("cpu") == MemoryStatus.NORMAL

    def test_memory_status_warning(self, guard):
        guard.get_cpu_memory = MagicMock(
            return_value=MemoryInfo(
                total_mb=8192, used_mb=6144, free_mb=2048, usage_ratio=0.75, device="cpu"
            )
        )
        assert guard.get_memory_status("cpu") == MemoryStatus.WARNING

    def test_memory_status_critical(self, guard):
        guard.get_cpu_memory = MagicMock(
            return_value=MemoryInfo(
                total_mb=8192, used_mb=7168, free_mb=1024, usage_ratio=0.875, device="cpu"
            )
        )
        assert guard.get_memory_status("cpu") == MemoryStatus.CRITICAL

    def test_memory_status_oom_risk(self, guard):
        guard.get_cpu_memory = MagicMock(
            return_value=MemoryInfo(
                total_mb=8192, used_mb=7987, free_mb=205, usage_ratio=0.975, device="cpu"
            )
        )
        assert guard.get_memory_status("cpu") == MemoryStatus.OOM_RISK

    def test_memory_status_none_returns_normal(self, guard):
        guard.get_cpu_memory = MagicMock(return_value=None)
        assert guard.get_memory_status("cpu") == MemoryStatus.NORMAL

    def test_memory_status_gpu_device_parsed(self, guard):
        guard.get_gpu_memory = MagicMock(return_value=None)
        result = guard.get_memory_status("gpu:1")
        guard.get_gpu_memory.assert_called_once_with(1)
        assert result == MemoryStatus.NORMAL

    # --- can_allocate ----------------------------------------------------

    def test_can_allocate_true(self, guard):
        guard.get_cpu_memory = MagicMock(
            return_value=MemoryInfo(
                total_mb=8192, used_mb=2048, free_mb=6144, usage_ratio=0.25, device="cpu"
            )
        )
        assert guard.can_allocate(1024.0, device="cpu")

    def test_can_allocate_false(self, guard):
        guard.get_cpu_memory = MagicMock(
            return_value=MemoryInfo(
                total_mb=8192, used_mb=8000, free_mb=192, usage_ratio=0.97, device="cpu"
            )
        )
        assert not guard.can_allocate(1024.0, device="cpu")

    def test_can_allocate_none_returns_true(self, guard):
        guard.get_gpu_memory = MagicMock(return_value=None)
        assert guard.can_allocate(99999.0, device="gpu:0")

    def test_can_allocate_gpu(self, guard):
        guard.get_gpu_memory = MagicMock(
            return_value=MemoryInfo(
                total_mb=16384, used_mb=4096, free_mb=12288, usage_ratio=0.25, device="gpu:0"
            )
        )
        assert guard.can_allocate(2048.0, device="gpu:0")

    # --- get_safe_batch_size ----------------------------------------------

    def test_safe_batch_size_normal(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.NORMAL)
        assert guard.get_safe_batch_size(32) == 32

    def test_safe_batch_size_warning(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.WARNING)
        result = guard.get_safe_batch_size(32)
        assert result < 32
        assert result >= guard.config.min_batch_size

    def test_safe_batch_size_critical(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.CRITICAL)
        result = guard.get_safe_batch_size(32)
        assert result < 32

    def test_safe_batch_size_oom_risk(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.OOM_RISK)
        result = guard.get_safe_batch_size(32)
        assert result <= guard.config.min_batch_size or result < 32

    def test_safe_batch_size_respects_min(self, guard):
        guard.config.min_batch_size = 4
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.OOM_RISK)
        result = guard.get_safe_batch_size(4)
        assert result >= 4

    def test_safe_batch_size_increments_batch_reductions(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.WARNING)
        guard.get_safe_batch_size(64)
        assert guard._batch_reductions == 1

    # --- cleanup_memory ---------------------------------------------------

    @patch("gc.collect")
    def test_cleanup_memory_gc_on_warning(self, mock_gc, guard):
        guard.config.gc_on_warning = True
        guard.cleanup_memory()
        mock_gc.assert_called_once()
        assert guard._gc_runs == 1

    @patch("gc.collect")
    def test_cleanup_memory_force_gc(self, mock_gc, guard):
        guard.config.gc_on_warning = False
        guard.cleanup_memory(force_gc=True)
        mock_gc.assert_called_once()

    @patch("gc.collect")
    def test_cleanup_memory_no_gc(self, mock_gc, guard):
        guard.config.gc_on_warning = False
        guard.cleanup_memory(force_gc=False)
        mock_gc.assert_not_called()

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", True)
    @patch("alpha_zero.utils.memory_guard.torch")
    @patch("gc.collect")
    def test_cleanup_memory_clears_cuda_cache(self, mock_gc, mock_torch, guard):
        mock_torch.cuda.is_available.return_value = True
        guard.config.enable_cuda_memory_management = True
        guard.cleanup_memory(force_gc=True)
        mock_torch.cuda.empty_cache.assert_called_once()

    # --- check_and_respond -----------------------------------------------

    def _make_info(self, usage_ratio):
        return MemoryInfo(
            total_mb=8192,
            used_mb=8192 * usage_ratio,
            free_mb=8192 * (1 - usage_ratio),
            usage_ratio=usage_ratio,
            device="cpu",
        )

    def test_check_and_respond_normal(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.NORMAL)
        guard.get_cpu_memory = MagicMock(return_value=self._make_info(0.5))
        status = guard.check_and_respond("cpu")
        assert status == MemoryStatus.NORMAL

    def test_check_and_respond_warning_with_callback(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.WARNING)
        guard.get_cpu_memory = MagicMock(return_value=self._make_info(0.75))
        callback = MagicMock()
        guard.register_warning_callback(callback)
        guard.check_and_respond("cpu")
        callback.assert_called_once()

    def test_check_and_respond_warning_callback_error(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.WARNING)
        guard.get_cpu_memory = MagicMock(return_value=self._make_info(0.75))
        bad_callback = MagicMock(side_effect=RuntimeError("boom"))
        guard.register_warning_callback(bad_callback)
        # Should not raise
        guard.check_and_respond("cpu")

    def test_check_and_respond_critical_with_callback(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.CRITICAL)
        guard.get_cpu_memory = MagicMock(return_value=self._make_info(0.9))
        callback = MagicMock()
        guard.register_critical_callback(callback)
        guard.cleanup_memory = MagicMock()
        guard.check_and_respond("cpu")
        callback.assert_called_once()
        guard.cleanup_memory.assert_called_once_with(force_gc=True)

    def test_check_and_respond_critical_callback_error(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.CRITICAL)
        guard.get_cpu_memory = MagicMock(return_value=self._make_info(0.9))
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        guard.register_critical_callback(bad_cb)
        guard.cleanup_memory = MagicMock()
        guard.check_and_respond("cpu")

    def test_check_and_respond_oom_risk_with_callback(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.OOM_RISK)
        guard.get_cpu_memory = MagicMock(return_value=self._make_info(0.97))
        callback = MagicMock()
        guard.register_oom_risk_callback(callback)
        guard.cleanup_memory = MagicMock()
        guard.check_and_respond("cpu")
        callback.assert_called_once()
        assert guard._oom_prevented == 1

    def test_check_and_respond_oom_risk_callback_error(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.OOM_RISK)
        guard.get_cpu_memory = MagicMock(return_value=self._make_info(0.97))
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        guard.register_oom_risk_callback(bad_cb)
        guard.cleanup_memory = MagicMock()
        guard.check_and_respond("cpu")

    def test_check_and_respond_oom_no_mem_info(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.OOM_RISK)
        guard.get_cpu_memory = MagicMock(return_value=None)
        guard.cleanup_memory = MagicMock()
        # Should still not raise
        guard.check_and_respond("cpu")

    def test_check_and_respond_trims_history(self, guard):
        guard._max_history = 5
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.NORMAL)
        guard.get_cpu_memory = MagicMock(return_value=self._make_info(0.3))
        for _ in range(10):
            guard.check_and_respond("cpu")
        assert len(guard._memory_history) <= 5

    def test_check_and_respond_gpu_device(self, guard):
        guard.get_memory_status = MagicMock(return_value=MemoryStatus.NORMAL)
        guard.get_gpu_memory = MagicMock(return_value=self._make_info(0.3))
        guard.check_and_respond("gpu:0")
        guard.get_gpu_memory.assert_called_once_with(0)

    # --- gradient checkpointing ------------------------------------------

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", True)
    def test_enable_gradient_checkpointing(self, guard):
        model = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()
        guard.enable_gradient_checkpointing(model, enable=True)
        model.gradient_checkpointing_enable.assert_called_once()
        assert guard._gradient_checkpointing_enabled

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", True)
    def test_disable_gradient_checkpointing(self, guard):
        model = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        guard._gradient_checkpointing_enabled = True
        guard.enable_gradient_checkpointing(model, enable=False)
        model.gradient_checkpointing_disable.assert_called_once()
        assert not guard._gradient_checkpointing_enabled

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", False)
    def test_gradient_checkpointing_no_torch(self, guard):
        model = MagicMock()
        guard.enable_gradient_checkpointing(model)
        model.gradient_checkpointing_enable.assert_not_called()

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", True)
    def test_gradient_checkpointing_config_disabled(self, guard):
        guard.config.enable_gradient_checkpointing = False
        model = MagicMock()
        guard.enable_gradient_checkpointing(model)
        model.gradient_checkpointing_enable.assert_not_called()

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", True)
    def test_gradient_checkpointing_no_method(self, guard):
        model = MagicMock(spec=[])  # no gradient_checkpointing_enable
        guard.enable_gradient_checkpointing(model)
        # Should not raise

    @patch("alpha_zero.utils.memory_guard.TORCH_AVAILABLE", True)
    def test_gradient_checkpointing_exception(self, guard):
        model = MagicMock()
        model.gradient_checkpointing_enable.side_effect = RuntimeError("err")
        guard.enable_gradient_checkpointing(model)
        # Should not raise

    # --- monitoring -------------------------------------------------------

    def test_start_stop_monitoring(self, guard):
        guard.config.monitor_interval_seconds = 0.05
        guard.check_and_respond = MagicMock()
        guard.start_monitoring("cpu")
        assert guard._monitoring
        time.sleep(0.15)
        guard.stop_monitoring()
        assert not guard._monitoring
        assert guard.check_and_respond.call_count >= 1

    def test_start_monitoring_twice_warns(self, guard):
        guard.config.monitor_interval_seconds = 0.05
        guard.check_and_respond = MagicMock()
        guard.start_monitoring("cpu")
        guard.start_monitoring("cpu")  # should warn, not start a second thread
        guard.stop_monitoring()

    def test_stop_monitoring_when_not_running(self, guard):
        guard.stop_monitoring()  # should not raise

    def test_monitoring_exception_in_loop(self, guard):
        guard.config.monitor_interval_seconds = 0.05
        guard.check_and_respond = MagicMock(side_effect=RuntimeError("fail"))
        guard.start_monitoring("cpu")
        time.sleep(0.15)
        guard.stop_monitoring()

    # --- get_memory_trend -------------------------------------------------

    def test_memory_trend_insufficient_data(self, guard):
        assert guard.get_memory_trend(window_size=10) is None

    def test_memory_trend_positive(self, guard):
        for i in range(10):
            guard._memory_history.append(
                MemoryInfo(
                    total_mb=8192,
                    used_mb=4096 + i * 100,
                    free_mb=4096 - i * 100,
                    usage_ratio=0.5 + i * 0.02,
                    device="cpu",
                )
            )
        trend = guard.get_memory_trend(window_size=10)
        assert trend is not None
        assert trend > 0

    def test_memory_trend_negative(self, guard):
        for i in range(10):
            guard._memory_history.append(
                MemoryInfo(
                    total_mb=8192,
                    used_mb=6144 - i * 100,
                    free_mb=2048 + i * 100,
                    usage_ratio=0.75 - i * 0.02,
                    device="cpu",
                )
            )
        trend = guard.get_memory_trend(window_size=10)
        assert trend is not None
        assert trend < 0

    # --- get_statistics ---------------------------------------------------

    def test_get_statistics(self, guard):
        stats = guard.get_statistics()
        assert "gc_runs" in stats
        assert "batch_reductions" in stats
        assert "oom_prevented" in stats
        assert "gradient_checkpointing_enabled" in stats
        assert "monitoring_active" in stats
        assert "history_size" in stats

    # --- callback registration -------------------------------------------

    def test_register_callbacks(self, guard):
        cb1 = MagicMock()
        cb2 = MagicMock()
        cb3 = MagicMock()
        guard.register_warning_callback(cb1)
        guard.register_critical_callback(cb2)
        guard.register_oom_risk_callback(cb3)
        assert cb1 in guard._on_warning
        assert cb2 in guard._on_critical
        assert cb3 in guard._on_oom_risk
