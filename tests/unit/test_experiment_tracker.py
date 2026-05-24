"""Public experiment-tracker import surface."""
from __future__ import annotations

import importlib
import warnings


def test_experiment_tracker_get_returns_adapter():
    from alpha_zero.utils import experiment_tracker
    experiment_tracker.reset_tracker_for_tests()
    tracker = experiment_tracker.get_tracker()
    assert hasattr(tracker, "log_training_step")
    assert hasattr(tracker, "log_evaluation")
    assert hasattr(tracker, "finish")


def test_wandb_logger_import_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import alpha_zero.utils.wandb_logger as wl
        importlib.reload(wl)

    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("MLflow" in m or "wandb_logger" in m for m in msgs), \
        f"no deprecation warning found in {msgs}"


def test_wandb_logger_class_is_adapter_alias():
    from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter
    from alpha_zero.utils.wandb_logger import WandbLogger
    assert WandbLogger is WandbCompatAdapter or issubclass(WandbLogger, WandbCompatAdapter)


def test_wandb_config_class_is_monitoring_alias():
    from alpha_zero.monitoring.config import MonitoringConfig
    from alpha_zero.utils.wandb_logger import WandbConfig
    assert WandbConfig is MonitoringConfig or issubclass(WandbConfig, MonitoringConfig)
