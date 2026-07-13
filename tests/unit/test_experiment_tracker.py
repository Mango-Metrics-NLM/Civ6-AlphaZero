"""Public experiment-tracker import surface."""
from __future__ import annotations

import warnings


def test_experiment_tracker_get_returns_adapter():
    from alpha_zero.utils import experiment_tracker
    experiment_tracker.reset_tracker_for_tests()
    tracker = experiment_tracker.get_tracker()
    assert hasattr(tracker, "log_training_step")
    assert hasattr(tracker, "log_evaluation")
    assert hasattr(tracker, "finish")


def test_legacy_tracker_alias_emits_deprecation_warning():
    """The W&B-era names survive only as lazy aliases on alpha_zero.utils."""
    import alpha_zero.utils as utils_pkg

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = utils_pkg.WandbLogger

    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("deprecated" in m for m in msgs), f"no deprecation warning found in {msgs}"


def test_legacy_wandb_logger_alias_is_adapter():
    import alpha_zero.utils as utils_pkg
    from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert utils_pkg.WandbLogger is WandbCompatAdapter
        assert callable(utils_pkg.get_wandb_logger)
        assert callable(utils_pkg.init_wandb)


def test_utils_getattr_raises_for_unknown_names():
    import pytest

    import alpha_zero.utils as utils_pkg

    with pytest.raises(AttributeError):
        _ = utils_pkg.definitely_not_a_real_symbol


def test_shim_module_is_gone():
    """utils/wandb_logger.py was deleted in the W&B retirement sweep."""
    import importlib.util

    assert importlib.util.find_spec("alpha_zero.utils.wandb_logger") is None
