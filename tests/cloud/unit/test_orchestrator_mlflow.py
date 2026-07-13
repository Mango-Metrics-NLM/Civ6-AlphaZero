"""Orchestrator MLFLOW_* env coordination."""

from __future__ import annotations

from alpha_zero.cloud.gcp import vertex_jobs


def test_tracker_env_vars_includes_mlflow_run_name(monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_NAME", "selfplay-iter003")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    env = vertex_jobs._tracker_env_vars()
    assert env.get("MLFLOW_RUN_NAME") == "selfplay-iter003"
    assert env.get("MLFLOW_TRACKING_URI") == "file:./mlruns"


def test_tracker_env_vars_omits_empty_values(monkeypatch):
    """Vertex AI rejects empty env values; filter them."""
    for var in (
        "MLFLOW_RUN_NAME",
        "MLFLOW_TAGS",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_ENABLED",
        "MLFLOW_REGISTRY_URI",
    ):
        monkeypatch.delenv(var, raising=False)
    env = vertex_jobs._tracker_env_vars()
    assert "" not in env.values()
    for v in env.values():
        assert v != ""


def test_tracker_env_vars_collects_all_mlflow_keys(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "exp")
    monkeypatch.setenv("MLFLOW_RUN_NAME", "run")
    monkeypatch.setenv("MLFLOW_TAGS", "k=v")
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    env = vertex_jobs._tracker_env_vars()
    assert env["MLFLOW_TRACKING_URI"] == "file:./mlruns"
    assert env["MLFLOW_EXPERIMENT_NAME"] == "exp"
    assert env["MLFLOW_RUN_NAME"] == "run"
    assert env["MLFLOW_TAGS"] == "k=v"
    assert env["MLFLOW_ENABLED"] == "true"


def test_wandb_era_aliases_are_gone():
    """The W&B-era vertex_jobs surface was deleted in the retirement sweep."""
    assert not hasattr(vertex_jobs, "_wandb_env_vars")
    assert not hasattr(vertex_jobs, "ensure_wandb_api_key")
    assert not hasattr(vertex_jobs, "resolve_wandb_key_from_secret_manager")


# ---------------------------------------------------------------------------
# vertex_orchestrator main() tracker lifecycle
# (ports the retired test_orchestrator_wandb.py onto WandbCompatAdapter)
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

_VO = "alpha_zero.cloud.gcp.vertex_orchestrator"
_ADAPTER = "alpha_zero.monitoring.mlflow_logger.WandbCompatAdapter"


@pytest.fixture()
def _orchestrator_sandbox(monkeypatch):
    """Patch heavy GCP surfaces so main() can be driven in-process.

    main() itself writes MLFLOW_RUN_GROUP / MLFLOW_TAGS into os.environ
    (run-group propagation); monkeypatch only restores variables *it*
    touched, so those must be scrubbed explicitly after the test to avoid
    polluting later from_env() tests.
    """
    import os as _os

    monkeypatch.delenv("CLOUD_ML_JOB_ID", raising=False)
    monkeypatch.delenv("MLFLOW_RUN_GROUP", raising=False)
    monkeypatch.delenv("MLFLOW_TAGS", raising=False)
    try:
        with (
            patch(f"{_VO}.GCPConfig") as mock_config,
            patch(f"{_VO}.VertexAIClient"),
            patch(f"{_VO}.GCSModelStorage") as mock_storage,
        ):
            cfg = MagicMock()
            cfg.project_id = "test-project"
            cfg.region = "us-central1"
            cfg.bucket = "test-bucket"
            mock_config.return_value = cfg
            mock_storage.return_value.get_latest_version.return_value = 1
            yield
    finally:
        _os.environ.pop("MLFLOW_RUN_GROUP", None)
        _os.environ.pop("MLFLOW_TAGS", None)


def _tracker_mock(enabled: bool = True, run_name: str | None = None) -> MagicMock:
    tracker = MagicMock()
    tracker.is_enabled = enabled
    tracker.run_name = run_name
    return tracker


def _run_main_interrupted(tracker: MagicMock) -> int:
    """Drive main() to the KeyboardInterrupt path with the tracker patched in."""
    from alpha_zero.cloud.gcp.vertex_orchestrator import main

    with patch(_ADAPTER) as mock_cls:
        mock_cls.from_env.return_value = tracker
        with patch(f"{_VO}.run_selfplay_phase", side_effect=KeyboardInterrupt("stop")):
            with patch("sys.argv", ["orchestrator", "--iterations=1", "--dry_run"]):
                return main()


def test_tracker_init_called_when_enabled(_orchestrator_sandbox):
    tracker = _tracker_mock(enabled=True)
    result = _run_main_interrupted(tracker)
    assert result == 130
    config_dict = tracker.init.call_args.kwargs["config_dict"]
    assert config_dict["iterations"] == 1
    assert config_dict["mode"] == "orchestrator"


def test_tracker_not_initialized_when_disabled(_orchestrator_sandbox):
    tracker = _tracker_mock(enabled=False)
    _run_main_interrupted(tracker)
    tracker.init.assert_not_called()


def test_finish_exit_code_130_on_keyboard_interrupt(_orchestrator_sandbox):
    tracker = _tracker_mock(enabled=True)
    result = _run_main_interrupted(tracker)
    assert result == 130
    tracker.finish.assert_called_with(exit_code=130)


def test_finish_exit_code_1_on_exception(_orchestrator_sandbox):
    from alpha_zero.cloud.gcp.vertex_orchestrator import main

    tracker = _tracker_mock(enabled=True)
    with patch(_ADAPTER) as mock_cls:
        mock_cls.from_env.return_value = tracker
        with patch(f"{_VO}.run_selfplay_phase", side_effect=RuntimeError("boom")):
            with patch("sys.argv", ["orchestrator", "--iterations=1", "--dry_run"]):
                result = main()
    assert result == 1
    tracker.finish.assert_called_with(exit_code=1)


def test_main_survives_tracker_import_error(_orchestrator_sandbox):
    """ImportError on the monitoring backend must degrade, not crash."""
    import sys as _sys

    from alpha_zero.cloud.gcp.vertex_orchestrator import main

    with patch.dict(_sys.modules, {"alpha_zero.monitoring.mlflow_logger": None}):
        with patch(f"{_VO}.run_selfplay_phase", side_effect=KeyboardInterrupt("stop")):
            with patch("sys.argv", ["orchestrator", "--iterations=1", "--dry_run"]):
                result = main()
    assert result == 130


def test_run_group_propagated_from_run_name(_orchestrator_sandbox, monkeypatch):
    """The orchestrator run name becomes MLFLOW_RUN_GROUP + a group= tag."""
    import os as _os

    tracker = _tracker_mock(enabled=True, run_name="orch-run-7")
    _run_main_interrupted(tracker)
    assert _os.environ.get("MLFLOW_RUN_GROUP") == "orch-run-7"
    assert "group=orch-run-7" in _os.environ.get("MLFLOW_TAGS", "")


def test_run_group_skipped_when_no_run_name(_orchestrator_sandbox):
    import os as _os

    tracker = _tracker_mock(enabled=True, run_name=None)
    _run_main_interrupted(tracker)
    assert "MLFLOW_RUN_GROUP" not in _os.environ
