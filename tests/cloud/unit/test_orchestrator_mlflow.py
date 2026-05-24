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
    for var in ("MLFLOW_RUN_NAME", "MLFLOW_TAGS", "MLFLOW_TRACKING_URI",
                "MLFLOW_EXPERIMENT_NAME", "MLFLOW_ENABLED", "MLFLOW_REGISTRY_URI"):
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


def test_wandb_env_vars_is_deprecated_alias(monkeypatch, recwarn):
    """The old _wandb_env_vars name still works but warns."""
    monkeypatch.setenv("MLFLOW_RUN_NAME", "x")
    env = vertex_jobs._wandb_env_vars()
    assert env.get("MLFLOW_RUN_NAME") == "x"
    deprecation = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
    assert deprecation, "expected DeprecationWarning"
