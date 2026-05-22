"""Tests for ``alpha_zero.monitoring.config``.

Covers dataclass defaults, ``from_dict`` schema handling, ``from_env``
parsing (CIV6AZ_ + MLFLOW_), and the tag-string parser edge cases.
"""

from __future__ import annotations

import pytest

from alpha_zero.monitoring.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    ENV_PREFIX,
    MLflowConfig,
    parse_tag_string,
)

# ---------------------------------------------------------------------
# parse_tag_string
# ---------------------------------------------------------------------


def test_parse_tag_string_empty():
    assert parse_tag_string("") == {}


def test_parse_tag_string_single_pair():
    assert parse_tag_string("env=prod") == {"env": "prod"}


def test_parse_tag_string_multi_pair():
    assert parse_tag_string("env=prod,run=test") == {"env": "prod", "run": "test"}


def test_parse_tag_string_whitespace():
    assert parse_tag_string(" env = prod ,  run = test ") == {
        "env": "prod",
        "run": "test",
    }


def test_parse_tag_string_skips_malformed():
    # No "=" → skipped silently.
    assert parse_tag_string("env=prod,malformed,run=test") == {
        "env": "prod",
        "run": "test",
    }


def test_parse_tag_string_handles_value_with_equals():
    assert parse_tag_string("url=https://x?a=b") == {"url": "https://x?a=b"}


def test_parse_tag_string_skips_empty_key():
    assert parse_tag_string("=value,k=v") == {"k": "v"}


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------


def test_default_config():
    cfg = MLflowConfig()
    assert cfg.enabled is False
    assert cfg.tracking_uri == DEFAULT_TRACKING_URI
    assert cfg.experiment_name == DEFAULT_EXPERIMENT_NAME
    assert cfg.run_name is None
    assert cfg.tags == {}
    assert cfg.fail_silently is True
    assert cfg.nested_runs is True
    assert cfg.log_models is True


# ---------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------


def test_from_dict_empty():
    cfg = MLflowConfig.from_dict({})
    assert cfg == MLflowConfig()


def test_from_dict_full():
    cfg = MLflowConfig.from_dict(
        {
            "enabled": True,
            "tracking_uri": "file:///tmp/mlruns",
            "experiment_name": "my_experiment",
            "run_name": "trial_1",
            "tags": {"env": "prod", "version": "v2"},
            "description": "hello",
            "artifact_location": "gs://bucket/path",
            "log_frequency": 5,
            "log_models": False,
            "log_artifacts": False,
            "nested_runs": False,
            "fail_silently": False,
            "request_timeout_seconds": 60.0,
        }
    )
    assert cfg.enabled is True
    assert cfg.tracking_uri == "file:///tmp/mlruns"
    assert cfg.experiment_name == "my_experiment"
    assert cfg.run_name == "trial_1"
    assert cfg.tags == {"env": "prod", "version": "v2"}
    assert cfg.artifact_location == "gs://bucket/path"
    assert cfg.log_frequency == 5
    assert cfg.log_models is False
    assert cfg.log_artifacts is False
    assert cfg.nested_runs is False
    assert cfg.fail_silently is False
    assert cfg.request_timeout_seconds == 60.0


def test_from_dict_tags_as_string():
    cfg = MLflowConfig.from_dict({"tags": "env=prod,role=trainer"})
    assert cfg.tags == {"env": "prod", "role": "trainer"}


def test_from_dict_tags_as_list():
    cfg = MLflowConfig.from_dict({"tags": ["env=prod", "role=trainer"]})
    assert cfg.tags == {"env": "prod", "role": "trainer"}


def test_from_dict_tags_invalid_type_defaults_to_empty():
    cfg = MLflowConfig.from_dict({"tags": 42})
    assert cfg.tags == {}


# ---------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------


def test_from_env_defaults_when_empty(monkeypatch: pytest.MonkeyPatch):
    # Strip any inherited env vars.
    for name in (
        "CIV6AZ_MLFLOW_ENABLED",
        "CIV6AZ_MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_URI",
        "CIV6AZ_MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_EXPERIMENT_NAME",
        "CIV6AZ_MLFLOW_RUN_NAME",
        "CIV6AZ_MLFLOW_TAGS",
        "CIV6AZ_MLFLOW_ARTIFACT_LOCATION",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = MLflowConfig.from_env()
    assert cfg.enabled is False
    assert cfg.tracking_uri == DEFAULT_TRACKING_URI
    assert cfg.experiment_name == DEFAULT_EXPERIMENT_NAME
    assert cfg.tags == {}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("ON", True),
        ("false", False),
        ("0", False),
        ("", False),
        ("nonsense", False),
    ],
)
def test_from_env_enabled_parsing(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
):
    monkeypatch.setenv(f"{ENV_PREFIX}ENABLED", value)
    cfg = MLflowConfig.from_env()
    assert cfg.enabled is expected


def test_from_env_civ6_prefix_wins_over_mlflow_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.std")
    monkeypatch.setenv(f"{ENV_PREFIX}TRACKING_URI", "http://civ6.specific")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "std_exp")
    monkeypatch.setenv(f"{ENV_PREFIX}EXPERIMENT_NAME", "civ6_exp")
    cfg = MLflowConfig.from_env()
    assert cfg.tracking_uri == "http://civ6.specific"
    assert cfg.experiment_name == "civ6_exp"


def test_from_env_falls_back_to_mlflow_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(f"{ENV_PREFIX}TRACKING_URI", raising=False)
    monkeypatch.delenv(f"{ENV_PREFIX}EXPERIMENT_NAME", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.std")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "std_exp")
    cfg = MLflowConfig.from_env()
    assert cfg.tracking_uri == "http://mlflow.std"
    assert cfg.experiment_name == "std_exp"


def test_from_env_tags_roundtrip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(f"{ENV_PREFIX}TAGS", "env=prod,role=trainer,version=1.3.0")
    cfg = MLflowConfig.from_env()
    assert cfg.tags == {"env": "prod", "role": "trainer", "version": "1.3.0"}


def test_from_env_full_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(f"{ENV_PREFIX}ENABLED", "true")
    monkeypatch.setenv(f"{ENV_PREFIX}TRACKING_URI", "file:///tmp/x")
    monkeypatch.setenv(f"{ENV_PREFIX}EXPERIMENT_NAME", "exp_1")
    monkeypatch.setenv(f"{ENV_PREFIX}RUN_NAME", "run_1")
    monkeypatch.setenv(f"{ENV_PREFIX}TAGS", "a=1,b=2")
    monkeypatch.setenv(f"{ENV_PREFIX}ARTIFACT_LOCATION", "gs://bucket/x")
    monkeypatch.setenv(f"{ENV_PREFIX}LOG_FREQUENCY", "10")
    monkeypatch.setenv(f"{ENV_PREFIX}LOG_MODELS", "false")
    monkeypatch.setenv(f"{ENV_PREFIX}NESTED_RUNS", "false")
    monkeypatch.setenv(f"{ENV_PREFIX}FAIL_SILENTLY", "false")
    monkeypatch.setenv(f"{ENV_PREFIX}REQUEST_TIMEOUT_SECONDS", "45.5")
    cfg = MLflowConfig.from_env()
    assert cfg.enabled is True
    assert cfg.tracking_uri == "file:///tmp/x"
    assert cfg.experiment_name == "exp_1"
    assert cfg.run_name == "run_1"
    assert cfg.tags == {"a": "1", "b": "2"}
    assert cfg.artifact_location == "gs://bucket/x"
    assert cfg.log_frequency == 10
    assert cfg.log_models is False
    assert cfg.nested_runs is False
    assert cfg.fail_silently is False
    assert cfg.request_timeout_seconds == 45.5
