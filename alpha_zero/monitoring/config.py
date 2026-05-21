"""MLflow monitoring configuration.

Single-source configuration for the MLflow tracking backend. Re-exported by
``alpha_zero.orchestrator.config`` to avoid duplicating field definitions.

Environment variable contract (precedence: ``CIV6AZ_MLFLOW_*`` > ``MLFLOW_*``):

- ``CIV6AZ_MLFLOW_ENABLED``: bool ("true"/"false"/"1"/"0")
- ``CIV6AZ_MLFLOW_TRACKING_URI`` / ``MLFLOW_TRACKING_URI``: tracking URI
- ``CIV6AZ_MLFLOW_EXPERIMENT_NAME`` / ``MLFLOW_EXPERIMENT_NAME``
- ``CIV6AZ_MLFLOW_RUN_NAME``
- ``CIV6AZ_MLFLOW_TAGS``: comma-separated ``"k1=v1,k2=v2"``
- ``CIV6AZ_MLFLOW_ARTIFACT_LOCATION``: e.g. ``"gs://bucket/civ6-mlflow"``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TRACKING_URI = "file://./mlruns"
DEFAULT_EXPERIMENT_NAME = "civ6-alphazero"
ENV_PREFIX = "CIV6AZ_MLFLOW_"


def parse_tag_string(tag_string: str) -> dict[str, str]:
    """Parse a comma-separated tag string into a dict.

    Form: ``"k1=v1,k2=v2"``. Malformed pairs are silently dropped.
    """
    parsed: dict[str, str] = {}
    if not tag_string:
        return parsed
    for pair in tag_string.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k:
                parsed[k] = v
    return parsed


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking.

    All fields are overridable via YAML or environment variables. ``MLflowLogger``
    swallows tracking exceptions when ``fail_silently=True`` so training never
    crashes if the tracking server is unreachable.
    """

    enabled: bool = False

    tracking_uri: str = DEFAULT_TRACKING_URI
    registry_uri: str | None = None
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    description: str = ""

    artifact_location: str | None = None

    log_frequency: int = 1
    log_models: bool = True
    log_artifacts: bool = True
    log_system_metrics: bool = False
    nested_runs: bool = True

    fail_silently: bool = True
    request_timeout_seconds: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLflowConfig:
        """Construct from a YAML/dict payload."""
        tags_data = data.get("tags", {}) or {}
        if isinstance(tags_data, str):
            tags_dict = parse_tag_string(tags_data)
        elif isinstance(tags_data, list):
            # Tolerate W&B-style list form for migration: ["k=v", ...]
            tags_dict = {}
            for entry in tags_data:
                if isinstance(entry, str) and "=" in entry:
                    k, v = entry.split("=", 1)
                    tags_dict[k.strip()] = v.strip()
        elif isinstance(tags_data, dict):
            tags_dict = {str(k): str(v) for k, v in tags_data.items()}
        else:
            tags_dict = {}

        return cls(
            enabled=bool(data.get("enabled", False)),
            tracking_uri=str(data.get("tracking_uri", DEFAULT_TRACKING_URI)),
            registry_uri=data.get("registry_uri"),
            experiment_name=str(data.get("experiment_name", DEFAULT_EXPERIMENT_NAME)),
            run_name=data.get("run_name"),
            tags=tags_dict,
            description=str(data.get("description", "")),
            artifact_location=data.get("artifact_location"),
            log_frequency=int(data.get("log_frequency", 1)),
            log_models=bool(data.get("log_models", True)),
            log_artifacts=bool(data.get("log_artifacts", True)),
            log_system_metrics=bool(data.get("log_system_metrics", False)),
            nested_runs=bool(data.get("nested_runs", True)),
            fail_silently=bool(data.get("fail_silently", True)),
            request_timeout_seconds=float(data.get("request_timeout_seconds", 30.0)),
        )

    @classmethod
    def from_env(cls) -> MLflowConfig:
        """Construct from environment variables.

        Honors both ``CIV6AZ_MLFLOW_*`` (project convention) and standard
        ``MLFLOW_*`` env vars (project takes precedence).
        """
        tracking_uri = (
            os.environ.get(f"{ENV_PREFIX}TRACKING_URI")
            or os.environ.get("MLFLOW_TRACKING_URI")
            or DEFAULT_TRACKING_URI
        )
        experiment_name = (
            os.environ.get(f"{ENV_PREFIX}EXPERIMENT_NAME")
            or os.environ.get("MLFLOW_EXPERIMENT_NAME")
            or DEFAULT_EXPERIMENT_NAME
        )
        return cls(
            enabled=_bool_env(f"{ENV_PREFIX}ENABLED", default=False),
            tracking_uri=tracking_uri,
            registry_uri=os.environ.get(f"{ENV_PREFIX}REGISTRY_URI"),
            experiment_name=experiment_name,
            run_name=os.environ.get(f"{ENV_PREFIX}RUN_NAME"),
            tags=parse_tag_string(os.environ.get(f"{ENV_PREFIX}TAGS", "")),
            description=os.environ.get(f"{ENV_PREFIX}DESCRIPTION", ""),
            artifact_location=os.environ.get(f"{ENV_PREFIX}ARTIFACT_LOCATION"),
            log_frequency=int(os.environ.get(f"{ENV_PREFIX}LOG_FREQUENCY", "1")),
            log_models=_bool_env(f"{ENV_PREFIX}LOG_MODELS", default=True),
            log_artifacts=_bool_env(f"{ENV_PREFIX}LOG_ARTIFACTS", default=True),
            log_system_metrics=_bool_env(f"{ENV_PREFIX}LOG_SYSTEM_METRICS", default=False),
            nested_runs=_bool_env(f"{ENV_PREFIX}NESTED_RUNS", default=True),
            fail_silently=_bool_env(f"{ENV_PREFIX}FAIL_SILENTLY", default=True),
            request_timeout_seconds=float(
                os.environ.get(f"{ENV_PREFIX}REQUEST_TIMEOUT_SECONDS", "30.0")
            ),
        )
