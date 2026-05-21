"""MLflow monitoring configuration.

Single-source configuration for the MLflow tracking backend. Re-exported by
``alpha_zero.orchestrator.config`` to avoid duplicating field definitions.

This module exposes :class:`MLflowConfig` (canonical) and the historical
alias :class:`MonitoringConfig` (kept for backwards compatibility with code
written against the original PR 79 monitoring package). Both names refer to
the same dataclass.

Environment variable contract (precedence: ``CIV6AZ_MLFLOW_*`` > ``MLFLOW_*``):

- ``CIV6AZ_MLFLOW_ENABLED``: bool ("true"/"false"/"1"/"0")
- ``CIV6AZ_MLFLOW_TRACKING_URI`` / ``MLFLOW_TRACKING_URI``: tracking URI
- ``CIV6AZ_MLFLOW_EXPERIMENT_NAME`` / ``MLFLOW_EXPERIMENT_NAME``
- ``CIV6AZ_MLFLOW_RUN_NAME``
- ``CIV6AZ_MLFLOW_TAGS``: comma-separated ``"k1=v1,k2=v2"``
- ``CIV6AZ_MLFLOW_ARTIFACT_LOCATION``: e.g. ``"gs://bucket/civ6-mlflow"``
- ``CIV6AZ_MLFLOW_LOG_FREQUENCY``: int, default 1
- ``CIV6AZ_MLFLOW_LOG_MODELS`` / ``LOG_ARTIFACTS`` / ``LOG_SYSTEM_METRICS``
- ``CIV6AZ_MLFLOW_NESTED_RUNS`` / ``FAIL_SILENTLY``: bool
- ``CIV6AZ_MLFLOW_REQUEST_TIMEOUT_SECONDS``: float
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults & helpers
# ---------------------------------------------------------------------------

#: Default MLflow tracking URI (local file store, MLflow-canonical form).
DEFAULT_TRACKING_URI: str = "file://./mlruns"

#: Default experiment name used when none is specified.
DEFAULT_EXPERIMENT_NAME: str = "civ6-alphazero"

#: Environment-variable prefix for project-specific overrides.
ENV_PREFIX: str = "CIV6AZ_MLFLOW_"


def parse_tag_string(tag_string: str) -> dict[str, str]:
    """Parse a comma-separated tag string into a dict.

    Form: ``"k1=v1,k2=v2"``. Malformed pairs (no ``=``) are silently dropped.
    Whitespace around keys and values is stripped. Empty keys are skipped.
    Values containing ``=`` are preserved (split only on the first ``=``).

    Args:
        tag_string: Raw tag string, e.g. from an env var.

    Returns:
        Mapping ``{key: value}`` of parsed tag pairs (may be empty).
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
    """Read a boolean environment variable with permissive truthy parsing."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


# ---------------------------------------------------------------------------
# Canonical dataclass
# ---------------------------------------------------------------------------


@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking.

    All fields are overridable via YAML or environment variables. ``MLflowLogger``
    swallows tracking exceptions when ``fail_silently=True`` so training never
    crashes if the tracking server is unreachable.

    Attributes:
        enabled: Master switch. False by default - must opt-in via YAML, env
            var, or the orchestrator ``--mlflow-*`` CLI flags.
        tracking_uri: MLflow tracking URI (file://, http://, sqlite://, etc.).
        registry_uri: Optional model-registry URI; None disables the registry.
        experiment_name: MLflow experiment name (created if missing).
        run_name: Optional human-readable run name (auto-generated if None).
        tags: Key-value tags attached to every MLflow run.
        description: Free-text description for the MLflow run.
        artifact_location: Optional remote artifact root (e.g. ``gs://bucket/path``).
        log_frequency: Iteration cadence for metric logging (1 = every iter).
        log_models: When True, also log model checkpoints as artifacts.
        log_artifacts: When True, allow ``log_artifact`` calls (else no-op).
        log_system_metrics: When True, enable MLflow's built-in system metrics.
        nested_runs: When True, open a nested run per ``log_iteration`` call
            while continuing to mirror metrics on the parent (campaign) run.
        fail_silently: When True (default), any MLflow API exception is logged
            and swallowed; training continues.
        request_timeout_seconds: Soft client timeout (informational; MLflow
            client honors this where supported).
        eval_interval: Iteration cadence for mid-run policy evaluation
            (0 disables). Consumed by orchestrator wiring, not the logger.
        log_grad_norms: When True, the orchestrator should log per-epoch L2
            gradient norms to MLflow.
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

    # ------------------------------------------------------------------
    # Orchestrator-side hyperparameters (consumed by orchestrator, not
    # the logger). Kept here so the YAML schema stays single-source.
    # ------------------------------------------------------------------
    eval_interval: int = 0
    log_grad_norms: bool = False

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLflowConfig:
        """Create from a YAML/dict payload.

        Tag input is tolerant: accepts a dict, a comma-separated string
        (``"k=v,k2=v2"``), a list of ``"k=v"`` strings, or a list of
        ``{"key": k, "value": v}`` mappings.

        Uses ``or``-based null coalescing on string fields so empty YAML
        keys (parsed as ``None``) fall back to module defaults instead of
        the literal string ``"None"``.
        """
        raw_tags = data.get("tags") or {}
        tags_dict: dict[str, str]
        if isinstance(raw_tags, str):
            tags_dict = parse_tag_string(raw_tags)
        elif isinstance(raw_tags, list):
            tags_dict = {}
            for entry in raw_tags:
                if isinstance(entry, str) and "=" in entry:
                    k, v = entry.split("=", 1)
                    tags_dict[k.strip()] = v.strip()
                elif isinstance(entry, dict) and "key" in entry and "value" in entry:
                    tags_dict[str(entry["key"])] = str(entry["value"])
        elif isinstance(raw_tags, dict):
            tags_dict = {str(k): str(v) for k, v in raw_tags.items()}
        else:
            logger.warning(
                "MLflowConfig.from_dict: unexpected 'tags' type %s; ignoring.",
                type(raw_tags).__name__,
            )
            tags_dict = {}

        return cls(
            enabled=bool(data.get("enabled", False)),
            tracking_uri=str(data.get("tracking_uri") or DEFAULT_TRACKING_URI),
            registry_uri=data.get("registry_uri"),
            experiment_name=str(data.get("experiment_name") or DEFAULT_EXPERIMENT_NAME),
            run_name=data.get("run_name"),
            tags=tags_dict,
            description=str(data.get("description") or ""),
            artifact_location=data.get("artifact_location"),
            log_frequency=int(data.get("log_frequency") or 1),
            log_models=bool(data.get("log_models", True)),
            log_artifacts=bool(data.get("log_artifacts", True)),
            log_system_metrics=bool(data.get("log_system_metrics", False)),
            nested_runs=bool(data.get("nested_runs", True)),
            fail_silently=bool(data.get("fail_silently", True)),
            request_timeout_seconds=float(data.get("request_timeout_seconds") or 30.0),
            eval_interval=int(data.get("eval_interval") or 0),
            log_grad_norms=bool(data.get("log_grad_norms", False)),
        )

    @classmethod
    def from_env(cls) -> MLflowConfig:
        """Create from environment variables.

        Honors both ``CIV6AZ_MLFLOW_*`` (project convention) and standard
        ``MLFLOW_*`` env vars; the project-specific names take precedence.
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
            eval_interval=int(os.environ.get(f"{ENV_PREFIX}EVAL_INTERVAL", "0")),
            log_grad_norms=_bool_env(f"{ENV_PREFIX}LOG_GRAD_NORMS", default=False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary suitable for YAML/JSON output."""
        return {
            "enabled": self.enabled,
            "tracking_uri": self.tracking_uri,
            "registry_uri": self.registry_uri,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tags": dict(self.tags),
            "description": self.description,
            "artifact_location": self.artifact_location,
            "log_frequency": self.log_frequency,
            "log_models": self.log_models,
            "log_artifacts": self.log_artifacts,
            "log_system_metrics": self.log_system_metrics,
            "nested_runs": self.nested_runs,
            "fail_silently": self.fail_silently,
            "request_timeout_seconds": self.request_timeout_seconds,
            "eval_interval": self.eval_interval,
            "log_grad_norms": self.log_grad_norms,
        }


# ---------------------------------------------------------------------------
# Backwards-compatibility alias
# ---------------------------------------------------------------------------

#: Historical name from the original PR 79 monitoring package. Kept as an
#: alias so existing imports (``from alpha_zero.monitoring import
#: MonitoringConfig``) continue to work after the PR 78 + PR 79 merge.
MonitoringConfig = MLflowConfig

__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_TRACKING_URI",
    "ENV_PREFIX",
    "MLflowConfig",
    "MonitoringConfig",
    "parse_tag_string",
]
