"""
MLflow Monitoring Configuration.

Provides MonitoringConfig for configuring MLflow tracking URIs, experiment
names, and run metadata.  Uses ``or``-based null coalescing to safely handle
YAML keys that are defined without values (i.e. parsed as Python ``None``).

Example YAML (dangerous: tracking_uri: -> becomes None, NOT converted to
"None" via str()):
    monitoring:
      tracking_uri:
      experiment_name: my-experiment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

#: Default MLflow tracking URI (local file store).
DEFAULT_TRACKING_URI: str = "mlruns"

#: Default experiment name used when none is specified.
DEFAULT_EXPERIMENT_NAME: str = "civ6az-alphazero"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class MonitoringConfig:
    """Configuration for MLflow experiment tracking.

    Attributes:
        tracking_uri: URI for the MLflow tracking server or local directory.
        registry_uri: Optional model registry URI (None disables registry).
        experiment_name: Name of the MLflow experiment.
        run_name: Optional human-readable run name (auto-generated if None).
        tags: Key-value tags attached to every MLflow run.
        description: Free-text description for the MLflow experiment/run.
        enabled: Master switch to disable MLflow without removing config.
        eval_interval: Log mid-run evaluation metrics every N iterations.
            0 disables mid-run evaluation (default).
        log_grad_norms: When True, log L2 gradient norms per epoch to MLflow.
    """

    tracking_uri: str = DEFAULT_TRACKING_URI
    registry_uri: str | None = None
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True
    eval_interval: int = 0       # 0 = disabled; >0 = evaluate every N iterations
    log_grad_norms: bool = False  # Log gradient L2 norms per epoch

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonitoringConfig:
        """Create a MonitoringConfig from a raw dictionary (e.g. from YAML).

        Uses ``or``-based null coalescing to safely handle YAML keys that
        are present but have no value (parsed as Python ``None``).  This
        prevents the ``str(None)`` → ``"None"`` footgun which would cause
        downstream MLflow failures when using the value as a URI or name.

        Args:
            data: Dictionary of configuration values.

        Returns:
            A fully-initialised MonitoringConfig instance.

        Example::

            cfg = MonitoringConfig.from_dict(yaml.safe_load(open("config.yaml")))
        """
        # Tags: accept dict or list-of-pairs; fall back to empty dict.
        raw_tags = data.get("tags") or {}
        if isinstance(raw_tags, list):
            # Support [{key: k, value: v}, ...] YAML list format.
            tags_dict: dict[str, str] = {}
            for item in raw_tags:
                if isinstance(item, dict) and "key" in item and "value" in item:
                    tags_dict[str(item["key"])] = str(item["value"])
        elif isinstance(raw_tags, dict):
            tags_dict = {str(k): str(v) for k, v in raw_tags.items()}
        else:
            logger.warning(
                "MonitoringConfig.from_dict: unexpected 'tags' type %s; ignoring.",
                type(raw_tags).__name__,
            )
            tags_dict = {}

        return cls(
            # ``or`` replaces None (or empty string) with the module default.
            tracking_uri=str(data.get("tracking_uri") or DEFAULT_TRACKING_URI),
            registry_uri=data.get("registry_uri"),  # None is valid (disables registry)
            experiment_name=str(data.get("experiment_name") or DEFAULT_EXPERIMENT_NAME),
            run_name=data.get("run_name"),  # None is valid (auto-generated)
            tags=tags_dict,
            description=str(data.get("description") or ""),
            enabled=bool(data.get("enabled", True)),
            eval_interval=int(data.get("eval_interval") or 0),
            log_grad_norms=bool(data.get("log_grad_norms", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary suitable for YAML/JSON output.

        Returns:
            Dictionary representation of this config.
        """
        return {
            "tracking_uri": self.tracking_uri,
            "registry_uri": self.registry_uri,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tags": self.tags,
            "description": self.description,
            "enabled": self.enabled,
            "eval_interval": self.eval_interval,
            "log_grad_norms": self.log_grad_norms,
        }
