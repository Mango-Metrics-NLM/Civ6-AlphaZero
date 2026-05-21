"""MLflow experiment-tracking integration.

This module provides a hardened, optional wrapper around the MLflow Python
client. Design constraints:

- ``import mlflow`` is optional; missing dep degrades to a no-op logger.
- Every public method is guarded by ``is_initialized`` and wrapped in
  ``try/except``. With ``fail_silently=True`` (default) tracking failures
  never crash training.
- Nested run pattern: one parent run per campaign + one nested run per
  iteration, with metrics also pushed to the parent so the 50-point trend
  is visible in the parent run view.
- No hardcoded URIs, experiment names, or tags: everything comes from
  ``MLflowConfig`` which is itself populated from YAML + env vars.
"""

from __future__ import annotations

import atexit
import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alpha_zero.monitoring.config import MLflowConfig
from alpha_zero.monitoring.metrics import compute_elo_spread

if TYPE_CHECKING:
    from alpha_zero.evaluation.evaluation_pipeline import EvaluationResult
    from alpha_zero.tournament.elo import RatingLeaderboard, RatingUpdate

# Optional import — single import avoids isort ambiguity across ruff versions.
try:
    import mlflow as _mlflow
    from mlflow.tracking import MlflowClient as _MlflowClient

    MLFLOW_AVAILABLE = True
    mlflow: Any = _mlflow
    MlflowClient: Any = _MlflowClient
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

logger = logging.getLogger(__name__)


class MLflowLogger:
    """Hardened MLflow tracking client for the AlphaZero training loop.

    Lifecycle::

        logger = MLflowLogger(config)   # cheap; no I/O
        logger.init(config_dict=hparams)  # opens parent run
        for it in range(num_iterations):
            logger.log_iteration(it, train_metrics, game_metrics, scheduler_metrics)
            # periodically:
            logger.log_evaluation(eval_results, it)
            logger.log_elo_update(updates, leaderboard, it)
        logger.finish()  # idempotent; closes parent run

    All metric-logging methods are no-ops when ``MLFLOW_AVAILABLE`` is False,
    when ``config.enabled`` is False, or when ``init()`` has not been called.
    """

    def __init__(self, config: MLflowConfig | None = None) -> None:
        self.config: MLflowConfig = config if config is not None else MLflowConfig()
        self._initialized: bool = False
        self._parent_run: Any = None
        self._parent_run_id: str | None = None
        self._client: Any = None
        self._atexit_registered: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """True if the logger is configured to run AND the dep is importable."""
        return bool(self.config.enabled and MLFLOW_AVAILABLE)

    @property
    def is_initialized(self) -> bool:
        """True once ``init()`` has successfully opened a parent run."""
        return self._initialized

    @property
    def run(self) -> Any | None:
        """The active parent ``mlflow.ActiveRun`` or ``None``."""
        return self._parent_run

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> MLflowLogger:
        """Build a logger from environment variables."""
        return cls(MLflowConfig.from_env())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, config_dict: dict[str, Any] | None = None) -> Any | None:
        """Open the parent run and log initial hyperparameters.

        Args:
            config_dict: Flat hyperparameter dict logged as MLflow params.

        Returns:
            The active parent ``mlflow.ActiveRun`` or ``None`` on no-op.
        """
        if not self.is_enabled:
            return None
        if self._initialized:
            return self._parent_run
        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            if self.config.registry_uri:
                mlflow.set_registry_uri(self.config.registry_uri)
            self._client = MlflowClient(tracking_uri=self.config.tracking_uri)
            self._ensure_experiment()
            self._parent_run = mlflow.start_run(
                run_name=self.config.run_name,
                tags=dict(self.config.tags) or None,
                description=self.config.description or None,
            )
            self._parent_run_id = self._parent_run.info.run_id
            if config_dict:
                self._log_params_safe(config_dict)
            self._initialized = True
            if not self._atexit_registered:
                atexit.register(self._atexit_finish)
                self._atexit_registered = True
            logger.info(
                "MLflow initialized: uri=%s experiment=%s run_id=%s",
                self.config.tracking_uri,
                self.config.experiment_name,
                self._parent_run_id,
            )
            return self._parent_run
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow init failed; disabling logger: %s", exc)
            self.config.enabled = False
            self._initialized = False
            return None

    def _ensure_experiment(self) -> None:
        """Create the experiment if missing. Idempotent."""
        existing = self._client.get_experiment_by_name(self.config.experiment_name)
        if existing is None:
            self._client.create_experiment(
                name=self.config.experiment_name,
                artifact_location=self.config.artifact_location,
            )
        mlflow.set_experiment(self.config.experiment_name)

    def start_run(
        self,
        run_name: str | None = None,
        nested: bool = False,
    ) -> Any | None:
        """Open a (possibly nested) run. Caller owns the context lifetime."""
        if not self.is_enabled:
            return None
        try:
            return mlflow.start_run(run_name=run_name, nested=nested)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow start_run failed: %s", exc)
            return None

    def finish(self, status: str = "FINISHED") -> None:
        """Close the parent run. Idempotent — safe to call multiple times."""
        if not self._initialized:
            return
        try:
            mlflow.end_run(status=status)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow end_run failed: %s", exc)
        finally:
            self._initialized = False
            self._parent_run = None
            self._parent_run_id = None

    def _atexit_finish(self) -> None:
        # At interpreter shutdown the ``mlflow`` symbol may have been torn down
        # (e.g. by test mock teardown). Bail out silently in that case so we
        # don't spam warnings during shutdown.
        if not self._initialized or mlflow is None:
            return
        with suppress(Exception):
            self.finish()

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        data: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log a batch of scalar metrics."""
        if not self.is_initialized:
            return
        cleaned = self._clean_metrics(data)
        if not cleaned:
            return
        try:
            mlflow.log_metrics(cleaned, step=step)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow log_metrics failed: %s", exc)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a batch of hyperparameters."""
        if not self.is_initialized:
            return
        self._log_params_safe(params)

    def _log_params_safe(self, params: dict[str, Any]) -> None:
        try:
            cleaned = {k: self._stringify_param(v) for k, v in params.items()}
            mlflow.log_params(cleaned)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow log_params failed: %s", exc)

    def log_iteration(
        self,
        iteration: int,
        train_metrics: dict[str, float] | None = None,
        game_metrics: dict[str, float] | None = None,
        scheduler_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log one training iteration's metrics.

        Composes a single flat dict with namespace prefixes already supplied
        by the caller (``train/*``, ``game/*``, etc.) and emits it as one
        ``log_metrics(step=iteration)`` call on the parent run.

        When ``config.nested_runs`` is True, also opens a nested run
        ``iter_<iteration>`` and logs the same metrics there.
        """
        if not self.is_initialized:
            return
        flat: dict[str, float] = {}
        for d in (train_metrics, game_metrics, scheduler_metrics):
            if d:
                flat.update(d)
        if not flat:
            return
        cleaned = self._clean_metrics(flat)
        self._log_to_parent(cleaned, step=iteration)
        if self.config.nested_runs:
            self._log_in_nested_run(
                run_name=f"iter_{iteration:03d}",
                metrics=cleaned,
                step=iteration,
            )

    def log_evaluation(
        self,
        results: dict[str, EvaluationResult],
        iteration: int,
    ) -> None:
        """Log a dict of per-opponent ``EvaluationResult`` objects.

        Emits keys ``eval/<opponent>/win_rate``, ``eval/<opponent>/loss_rate``,
        ``eval/<opponent>/draw_rate``, ``eval/<opponent>/value_correlation``,
        ``eval/<opponent>/avg_game_length`` on the parent run.
        """
        if not self.is_initialized:
            return
        metrics: dict[str, float] = {}
        for opponent, result in results.items():
            prefix = f"eval/{opponent}"
            win_rate_obj: Any = getattr(result, "win_rate", None)
            if hasattr(win_rate_obj, "win_rate"):
                wr_value = float(win_rate_obj.win_rate)
                loss_value = float(getattr(win_rate_obj, "loss_rate", 0.0))
                draw_value = float(getattr(win_rate_obj, "draw_rate", 0.0))
            else:
                wr_value = float(win_rate_obj or 0.0)
                loss_value = 0.0
                draw_value = 0.0
            metrics[f"{prefix}/win_rate"] = wr_value
            metrics[f"{prefix}/loss_rate"] = loss_value
            metrics[f"{prefix}/draw_rate"] = draw_value
            metrics[f"{prefix}/value_correlation"] = float(
                getattr(result, "value_correlation", 0.0) or 0.0
            )
            metrics[f"{prefix}/avg_game_length"] = float(
                getattr(result, "avg_game_length", 0.0) or 0.0
            )
        if metrics:
            self._log_to_parent(self._clean_metrics(metrics), step=iteration)

    def log_elo_update(
        self,
        updates: list[RatingUpdate],
        leaderboard: RatingLeaderboard,
        iteration: int,
    ) -> None:
        """Log per-player ELO changes and aggregate spread."""
        if not self.is_initialized:
            return
        metrics: dict[str, float] = {}
        for update in updates:
            prefix = f"elo/{update.player_id}"
            metrics[f"{prefix}/rating"] = float(update.rating_after)
            metrics[f"{prefix}/rating_change"] = float(update.rating_change)
        spread = compute_elo_spread(leaderboard)
        for key, value in spread.items():
            metrics[f"elo/{key}"] = float(value)
        if metrics:
            self._log_to_parent(self._clean_metrics(metrics), step=iteration)

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Upload a file artifact to the active run."""
        if not self.is_initialized:
            return
        if not self.config.log_artifacts:
            return
        try:
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow log_artifact failed: %s", exc)

    def log_model_checkpoint(
        self,
        checkpoint_path: str | Path,
        iteration: int,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Upload a model checkpoint file under ``checkpoints/iter_NNNNNN/``."""
        if not self.is_initialized:
            return
        if not self.config.log_models:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning("Checkpoint not found, skipping artifact log: %s", path)
            return
        artifact_dir = f"checkpoints/iter_{iteration:06d}"
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_dir)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow log_model_checkpoint failed: %s", exc)
            return
        if metrics:
            self._log_to_parent(
                self._clean_metrics({f"checkpoint/{k}": v for k, v in metrics.items()}),
                step=iteration,
            )

    def log_text(self, text: str, artifact_file: str) -> None:
        """Log a string as an artifact file."""
        if not self.is_initialized:
            return
        try:
            mlflow.log_text(text, artifact_file)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow log_text failed: %s", exc)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on the active run."""
        if not self.is_initialized:
            return
        try:
            mlflow.set_tags({str(k): str(v) for k, v in tags.items()})
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow set_tags failed: %s", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_to_parent(self, metrics: dict[str, float], step: int | None) -> None:
        """Log metrics to the parent run via the client API.

        Using ``MlflowClient.log_metric`` keeps the metrics attached to the
        parent run even while a nested run is the active context.
        """
        if not metrics or self._client is None or self._parent_run_id is None:
            return
        try:
            timestamp = int(self._now_ms())
            for name, value in metrics.items():
                self._client.log_metric(
                    run_id=self._parent_run_id,
                    key=name,
                    value=float(value),
                    timestamp=timestamp,
                    step=step,
                )
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow client.log_metric failed: %s", exc)

    def _log_in_nested_run(
        self,
        run_name: str,
        metrics: dict[str, float],
        step: int,
    ) -> None:
        try:
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.log_metrics(metrics, step=step)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow nested run %s failed: %s", run_name, exc)

    @staticmethod
    def _clean_metrics(data: dict[str, Any]) -> dict[str, float]:
        """Coerce values to float and drop non-numerics / NaN."""
        cleaned: dict[str, float] = {}
        for key, value in data.items():
            if value is None:
                continue
            try:
                fval = float(value)
            except (TypeError, ValueError):
                continue
            if fval != fval:  # NaN check without importing math
                continue
            cleaned[str(key)] = fval
        return cleaned

    @staticmethod
    def _stringify_param(value: Any) -> str:
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return repr(value)

    @staticmethod
    def _now_ms() -> int:
        # Imported locally so tests can freeze time without patching at module
        # import time.
        import time

        return int(time.time() * 1000)


def get_default_tracking_uri() -> str:
    """Return the tracking URI from env vars or the project default."""
    return (
        os.environ.get("CIV6AZ_MLFLOW_TRACKING_URI")
        or os.environ.get("MLFLOW_TRACKING_URI")
        or MLflowConfig().tracking_uri
    )
