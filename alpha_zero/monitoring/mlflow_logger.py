"""
MLflow Logger for Civ6 AlphaZero Training.

Provides a structured wrapper around MLflow with:
- Safe parameter stringification with 250-character truncation (MLflow limit).
- Null-safe metric and parameter logging.
- Context-manager-style run lifecycle management.
- Graceful degradation when mlflow is not installed.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)

#: MLflow maximum allowed length for a parameter value string.
MLFLOW_MAX_PARAM_VALUE_LEN: int = 250

#: MLflow maximum allowed length for a metric key.
MLFLOW_MAX_METRIC_KEY_LEN: int = 250


def _truncate(s: str, max_len: int = MLFLOW_MAX_PARAM_VALUE_LEN) -> str:
    """Truncate *s* to *max_len* characters, appending ``...`` if truncated.

    Args:
        s: String to truncate.
        max_len: Maximum length (inclusive). Must be >= 3.

    Returns:
        Truncated string, at most *max_len* characters long.
    """
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


class MLflowLogger:
    """Structured MLflow logger for AlphaZero training runs.

    Wraps ``mlflow`` with safe defaults, parameter value truncation,
    and graceful no-op behaviour when mlflow is unavailable.

    Example::

        from alpha_zero.monitoring.config import MonitoringConfig
        from alpha_zero.monitoring.mlflow_logger import MLflowLogger

        cfg = MonitoringConfig(experiment_name="my-experiment")
        with MLflowLogger.from_config(cfg) as ml:
            ml.log_params({"lr": 0.001, "batch_size": 256})
            for step, loss in enumerate(training_loop()):
                ml.log_metrics({"train/loss": loss}, step=step)
    """

    def __init__(
        self,
        tracking_uri: str = "mlruns",
        experiment_name: str = "civ6az-alphazero",
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
        enabled: bool = True,
        registry_uri: str | None = None,
    ) -> None:
        """Initialise the logger.

        Args:
            tracking_uri: MLflow tracking server URI or local directory.
            experiment_name: Experiment to log runs under.
            run_name: Human-readable run name (auto-generated if None).
            tags: Extra tags attached to the run.
            description: Free-text description for the run.
            enabled: When False the logger becomes a no-op.
            registry_uri: Optional model registry URI.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.description = description
        self.enabled = enabled
        self.registry_uri = registry_uri

        self._run: Any | None = None  # mlflow.ActiveRun
        self._mlflow: Any | None = None  # mlflow module

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> MLflowLogger:
        """Create from a :class:`~alpha_zero.monitoring.config.MonitoringConfig`.

        Args:
            config: MonitoringConfig instance.

        Returns:
            Configured MLflowLogger.
        """
        return cls(
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            run_name=config.run_name,
            tags=config.tags,
            description=config.description,
            enabled=config.enabled,
            registry_uri=config.registry_uri,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        """Start an MLflow run.  No-op when disabled or mlflow unavailable."""
        if not self.enabled:
            return

        try:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            if self.registry_uri:
                mlflow.set_registry_uri(self.registry_uri)

            mlflow.set_experiment(self.experiment_name)
            run_tags = {
                "mlflow.note.content": self.description,
                **self.tags,
            }
            self._run = mlflow.start_run(run_name=self.run_name, tags=run_tags)
            logger.info(
                "MLflow run started: experiment=%s run_id=%s",
                self.experiment_name,
                self._run.info.run_id,
            )
        except ImportError:
            logger.warning("mlflow not installed — MLflowLogger is a no-op for this run.")
            self.enabled = False
        except Exception as exc:
            logger.warning("MLflow start_run failed (%s); continuing without MLflow.", exc)
            self.enabled = False

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active MLflow run.

        Args:
            status: MLflow run status string (FINISHED, FAILED, KILLED).
        """
        if not self.enabled or self._mlflow is None:
            return
        with contextlib.suppress(Exception):
            self._mlflow.end_run(status=status)
            self._run = None
            logger.info("MLflow run ended with status=%s", status)

    def __enter__(self) -> MLflowLogger:
        self.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end_run(status=status)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stringify_param(value: Any) -> str:
        """Convert *value* to a string safe for MLflow parameter logging.

        MLflow enforces a 250-character limit on parameter values.  Long
        strings (e.g. full config repr) are truncated with a trailing
        ellipsis to stay within the limit.

        Args:
            value: Arbitrary Python value to stringify.

        Returns:
            String representation, at most 250 characters.
        """
        if isinstance(value, (str, int, float, bool)):
            s = str(value)
        else:
            s = repr(value)
        return _truncate(s, MLFLOW_MAX_PARAM_VALUE_LEN)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of parameters to MLflow.

        Values are stringified and truncated to 250 chars automatically.

        Args:
            params: Key-value parameter dictionary.
        """
        if not self.enabled or self._mlflow is None:
            return
        safe_params = {k: self._stringify_param(v) for k, v in params.items()}
        with contextlib.suppress(Exception):
            self._mlflow.log_params(safe_params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dictionary of numeric metrics to MLflow.

        Args:
            metrics: Key-value metrics dictionary (values must be numeric).
            step: Optional training step / iteration number.
        """
        if not self.enabled or self._mlflow is None:
            return
        with contextlib.suppress(Exception):
            self._mlflow.log_metrics(metrics, step=step)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name.
            value: Parameter value (will be stringified + truncated).
        """
        self.log_params({key: value})

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric.

        Args:
            key: Metric name.
            value: Numeric metric value.
            step: Optional step number.
        """
        self.log_metrics({key: value}, step=step)

    def set_tag(self, key: str, value: str) -> None:
        """Set a run tag.

        Args:
            key: Tag key.
            value: Tag value.
        """
        if not self.enabled or self._mlflow is None:
            return
        with contextlib.suppress(Exception):
            self._mlflow.set_tag(key, _truncate(str(value)))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Upload a local file as an MLflow artifact.

        Args:
            local_path: Path to the local file.
            artifact_path: Optional subdirectory within the artifact store.
        """
        if not self.enabled or self._mlflow is None:
            return
        if not os.path.exists(local_path):
            logger.warning("MLflowLogger.log_artifact: file not found: %s", local_path)
            return
        with contextlib.suppress(Exception):
            self._mlflow.log_artifact(local_path, artifact_path=artifact_path)

    @property
    def run_id(self) -> str | None:
        """Active MLflow run ID, or None if no run is active."""
        if self._run is None:
            return None
        with contextlib.suppress(Exception):
            run_id = self._run.info.run_id
            return str(run_id) if run_id is not None else None
        return None

    def log_figure(
        self,
        fig: Any,
        filename: str,
        artifact_path: str | None = None,
    ) -> None:
        """Save a matplotlib figure as an MLflow artifact.

        The figure is saved to a temporary file and uploaded.  Supports
        any format that matplotlib accepts (png, pdf, svg, etc.) — inferred
        from the *filename* extension.

        Args:
            fig: A ``matplotlib.figure.Figure`` instance.
            filename: Filename for the artifact (e.g. ``"loss_curve.png"``).
            artifact_path: Optional subdirectory in the artifact store.
        """
        if not self.enabled or self._mlflow is None:
            return
        import tempfile

        with contextlib.suppress(Exception):
            suffix = os.path.splitext(filename)[-1] or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
            try:
                fig.savefig(tmp_path, bbox_inches="tight", dpi=150)
                self._mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    def log_table(
        self,
        data: Any,  # dict[str, list] | pd.DataFrame
        artifact_filename: str,
        artifact_path: str | None = None,
    ) -> None:
        """Log a table (dict or DataFrame) as a CSV artifact.

        Args:
            data: A dictionary of ``{column: [values]}`` or a pandas DataFrame.
            artifact_filename: Filename for the CSV artifact (e.g. ``"metrics.csv"``).
            artifact_path: Optional subdirectory in the artifact store.
        """
        if not self.enabled or self._mlflow is None:
            return
        import csv
        import tempfile

        with contextlib.suppress(Exception):
            # Normalise to list-of-rows
            try:
                # pandas DataFrame
                rows = data.to_dict(orient="records")
                columns = list(data.columns)
            except AttributeError:
                # plain dict {col: [vals]}
                columns = list(data.keys())
                rows = [
                    {col: data[col][i] for col in columns}
                    for i in range(len(next(iter(data.values()))))
                ]

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                delete=False,
                newline="",
            ) as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
                tmp_path = tmp.name

            try:
                self._mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    @contextlib.contextmanager
    def step_context(self, step: int) -> Generator[None, None, None]:
        """Context manager that automatically attaches a step to metrics.

        Example::

            with ml.step_context(step=epoch):
                ml.log_metric("loss", loss_value)

        Args:
            step: Step number to attach.

        Yields:
            Nothing.
        """
        # Store step for use in nested log_metric calls (future extension).
        self._current_step: int | None = step
        try:
            yield
        finally:
            self._current_step = None

    def log_evaluation_result(
        self,
        result: Any,
        *,
        prefix: str = "baseline",
        step: int | None = None,
    ) -> dict[str, float]:
        """Log an ``EvaluationResult`` to MLflow, handling the nested ``WinRateResult``.

        Extracts and logs:

        - ``{prefix}/wins``, ``/losses``, ``/draws``, ``/total_games``  (raw counts)
        - ``{prefix}/win_rate``, ``/draw_rate``, ``/loss_rate``         (fractions 0-1)
        - ``{prefix}/avg_game_length``, ``/avg_cities_won``, etc.       (floats)
        - ``{prefix}/value_correlation``                                 (float)

        Args:
            result: ``EvaluationResult`` from
                ``alpha_zero.evaluation.evaluation_pipeline``.
            prefix: Metric key prefix (e.g. ``"baseline/random"``).
            step: Optional MLflow step number.

        Returns:
            Dict of all metrics that were logged (empty if logger disabled).
        """
        if not self.enabled or self._mlflow is None:
            return {}

        metrics: dict[str, float] = {}

        # --- Win-rate metrics (nested WinRateResult) ---
        wr = getattr(result, "win_rate", None)
        if wr is not None:
            for count_attr in ("wins", "losses", "draws", "total_games"):
                val = getattr(wr, count_attr, None)
                if val is not None:
                    try:
                        metrics[f"{prefix}/{count_attr}"] = float(val)
                    except (TypeError, ValueError):
                        logger.debug("Cannot convert %s.%s to float", count_attr, val)

            # win_rate/draw_rate/loss_rate are 0-100 on WinRateResult; normalise to 0-1
            for rate_attr in ("win_rate", "draw_rate", "loss_rate"):
                val = getattr(wr, rate_attr, None)
                if val is not None:
                    try:
                        metrics[f"{prefix}/{rate_attr}"] = float(val) / 100.0
                    except (TypeError, ValueError):
                        logger.debug("Cannot convert %s.%s to float", rate_attr, val)
        else:
            logger.warning(
                "log_evaluation_result: result has no 'win_rate' attribute "
                "(type=%s); skipping rate metrics",
                type(result).__name__,
            )

        # --- Direct float attributes ---
        for attr in (
            "avg_game_length",
            "avg_cities_won",
            "avg_cities_lost",
            "avg_units_won",
            "avg_units_lost",
            "total_evaluation_time",
            "value_correlation",
        ):
            val = getattr(result, attr, None)
            if val is not None:
                try:
                    metrics[f"{prefix}/{attr}"] = float(val)
                except (TypeError, ValueError):
                    logger.debug("Cannot convert result.%s (%r) to float", attr, val)

        if metrics:
            self.log_metrics(metrics, step=step)
            logger.debug("Logged %d evaluation metrics under prefix %r", len(metrics), prefix)
        else:
            logger.warning("log_evaluation_result: no metrics extracted from result")

        return metrics

