"""MLflow experiment-tracking integration.

This module unifies two independent MLflow logger implementations (PR #78 +
PR #79) into a single class with a superset surface area. Design constraints:

- ``import mlflow`` is optional; missing dep degrades to a no-op logger.
- The logger may be constructed from a :class:`MLflowConfig` (PR #78 style)
  *or* from individual keyword arguments (PR #79 style). Both produce the
  same internal state.
- Every public method that touches the MLflow API is wrapped in
  ``try / except``. When ``config.fail_silently`` is ``True`` (default)
  tracking failures never crash training.
- Nested run pattern: one parent run per campaign + one nested run per
  iteration. Iteration metrics are mirrored on the parent so the long-form
  trend is visible there.
- No hardcoded URIs, experiment names, or tags: everything flows through
  :class:`MLflowConfig`, which is itself populated from YAML + env vars.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
from collections.abc import Generator
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alpha_zero.monitoring.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    ENV_PREFIX,
    MLflowConfig,
)
from alpha_zero.monitoring.metrics import compute_elo_spread

if TYPE_CHECKING:
    from alpha_zero.evaluation.evaluation_pipeline import EvaluationResult
    from alpha_zero.tournament.elo import RatingLeaderboard, RatingUpdate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MLflow import
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Module-level constants & helpers
# ---------------------------------------------------------------------------

#: MLflow maximum allowed length for a parameter value string.
MLFLOW_MAX_PARAM_VALUE_LEN: int = 250

#: MLflow maximum allowed length for a metric key.
MLFLOW_MAX_METRIC_KEY_LEN: int = 250


def _truncate(s: str, max_len: int = MLFLOW_MAX_PARAM_VALUE_LEN) -> str:
    """Truncate *s* to *max_len* characters, appending ``...`` if truncated."""
    if len(s) <= max_len:
        return s
    if max_len < 3:
        return s[:max_len]
    return s[: max_len - 3] + "..."


def get_default_tracking_uri() -> str:
    """Return the tracking URI from env vars or the project default."""
    return (
        os.environ.get(f"{ENV_PREFIX}TRACKING_URI")
        or os.environ.get("MLFLOW_TRACKING_URI")
        or DEFAULT_TRACKING_URI
    )


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class MLflowLogger:
    """Hardened MLflow tracking client for the AlphaZero training loop.

    Construction is dual-mode for backwards compatibility:

    Config-object style (PR #78)::

        cfg = MLflowConfig(enabled=True, experiment_name="my-exp")
        ml = MLflowLogger(cfg)
        ml.init(config_dict={"lr": 1e-3})
        ml.log_iteration(0, train_metrics={"loss": 1.2})
        ml.finish()

    Keyword-argument style (PR #79)::

        with MLflowLogger(
            tracking_uri="file://./mlruns",
            experiment_name="my-exp",
            enabled=True,
        ) as ml:
            ml.log_metrics({"train/loss": 1.2}, step=0)
            ml.log_evaluation_result(result, prefix="baseline/random", step=0)

    Both styles share the same internal state; either lifecycle API
    (``init/finish`` or ``start_run/end_run`` / context manager) works.

    All metric-logging methods are no-ops when ``MLFLOW_AVAILABLE`` is
    ``False``, when ``config.enabled`` is ``False``, or when no run has
    been opened yet.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: MLflowConfig | None = None,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Initialise the logger.

        Either pass a fully-formed ``MLflowConfig`` as the positional arg,
        or pass individual keyword arguments. Passing both raises
        ``TypeError`` to avoid silent precedence bugs.
        """
        kw_overrides = {
            "tracking_uri": tracking_uri,
            "registry_uri": registry_uri,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "tags": tags,
            "description": description,
            "enabled": enabled,
        }
        kw_set = {k: v for k, v in kw_overrides.items() if v is not None}

        if config is None:
            # PR #79 style: build a config from individual kwargs.
            self.config: MLflowConfig = MLflowConfig(
                tracking_uri=tracking_uri or DEFAULT_TRACKING_URI,
                registry_uri=registry_uri,
                experiment_name=experiment_name or DEFAULT_EXPERIMENT_NAME,
                run_name=run_name,
                tags=dict(tags) if tags else {},
                description=description or "",
                enabled=bool(enabled) if enabled is not None else False,
            )
        elif kw_set:
            raise TypeError(
                "MLflowLogger: pass either a MLflowConfig OR keyword "
                f"arguments, not both. Got config + {sorted(kw_set)}."
            )
        else:
            self.config = config

        self._initialized: bool = False
        self._parent_run: Any = None
        self._parent_run_id: str | None = None
        self._client: Any = None
        self._atexit_registered: bool = False
        self._current_step: int | None = None
        # ``_mlflow`` and ``_run`` mirror the module-level symbols once the
        # logger has been initialised. They are also writable so legacy PR #79
        # tests can inject mocks directly without going through ``init()``.
        self._mlflow: Any = None
        self._run: Any = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """True if the logger is configured to run AND the dep is importable."""
        return bool(self.config.enabled and MLFLOW_AVAILABLE)

    @property
    def enabled(self) -> bool:
        """Backwards-compat alias for :attr:`is_enabled` (PR #79 surface)."""
        return self.is_enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Setter used by legacy code that toggles enablement at runtime."""
        self.config.enabled = bool(value)

    @property
    def is_initialized(self) -> bool:
        """True once a run has been successfully opened.

        Returns True if either the canonical ``init()`` path succeeded *or* a
        caller has manually injected an ``_mlflow`` module reference (legacy
        PR #79 test pattern).
        """
        return bool(self._initialized) or self._mlflow is not None

    @property
    def run(self) -> Any | None:
        """The active parent ``mlflow.ActiveRun`` (or ``None``)."""
        return self._parent_run if self._parent_run is not None else self._run

    @property
    def run_id(self) -> str | None:
        """Active MLflow run ID, or ``None`` if no run is active."""
        if self._parent_run_id is not None:
            return self._parent_run_id
        active = self._parent_run if self._parent_run is not None else self._run
        if active is None:
            return None
        with contextlib.suppress(Exception):
            run_id = active.info.run_id
            return str(run_id) if run_id is not None else None
        return None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> MLflowLogger:
        """Build a logger from environment variables."""
        return cls(MLflowConfig.from_env())

    @classmethod
    def from_config(cls, config: MLflowConfig) -> MLflowLogger:
        """Build a logger from an :class:`MLflowConfig` instance."""
        return cls(config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, config_dict: dict[str, Any] | None = None) -> Any | None:
        """Open the parent run and (optionally) log initial hyperparameters.

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
            # MlflowClient handles experiment management and parent-run
            # metric mirroring. Wrap in its own try/except so a transient
            # backend issue (or a test that only patches the bare ``mlflow``
            # module) does not prevent the parent run from opening.
            try:
                self._client = (
                    MlflowClient(tracking_uri=self.config.tracking_uri)
                    if MlflowClient is not None
                    else None
                )
                if self._client is not None:
                    self._ensure_experiment()
                else:
                    mlflow.set_experiment(self.config.experiment_name)
            except Exception as exc:
                if not self.config.fail_silently:
                    raise
                # Escalate to a real warning when the user explicitly asked
                # for a custom artifact_location (e.g. ``gs://bucket/...``);
                # falling back to ``mlflow.set_experiment(name)`` drops that
                # configuration silently and lands artifacts in the default
                # local store, which is almost never what the user wants.
                if self.config.artifact_location:
                    logger.warning(
                        "MlflowClient setup failed; artifact_location=%r "
                        "will NOT be honored - artifacts will land in the "
                        "default store. Cause: %s",
                        self.config.artifact_location,
                        exc,
                    )
                else:
                    logger.debug(
                        "MlflowClient setup failed (continuing without it): %s",
                        exc,
                    )
                self._client = None
                with suppress(Exception):
                    mlflow.set_experiment(self.config.experiment_name)
            run_tags: dict[str, str] = dict(self.config.tags)
            if self.config.description:
                run_tags.setdefault("mlflow.note.content", self.config.description)
            self._parent_run = mlflow.start_run(
                run_name=self.config.run_name,
                tags=run_tags or None,
                description=self.config.description or None,
            )
            self._parent_run_id = self._parent_run.info.run_id
            # Cache module/run references for instance-attribute access (PR #79).
            self._mlflow = mlflow
            self._run = self._parent_run
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
            self._mlflow = None
            self._run = None
            return None

    def _ensure_experiment(self) -> None:
        """Create the experiment if missing. Idempotent."""
        if self._client is None or mlflow is None:
            return
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
        """Open a run.

        - Called with **no arguments**, this is equivalent to :meth:`init`
          (PR #79 lifecycle): it opens the parent / campaign run.
        - Called with ``run_name=...`` or ``nested=True`` it opens an
          additional run handle (PR #78 lifecycle) and returns it. The
          caller owns the lifetime in that case.

        The PR #78 path additionally requires the parent run to be open
        already; calling ``start_run(run_name=...)`` without a preceding
        ``init()`` would otherwise create an orphan top-level run outside
        the managed lifecycle. We log a warning and return ``None`` in that
        case to surface the misuse without crashing training.
        """
        if run_name is None and not nested:
            self.init()
            return self._parent_run
        if not self.is_initialized:
            logger.warning(
                "MLflowLogger.start_run(run_name=%r, nested=%s) called before "
                "init(); refusing to open an orphan run. Call init() first.",
                run_name,
                nested,
            )
            return None
        if not self.is_enabled:
            return None
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is None:
            return None
        try:
            return client.start_run(run_name=run_name, nested=nested)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow start_run failed: %s", exc)
            return None

    def finish(self, status: str = "FINISHED") -> None:
        """Close the parent run. Idempotent - safe to call multiple times.

        Honors a logger that has been ``__enter__``-ed via the legacy PR #79
        pattern (manually-injected ``_mlflow``) as well as the canonical
        ``init()`` lifecycle.
        """
        if not self.is_initialized:
            return
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is not None:
            try:
                client.end_run(status=status)
            except Exception as exc:
                if not self.config.fail_silently:
                    raise
                logger.warning("MLflow end_run failed: %s", exc)
        self._initialized = False
        self._parent_run = None
        self._parent_run_id = None
        self._mlflow = None
        self._run = None

    def end_run(self, status: str = "FINISHED") -> None:
        """Backwards-compat alias for :meth:`finish` (PR #79 surface)."""
        self.finish(status=status)

    def _atexit_finish(self) -> None:
        # At interpreter shutdown the ``mlflow`` symbol may have been torn
        # down (e.g. by test-mock teardown). Bail out silently in that case
        # so we don't spam warnings during shutdown.
        if not self._initialized or mlflow is None:
            return
        with suppress(Exception):
            self.finish()

    # ------------------------------------------------------------------
    # Context-manager protocol (PR #79 surface)
    # ------------------------------------------------------------------

    def __enter__(self) -> MLflowLogger:
        self.init()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.finish(status=status)

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a batch of hyperparameters (stringified, 250-char truncated)."""
        if not self.is_initialized:
            return
        self._log_params_safe(params)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self.log_params({key: value})

    def log_metrics(self, data: dict[str, float], step: int | None = None) -> None:
        """Log a batch of scalar metrics.

        Non-numeric values and NaNs are dropped. ``step`` defaults to the
        active :meth:`step_context` value when set, else ``None``.
        """
        if not self.is_initialized or self._mlflow is None:
            return
        cleaned = self._clean_metrics(data)
        if not cleaned:
            return
        if step is None:
            step = self._current_step
        try:
            self._mlflow.log_metrics(cleaned, step=step)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow log_metrics failed: %s", exc)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric."""
        self.log_metrics({key: value}, step=step)

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
        """Log a dict of per-opponent ``EvaluationResult`` objects (PR #78).

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

    def log_evaluation_result(
        self,
        result: Any,
        *,
        prefix: str = "baseline",
        step: int | None = None,
    ) -> dict[str, float]:
        """Log a single ``EvaluationResult`` (PR #79: 14-metric breakdown).

        Extracts and logs:

        - ``{prefix}/wins``, ``/losses``, ``/draws``, ``/total_games``  (raw counts)
        - ``{prefix}/win_rate``, ``/draw_rate``, ``/loss_rate``         (fractions 0-1)
        - ``{prefix}/avg_game_length``, ``/avg_cities_won``,
          ``/avg_cities_lost``, ``/avg_units_won``, ``/avg_units_lost``,
          ``/total_evaluation_time``, ``/value_correlation``            (floats)

        The win/draw/loss rates on ``WinRateResult`` are stored as 0-100
        percentages internally; we normalise to fractions 0-1 here to match
        MLflow conventions.

        Returns:
            Dict of all metrics that were logged (empty if logger disabled).
        """
        if not self.is_initialized:
            return {}

        metrics: dict[str, float] = {}

        wr = getattr(result, "win_rate", None)
        if wr is not None:
            for count_attr in ("wins", "losses", "draws", "total_games"):
                val = getattr(wr, count_attr, None)
                if val is not None:
                    try:
                        metrics[f"{prefix}/{count_attr}"] = float(val)
                    except (TypeError, ValueError):
                        logger.debug("Cannot convert %s.%s to float", count_attr, val)

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
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is None or not self.config.log_artifacts:
            return
        path_str = str(local_path)
        if not os.path.exists(path_str):
            logger.warning("MLflowLogger.log_artifact: file not found: %s", path_str)
            return
        try:
            client.log_artifact(path_str, artifact_path=artifact_path)
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
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is None or not self.config.log_models:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning("Checkpoint not found, skipping artifact log: %s", path)
            return
        artifact_dir = f"checkpoints/iter_{iteration:06d}"
        try:
            client.log_artifact(str(path), artifact_path=artifact_dir)
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
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is None:
            return
        try:
            client.log_text(text, artifact_file)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow log_text failed: %s", exc)

    def log_figure(
        self,
        fig: Any,
        filename: str,
        artifact_path: str | None = None,
    ) -> None:
        """Save a matplotlib figure as an MLflow artifact.

        The figure is saved to a temporary file and uploaded. The format is
        inferred from the *filename* extension (png, pdf, svg, ...).
        """
        if not self.is_initialized:
            return
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is None:
            return
        import tempfile

        try:
            suffix = os.path.splitext(filename)[-1] or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
            try:
                fig.savefig(tmp_path, bbox_inches="tight", dpi=150)
                client.log_artifact(tmp_path, artifact_path=artifact_path)
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow log_figure failed: %s", exc)

    def log_table(
        self,
        data: Any,  # dict[str, list] | pandas.DataFrame
        artifact_filename: str,
        artifact_path: str | None = None,
    ) -> None:
        """Log a table (dict of lists or DataFrame) as a CSV artifact."""
        if not self.is_initialized:
            return
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is None:
            return
        import csv
        import tempfile

        try:
            try:
                rows = data.to_dict(orient="records")
                columns = list(data.columns)
            except AttributeError:
                columns = list(data.keys())
                rows = [
                    {col: data[col][i] for col in columns}
                    for i in range(len(next(iter(data.values()))))
                ]

            suffix = os.path.splitext(artifact_filename)[-1] or ".csv"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, newline=""
            ) as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
                tmp_path = tmp.name

            try:
                client.log_artifact(tmp_path, artifact_path=artifact_path)
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow log_table failed: %s", exc)

    def set_tag(self, key: str, value: str) -> None:
        """Set a single tag on the active run."""
        if not self.is_initialized:
            return
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is None:
            return
        try:
            client.set_tag(key, value)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow set_tag failed: %s", exc)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set multiple tags on the active run."""
        if not self.is_initialized:
            return
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is None:
            return
        try:
            client.set_tags({str(k): _truncate(str(v)) for k, v in tags.items()})
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow set_tags failed: %s", exc)

    @contextlib.contextmanager
    def step_context(self, step: int) -> Generator[None, None, None]:
        """Context manager that attaches an implicit step to metric calls."""
        prev = self._current_step
        self._current_step = step
        try:
            yield
        finally:
            self._current_step = prev

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_to_parent(
        self,
        metrics: dict[str, float],
        step: int | None,
    ) -> None:
        """Log metrics to the parent run via the MlflowClient API.

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
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is None:
            return
        try:
            with client.start_run(run_name=run_name, nested=True):
                client.log_metrics(metrics, step=step)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow nested run %s failed: %s", run_name, exc)

    def _log_params_safe(self, params: dict[str, Any]) -> None:
        client: Any = self._mlflow if self._mlflow is not None else mlflow
        if client is None:
            return
        try:
            cleaned = {k: self._stringify_param(v) for k, v in params.items()}
            client.log_params(cleaned)
        except Exception as exc:
            if not self.config.fail_silently:
                raise
            logger.warning("MLflow log_params failed: %s", exc)

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
        """Stringify *value* for MLflow param logging with 250-char truncation."""
        if isinstance(value, (str, int, float, bool)):
            s = str(value)
        else:
            s = repr(value)
        return _truncate(s, MLFLOW_MAX_PARAM_VALUE_LEN)

    @staticmethod
    def _now_ms() -> int:
        # Imported locally so tests can freeze time without patching at
        # module import time.
        import time

        return int(time.time() * 1000)


class WandbCompatAdapter(MLflowLogger):
    """Drop-in replacement for the retired W&B ``WandbLogger``.

    Re-exposes every public method of the legacy ``WandbLogger`` so worker
    code can switch the import line and gain MLflow tracking with no other
    changes.  Method signatures mirror ``WandbLogger`` exactly to avoid
    silent breakage at PBT/tournament/eval call sites.

    Fidelity map (W&B feature -> adapter behavior):

    ========================  =============================================
    ``log`` / ``log_*``       full parity (numpy scalars/0-d arrays coerced,
                              one level of nested dicts flattened)
    ``alert``                 MLflow tag ``alert/<level>/<title>`` **plus**
                              ``logger.warning`` so alerts are never silent
    ``summary``               MLflow tags (stringified)
    ``log_table``             CSV artifact (rows must be rectangular)
    ``log_model``             plain file artifact under ``model/``
    ``watch_model``           accepted no-op: sets ``watch_model_requested``
                              tag + info log; use ``MLflowConfig.log_grad_norms``
                              for gradient-norm metrics
    ``define_metric`` /       accepted no-ops (MLflow has no chart
    ``define_*_charts``       pre-registration)
    ``log_histogram`` /       accepted no-ops
    ``log_image``
    ========================  =============================================
    """

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> WandbCompatAdapter:
        """Construct from ``MLFLOW_*`` / ``CIV6AZ_MLFLOW_*`` environment variables."""
        from typing import cast as _cast

        from alpha_zero.monitoring.config import MLflowConfig

        return _cast(WandbCompatAdapter, cls.from_config(MLflowConfig.from_env()))

    # ------------------------------------------------------------------
    # Lifecycle parity (WandbLogger.finish; init inherited from MLflowLogger)
    # ------------------------------------------------------------------

    def finish(  # type: ignore[override]
        self,
        exit_code: int | None = None,
        status: str | None = None,
    ) -> None:
        """Mirror ``WandbLogger.finish`` while preserving ``MLflowLogger.finish``.

        Accepts both the W&B ``exit_code`` kwarg and the MLflow ``status`` kwarg
        so base-style ``end_run`` callbacks (which forward ``status=...``) and
        worker code (which passes nothing or ``exit_code=0``) both work.
        """
        if status is None:
            status = "FAILED" if exit_code not in (None, 0) else "FINISHED"
        # Delegate to the parent finish() (which handles the actual MLflow
        # teardown) via super() to avoid infinite recursion through end_run.
        super().finish(status=status)

    @property
    def step(self) -> int:
        return int(getattr(self, "_current_step", 0) or 0)

    @property
    def run(self) -> Any | None:
        return self._run

    @property
    def run_name(self) -> str | None:
        """Human-readable name of the active run (``None`` before ``init``).

        MLflow keeps run metadata under ``run.info`` — callers must not read
        ``.name``/``.id`` off the run object directly (they don't exist there,
        unlike on a W&B ``Run``). Falls back to the run id when the backend
        assigned no name.
        """
        info = getattr(self._run, "info", None)
        if info is None:
            return None
        return getattr(info, "run_name", None) or getattr(info, "run_id", None)

    # ------------------------------------------------------------------
    # Generic log()
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_metric_value(value: Any) -> float | None:
        """Best-effort coercion of a logged value to an MLflow float metric.

        Accepts Python ints/floats/bools and numpy scalars / 0-d / size-1
        arrays (via ``.item()``). Returns ``None`` for anything that is not
        a single numeric value (strings, sequences, larger arrays, ...).
        """
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        # numpy scalars (np.int64, np.float32, np.bool_) and 0-d / size-1
        # arrays all expose .item(); duck-type so numpy stays optional here.
        item = getattr(value, "item", None)
        if callable(item):
            size = getattr(value, "size", 1)
            if size == 1:
                try:
                    coerced = value.item()
                except (ValueError, TypeError):
                    return None
                if isinstance(coerced, bool):
                    return float(coerced)
                if isinstance(coerced, (int, float)):
                    return float(coerced)
        return None

    def log(
        self,
        data: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Mirror ``WandbLogger.log(data, step, commit)``.

        Numeric values (including numpy scalars and 0-d/size-1 arrays) are
        forwarded to ``log_metrics``; one level of nested dicts is flattened
        as ``parent/child``. Anything else is dropped with a debug log,
        because MLflow metrics must be floats.
        """
        numeric: dict[str, float] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    coerced = self._coerce_metric_value(sub_value)
                    if coerced is not None:
                        numeric[f"{key}/{sub_key}"] = coerced
                    else:
                        logger.debug("Dropping non-numeric metric %s/%s", key, sub_key)
                continue
            coerced = self._coerce_metric_value(value)
            if coerced is not None:
                numeric[key] = coerced
            else:
                logger.debug("Dropping non-numeric metric %s (%s)", key, type(value).__name__)
        if numeric:
            self.log_metrics(numeric, step=step)

    # ------------------------------------------------------------------
    # Training / selfplay / mcts / eval / imitation / pbt / game
    # ------------------------------------------------------------------

    def log_training_step(
        self,
        step: int,
        loss: float | None = None,
        policy_loss: float | None = None,
        value_loss: float | None = None,
        learning_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Mirror ``WandbLogger.log_training_step``."""
        payload: dict[str, float] = {}
        for name, val in (
            ("loss", loss),
            ("policy_loss", policy_loss),
            ("value_loss", value_loss),
            ("learning_rate", learning_rate),
        ):
            if val is not None:
                payload[f"train/{name}"] = float(val)
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                key = k if k.startswith("train/") else f"train/{k}"
                payload[key] = float(v)
        if payload:
            self.log_metrics(payload, step=int(step))

    def log_selfplay(
        self,
        games_completed: int,
        buffer_size: int,
        samples_generated: int,
        **kwargs: Any,
    ) -> None:
        """Mirror ``WandbLogger.log_selfplay``."""
        payload: dict[str, float] = {
            "selfplay/games_completed": float(games_completed),
            "selfplay/buffer_size": float(buffer_size),
            "selfplay/samples_generated": float(samples_generated),
        }
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                key = k if k.startswith("selfplay/") else f"selfplay/{k}"
                payload[key] = float(v)
        self.log_metrics(payload)

    def log_mcts(
        self,
        average_depth: float,
        average_visits: float,
        tree_reuse_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Mirror ``WandbLogger.log_mcts``."""
        payload: dict[str, float] = {
            "mcts/average_depth": float(average_depth),
            "mcts/average_visits": float(average_visits),
        }
        if tree_reuse_rate is not None:
            payload["mcts/tree_reuse_rate"] = float(tree_reuse_rate)
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                key = k if k.startswith("mcts/") else f"mcts/{k}"
                payload[key] = float(v)
        self.log_metrics(payload)

    def log_evaluation(  # type: ignore[override]
        self,
        win_rate: float,
        games_played: int,
        average_reward: float,
        average_turns: float,
        **kwargs: Any,
    ) -> None:
        """Mirror ``WandbLogger.log_evaluation``.

        Intentionally overrides ``MLflowLogger.log_evaluation`` (PR 79 surface,
        which takes ``dict[str, EvaluationResult]``) to provide W&B parity at
        the adapter layer.  Worker code that wants the dict form can call
        :py:meth:`log_baseline_evaluation` instead.
        """
        iteration = kwargs.pop("iteration", None)
        payload: dict[str, float] = {
            "eval/win_rate": float(win_rate),
            "eval/games_played": float(games_played),
            "eval/average_reward": float(average_reward),
            "eval/average_turns": float(average_turns),
        }
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                key = k if k.startswith("eval/") else f"eval/{k}"
                payload[key] = float(v)
        step = int(iteration) if iteration is not None else None
        self.log_metrics(payload, step=step)

    def log_baseline_evaluation(
        self,
        results: dict[str, Any],
        iteration: int | None = None,
    ) -> dict[str, float]:
        """Mirror ``WandbLogger.log_baseline_evaluation``."""
        combined: dict[str, float] = {}
        for opponent_name, result in results.items():
            prefix = f"eval/{opponent_name}"
            combined.update(self.log_evaluation_result(result, prefix=prefix, step=iteration))
        return combined

    def log_imitation_training(
        self,
        epoch: int,
        loss: float,
        policy_loss: float,
        value_loss: float,
        policy_accuracy: float | None = None,
        value_mse: float | None = None,
        games_used: int | None = None,
        samples_per_epoch: int | None = None,
    ) -> None:
        """Mirror ``WandbLogger.log_imitation_training``."""
        payload: dict[str, float] = {
            "imitation/epoch": float(epoch),
            "imitation/loss": float(loss),
            "imitation/policy_loss": float(policy_loss),
            "imitation/value_loss": float(value_loss),
        }
        if policy_accuracy is not None:
            payload["imitation/policy_accuracy"] = float(policy_accuracy)
        if value_mse is not None:
            payload["imitation/value_mse"] = float(value_mse)
        if games_used is not None:
            payload["imitation/games_used"] = float(games_used)
        if samples_per_epoch is not None:
            payload["imitation/samples_per_epoch"] = float(samples_per_epoch)
        self.log_metrics(payload, step=int(epoch))

    def log_reflection_stats(
        self,
        total_entries: int,
        unique_patterns: int | None = None,
        recent_additions: int | None = None,
        avg_critique_length: float | None = None,
        iteration: int | None = None,
    ) -> None:
        """Mirror ``WandbLogger.log_reflection_stats``."""
        payload: dict[str, float] = {"reflection/total_entries": float(total_entries)}
        if unique_patterns is not None:
            payload["reflection/unique_patterns"] = float(unique_patterns)
        if recent_additions is not None:
            payload["reflection/recent_additions"] = float(recent_additions)
        if avg_critique_length is not None:
            payload["reflection/avg_critique_length"] = float(avg_critique_length)
        step = int(iteration) if iteration is not None else None
        self.log_metrics(payload, step=step)

    def log_population_stats(
        self,
        generation: int,
        mean_fitness: float,
        max_fitness: float,
        min_fitness: float,
        mean_elo: float,
        max_elo: float,
        min_elo: float,
        population_size: int,
        mutations_count: int = 0,
        **kwargs: Any,
    ) -> None:
        """Mirror ``WandbLogger.log_population_stats`` (PBT)."""
        payload: dict[str, float] = {
            "pbt/generation": float(generation),
            "pbt/fitness_mean": float(mean_fitness),
            "pbt/fitness_max": float(max_fitness),
            "pbt/fitness_min": float(min_fitness),
            "pbt/elo_mean": float(mean_elo),
            "pbt/elo_max": float(max_elo),
            "pbt/elo_min": float(min_elo),
            "pbt/population_size": float(population_size),
            "pbt/mutations_count": float(mutations_count),
        }
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                key = k if k.startswith("pbt/") else f"pbt/{k}"
                payload[key] = float(v)
        self.log_metrics(payload, step=int(generation))

    def log_game(
        self,
        game_id: str,
        winner: int,
        total_turns: int,
        reward: float,
        game_length_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Mirror ``WandbLogger.log_game``."""
        self.set_tag(f"game/{game_id}", "logged")
        payload: dict[str, float] = {
            "game/winner": float(winner),
            "game/total_turns": float(total_turns),
            "game/reward": float(reward),
        }
        if game_length_seconds is not None:
            payload["game/duration_seconds"] = float(game_length_seconds)
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                key = k if k.startswith("game/") else f"game/{k}"
                payload[key] = float(v)
        self.log_metrics(payload)

    # ------------------------------------------------------------------
    # Tables / artifacts / summary / alerts
    # ------------------------------------------------------------------

    def summary(self, data: dict[str, Any]) -> None:
        """Mirror ``WandbLogger.summary`` - write summary values as MLflow tags."""
        for k, v in data.items():
            self.set_tag(f"summary/{k}", str(v))

    def alert(self, title: str, text: str, level: str = "INFO") -> None:
        """Mirror ``WandbLogger.alert`` - write alert as an MLflow tag.

        Also emits a ``logger.warning`` so alerts are visible in worker logs
        even when MLflow is disabled or the tag is later overwritten by a
        repeat alert with the same level/title.
        """
        logger.warning("tracker alert [%s] %s: %s", level, title, text)
        self.set_tag(f"alert/{level}/{title}", text)

    def log_table(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Polymorphic mirror of ``WandbLogger.log_table`` and ``MLflowLogger.log_table``.

        Three call shapes are supported so worker code that uses either the
        W&B-style or MLflow-style signature keeps working:

        * ``log_table(name, columns, data)`` - W&B row-oriented form.
        * ``log_table(column_data, artifact_filename=...)`` - MLflow form.
        * ``log_table(data=column_data, artifact_filename=...)`` - kwargs form.
        """
        # W&B-style: (name: str, columns: list[str], data: list[list])
        if (
            len(args) == 3
            and isinstance(args[0], str)
            and isinstance(args[1], list)
            and isinstance(args[2], list)
        ):
            name, columns, rows = args
            if not rows or not columns:
                return
            # Validate rectangular shape; refusing non-rectangular data
            # loudly surfaces malformed calls instead of corrupting the CSV
            # via the previous strict=False zip (which silently dropped cells).
            n_cols = len(columns)
            column_data: dict[str, list[Any]] = {col: [] for col in columns}
            for row_idx, row in enumerate(rows):
                if len(row) != n_cols:
                    raise ValueError(
                        f"log_table: row {row_idx} has {len(row)} values but "
                        f"columns has {n_cols} entries; refusing to log "
                        f"non-rectangular data"
                    )
                for col, val in zip(columns, row, strict=True):
                    column_data[col].append(val)
            super().log_table(column_data, artifact_filename=f"{name}.csv")
            return
        # MLflow-style: delegate verbatim.
        super().log_table(*args, **kwargs)

    # ------------------------------------------------------------------
    # No-op parity (W&B-only features without a clean MLflow equivalent)
    # ------------------------------------------------------------------

    def watch_model(self, *args: Any, **kwargs: Any) -> None:
        """Accepted no-op: MLflow has no direct equivalent of ``wandb.watch``.

        Records the request as a tag and an info log so the intent is
        auditable. For gradient-norm metrics enable
        ``MLflowConfig.log_grad_norms`` instead.
        """
        logger.info(
            "watch_model() requested; gradient watching is not supported by the "
            "MLflow backend — enable MLflowConfig.log_grad_norms for grad-norm metrics."
        )
        self.set_tag("watch_model_requested", "true")
        return None

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        """No-op: MLflow infers metric axes from logged step values."""
        return None

    def define_baseline_charts(self) -> None:
        """No-op: MLflow dashboards do not need chart pre-registration."""
        return None

    def define_pbt_charts(self) -> None:
        """No-op: MLflow dashboards do not need chart pre-registration."""
        return None

    def log_histogram(
        self,
        name: str,
        values: Any,
        num_bins: int = 64,
    ) -> None:
        """No-op stub for parity with ``WandbLogger.log_histogram``."""
        return None

    def log_image(
        self,
        name: str,
        image: Any,
        caption: str | None = None,
    ) -> None:
        """No-op stub for parity with ``WandbLogger.log_image``."""
        return None

    def log_model(
        self,
        model_path: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mirror ``WandbLogger.log_model`` - upload checkpoint as an artifact."""
        self.log_artifact(str(model_path), artifact_path="model")


__all__ = [
    "MLFLOW_AVAILABLE",
    "MLFLOW_MAX_METRIC_KEY_LEN",
    "MLFLOW_MAX_PARAM_VALUE_LEN",
    "MLflowLogger",
    "MlflowClient",
    "WandbCompatAdapter",
    "get_default_tracking_uri",
    "mlflow",
]
