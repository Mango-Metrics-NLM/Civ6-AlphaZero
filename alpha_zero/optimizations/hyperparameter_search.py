"""Optuna-backed hyperparameter search with MLflow trial logging.

Replaces the retired W&B sweep pathway (``wandb.sweep`` / ``wandb.agent``,
driven through ``scripts/run_sweep.py``). The design is deliberately
decoupled so each piece is reusable on its own:

- :class:`SweepConfig` — declarative, YAML-loadable configuration with
  ``CIV6AZ_SWEEP_*`` environment overrides (legacy ``SWEEP_TIMEOUT_SECONDS``
  is honored for backwards compatibility with the W&B-era wrapper).
- ``suggest_params`` — maps a declarative search space onto an Optuna trial.
- :class:`SubprocessTrainingObjective` — runs a training command built from
  a template, extracts the result from a ``SWEEP_METRICS_JSON=`` stdout
  marker, and prunes failed/timed-out trials instead of aborting the study.
- :class:`HyperparameterSearch` — orchestrates the study; both the objective
  and the experiment tracker are injected, so unit tests need no network,
  no subprocesses, and no MLflow server.

Optuna is an optional dependency (``pip install -e '.[tuning]'``); the
import is lazy and failure raises a clear, actionable error.

Example::

    from alpha_zero.optimizations.hyperparameter_search import (
        HyperparameterSearch,
        SweepConfig,
    )

    config = SweepConfig.from_yaml("configs/sweep_muzero.yaml")
    search = HyperparameterSearch(config)
    result = search.run()
    print(result.best_params, result.best_value)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Environment prefix for SweepConfig overrides (project CIV6AZ_ convention).
ENV_PREFIX = "CIV6AZ_SWEEP_"

#: Legacy env var honored by the retired W&B sweep wrapper; kept working.
LEGACY_TIMEOUT_ENV = "SWEEP_TIMEOUT_SECONDS"

#: Stdout marker emitted by training entry points for subprocess objectives.
METRICS_MARKER = "SWEEP_METRICS_JSON="

#: Default per-trial timeout (12h) — same default the W&B wrapper used.
DEFAULT_TRIAL_TIMEOUT_SECONDS = 12 * 60 * 60

_VALID_PARAM_TYPES = ("float", "int", "categorical")
_VALID_DIRECTIONS = ("maximize", "minimize")
_VALID_SAMPLERS = ("tpe", "random")


class SearchSpaceError(ValueError):
    """Raised when a declarative search space is malformed."""


def validate_search_space(space: Mapping[str, Any]) -> None:
    """Validate a declarative search space definition.

    Each entry maps a parameter name to a spec dict:

    - ``{"type": "float", "low": ..., "high": ..., "log": bool}``
    - ``{"type": "int", "low": ..., "high": ..., "step": int}``
    - ``{"type": "categorical", "choices": [...]}``

    Raises:
        SearchSpaceError: On unknown types, missing bounds/choices, or
            inverted ranges — with the offending parameter named.
    """
    if not space:
        raise SearchSpaceError("search_space must define at least one parameter")
    for name, spec in space.items():
        if not isinstance(spec, Mapping):
            raise SearchSpaceError(f"{name}: spec must be a mapping, got {type(spec).__name__}")
        ptype = spec.get("type")
        if ptype not in _VALID_PARAM_TYPES:
            raise SearchSpaceError(
                f"{name}: type must be one of {_VALID_PARAM_TYPES}, got {ptype!r}"
            )
        if ptype == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, (list, tuple)) or not choices:
                raise SearchSpaceError(f"{name}: categorical requires a non-empty 'choices' list")
        else:
            if "low" not in spec or "high" not in spec:
                raise SearchSpaceError(f"{name}: {ptype} requires 'low' and 'high'")
            if spec["low"] > spec["high"]:
                raise SearchSpaceError(
                    f"{name}: low ({spec['low']}) must be <= high ({spec['high']})"
                )


def suggest_params(trial: Any, space: Mapping[str, Any]) -> dict[str, Any]:
    """Draw one parameter set from *space* using an Optuna trial.

    Args:
        trial: ``optuna.trial.Trial`` (duck-typed; tests may pass fakes).
        space: Validated declarative search space (see
            :func:`validate_search_space`).

    Returns:
        Mapping of parameter name to the suggested value.
    """
    params: dict[str, Any] = {}
    for name, spec in space.items():
        ptype = spec["type"]
        if ptype == "float":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
            )
        else:  # categorical — validated upstream
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
    logger.debug("Trial %s suggested params: %s", getattr(trial, "number", "?"), params)
    return params


@dataclass
class SweepConfig:
    """Configuration for an Optuna hyperparameter search.

    Every operational value is a field with an explicit default; overridable
    via YAML (:meth:`from_yaml`) and ``CIV6AZ_SWEEP_*`` environment variables
    (:meth:`apply_env_overrides`).
    """

    study_name: str = "civ6az-sweep"
    direction: str = "maximize"
    metric: str = "best_avg_length"
    n_trials: int = 20
    trial_timeout_seconds: int = DEFAULT_TRIAL_TIMEOUT_SECONDS
    sampler: str = "tpe"
    seed: int | None = None
    #: Optuna storage URL (e.g. ``sqlite:///sweeps.db``); None = in-memory.
    storage: str | None = None
    #: Declarative search space; see :func:`validate_search_space`.
    search_space: dict[str, Any] = field(default_factory=dict)
    #: Command template for subprocess objectives. ``{python}`` expands to
    #: the current interpreter; ``{param}`` placeholders expand per trial.
    command: list[str] = field(default_factory=list)
    #: Enable MLflow logging of trials (uses the monitoring package).
    tracking_enabled: bool = True

    def __post_init__(self) -> None:
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}: {self.direction!r}")
        if self.sampler not in _VALID_SAMPLERS:
            raise ValueError(f"sampler must be one of {_VALID_SAMPLERS}: {self.sampler!r}")
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.search_space:
            validate_search_space(self.search_space)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SweepConfig:
        """Create from a plain mapping (YAML-shaped)."""
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown SweepConfig keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> SweepConfig:
        """Load from a YAML file and apply environment overrides."""
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        logger.debug("Loaded sweep config from %s: %s", path, sorted(data))
        return cls.from_dict(data).apply_env_overrides()

    def apply_env_overrides(self) -> SweepConfig:
        """Apply ``CIV6AZ_SWEEP_*`` overrides (returns self for chaining).

        Precedence for the per-trial timeout:
        ``CIV6AZ_SWEEP_TRIAL_TIMEOUT_SECONDS`` > legacy
        ``SWEEP_TIMEOUT_SECONDS`` > configured value.
        """
        env_int = _env_int
        self.n_trials = env_int(f"{ENV_PREFIX}N_TRIALS", self.n_trials)
        canonical_timeout = os.environ.get(f"{ENV_PREFIX}TRIAL_TIMEOUT_SECONDS")
        if canonical_timeout is not None:
            self.trial_timeout_seconds = env_int(
                f"{ENV_PREFIX}TRIAL_TIMEOUT_SECONDS", self.trial_timeout_seconds
            )
        elif os.environ.get(LEGACY_TIMEOUT_ENV) is not None:
            self.trial_timeout_seconds = env_int(LEGACY_TIMEOUT_ENV, self.trial_timeout_seconds)
        self.study_name = os.environ.get(f"{ENV_PREFIX}STUDY_NAME", self.study_name)
        self.storage = os.environ.get(f"{ENV_PREFIX}STORAGE", self.storage)
        self.metric = os.environ.get(f"{ENV_PREFIX}METRIC", self.metric)
        return self


def _env_int(name: str, default: int) -> int:
    """Read an int env var, warning (not raising) on garbage values."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s value %r; using %d", name, raw, default)
        return default


@dataclass
class SweepResult:
    """Outcome of a completed search."""

    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    n_pruned: int
    study_name: str


class SubprocessTrainingObjective:
    """Objective that runs a training command and reads a metric from stdout.

    The command is a template list; ``{python}`` expands to
    ``sys.executable`` and any ``{name}`` placeholder expands to the trial's
    suggested parameter value. The training process must print a line
    containing ``SWEEP_METRICS_JSON={...}``; *metric* selects the key to
    optimize.

    Failed or timed-out trials raise ``optuna.TrialPruned`` so a bad
    configuration never aborts the whole study.
    """

    def __init__(
        self,
        command: list[str],
        metric: str,
        timeout_seconds: int = DEFAULT_TRIAL_TIMEOUT_SECONDS,
        marker: str = METRICS_MARKER,
    ) -> None:
        if not command:
            raise ValueError("command template must not be empty")
        self.command = list(command)
        self.metric = metric
        self.timeout_seconds = timeout_seconds
        self.marker = marker
        self._pattern = re.compile(re.escape(marker) + r"(\{.*\})")

    def build_command(self, params: Mapping[str, Any]) -> list[str]:
        """Expand the command template with *params* (and ``{python}``)."""
        values = {"python": sys.executable, **params}
        try:
            return [part.format(**values) for part in self.command]
        except KeyError as exc:
            raise ValueError(
                f"Command template references unknown placeholder {exc}; "
                f"available: {sorted(values)}"
            ) from exc

    def extract_metric(self, stdout: str) -> float:
        """Pull the optimized metric from captured stdout.

        Uses the LAST marker line so incremental progress prints don't
        shadow the final result.
        """
        matches = self._pattern.findall(stdout)
        if not matches:
            raise ValueError(f"no '{self.marker}' line found in training output")
        payload = json.loads(matches[-1])
        if self.metric not in payload:
            raise ValueError(f"metric {self.metric!r} not in reported metrics {sorted(payload)}")
        return float(payload[self.metric])

    def __call__(self, params: Mapping[str, Any]) -> float:
        import optuna

        cmd = self.build_command(params)
        logger.info("Trial command: %s (timeout %ds)", " ".join(cmd), self.timeout_seconds)
        try:
            # S603: the command is built from the operator's own sweep
            # config (configs/sweep_*.yaml), not untrusted input.
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "Trial exceeded timeout of %ds (raise %s%s to allow longer runs); pruning",
                self.timeout_seconds,
                ENV_PREFIX,
                "TRIAL_TIMEOUT_SECONDS",
            )
            raise optuna.TrialPruned("trial timed out") from None
        if proc.returncode != 0:
            logger.warning(
                "Trial exited with code %d; pruning. stderr tail: %s",
                proc.returncode,
                proc.stderr[-500:] if proc.stderr else "<empty>",
            )
            raise optuna.TrialPruned(f"training exited with code {proc.returncode}")
        try:
            value = self.extract_metric(proc.stdout)
        except ValueError as exc:
            logger.warning("Trial produced no readable metric (%s); pruning", exc)
            raise optuna.TrialPruned(str(exc)) from exc
        logger.debug("Trial metric %s=%f", self.metric, value)
        return value


class HyperparameterSearch:
    """Orchestrates an Optuna study with optional MLflow trial logging.

    Args:
        config: Search configuration.
        objective: Callable mapping a suggested-parameter dict to the metric
            value. Defaults to a :class:`SubprocessTrainingObjective` built
            from ``config.command``.
        tracker: Duck-typed experiment tracker exposing ``log(dict, step=)``
            and ``set_tag(key, value)`` (e.g. ``WandbCompatAdapter``).
            Defaults to ``WandbCompatAdapter.from_env()`` when
            ``config.tracking_enabled``; pass a mock in tests.
    """

    def __init__(
        self,
        config: SweepConfig,
        objective: Callable[[dict[str, Any]], float] | None = None,
        tracker: Any | None = None,
    ) -> None:
        self.config = config
        self._objective = objective
        self._tracker = tracker

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_optuna(self) -> Any:
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover - exercised via message test
            raise ImportError(
                "optuna is required for hyperparameter search; install it with "
                "pip install -e '.[tuning]'"
            ) from exc
        return optuna

    def _resolve_objective(self) -> Callable[[dict[str, Any]], float]:
        if self._objective is not None:
            return self._objective
        return SubprocessTrainingObjective(
            command=self.config.command,
            metric=self.config.metric,
            timeout_seconds=self.config.trial_timeout_seconds,
        )

    def _resolve_tracker(self) -> Any | None:
        if self._tracker is not None:
            return self._tracker
        if not self.config.tracking_enabled:
            return None
        try:
            from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter

            tracker = WandbCompatAdapter.from_env()
            if tracker.is_enabled:
                tracker.init(
                    config_dict={
                        "sweep/study_name": self.config.study_name,
                        "sweep/direction": self.config.direction,
                        "sweep/metric": self.config.metric,
                        "sweep/n_trials": self.config.n_trials,
                        "sweep/sampler": self.config.sampler,
                    }
                )
                return tracker
        except Exception as exc:  # tracking must never sink the sweep
            logger.warning("Experiment tracking unavailable for sweep: %s", exc)
        return None

    def _build_sampler(self, optuna: Any) -> Any:
        if self.config.sampler == "random":
            return optuna.samplers.RandomSampler(seed=self.config.seed)
        return optuna.samplers.TPESampler(seed=self.config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SweepResult:
        """Execute the search and return the best trial found."""
        optuna = self._load_optuna()
        validate_search_space(self.config.search_space)
        objective = self._resolve_objective()
        tracker = self._resolve_tracker()

        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=self._build_sampler(optuna),
            storage=self.config.storage,
            load_if_exists=self.config.storage is not None,
        )
        logger.info(
            "Starting sweep %s: %d trials, %s %s, sampler=%s",
            self.config.study_name,
            self.config.n_trials,
            self.config.direction,
            self.config.metric,
            self.config.sampler,
        )

        def _optuna_objective(trial: Any) -> float:
            params = suggest_params(trial, self.config.search_space)
            value = objective(params)
            if tracker is not None:
                tracker.log(
                    {f"sweep/{self.config.metric}": value, "sweep/trial": trial.number},
                    step=trial.number,
                )
            return value

        study.optimize(_optuna_objective, n_trials=self.config.n_trials)

        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if completed == 0:
            raise RuntimeError(
                f"Sweep {self.config.study_name!r} produced no completed trials "
                f"({pruned} pruned of {len(study.trials)}); every training run "
                "failed, timed out, or emitted no readable metric. Check the "
                "trial command, the SWEEP_METRICS_JSON output, and the "
                f"per-trial timeout ({self.config.trial_timeout_seconds}s)."
            )
        result = SweepResult(
            best_params=dict(study.best_params),
            best_value=float(study.best_value),
            n_trials=len(study.trials),
            n_pruned=pruned,
            study_name=self.config.study_name,
        )
        logger.info(
            "Sweep complete: best %s=%f with %s (%d trials, %d pruned)",
            self.config.metric,
            result.best_value,
            result.best_params,
            result.n_trials,
            result.n_pruned,
        )
        if tracker is not None:
            tracker.log({f"sweep/best_{self.config.metric}": result.best_value})
            tracker.set_tag("sweep/best_params", json.dumps(result.best_params))
            tracker.finish()
        return result
