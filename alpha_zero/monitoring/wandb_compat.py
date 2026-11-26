"""Weights & Biases compatibility adapter backed by :class:`MLflowLogger`."""

from __future__ import annotations

from typing import Any
from typing import cast as _cast

from alpha_zero.monitoring.config import MLflowConfig
from alpha_zero.monitoring.mlflow_logger import MLflowLogger, logger


class WandbCompatAdapter(MLflowLogger):
    """Drop-in replacement for the retired W&B ``WandbLogger``.

    Re-exposes every public method of the legacy ``WandbLogger`` so worker
    code can switch the import line and gain MLflow tracking with no other
    changes. Method signatures mirror ``WandbLogger`` exactly to avoid
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

    @classmethod
    def from_env(cls) -> WandbCompatAdapter:
        """Construct from ``MLFLOW_*`` / ``CIV6AZ_MLFLOW_*`` environment variables."""
        return _cast(WandbCompatAdapter, cls.from_config(MLflowConfig.from_env()))

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
        the adapter layer. Worker code that wants the dict form can call
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
        if (
            len(args) == 3
            and isinstance(args[0], str)
            and isinstance(args[1], list)
            and isinstance(args[2], list)
        ):
            name, columns, rows = args
            if not rows or not columns:
                return
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
        super().log_table(*args, **kwargs)

    def watch_model(self, *args: Any, **kwargs: Any) -> None:
        """Accepted no-op: MLflow has no direct equivalent of ``wandb.watch``."""
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


__all__ = ["WandbCompatAdapter"]
