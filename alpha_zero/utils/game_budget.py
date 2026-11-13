"""Worst-case game-count cost models and the 1,000-game cap (LF-G11).

Policy
------
Every run is at most ``DEFAULT_MAX_GAMES`` (1,000) games unless a maintainer
approved a larger one in an issue. The cap is only meaningful if something
*computes* the number, so each runner prices itself: after it has resolved its
CLI arguments, preset, YAML and environment overrides, it hands the resolved
values to :func:`enforce`, which refuses (raises :class:`GameBudgetExceeded`)
when the worst case is over the cap and no ``--approval_ref`` was given.

Design notes
------------
* The cost models are **pure functions of resolved values**. They never import
  torch, read files or consult the environment, so they are cheap to call at
  the top of ``main()`` and trivially testable.
* A model is a worst case, not a prediction: it assumes every scheduled
  evaluation runs and no run stops early. Where a runner can resume part-way
  the model prices from the resolved start point.
* A runner whose cost cannot be derived (for example a sweep whose trial
  command is not a recognised runner) is **refused**, not waved through: an
  unpriceable run is exactly the run the cap exists to catch. It is
  :class:`GameBudgetUnpriceable`, a subclass of :class:`GameBudgetExceeded`, so
  one ``except`` clause handles both and ``--approval_ref`` still overrides it.
* Evaluation terms are keyed (``eval_games``, ...) so a later change that adds
  or retunes an evaluation schedule extends one model rather than every caller.

Public surface: :func:`worst_case_games`, :func:`enforce`,
:func:`add_budget_arguments`, :func:`enforce_cli`.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: A run above this many games needs an approval reference.
DEFAULT_MAX_GAMES = 1000

#: Evaluation games ``training_civ6`` plays per evaluation on a normal run.
#: ``training_civ6.DEFAULT_EVAL_GAMES`` re-exports this value.
DEFAULT_TRAINING_EVAL_GAMES = 20

#: Process exit status used when a runner refuses to start.
EXIT_CODE_REFUSED = 2


class GameBudgetExceededError(RuntimeError):
    """A run would play more games than the cap allows, with no approval."""

    def __init__(self, runner: str, games: int, max_games: int, detail: str = "") -> None:
        self.runner = runner
        self.games = games
        self.max_games = max_games
        message = (
            f"{runner}: worst case {games:,} games exceeds --max_games {max_games:,}. "
            "Reduce the run, or pass --approval_ref <issue> once the run is approved "
            "in that issue (CLAUDE.md section 3.2)."
        )
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


class GameBudgetUnpriceableError(GameBudgetExceededError):
    """The runner's cost could not be derived, so it is treated as over budget."""

    def __init__(self, runner: str, reason: str, max_games: int = DEFAULT_MAX_GAMES) -> None:
        RuntimeError.__init__(
            self,
            f"{runner}: cannot price this run ({reason}). Refusing rather than guessing; "
            "pass --approval_ref <issue> if the run is approved.",
        )
        self.runner = runner
        self.games = -1
        self.max_games = max_games


class GameBudgetCapRaisedError(GameBudgetExceededError):
    """``--max_games`` above the default cap was given without an approval reference."""

    def __init__(self, runner: str, max_games: int) -> None:
        RuntimeError.__init__(
            self,
            f"{runner}: --max_games {max_games:,} is above the {DEFAULT_MAX_GAMES:,}-game "
            "cap. --max_games may only lower the cap; raising it needs --approval_ref <issue>.",
        )
        self.runner = runner
        self.games = -1
        self.max_games = max_games


#: The name callers and docs use (LF-G11 interface). ``GameBudgetExceededError`` is
#: the class itself, spelled to satisfy the ``N818`` naming rule without a waiver.
GameBudgetExceeded = GameBudgetExceededError
GameBudgetUnpriceable = GameBudgetUnpriceableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nonneg(value: Any) -> int:
    """Coerce to a non-negative int; the models price what would actually run."""
    return max(int(value), 0)


def _count_multiples(low: int, high: int, step: int) -> int:
    """Count multiples of *step* in the half-open range ``[low, high)``, ``low >= 0``."""
    if high <= low:
        return 0
    return -(-high // step) - -(-low // step)


def _scheduled_hits(start: int, end: int, every: int) -> int:
    """Iterations ``i`` in ``[start, end)`` with ``i % every == 0``."""
    return _count_multiples(start, end, max(every, 1))


# ---------------------------------------------------------------------------
# Cost models. Each takes the runner's *resolved* values.
# ---------------------------------------------------------------------------


def _training_civ6(r: Mapping[str, Any]) -> int:
    """``G*(I-S) + eval_games * evaluations`` for ``alpha_zero.training_civ6``.

    The trainer evaluates when ``(iteration + 1) % eval_interval == 0`` for
    iteration in ``[S, I)`` -- or on every iteration under ``--dry_run``
    (``eval_always``) -- and each evaluation plays ``eval_games`` games, so
    ``eval_games == 0`` contributes nothing. ``eval_interval`` defaults to
    ``save_interval``, the trainer's own default (LF-A14), so inputs written
    before the two were decoupled price exactly as they did.
    """
    iterations = _nonneg(r["num_iterations"])
    start = min(_nonneg(r.get("start_iteration", 0)), iterations)
    games = _nonneg(r["num_games"])
    eval_games = _nonneg(r.get("eval_games", DEFAULT_TRAINING_EVAL_GAMES))
    remaining = iterations - start
    if eval_games == 0:
        evaluations = 0
    elif r.get("eval_always"):
        evaluations = remaining
    else:
        interval = r.get("eval_interval")
        if interval is None:
            interval = r.get("save_interval", 1)
        every = max(_nonneg(interval), 1)
        # (i + 1) % every == 0  <=>  i + 1 is a multiple of `every`, i + 1 in [S+1, I].
        evaluations = iterations // every - start // every
    return games * remaining + eval_games * evaluations


def _orchestrator(r: Mapping[str, Any]) -> int:
    """Self-play plus the orchestrator's two evaluation cadences.

    ``workers*games*iterations*(1+retries)`` of self-play. ``_evaluate`` plays
    ``evaluation_games`` games on iterations that are multiples of
    ``evaluation_interval``; on those that are also multiples of
    ``full_eval_interval`` it additionally plays that many games against each
    of two baselines (worst case: the evaluation pipeline is present).
    ``--dry_run`` plays one game.
    """
    if r.get("dry_run"):
        return 1
    iterations = _nonneg(r["num_iterations"])
    selfplay = (
        max(_nonneg(r.get("workers", 1)), 1)
        * _nonneg(r["games_per_iteration"])
        * iterations
        * (1 + _nonneg(r.get("retries", 0)))
    )
    eval_games = _nonneg(r.get("evaluation_games", 0))
    eval_every = max(_nonneg(r.get("evaluation_interval", 1)), 1)
    full_every = max(_nonneg(r.get("full_eval_interval", 1)), 1)
    evals = _scheduled_hits(0, iterations, eval_every)
    full_evals = _scheduled_hits(0, iterations, math.lcm(eval_every, full_every))
    return selfplay + eval_games * (evals + 2 * full_evals)


def _benchmark(r: Mapping[str, Any]) -> int:
    """``n_games * (len(opponents) + len(opponent_checkpoints))``."""
    return _nonneg(r["n_games"]) * (
        _nonneg(r["num_opponents"]) + _nonneg(r.get("num_opponent_checkpoints", 0))
    )


def _baseline_eval(r: Mapping[str, Any]) -> int:
    """One batch of ``n_games`` against each of random and heuristic."""
    return 2 * _nonneg(r["n_games"])


def _encoder_ab_train(r: Mapping[str, Any]) -> int:
    """Each arm self-plays ``iterations * games_per_iter`` games."""
    return _nonneg(r["num_arms"]) * _nonneg(r["iterations"]) * _nonneg(r["games_per_iter"])


def _encoder_ab_arena(r: Mapping[str, Any]) -> int:
    return _nonneg(r["num_games"])


def _chess_training(r: Mapping[str, Any]) -> int:
    """Games played; ``--dry-run`` builds the env and returns without playing."""
    if r.get("dry_run"):
        return 0
    return _nonneg(r["iterations"]) * _nonneg(r["games_per_iter"])


def _enhanced_training(r: Mapping[str, Any]) -> int:
    """Self-play + in-loop evaluations (10 games) + the final 20-game evaluation."""
    iterations = _nonneg(r["num_iterations"])
    interval = max(_nonneg(r["eval_interval"]), 1)
    return (
        iterations * _nonneg(r["games_per_iteration"])
        + _nonneg(r.get("eval_games", 10)) * (iterations // interval)
        + _nonneg(r.get("final_eval_games", 20))
    )


def _muzero_training(r: Mapping[str, Any]) -> int:
    """``iterations * games``; parallel workers split the games, not multiply."""
    return _nonneg(r["num_iterations"]) * _nonneg(r["games_per_iteration"])


def _sweep(r: Mapping[str, Any]) -> int:
    """``n_trials`` times the games one trial command plays; 0 under ``--dry-run``."""
    if r.get("dry_run"):
        return 0
    per_trial = r.get("trial_games")
    if per_trial is None:
        raise GameBudgetUnpriceable(
            "run_sweep", "the trial command is not a recognised, fully-resolved runner"
        )
    return _nonneg(r["n_trials"]) * _nonneg(per_trial)


def _post_run_evaluation(r: Mapping[str, Any]) -> int:
    """Round-robin tournament plus two baseline batches.

    The tournament plays ``C(n, 2) * tournament_games`` even under ``--dry_run``
    (dummy models still play); only the baseline batch is skipped by it.
    """
    total = 0
    if not r.get("skip_tournament"):
        models = _nonneg(r["num_models"])
        total += models * (models - 1) // 2 * _nonneg(r["tournament_games"])
    if not r.get("skip_baseline_eval") and not r.get("dry_run"):
        total += 2 * _nonneg(r["baseline_games"])
    return total


def _validate_convergence(r: Mapping[str, Any]) -> int:
    """Per iteration: self-play games plus the evaluation against random."""
    return _nonneg(r["iterations"]) * (_nonneg(r["games_per_iter"]) + _nonneg(r["eval_games"]))


def _benchmark_batch_selfplay(r: Mapping[str, Any]) -> int:
    """The sequential and the batched variant each play ``games`` plus warmup."""
    return 2 * (_nonneg(r["games"]) + _nonneg(r.get("warmup_games", 0)))


def _calibrate_throughput(r: Mapping[str, Any]) -> int:
    """``games`` complete self-play games (no evaluation phase)."""
    return _nonneg(r["games"])


def _probe_target_sharpness(r: Mapping[str, Any]) -> int:
    """No games: it searches positions from random play and steps only clones.

    Priced explicitly (rather than exempt) so a later change that makes it play
    games has to change this model, and a test, to say so.
    """
    _nonneg(r["positions"])
    return 0


def _enhanced_env_smoke_test(r: Mapping[str, Any]) -> int:
    """``num_selfplay_games`` complete games; the vectorised check takes one step only."""
    return _nonneg(r["num_selfplay_games"])


def _local_muzero_launcher(r: Mapping[str, Any]) -> int:
    return (
        _nonneg(r["iterations"])
        * _nonneg(r["num_selfplay_workers"])
        * _nonneg(r["games_per_worker"])
    )


_MODELS: dict[str, Callable[[Mapping[str, Any]], int]] = {
    "training_civ6": _training_civ6,
    "orchestrator": _orchestrator,
    "benchmark": _benchmark,
    "run_baseline_eval": _baseline_eval,
    "run_encoder_ab.train": _encoder_ab_train,
    "run_encoder_ab.arena": _encoder_ab_arena,
    "run_chess_training": _chess_training,
    "run_enhanced_training": _enhanced_training,
    "run_muzero_training": _muzero_training,
    "run_sweep": _sweep,
    "post_run_evaluation": _post_run_evaluation,
    "validate_convergence": _validate_convergence,
    "benchmark_batch_selfplay": _benchmark_batch_selfplay,
    "local_muzero_launcher": _local_muzero_launcher,
    "calibrate_throughput": _calibrate_throughput,
    "probe_target_sharpness": _probe_target_sharpness,
    "enhanced_env_smoke_test": _enhanced_env_smoke_test,
}


def orchestrator_budget_inputs(config: Any, dry_run: bool) -> dict[str, Any]:
    """Resolved ``orchestrator`` cost-model inputs from an ``OrchestratorConfig``.

    Duck-typed on ``config.training`` so this module needs no orchestrator import.
    Call it after preset, YAML, environment and CLI overrides have been applied.
    """
    training = config.training
    return {
        "dry_run": dry_run,
        "num_iterations": training.num_iterations,
        "games_per_iteration": training.games_per_iteration,
        "evaluation_interval": training.evaluation_interval,
        "evaluation_games": training.evaluation_games,
        "full_eval_interval": training.full_eval_interval,
    }


def priced_runners() -> tuple[str, ...]:
    """Names of the runners that have a cost model."""
    return tuple(sorted(_MODELS))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def worst_case_games(runner: str, resolved: Mapping[str, Any]) -> int:
    """Return the worst-case number of games *runner* would play.

    Args:
        runner: A name from :func:`priced_runners`.
        resolved: The runner's values *after* CLI, preset, YAML and environment
            resolution. Keys are model-specific; see each ``_<runner>`` function.

    Raises:
        KeyError: *runner* has no cost model. Callers that must fail closed on
            unknown runners should treat this as a refusal.
    """
    try:
        model = _MODELS[runner]
    except KeyError:
        raise KeyError(
            f"no game-budget model for runner {runner!r}; known: {', '.join(priced_runners())}"
        ) from None
    return model(resolved)


def enforce(
    runner: str,
    resolved: Mapping[str, Any],
    max_games: int = DEFAULT_MAX_GAMES,
    approval_ref: str | None = None,
) -> int:
    """Refuse a run whose worst case exceeds *max_games* without an approval.

    Args:
        runner: Cost-model name.
        resolved: Resolved runner values (see :func:`worst_case_games`).
        max_games: The cap. Defaults to 1,000 and may only be lowered; a higher
            value requires *approval_ref*.
        approval_ref: Issue reference (for example ``#142``) recording the
            maintainer's approval of a larger run. Logged when it is used.

    Returns:
        The priced worst-case game count.

    Raises:
        GameBudgetExceeded: Over the cap and no *approval_ref*, or *max_games*
            above the default cap and no *approval_ref*.
    """
    if max_games < 0:
        raise ValueError(f"max_games must be >= 0, got {max_games}")
    ref = (approval_ref or "").strip()
    if max_games > DEFAULT_MAX_GAMES:
        # The flag alone never authorises a bigger run: only a recorded approval does.
        if not ref:
            raise GameBudgetCapRaisedError(runner, max_games)
        logger.warning(
            "%s: cap raised to %s (default %s) on approval %s",
            runner,
            f"{max_games:,}",
            f"{DEFAULT_MAX_GAMES:,}",
            ref,
        )
    try:
        games = worst_case_games(runner, resolved)
    except GameBudgetUnpriceable as exc:
        if ref:
            logger.warning(
                "%s: run could not be priced; proceeding on approval %s (%s)", runner, ref, exc
            )
            return -1
        raise
    if games <= max_games:
        logger.info("%s: worst case %s games (cap %s)", runner, f"{games:,}", f"{max_games:,}")
        return games
    if ref:
        logger.warning(
            "%s: worst case %s games exceeds cap %s; proceeding on approval %s",
            runner,
            f"{games:,}",
            f"{max_games:,}",
            ref,
        )
        return games
    raise GameBudgetExceeded(runner, games, max_games)


def add_budget_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--max_games`` and ``--approval_ref`` to *parser*.

    The underscore spelling matches the repository's other budget flags (and
    the PreToolUse guard that asks on ``--approval_ref``).
    """
    group = parser.add_argument_group("game budget")
    group.add_argument(
        "--max_games",
        type=int,
        default=DEFAULT_MAX_GAMES,
        help=(
            f"Refuse to start if the worst-case game count exceeds this "
            f"(default {DEFAULT_MAX_GAMES:,}). Can only lower the cap; "
            "raising it needs --approval_ref."
        ),
    )
    group.add_argument(
        "--approval_ref",
        type=str,
        default=None,
        help="Issue reference recording approval of a run above --max_games.",
    )


def enforce_cli(runner: str, resolved: Mapping[str, Any], args: argparse.Namespace) -> int:
    """:func:`enforce` for a ``main()``: print the refusal and exit non-zero.

    Reads ``args.max_games`` and ``args.approval_ref`` (as added by
    :func:`add_budget_arguments`). On refusal writes the message to stderr and
    raises ``SystemExit(EXIT_CODE_REFUSED)`` so every runner refuses the same
    way regardless of whether its ``main`` returns an int or ``None``.
    """
    try:
        return enforce(
            runner,
            resolved,
            max_games=int(getattr(args, "max_games", DEFAULT_MAX_GAMES)),
            approval_ref=getattr(args, "approval_ref", None),
        )
    except GameBudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_CODE_REFUSED) from exc
