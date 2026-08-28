"""Cloud-native model evaluator for Vertex AI.

.. warning::

   **LEGACY -- not the deployed evaluation path.** ``cloudbuild.yaml`` builds
   only ``dockerfiles/Dockerfile.vertex``, whose entrypoint routes through
   ``alpha_zero.cloud.gcp.entrypoint``. This module is reached only by
   ``dockerfiles/Dockerfile.evaluator``, which nothing but the ``Makefile``
   and ``docker-compose.yml`` (local development) builds. The deployed
   promotion decision lives in ``alpha_zero/cloud/gcp/vertex_orchestrator.py``
   (``resolve_promotion``). Changes here do not affect production; they keep
   this path from diverging if it is ever revived.

Compares candidate models against the current champion using
head-to-head matches.

Usage:
    python -m alpha_zero.cloud.evaluator \\
        --num-games=400 \\
        --iteration=5 \\
        --promotion-threshold=0.55

Environment Variables:
    CIV6_GCS_BUCKET: GCS bucket for storage
    CIV6_GCS_PREFIX: Path prefix within bucket
    CIV6_ITERATION: Training iteration
    CIV6_EVAL_GAMES: Number of evaluation games
    CIV6_PROMOTION_THRESHOLD: Win rate required for promotion
    CIV6_DEVICE: PyTorch device (cuda/cpu)
    CIV6AZ_PROMOTION_POLICY: Promotion policy name (default ``legacy_win_rate``,
        which reproduces the historical ``wins / total`` gate exactly). See
        ``alpha_zero/evaluation/promotion.py`` for the registry.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from alpha_zero.evaluation.promotion import (
    PromotionConfig,
    PromotionVerdict,
    evaluate_promotion,
    log_promotion_decision,
)
from alpha_zero.storage import StorageConfig, create_storage_backend
from alpha_zero.storage.models import EvaluationResult, ModelMetadata
from alpha_zero.utils.compat import UTC

if TYPE_CHECKING:
    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


def count_consecutive_non_promotions(evaluations: Sequence[EvaluationResult]) -> int:
    """Count trailing consecutive evaluations that did not promote.

    Evaluations are ordered by ``iteration``; the count is taken from the most
    recent backwards and resets at the first promotion encountered. This is the
    signal a training run was missing in the v1.2.0 "0 promotions" regression
    (see CLAUDE.md S13): the loop kept iterating while the champion never moved.

    Args:
        evaluations: Evaluation results (any order); only ``iteration`` and
            ``promoted`` are read.

    Returns:
        Number of consecutive most-recent evaluations with ``promoted is False``.
    """
    ordered = sorted(evaluations, key=lambda e: e.iteration)
    streak = 0
    for result in reversed(ordered):
        if result.promoted:
            break
        streak += 1
    return streak


@dataclass
class EvaluatorConfig:
    """Configuration for model evaluator.

    Attributes:
        num_games: Number of evaluation games.
        iteration: Current training iteration.
        promotion_threshold: Win rate required for promotion.
        num_simulations: MCTS simulations per move.
        c_puct: MCTS exploration constant.
        max_moves: Maximum moves per game.
        board_size: Game board size.
        device: PyTorch device.
        early_stop_threshold: Games to check for early stopping.
        early_stop_margin: Win rate margin for early stopping.
        stagnation_patience: Consecutive non-promotions that trigger a
            stagnation warning. ``0`` (default) disables the check.
    """

    num_games: int = 400
    iteration: int = 0
    promotion_threshold: float = 0.55
    num_simulations: int = 200  # Lower for faster evaluation
    c_puct: float = 1.5
    max_moves: int = 200
    board_size: int = 8
    device: str = "cuda"
    early_stop_threshold: int = 200
    early_stop_margin: float = 0.15
    stagnation_patience: int = 0

    @classmethod
    def from_env(cls) -> EvaluatorConfig:
        """Create configuration from environment variables."""
        return cls(
            num_games=int(os.environ.get("CIV6_EVAL_GAMES", "400")),
            iteration=int(os.environ.get("CIV6_ITERATION", "0")),
            promotion_threshold=float(os.environ.get("CIV6_PROMOTION_THRESHOLD", "0.55")),
            num_simulations=int(os.environ.get("CIV6_NUM_SIMULATIONS", "200")),
            c_puct=float(os.environ.get("CIV6_MCTS_C_PUCT", "1.5")),
            max_moves=int(os.environ.get("CIV6_MAX_MOVES", "200")),
            board_size=int(os.environ.get("CIV6_BOARD_SIZE", "8")),
            device=os.environ.get("CIV6_DEVICE", "cuda"),
            stagnation_patience=int(os.environ.get("CIV6_STAGNATION_PATIENCE", "0")),
        )


class CloudEvaluator:
    """Cloud-native evaluator for model comparison.

    Plays matches between the candidate model (from current iteration)
    and the current champion model to determine if the candidate
    should be promoted.

    Attributes:
        config: Evaluator configuration.
        storage: Storage backend.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        storage: StorageBackend,
    ) -> None:
        """Initialize evaluator.

        Args:
            config: Evaluator configuration.
            storage: Storage backend.
        """
        self.config = config
        self.storage = storage

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # One promotion config for both the early-stop check and the final
        # gate, so they can never measure different statistics. The threshold
        # comes from this evaluator's own config (preserving --promotion_threshold
        # and CIV6_PROMOTION_THRESHOLD precedence); policy/confidence/min_games
        # resolve from CIV6AZ_PROMOTION_* via PromotionConfig's field defaults.
        self._promotion_config = PromotionConfig(threshold=config.promotion_threshold)

        self._candidate_model: MicroCivNetwork | None = None
        self._champion_model: MicroCivNetwork | None = None
        self._candidate_version: int = 0
        self._champion_version: int = 0

        logger.info(
            f"CloudEvaluator initialized (iteration={config.iteration}, games={config.num_games})"
        )

    def _load_models(self) -> bool:
        """Load candidate and champion models.

        Returns:
            True if both models were loaded successfully.
        """
        from alpha_zero.core.civ6_network import MicroCivNetwork

        # Load candidate model from current iteration
        try:
            candidate_state, candidate_meta = self.storage.load_model(
                iteration=self.config.iteration,
                map_location=self.device,
            )
            self._candidate_version = candidate_meta.version

            self._candidate_model = MicroCivNetwork(
                board_size=self.config.board_size,
                action_space_size=356,
            ).to(self.device)
            self._candidate_model.load_state_dict(candidate_state)
            self._candidate_model.eval()

            logger.info(f"Loaded candidate model v{self._candidate_version}")

        except FileNotFoundError:
            logger.error(f"Candidate model not found for iteration {self.config.iteration}")
            return False

        # Load champion model
        try:
            champion_state, champion_meta = self.storage.load_champion(
                map_location=self.device,
            )
            self._champion_version = champion_meta.version

            self._champion_model = MicroCivNetwork(
                board_size=self.config.board_size,
                action_space_size=356,
            ).to(self.device)
            self._champion_model.load_state_dict(champion_state)
            self._champion_model.eval()

            logger.info(f"Loaded champion model v{self._champion_version}")

        except FileNotFoundError:
            # No champion yet - candidate automatically wins
            logger.info("No champion model found, candidate will be promoted")
            self._champion_model = None
            self._champion_version = 0

        return True

    def _create_mcts(self) -> Any:
        """Create MCTS instance for evaluation."""
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        return SimpleMCTS(  # type: ignore[call-arg]
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=0.0,  # No exploration noise during evaluation
            dirichlet_epsilon=0.0,
        )

    def _create_environment(self) -> Any:
        """Create game environment."""
        from alpha_zero.envs.microciv import MicroCivEnv

        return MicroCivEnv(
            board_size=self.config.board_size,
            max_turns=self.config.max_moves,
        )

    def _play_match(
        self,
        candidate_plays_first: bool,
    ) -> dict[str, Any]:
        """Play a single evaluation match.

        Args:
            candidate_plays_first: Whether candidate plays as player 0.

        Returns:
            Match result dictionary.
        """
        env = self._create_environment()
        mcts = self._create_mcts()

        obs, _ = env.reset()
        move_count = 0
        done = False

        # Assign models to players
        if candidate_plays_first:
            players = [self._candidate_model, self._champion_model]
        else:
            players = [self._champion_model, self._candidate_model]

        while not done and move_count < self.config.max_moves:
            current_player = move_count % 2
            model = players[current_player]

            # If no champion (first evaluation), candidate plays both sides
            if model is None:
                model = self._candidate_model

            # Get action from MCTS
            with torch.no_grad():
                policy = mcts.search(env, model, obs)

            # Select action (deterministic during evaluation)
            action_mask = obs.get("action_mask", np.ones(len(policy)))
            policy_masked = policy * action_mask
            if policy_masked.sum() > 0:
                action = policy_masked.argmax()
            else:
                valid_actions = np.where(action_mask > 0)[0]
                action = valid_actions[0] if len(valid_actions) > 0 else 0

            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            move_count += 1

        # Determine winner
        winner = info.get("winner")

        # Convert to candidate perspective
        if winner is None:
            result = "draw"
            candidate_won = None
        elif candidate_plays_first:
            candidate_won = winner == 0
            result = "candidate_win" if candidate_won else "champion_win"
        else:
            candidate_won = winner == 1
            result = "candidate_win" if candidate_won else "champion_win"

        return {
            "result": result,
            "candidate_won": candidate_won,
            "moves": move_count,
            "candidate_played_first": candidate_plays_first,
        }

    def _promotion_verdict(
        self,
        candidate_wins: int,
        champion_wins: int,
        draws: int,
    ) -> PromotionVerdict:
        """Score a head-to-head record under the configured promotion policy.

        The single place this class decides anything about promotion, so the
        early-stop heuristic and the final gate cannot drift apart.

        Note the argument order: this class counts
        ``(candidate_wins, champion_wins, draws)`` while the policy signature is
        ``(wins, draws, losses)``. The mapping below is by keyword precisely
        because a positional call would transpose draws and losses silently.

        Args:
            candidate_wins: Games the candidate won.
            champion_wins: Games the champion won (losses, from the candidate's view).
            draws: Drawn games.

        Returns:
            The policy's verdict, carrying the decision, statistic and summary.
        """
        return evaluate_promotion(
            wins=candidate_wins,
            draws=draws,
            losses=champion_wins,
            config=self._promotion_config,
        )

    def _promotion_statistic(self, candidate_wins: int, champion_wins: int, draws: int) -> float:
        """The configured policy's statistic, computed without side effects.

        ``_promotion_verdict`` is the decision path and logs when a record makes
        promotion arithmetically impossible. Early stopping polls after every
        game, so calling it there would repeat that warning hundreds of times
        per evaluation; the statistic itself is what the heuristic needs.
        """
        logger_name = "alpha_zero.evaluation.promotion"
        promotion_logger = logging.getLogger(logger_name)
        previous_level = promotion_logger.level
        promotion_logger.setLevel(logging.ERROR)
        try:
            return self._promotion_verdict(candidate_wins, champion_wins, draws).statistic
        finally:
            promotion_logger.setLevel(previous_level)

    def _should_early_stop(
        self,
        candidate_wins: int,
        champion_wins: int,
        draws: int,
    ) -> tuple[bool, str]:
        """Check if evaluation can be stopped early.

        Args:
            candidate_wins: Number of candidate wins.
            champion_wins: Number of champion wins.
            draws: Number of draws.

        Returns:
            Tuple of (should_stop, reason).
        """
        total = candidate_wins + champion_wins + draws
        if total < self.config.early_stop_threshold:
            return False, ""

        # Use the configured policy's own statistic rather than a fixed
        # draws-as-losses win rate, so a draw-aware policy is not stopped early
        # by a heuristic measuring something else. Under the default
        # legacy_win_rate policy this is candidate_wins / total -- byte-identical
        # to the historical behaviour.
        #
        # This is a heuristic, deliberately not the gate: an interval policy
        # such as score_rate_lower_bound decides on confidence bounds, and
        # stopping early shrinks n, which can only widen them. Treat an early
        # stop as "no longer worth more games", not as the verdict.
        statistic = self._promotion_statistic(candidate_wins, champion_wins, draws)

        # If clearly winning
        if statistic > self.config.promotion_threshold + self.config.early_stop_margin:
            return True, "decisive_lead"

        # If clearly losing
        if statistic < self.config.promotion_threshold - self.config.early_stop_margin:
            return True, "decisive_loss"

        return False, ""

    def run(self) -> dict[str, Any]:
        """Run evaluation.

        Returns:
            Dictionary with evaluation results.
        """
        start_time = datetime.now(UTC)
        logger.info(
            f"Starting evaluation: {self.config.num_games} games, iteration {self.config.iteration}"
        )

        # Load models
        if not self._load_models():
            return {
                "status": "error",
                "reason": "model_load_failed",
            }

        # If no champion, auto-promote
        if self._champion_model is None:
            logger.info("No champion - promoting candidate automatically")
            self._promote_candidate()
            return {
                "status": "promoted",
                "reason": "no_champion",
                "candidate_version": self._candidate_version,
            }

        # Play evaluation matches
        candidate_wins = 0
        champion_wins = 0
        draws = 0
        game_results = []

        for game_num in range(self.config.num_games):
            # Alternate starting player
            candidate_plays_first = game_num % 2 == 0

            result = self._play_match(candidate_plays_first)
            game_results.append(result)

            if result["candidate_won"] is True:
                candidate_wins += 1
            elif result["candidate_won"] is False:
                champion_wins += 1
            else:
                draws += 1

            # Log progress
            if (game_num + 1) % 50 == 0:
                total = candidate_wins + champion_wins + draws
                win_rate = candidate_wins / total
                logger.info(
                    f"Progress: {game_num + 1}/{self.config.num_games} games, "
                    f"candidate win rate: {win_rate:.2%}"
                )

            # Check for early stopping
            should_stop, reason = self._should_early_stop(candidate_wins, champion_wins, draws)
            if should_stop:
                logger.info(f"Early stopping: {reason}")
                break

        # Calculate final statistics
        total_games = candidate_wins + champion_wins + draws
        win_rate = candidate_wins / total_games if total_games > 0 else 0.0

        # Route the decision through the promotion-policy registry. Keyword
        # arguments are deliberate: this function's locals are ordered
        # (candidate_wins, champion_wins, draws) while the policy signature is
        # (wins, draws, losses), so positional calls transpose silently.
        verdict = self._promotion_verdict(candidate_wins, champion_wins, draws)
        promoted = verdict.promoted

        # Create evaluation result. score_rate/ci_low/ci_high are shadow
        # statistics recorded alongside the legacy win_rate: they describe the
        # same games under draw-aware scoring, so a policy change can be
        # evaluated against history before it is adopted.
        #
        # These are the summary's *diagnostic* interval, which summarize()
        # always computes -- matching EvaluationResult's documented meaning
        # ("the score-rate confidence interval, if computed"). That is
        # deliberately NOT PromotionVerdict.ci_low, which carries a different
        # contract: the interval the policy actually consulted, None for a
        # point-estimate policy such as the default. Same field name, two
        # meanings, so the distinction is worth stating here.
        eval_result = EvaluationResult(
            iteration=self.config.iteration,
            candidate_version=self._candidate_version,
            champion_version=self._champion_version,
            num_games=total_games,
            candidate_wins=candidate_wins,
            champion_wins=champion_wins,
            draws=draws,
            win_rate=win_rate,
            promoted=promoted,
            game_results=game_results,
            score_rate=verdict.summary.score_rate,
            ci_low=verdict.summary.ci_low,
            ci_high=verdict.summary.ci_high,
        )

        # Save evaluation result
        self.storage.save_evaluation(eval_result)

        # Structured record of which policy decided, on what statistic.
        log_promotion_decision(
            verdict,
            target=logger,
            context={
                "iteration": self.config.iteration,
                "candidate_version": self._candidate_version,
                "champion_version": self._champion_version,
            },
        )

        # Promote if successful
        if promoted:
            self._promote_candidate()
            logger.info(
                f"Candidate promoted! Win rate: {win_rate:.2%} "
                f"(>= {self.config.promotion_threshold:.2%})"
            )
        else:
            logger.info(
                f"Candidate not promoted. Win rate: {win_rate:.2%} "
                f"(< {self.config.promotion_threshold:.2%})"
            )

        # Promotion-stagnation check (reads persisted history, including the
        # result just saved). Surfaces the "many iterations, 0 promotions" stall.
        stagnation = self._check_promotion_stagnation()

        elapsed = (datetime.now(UTC) - start_time).total_seconds()

        return {
            "status": "completed",
            "iteration": self.config.iteration,
            "candidate_version": self._candidate_version,
            "champion_version": self._champion_version,
            "games_played": total_games,
            "candidate_wins": candidate_wins,
            "champion_wins": champion_wins,
            "draws": draws,
            "win_rate": win_rate,
            "promoted": promoted,
            # The full decision, not just the boolean: score_rate_lower_bound
            # can return "inconclusive", which collapses to promoted=False and
            # would otherwise be indistinguishable from a genuine rejection.
            "decision": verdict.decision.value,
            "promotion_policy": verdict.policy,
            "score_rate": verdict.summary.score_rate,
            "ci_low": verdict.summary.ci_low,
            "ci_high": verdict.summary.ci_high,
            "stagnation_detected": stagnation["stagnation_detected"],
            "consecutive_non_promotions": stagnation["consecutive_non_promotions"],
            "elapsed_seconds": elapsed,
            "games_per_hour": total_games / (elapsed / 3600) if elapsed > 0 else 0,
        }

    def _check_promotion_stagnation(self) -> dict[str, Any]:
        """Detect a run of consecutive non-promotions across iterations.

        Reads ``storage.list_evaluations()`` and counts the trailing streak of
        non-promotions. When the streak reaches ``config.stagnation_patience``
        (and patience > 0) a WARNING is logged so an operator/monitor can
        intervene rather than burning iterations on a stalled champion.

        Returns:
            Mapping with ``stagnation_detected`` (bool) and
            ``consecutive_non_promotions`` (int).
        """
        patience = self.config.stagnation_patience
        if patience <= 0:
            return {"stagnation_detected": False, "consecutive_non_promotions": 0}

        try:
            evaluations = self.storage.list_evaluations()
        except Exception as exc:  # storage hiccup must not fail evaluation
            logger.warning("Stagnation check skipped (could not list evaluations): %s", exc)
            return {"stagnation_detected": False, "consecutive_non_promotions": 0}

        streak = count_consecutive_non_promotions(evaluations)
        detected = streak >= patience
        if detected:
            logger.warning(
                "Promotion stagnation detected: %d consecutive evaluations without a "
                "promotion (patience=%d). Champion has not improved; training may have "
                "stalled (see CLAUDE.md Section 13).",
                streak,
                patience,
            )
        return {"stagnation_detected": detected, "consecutive_non_promotions": streak}

    def _promote_candidate(self) -> None:
        """Promote candidate model to champion."""
        logger.info(f"Promoting candidate v{self._candidate_version} to champion")

        # Load candidate state dict
        candidate_state, _ = self.storage.load_model(
            iteration=self.config.iteration,
            map_location="cpu",
        )

        # Save as champion
        metadata = ModelMetadata(
            iteration=self.config.iteration,
            version=self._candidate_version,
        )
        self.storage.save_champion(candidate_state, metadata)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Civ6 AlphaZero Model Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=int(os.environ.get("CIV6_EVAL_GAMES", "400")),
        help="Number of evaluation games",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=int(os.environ.get("CIV6_ITERATION", "0")),
        help="Current training iteration",
    )
    parser.add_argument(
        "--promotion-threshold",
        type=float,
        default=float(os.environ.get("CIV6_PROMOTION_THRESHOLD", "0.55")),
        help="Win rate required for promotion",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=int(os.environ.get("CIV6_NUM_SIMULATIONS", "200")),
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=int(os.environ.get("CIV6_BOARD_SIZE", "8")),
        help="Game board size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("CIV6_DEVICE", "cuda"),
        help="PyTorch device (cuda/cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for evaluator."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    # Create configuration
    config = EvaluatorConfig(
        num_games=args.num_games,
        iteration=args.iteration,
        promotion_threshold=args.promotion_threshold,
        num_simulations=args.num_simulations,
        board_size=args.board_size,
        device=args.device,
    )

    if args.dry_run:
        logger.info("Dry run - configuration validated")
        logger.info(f"Config: {config}")
        return

    # Create storage backend
    storage_config = StorageConfig.from_env()
    storage = create_storage_backend(storage_config)

    # Create and run evaluator
    evaluator = CloudEvaluator(
        config=config,
        storage=storage,
    )

    try:
        result = evaluator.run()
        logger.info(f"Evaluator completed with result: {result}")
    except Exception as e:
        logger.error(f"Evaluator failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        storage.close()


if __name__ == "__main__":
    main()
