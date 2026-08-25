"""
Sequential Testing for Model Evaluation in Civ6 AlphaZero.

Implements sequential testing using SPRT (Sequential Probability Ratio Test)
to make early stopping decisions during evaluation, saving compute when
results are decisive.

Features:
- SPRT-based sequential testing
- Configurable significance levels
- Early stopping when results are decisive
- Win rate confidence intervals
- Cost savings tracking

All values configurable via environment variables with CIV6AZ_EVAL_ prefix.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class EvaluationDecision(str, Enum):
    """Decision from sequential testing."""

    CONTINUE = "continue"  # Need more games
    ACCEPT = "accept"  # New model is better
    REJECT = "reject"  # New model is not better
    INCONCLUSIVE = "inconclusive"  # Reached max games without decision


@dataclass
class GameResult:
    """Result of an evaluation game.

    Attributes:
        game_id: Unique game identifier.
        winner: 1 if new model won, -1 if baseline won, 0 if draw.
        turns: Number of turns in the game.
        duration_seconds: Game duration.
    """

    game_id: int
    winner: int  # 1 = new model, -1 = baseline, 0 = draw
    turns: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "winner": self.winner,
            "turns": self.turns,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class SequentialTestResult:
    """Result of sequential testing.

    Attributes:
        decision: Final decision.
        games_played: Number of games played.
        wins: New model wins.
        losses: New model losses.
        draws: Number of draws.
        win_rate: Win rate of new model.
        confidence: Statistical confidence (1 - p-value).
        reason: Human-readable reason for decision.
        cost_savings: Estimated cost savings from early stopping.
    """

    decision: EvaluationDecision
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    confidence: float
    reason: str
    cost_savings: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "confidence": self.confidence,
            "reason": self.reason,
            "cost_savings": self.cost_savings,
        }


@dataclass
class SequentialTestingConfig:
    """Configuration for sequential testing in evaluation.

    Environment Variables:
        CIV6AZ_EVAL_MIN_GAMES: Minimum games before decision (default: 20)
        CIV6AZ_EVAL_MAX_GAMES: Maximum games to play (default: 400)
        CIV6AZ_EVAL_EARLY_STOP_THRESHOLD: Games for early stop if decisive (default: 200)
        CIV6AZ_EVAL_SIGNIFICANCE_LEVEL: Alpha for statistical test (default: 0.05)
        CIV6AZ_EVAL_WIN_THRESHOLD: Win rate threshold to accept (default: 0.55)
        CIV6AZ_EVAL_DECISIVE_MARGIN: Margin for decisive result (default: 0.1)
        CIV6AZ_EVAL_COST_PER_GAME: Cost per evaluation game USD (default: 0.01)
    """

    min_games: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_EVAL_MIN_GAMES", "20"))
    )
    max_games: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_EVAL_MAX_GAMES", "400"))
    )
    early_stop_threshold: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_EVAL_EARLY_STOP_THRESHOLD", "200"))
    )
    significance_level: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_EVAL_SIGNIFICANCE_LEVEL", "0.05"))
    )
    win_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_EVAL_WIN_THRESHOLD", "0.55"))
    )
    decisive_margin: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_EVAL_DECISIVE_MARGIN", "0.1"))
    )
    cost_per_game_usd: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_EVAL_COST_PER_GAME", "0.01"))
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values."""
        errors: list[str] = []

        if self.min_games < 1:
            errors.append("min_games must be at least 1")

        if self.max_games < self.min_games:
            errors.append("max_games must be >= min_games")

        if self.early_stop_threshold < self.min_games:
            errors.append("early_stop_threshold must be >= min_games")

        if self.early_stop_threshold > self.max_games:
            errors.append("early_stop_threshold must be <= max_games")

        if not 0 < self.significance_level < 1:
            errors.append("significance_level must be between 0 and 1")

        if not 0.5 <= self.win_threshold < 1:
            errors.append("win_threshold must be between 0.5 and 1")

        if not 0 < self.decisive_margin < 0.5:
            errors.append("decisive_margin must be between 0 and 0.5")

        if self.cost_per_game_usd < 0:
            errors.append("cost_per_game_usd cannot be negative")

        if errors:
            from alpha_zero.cloud.gcp.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid sequential testing configuration: {'; '.join(errors)}",
                details={"errors": errors},
            )

    @classmethod
    def from_env(cls) -> SequentialTestingConfig:
        """Create from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> SequentialTestingConfig:
        """Load from YAML file."""
        path = Path(path)
        if not path.exists():
            from alpha_zero.cloud.gcp.exceptions import ConfigurationError

            raise ConfigurationError(f"Sequential testing config not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            from alpha_zero.cloud.gcp.exceptions import ConfigurationError

            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SequentialTestingConfig:
        """Create from dictionary."""
        return cls(
            min_games=data.get("min_games", 20),
            max_games=data.get("max_games", 400),
            early_stop_threshold=data.get("early_stop_threshold", 200),
            significance_level=data.get("significance_level", 0.05),
            win_threshold=data.get("win_threshold", 0.55),
            decisive_margin=data.get("decisive_margin", 0.1),
            cost_per_game_usd=data.get("cost_per_game_usd", 0.01),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_games": self.min_games,
            "max_games": self.max_games,
            "early_stop_threshold": self.early_stop_threshold,
            "significance_level": self.significance_level,
            "win_threshold": self.win_threshold,
            "decisive_margin": self.decisive_margin,
            "cost_per_game_usd": self.cost_per_game_usd,
        }


class SequentialTester:
    """Implements sequential testing for model evaluation.

    Uses SPRT (Sequential Probability Ratio Test) concepts to make early
    stopping decisions during evaluation, saving compute when results are
    decisive.

    Usage:
        config = SequentialTestingConfig()
        tester = SequentialTester(config)

        for game_id in range(config.max_games):
            winner = play_game(new_model, baseline)
            tester.record_game(game_id, winner)

            decision = tester.get_decision()
            if decision != EvaluationDecision.CONTINUE:
                break

        result = tester.get_result()
    """

    def __init__(self, config: SequentialTestingConfig | None = None) -> None:
        """Initialize tester.

        Args:
            config: Configuration (defaults to from env).
        """
        self.config = config or SequentialTestingConfig.from_env()
        self._games: list[GameResult] = []
        self._wins = 0
        self._losses = 0
        self._draws = 0

        logger.info(
            f"SequentialTester initialized: min={self.config.min_games}, "
            f"max={self.config.max_games}, threshold={self.config.win_threshold}"
        )

    def record_game(
        self,
        game_id: int,
        winner: int,
        turns: int = 0,
        duration_seconds: float = 0.0,
    ) -> None:
        """Record an evaluation game result.

        Args:
            game_id: Unique game identifier.
            winner: 1 if new model won, -1 if baseline won, 0 if draw.
            turns: Number of turns.
            duration_seconds: Game duration.
        """
        game = GameResult(
            game_id=game_id,
            winner=winner,
            turns=turns,
            duration_seconds=duration_seconds,
        )
        self._games.append(game)

        if winner == 1:
            self._wins += 1
        elif winner == -1:
            self._losses += 1
        else:
            self._draws += 1

    def get_decision(self) -> EvaluationDecision:
        """Get current decision based on games played.

        Returns:
            EvaluationDecision indicating whether to continue, accept, or reject.
        """
        n = len(self._games)

        # Need minimum games
        if n < self.config.min_games:
            return EvaluationDecision.CONTINUE

        # Calculate win rate (excluding draws for decision)
        decisive_games = self._wins + self._losses
        if decisive_games == 0:
            return EvaluationDecision.CONTINUE

        win_rate = self._wins / decisive_games

        # Check if we've reached max games
        if n >= self.config.max_games:
            if win_rate >= self.config.win_threshold:
                return EvaluationDecision.ACCEPT
            else:
                return EvaluationDecision.INCONCLUSIVE

        # Check for decisive result
        if self.is_decisive():
            if win_rate >= self.config.win_threshold:
                return EvaluationDecision.ACCEPT
            else:
                return EvaluationDecision.REJECT

        # Check early stop threshold
        if n >= self.config.early_stop_threshold:
            # Apply stricter criteria at early stop threshold
            ci_low, ci_high = self.get_confidence_interval()

            # Accept if lower bound of CI is above threshold
            if ci_low >= self.config.win_threshold:
                return EvaluationDecision.ACCEPT

            # Reject if upper bound of CI is below 0.5
            if ci_high < 0.5:
                return EvaluationDecision.REJECT

        return EvaluationDecision.CONTINUE

    def is_decisive(self) -> bool:
        """Check if result is statistically decisive.

        Uses the decisive margin to determine if the result is clear enough
        to make an early decision.

        Returns:
            True if result is decisive.
        """
        decisive_games = self._wins + self._losses
        if decisive_games < self.config.min_games:
            return False

        win_rate = self._wins / decisive_games

        # Check if win rate is decisively above or below threshold
        if win_rate >= self.config.win_threshold + self.config.decisive_margin:
            return True
        if win_rate <= 0.5 - self.config.decisive_margin:
            return True

        return False

    def get_confidence_interval(self) -> tuple[float, float]:
        """Get confidence interval for win rate.

        Uses Wilson score interval for binomial proportions.

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        decisive_games = self._wins + self._losses
        if decisive_games == 0:
            return (0.0, 1.0)

        # Wilson score interval
        z = self._get_z_score(self.config.significance_level / 2)
        p_hat = self._wins / decisive_games
        n = decisive_games

        denominator = 1 + z * z / n
        center = p_hat + z * z / (2 * n)
        spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))

        lower = (center - spread) / denominator
        upper = (center + spread) / denominator

        return (max(0.0, lower), min(1.0, upper))

    @staticmethod
    def _get_z_score(alpha: float) -> float:
        """Get z-score for given alpha level.

        Uses approximation for standard normal quantile.

        Args:
            alpha: Significance level.

        Returns:
            Z-score.
        """
        # Approximation for normal quantile
        # For common values:
        # alpha=0.025 -> z ≈ 1.96
        # alpha=0.05 -> z ≈ 1.645
        if alpha >= 0.5:
            return 0.0

        # Abramowitz & Stegun 26.2.23 returns the UPPER-tail quantile for the
        # tail mass min(alpha, 1 - alpha), i.e. it is already positive. A prior
        # implementation multiplied it by ``-1`` for every alpha < 0.5, which is
        # every real significance level. That made ``get_confidence_interval``
        # return its bounds swapped (low > high), inverting both decision
        # branches in ``get_decision``: the accept test fired on the upper bound
        # (anti-conservative) and the reject test on the lower bound.
        p = min(alpha, 1 - alpha)

        t = math.sqrt(-2 * math.log(p))

        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

    def get_result(self) -> SequentialTestResult:
        """Get full result including statistics.

        Returns:
            SequentialTestResult with all statistics.
        """
        n = len(self._games)
        decisive_games = self._wins + self._losses

        if decisive_games > 0:
            win_rate = self._wins / decisive_games
        else:
            win_rate = 0.5

        _ci_low, _ci_high = self.get_confidence_interval()

        # Calculate confidence from p-value
        confidence = 1.0 - self._calculate_p_value()

        # Get decision and reason
        decision = self.get_decision()
        reason = self._build_reason(decision, win_rate, n)

        # Calculate cost savings
        cost_savings = self._calculate_cost_savings(n)

        return SequentialTestResult(
            decision=decision,
            games_played=n,
            wins=self._wins,
            losses=self._losses,
            draws=self._draws,
            win_rate=round(win_rate, 4),
            confidence=round(confidence, 4),
            reason=reason,
            cost_savings=round(cost_savings, 2),
        )

    def _calculate_p_value(self) -> float:
        """Calculate p-value for current results.

        Uses binomial test approximation.

        Returns:
            P-value.
        """
        decisive_games = self._wins + self._losses
        if decisive_games < 2:
            return 1.0

        # Test against null hypothesis p = 0.5
        p_null = 0.5
        observed = self._wins / decisive_games

        # Normal approximation to binomial
        se = math.sqrt(p_null * (1 - p_null) / decisive_games)
        if se == 0:
            return 1.0

        z = abs(observed - p_null) / se

        # Two-tailed p-value (approximation)
        # P(|Z| > z) ≈ 2 * (1 - Φ(z))
        # Using approximation for normal CDF
        p_value = 2 * (1 - self._normal_cdf(z))

        return min(1.0, p_value)

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Approximate standard normal CDF.

        Args:
            z: Z-score.

        Returns:
            Cumulative probability.
        """
        # Approximation using error function
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def _build_reason(
        self,
        decision: EvaluationDecision,
        win_rate: float,
        games: int,
    ) -> str:
        """Build human-readable reason string."""
        ci_low, ci_high = self.get_confidence_interval()

        if decision == EvaluationDecision.ACCEPT:
            return (
                f"Model accepted with {win_rate:.1%} win rate "
                f"(CI: [{ci_low:.1%}, {ci_high:.1%}]) after {games} games"
            )
        elif decision == EvaluationDecision.REJECT:
            return (
                f"Model rejected with {win_rate:.1%} win rate "
                f"(CI: [{ci_low:.1%}, {ci_high:.1%}]) after {games} games"
            )
        elif decision == EvaluationDecision.INCONCLUSIVE:
            return (
                f"Inconclusive after {games} games with {win_rate:.1%} win rate "
                f"(CI: [{ci_low:.1%}, {ci_high:.1%}])"
            )
        else:
            return (
                f"Need more games: {games}/{self.config.max_games} played, win rate {win_rate:.1%}"
            )

    def _calculate_cost_savings(self, games_played: int) -> float:
        """Calculate cost savings from early stopping.

        Args:
            games_played: Number of games actually played.

        Returns:
            Estimated cost savings in USD.
        """
        games_saved = self.config.max_games - games_played
        if games_saved <= 0:
            return 0.0
        return games_saved * self.config.cost_per_game_usd

    def reset(self) -> None:
        """Reset tester for new evaluation."""
        self._games.clear()
        self._wins = 0
        self._losses = 0
        self._draws = 0
        logger.debug("SequentialTester reset")

    def get_statistics(self) -> dict[str, Any]:
        """Get testing statistics.

        Returns:
            Dictionary with statistics.
        """
        decisive_games = self._wins + self._losses
        win_rate = self._wins / decisive_games if decisive_games > 0 else 0.5
        ci_low, ci_high = self.get_confidence_interval()

        return {
            "games_played": len(self._games),
            "wins": self._wins,
            "losses": self._losses,
            "draws": self._draws,
            "win_rate": round(win_rate, 4),
            "confidence_interval": (round(ci_low, 4), round(ci_high, 4)),
            "is_decisive": self.is_decisive(),
            "current_decision": self.get_decision().value,
        }


def get_sequential_tester() -> SequentialTester:
    """Get a sequential tester instance.

    Convenience function for getting a tester.

    Returns:
        SequentialTester instance.
    """
    return SequentialTester(SequentialTestingConfig.from_env())
