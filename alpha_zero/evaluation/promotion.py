"""Config-driven promotion policies for champion-vs-candidate evaluation.

Separation of concerns: :mod:`alpha_zero.evaluation.scoring` owns the statistics
(score rate, ternary variance, confidence intervals); this module owns the
*decision policy* built on top of them, plus its configuration and logging.

Why a registry rather than a branch
-----------------------------------
Four sites in this repository independently decide whether to promote a model,
and they disagree with each other:

* ``cloud/gcp/arena.py`` -> ``vertex_orchestrator.py`` (the deployed path)
* ``cloud/evaluator.py``
* ``orchestrator/curriculum.py`` (curriculum rung promotion)

Each reimplements the comparison inline. Routing them through one registered
policy makes the decision rule a single, testable, swappable component -- the
same extension seam the project already uses for MCTS variants
(``alpha_zero/core/mcts_factory.py``) and storage backends
(``alpha_zero/storage/factory.py``), per charter Invariant 1.

Backwards compatibility
-----------------------
The default policy is :data:`DEFAULT_POLICY` -- ``legacy_win_rate`` -- which
reproduces the deployed comparison exactly, draws-counted-as-losses included.
Nothing changes until an operator opts in via ``CIV6AZ_PROMOTION_POLICY``. That
is deliberate: the statistically sound policies below change which candidates
are promoted, and that is a decision for a human, not a default.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from alpha_zero.evaluation.scoring import (
    PromotionDecision,
    ScoreSummary,
    score_rate,
    summarize,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MIN_GAMES",
    "DEFAULT_POLICY",
    "DEFAULT_PROMOTION_THRESHOLD",
    "PromotionConfig",
    "PromotionPolicy",
    "PromotionVerdict",
    "create_promotion_policy",
    "evaluate_promotion",
    "list_promotion_policies",
    "log_promotion_decision",
    "register_promotion_policy",
    "verdict_from_results",
]

# ---------------------------------------------------------------------------
# Defaults. Named constants rather than inline literals so every call site and
# every test refers to one definition.
# ---------------------------------------------------------------------------

#: Policy used when nothing is configured. Reproduces the deployed behaviour.
DEFAULT_POLICY = "legacy_win_rate"

#: Inherited from AlphaGo Zero, where Go with komi 7.5 is draw-free by
#: construction. Retained as the default so behaviour does not change here;
#: see the module docstring of :mod:`alpha_zero.evaluation.scoring`.
DEFAULT_PROMOTION_THRESHOLD = 0.55

#: Confidence level for interval-based policies.
DEFAULT_CONFIDENCE = 0.95

#: Minimum games before an interval-based policy will return a verdict.
DEFAULT_MIN_GAMES = 1

_ENV_PREFIX = "CIV6AZ_PROMOTION_"

#: Pre-``CIV6AZ_`` variable names still honoured, newest first. ``cloud/``
#: predates the ``CIV6AZ_`` convention and several deployment files still set
#: these, so dropping them would silently change a running pipeline.
_LEGACY_THRESHOLD_ENV = ("CIV6_PROMOTION_THRESHOLD",)


def _resolve_env(name: str, legacy_names: tuple[str, ...] = ()) -> str | None:
    """Read ``CIV6AZ_PROMOTION_<name>``, falling back to legacy names.

    Mirrors the precedence-with-deprecation-warning pattern used for tracker
    enablement: the modern name wins, a legacy name is honoured but warned
    about, and absence is not an error.
    """
    value = os.environ.get(f"{_ENV_PREFIX}{name}")
    if value is not None:
        return value
    for legacy in legacy_names:
        legacy_value = os.environ.get(legacy)
        if legacy_value is not None:
            logger.warning(
                "%s is deprecated; use %s%s instead. Honouring %s=%s for now.",
                legacy,
                _ENV_PREFIX,
                name,
                legacy,
                legacy_value,
            )
            return legacy_value
    return None


def _env_float(name: str, default: float, legacy_names: tuple[str, ...] = ()) -> float:
    raw = _resolve_env(name, legacy_names)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid float for %s%s: %r; falling back to %s", _ENV_PREFIX, name, raw, default
        )
        return default


def _env_int(name: str, default: int) -> int:
    raw = _resolve_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid int for %s%s: %r; falling back to %s", _ENV_PREFIX, name, raw, default
        )
        return default


@dataclass(frozen=True)
class PromotionConfig:
    """Configuration for a promotion decision.

    Every operational value is a field with an explicit default and a
    ``CIV6AZ_PROMOTION_*`` override (charter Invariant 5).

    Attributes:
        policy: Registered policy name. See :func:`list_promotion_policies`.
        threshold: Value the policy's statistic must clear.
        confidence: Confidence level for interval-based policies.
        min_games: Minimum games before an interval-based policy decides.
    """

    policy: str = field(default_factory=lambda: _resolve_env("POLICY") or DEFAULT_POLICY)
    threshold: float = field(
        default_factory=lambda: _env_float(
            "THRESHOLD", DEFAULT_PROMOTION_THRESHOLD, _LEGACY_THRESHOLD_ENV
        )
    )
    confidence: float = field(default_factory=lambda: _env_float("CONFIDENCE", DEFAULT_CONFIDENCE))
    min_games: int = field(default_factory=lambda: _env_int("MIN_GAMES", DEFAULT_MIN_GAMES))

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field is out of range."""
        errors: list[str] = []
        if self.policy not in _POLICY_REGISTRY:
            errors.append(
                f"unknown policy {self.policy!r}; valid policies are {list_promotion_policies()}"
            )
        if not 0.0 <= self.threshold <= 1.0:
            errors.append(f"threshold must be in [0, 1], got {self.threshold}")
        if not 0.0 < self.confidence < 1.0:
            errors.append(f"confidence must be in (0, 1), got {self.confidence}")
        if self.min_games < 0:
            errors.append(f"min_games must be non-negative, got {self.min_games}")
        if errors:
            raise ValueError(f"Invalid promotion configuration: {'; '.join(errors)}")

    @classmethod
    def from_env(cls) -> PromotionConfig:
        """Build from the environment. Field defaults already read it."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromotionConfig:
        """Build from a mapping, falling back to env-derived defaults per key."""
        defaults = cls()
        return cls(
            policy=data.get("policy", defaults.policy),
            threshold=data.get("threshold", defaults.threshold),
            confidence=data.get("confidence", defaults.confidence),
            min_games=data.get("min_games", defaults.min_games),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for artifacts and experiment tracking."""
        return {
            "policy": self.policy,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "min_games": self.min_games,
        }


@dataclass(frozen=True)
class PromotionVerdict:
    """The full reasoning behind one promotion decision.

    Carries everything needed to re-audit the call later, which a bare boolean
    does not: which policy ran, on what statistic, against what threshold, over
    what counts, and with what interval.
    """

    decision: PromotionDecision
    policy: str
    statistic_name: str
    statistic: float
    threshold: float
    summary: ScoreSummary
    reason: str
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def promoted(self) -> bool:
        """Boolean view for callers that gate on a flag.

        ``INCONCLUSIVE`` maps to *not promoted*: absence of evidence must never
        advance a champion.
        """
        return self.decision is PromotionDecision.PROMOTE

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping suitable for logging and metric backends.

        The summary is merged first and the verdict's own fields overlaid
        second, so ``ci_low``/``ci_high`` always mean *the interval the policy
        consulted* -- ``None`` for a point-estimate policy, never the
        summary's diagnostic interval masquerading as a decision input. The
        diagnostic interval stays recomputable from the counts and
        ``confidence`` the payload carries.
        """
        payload: dict[str, Any] = dict(self.summary.to_dict())
        payload.update(
            {
                "decision": self.decision.value,
                "promoted": self.promoted,
                "policy": self.policy,
                "statistic_name": self.statistic_name,
                "statistic": self.statistic,
                "threshold": self.threshold,
                "reason": self.reason,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
            }
        )
        return payload


@runtime_checkable
class PromotionPolicy(Protocol):
    """A rule that turns game counts into a :class:`PromotionVerdict`."""

    name: str

    def evaluate(
        self, wins: int, draws: int, losses: int, config: PromotionConfig
    ) -> PromotionVerdict:
        """Decide, given the record and the configuration."""
        ...


class LegacyWinRatePolicy:
    """Reproduces the deployed gate exactly, including its defect.

    ``wins / (wins + draws + losses) >= threshold`` -- every draw counted as a
    loss. The statistic's ceiling is therefore ``1 - draw_rate``, so promotion
    is unreachable once the draw rate exceeds ``1 - threshold``.

    Retained as the default so that adopting this module changes no behaviour.
    It emits a warning when the record makes promotion arithmetically
    impossible, which is the condition that previously passed unnoticed.
    """

    name = "legacy_win_rate"
    statistic_name = "win_rate"

    def evaluate(
        self, wins: int, draws: int, losses: int, config: PromotionConfig
    ) -> PromotionVerdict:
        summary = summarize(wins, draws, losses, config.confidence)
        total = summary.total_games
        statistic = wins / total if total else 0.0
        promoted = statistic >= config.threshold

        ceiling = 1.0 - summary.draw_rate
        if total and ceiling < config.threshold:
            logger.warning(
                "Promotion is arithmetically impossible for this record: draws are scored "
                "as losses, so win_rate cannot exceed %.3f against threshold %.3f "
                "(draw rate %.1f%% over %d games). Consider "
                "CIV6AZ_PROMOTION_POLICY=score_rate.",
                ceiling,
                config.threshold,
                summary.draw_rate * 100.0,
                total,
            )

        return PromotionVerdict(
            decision=PromotionDecision.PROMOTE if promoted else PromotionDecision.REJECT,
            policy=self.name,
            statistic_name=self.statistic_name,
            statistic=statistic,
            threshold=config.threshold,
            summary=summary,
            reason=(
                f"win_rate {statistic:.4f} "
                f"{'>=' if promoted else '<'} threshold {config.threshold:.4f} "
                f"(draws counted as losses)"
            ),
        )


class ScoreRatePolicy:
    """Point-estimate gate on the score rate, scoring a draw as half a point.

    Removes the draws-as-losses defect but remains a point estimate, so it
    still promotes on noise at small sample sizes. Prefer
    :class:`ScoreRateLowerBoundPolicy` where the game budget allows.
    """

    name = "score_rate"
    statistic_name = "score_rate"

    def evaluate(
        self, wins: int, draws: int, losses: int, config: PromotionConfig
    ) -> PromotionVerdict:
        summary = summarize(wins, draws, losses, config.confidence)
        statistic = score_rate(wins, draws, losses)
        promoted = statistic >= config.threshold
        return PromotionVerdict(
            decision=PromotionDecision.PROMOTE if promoted else PromotionDecision.REJECT,
            policy=self.name,
            statistic_name=self.statistic_name,
            statistic=statistic,
            threshold=config.threshold,
            summary=summary,
            ci_low=summary.ci_low,
            ci_high=summary.ci_high,
            reason=(
                f"score_rate {statistic:.4f} "
                f"{'>=' if promoted else '<'} threshold {config.threshold:.4f}"
            ),
        )


class ScoreRateLowerBoundPolicy:
    """Promote only when the confidence bound clears the threshold.

    Promotes when the interval's lower bound exceeds the threshold, rejects
    when its upper bound falls below, and otherwise returns ``INCONCLUSIVE`` --
    which is a distinct state from rejection and must not be collapsed into it.

    This is the policy with a controlled false-promotion rate; the two above
    both promote on noise at small ``n``.
    """

    name = "score_rate_lower_bound"
    statistic_name = "score_rate"

    def evaluate(
        self, wins: int, draws: int, losses: int, config: PromotionConfig
    ) -> PromotionVerdict:
        summary = summarize(wins, draws, losses, config.confidence)
        statistic = summary.score_rate

        if summary.total_games < max(1, config.min_games):
            decision = PromotionDecision.INCONCLUSIVE
            reason = f"only {summary.total_games} games, below min_games {config.min_games}"
        elif summary.ci_low > config.threshold:
            decision = PromotionDecision.PROMOTE
            reason = (
                f"{config.confidence:.0%} lower bound {summary.ci_low:.4f} > "
                f"threshold {config.threshold:.4f}"
            )
        elif summary.ci_high < config.threshold:
            decision = PromotionDecision.REJECT
            reason = (
                f"{config.confidence:.0%} upper bound {summary.ci_high:.4f} < "
                f"threshold {config.threshold:.4f}"
            )
        else:
            decision = PromotionDecision.INCONCLUSIVE
            reason = (
                f"{config.confidence:.0%} interval "
                f"[{summary.ci_low:.4f}, {summary.ci_high:.4f}] straddles "
                f"threshold {config.threshold:.4f}; needs more games"
            )

        return PromotionVerdict(
            decision=decision,
            policy=self.name,
            statistic_name=self.statistic_name,
            statistic=statistic,
            threshold=config.threshold,
            summary=summary,
            ci_low=summary.ci_low,
            ci_high=summary.ci_high,
            reason=reason,
        )


_POLICY_REGISTRY: dict[str, PromotionPolicy] = {}


def register_promotion_policy(policy: PromotionPolicy, *, replace: bool = False) -> None:
    """Register a policy under its ``name``.

    Public so a caller can add a policy without editing this module (charter
    Invariant 1: extension through the seam, not by editing core).

    Args:
        policy: Any object satisfying :class:`PromotionPolicy`.
        replace: Permit overwriting an existing registration.

    Raises:
        ValueError: If the name is taken and ``replace`` is False.
    """
    name = policy.name
    if name in _POLICY_REGISTRY and not replace:
        raise ValueError(f"Promotion policy {name!r} is already registered; pass replace=True")
    _POLICY_REGISTRY[name] = policy


def create_promotion_policy(name: str) -> PromotionPolicy:
    """Look up a registered policy by name.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown promotion policy: {name!r}. Valid policies are: {list_promotion_policies()}"
        )
    return _POLICY_REGISTRY[name]


def list_promotion_policies() -> list[str]:
    """Names of every registered policy, sorted."""
    return sorted(_POLICY_REGISTRY)


for _policy in (LegacyWinRatePolicy(), ScoreRatePolicy(), ScoreRateLowerBoundPolicy()):
    register_promotion_policy(_policy)


def evaluate_promotion(
    wins: int, draws: int, losses: int, config: PromotionConfig | None = None
) -> PromotionVerdict:
    """Decide promotion using the configured policy.

    The single entry point every decision site should call.

    Args:
        wins: Candidate wins.
        draws: Drawn games.
        losses: Candidate losses.
        config: Configuration; defaults to :meth:`PromotionConfig.from_env`.

    Returns:
        The verdict, including the reasoning behind it.
    """
    config = config or PromotionConfig.from_env()
    return create_promotion_policy(config.policy).evaluate(wins, draws, losses, config)


#: Keys an arena artifact must carry for a draw-aware decision to be possible.
_REQUIRED_COUNT_KEYS = ("wins", "draws", "losses")


def verdict_from_results(
    results: Mapping[str, Any], config: PromotionConfig | None = None
) -> PromotionVerdict | None:
    """Build a verdict from a persisted arena artifact.

    Returns ``None`` when the artifact predates per-outcome counts, which is
    the case for every evaluation written before ``score_rate`` was added. A
    caller that gets ``None`` must fall back to whatever scalar the artifact
    does carry rather than treating the absence as a rejection -- a stale
    artifact is a missing measurement, not evidence against the candidate.

    Args:
        results: Parsed ``evaluations/iter_NNN/results.json`` payload.
        config: Promotion configuration; defaults to the environment.

    Returns:
        The verdict, or ``None`` if the artifact lacks win/draw/loss counts.
    """
    missing = [key for key in _REQUIRED_COUNT_KEYS if results.get(key) is None]
    if missing:
        logger.warning(
            "Evaluation artifact lacks %s; cannot apply a draw-aware promotion "
            "policy. Falling back to the scalar the artifact carries.",
            ", ".join(missing),
        )
        return None

    try:
        wins = int(results["wins"])
        draws = int(results["draws"])
        losses = int(results["losses"])
    except (TypeError, ValueError):
        logger.warning(
            "Evaluation artifact has non-integer counts %s; falling back to the "
            "scalar the artifact carries.",
            {key: results.get(key) for key in _REQUIRED_COUNT_KEYS},
        )
        return None

    return evaluate_promotion(wins, draws, losses, config)


def log_promotion_decision(
    verdict: PromotionVerdict,
    *,
    target: logging.Logger | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Emit a promotion decision at INFO, with the full record at DEBUG.

    The INFO line is the one a human reads in a job log; the DEBUG payload is
    the machine-readable record for after-the-fact auditing. Metrics belong in
    MLflow -- charter NG-5 forbids new Weights & Biases instrumentation.
    """
    log = target or logger
    summary = verdict.summary
    log.info(
        "Promotion %s [%s]: %s=%.4f vs threshold %.4f over %d games "
        "(%dW/%dD/%dL, draw rate %.1f%%) -- %s",
        verdict.decision.value.upper(),
        verdict.policy,
        verdict.statistic_name,
        verdict.statistic,
        verdict.threshold,
        summary.total_games,
        summary.wins,
        summary.draws,
        summary.losses,
        summary.draw_rate * 100.0,
        verdict.reason,
    )
    payload = verdict.to_dict()
    if context:
        payload.update(context)
    log.debug("Promotion decision record: %s", payload)
