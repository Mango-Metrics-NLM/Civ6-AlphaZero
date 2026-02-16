"""Composable victory condition system for Civ6 AlphaZero.

Checks elimination, domination, and score-based victory conditions.
All thresholds and weights come from :class:`VictoryConfig` --
nothing is hardcoded.

Typical usage::

    system = VictorySystem(num_players=2)
    result = system.check_victory(game_state)
    if result is not None:
        print(f"Player {result.winner} wins by {result.victory_type.name}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


# ======================================================================
# Enums & Config
# ======================================================================

class VictoryType(Enum):
    """Types of victory that can end a game."""

    ELIMINATION = auto()
    SCORE = auto()
    DOMINATION = auto()  # control majority of cities


@dataclass
class VictoryConfig:
    """Data-driven victory configuration.

    Attributes:
        enabled_victories: List of victory type names that are active
            (e.g. ``["elimination", "score", "domination"]``).
        score_weights: Multipliers applied to each scoring category
            when computing a player's total score.
        domination_threshold: Fraction of total cities a player must
            control to win by domination.
        elimination_requires_all: If ``True``, a player is eliminated
            only when they have lost **all** cities **and** all units.
    """

    enabled_victories: list[str] = field(
        default_factory=lambda: ["elimination", "score"]
    )
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "unit": 1.0,
            "city": 5.0,
            "population": 2.0,
            "gold": 0.02,
            "territory": 0.5,
        }
    )
    domination_threshold: float = 0.66
    elimination_requires_all: bool = True


# ======================================================================
# Result dataclass
# ======================================================================

@dataclass
class VictoryCondition:
    """Describes the outcome when a victory is detected.

    Attributes:
        victory_type: Which victory condition was triggered.
        winner: Index of the winning player, or ``None`` for a draw.
        scores: Per-player scores at the time of victory.
        reason: Human-readable explanation.
    """

    victory_type: VictoryType
    winner: int | None = None
    scores: dict[int, float] = field(default_factory=dict)
    reason: str = ""


# ======================================================================
# Victory System
# ======================================================================

class VictorySystem:
    """Composable victory system driven by :class:`VictoryConfig`.

    The system supports three victory types checked in priority order:
    **elimination** > **domination** > **score** (at turn limit).

    Expected ``game_state`` dictionary structure::

        {
            "players": {
                0: {"units": [...], "cities": [...], "gold": 100, "territory": 12},
                1: {"units": [...], "cities": [...], "gold":  80, "territory":  9},
            },
            "turn_limit_reached": False,
            "total_cities": 6,       # optional, computed if absent
        }

    Each city dict should contain at least ``{"population": int}``.
    """

    def __init__(
        self, num_players: int = 2, config: VictoryConfig | None = None
    ) -> None:
        self.num_players = num_players
        self.config = config or VictoryConfig()
        self._enabled: set[str] = set(self.config.enabled_victories)
        logger.debug(
            "VictorySystem initialised for %d players, enabled=%s",
            num_players,
            self._enabled,
        )

    def reset(self) -> None:
        """Reset any internal state between games."""
        self._enabled = set(self.config.enabled_victories)
        logger.debug("VictorySystem reset")

    # ------------------------------------------------------------------
    # Public: main entry point
    # ------------------------------------------------------------------

    def check_victory(self, game_state: dict[str, Any]) -> VictoryCondition | None:
        """Check all enabled victory conditions in priority order.

        Priority: elimination > domination > score (only at turn limit).

        Args:
            game_state: Current game state dictionary.

        Returns:
            A :class:`VictoryCondition` if someone has won, otherwise
            ``None``.
        """
        if "elimination" in self._enabled:
            result = self.check_elimination(game_state)
            if result is not None:
                return result

        if "domination" in self._enabled:
            result = self.check_domination(game_state)
            if result is not None:
                return result

        if "score" in self._enabled and game_state.get("turn_limit_reached", False):
            result = self.check_score_victory(game_state)
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------
    # Elimination
    # ------------------------------------------------------------------

    def check_elimination(
        self, game_state: dict[str, Any]
    ) -> VictoryCondition | None:
        """Check whether any player has been eliminated.

        A player is eliminated when they have zero cities **and** zero
        units (if ``elimination_requires_all`` is ``True``), or zero
        cities alone otherwise.

        Args:
            game_state: Current game state dictionary.

        Returns:
            :class:`VictoryCondition` if exactly one player survives,
            else ``None``.
        """
        players: dict[int, dict[str, Any]] = game_state.get("players", {})
        survivors: list[int] = []

        for pid, pdata in players.items():
            cities = pdata.get("cities", [])
            units = pdata.get("units", [])

            if self.config.elimination_requires_all:
                if len(cities) > 0 or len(units) > 0:
                    survivors.append(pid)
            else:
                if len(cities) > 0:
                    survivors.append(pid)

        if len(survivors) == 1:
            winner = survivors[0]
            scores = self.calculate_scores(game_state)
            logger.info("Elimination victory for player %d", winner)
            return VictoryCondition(
                victory_type=VictoryType.ELIMINATION,
                winner=winner,
                scores=scores,
                reason=f"Player {winner} eliminated all opponents",
            )

        if len(survivors) == 0:
            scores = self.calculate_scores(game_state)
            logger.info("All players eliminated -- draw")
            return VictoryCondition(
                victory_type=VictoryType.ELIMINATION,
                winner=None,
                scores=scores,
                reason="All players eliminated simultaneously",
            )

        return None

    # ------------------------------------------------------------------
    # Domination
    # ------------------------------------------------------------------

    def check_domination(
        self, game_state: dict[str, Any]
    ) -> VictoryCondition | None:
        """Check whether any player controls enough cities to dominate.

        A player wins domination when they own at least
        ``domination_threshold`` of all cities on the map.

        Args:
            game_state: Current game state dictionary.

        Returns:
            :class:`VictoryCondition` if a player meets the threshold,
            else ``None``.
        """
        players: dict[int, dict[str, Any]] = game_state.get("players", {})
        total_cities = game_state.get("total_cities")
        if total_cities is None:
            total_cities = sum(
                len(p.get("cities", [])) for p in players.values()
            )

        if total_cities == 0:
            return None

        threshold = self.config.domination_threshold

        for pid, pdata in players.items():
            own_cities = len(pdata.get("cities", []))
            fraction = own_cities / total_cities
            if fraction >= threshold:
                scores = self.calculate_scores(game_state)
                logger.info(
                    "Domination victory for player %d (%.0f%% of cities)",
                    pid,
                    fraction * 100,
                )
                return VictoryCondition(
                    victory_type=VictoryType.DOMINATION,
                    winner=pid,
                    scores=scores,
                    reason=(
                        f"Player {pid} controls {own_cities}/{total_cities} "
                        f"cities ({fraction:.0%} >= {threshold:.0%})"
                    ),
                )

        return None

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def calculate_scores(
        self, game_state: dict[str, Any]
    ) -> dict[int, float]:
        """Calculate the score for every player.

        Score = sum(weight * count) for each category in
        ``score_weights``.

        Supported categories:
        - ``unit``:  number of units
        - ``city``:  number of cities
        - ``population``: total population across cities
        - ``gold``:  gold reserves
        - ``territory``: number of owned territory tiles

        Args:
            game_state: Current game state dictionary.

        Returns:
            Mapping of player id to total score.
        """
        players: dict[int, dict[str, Any]] = game_state.get("players", {})
        weights = self.config.score_weights
        scores: dict[int, float] = {}

        for pid, pdata in players.items():
            score = 0.0
            score += weights.get("unit", 0.0) * len(pdata.get("units", []))
            score += weights.get("city", 0.0) * len(pdata.get("cities", []))

            total_pop = sum(
                c.get("population", 1) for c in pdata.get("cities", [])
            )
            score += weights.get("population", 0.0) * total_pop
            score += weights.get("gold", 0.0) * pdata.get("gold", 0)
            score += weights.get("territory", 0.0) * pdata.get("territory", 0)

            scores[pid] = score

        return scores

    def check_score_victory(
        self, game_state: dict[str, Any]
    ) -> VictoryCondition | None:
        """Return the winner based on highest score.

        Intended for use when the turn limit has been reached.

        Args:
            game_state: Current game state dictionary.

        Returns:
            :class:`VictoryCondition` with the highest-scoring player
            as winner. ``None`` only if there are no players.
        """
        scores = self.calculate_scores(game_state)
        if not scores:
            return None

        winner = max(scores, key=lambda pid: scores[pid])
        logger.info(
            "Score victory for player %d (score %.1f)", winner, scores[winner]
        )
        return VictoryCondition(
            victory_type=VictoryType.SCORE,
            winner=winner,
            scores=scores,
            reason=f"Player {winner} has highest score ({scores[winner]:.1f})",
        )

    # ------------------------------------------------------------------
    # Progress & encoding
    # ------------------------------------------------------------------

    def get_victory_progress(
        self, player: int, game_state: dict[str, Any]
    ) -> dict[str, float]:
        """Return normalised progress toward each victory type.

        Values are in ``[0.0, 1.0]`` (clamped).  A value of ``1.0``
        means the condition is met or exceeded.

        Keys:
        - ``domination``: own_cities / (total_cities * threshold)
        - ``score``: own_score / max_possible_score (estimated)
        - ``elimination``: fraction of enemy assets destroyed

        Args:
            player: Player index.
            game_state: Current game state dictionary.

        Returns:
            Mapping of victory type name to progress float.
        """
        players: dict[int, dict[str, Any]] = game_state.get("players", {})
        progress: dict[str, float] = {}

        # -- domination --
        total_cities = game_state.get("total_cities")
        if total_cities is None:
            total_cities = sum(
                len(p.get("cities", [])) for p in players.values()
            )
        own_cities = len(players.get(player, {}).get("cities", []))
        dom_target = max(total_cities * self.config.domination_threshold, 1.0)
        progress["domination"] = min(1.0, own_cities / dom_target)

        # -- score --
        scores = self.calculate_scores(game_state)
        own_score = scores.get(player, 0.0)
        max_score = max(scores.values()) if scores else 1.0
        max_score = max(max_score, 1.0)  # avoid division by zero
        progress["score"] = min(1.0, own_score / max_score)

        # -- elimination --
        total_enemy_assets = 0
        destroyed_enemy_assets = 0
        for pid, pdata in players.items():
            if pid == player:
                continue
            units_alive = len(pdata.get("units", []))
            cities_alive = len(pdata.get("cities", []))
            # Estimate starting assets from game state metadata if available,
            # otherwise treat current alive count as partial information.
            starting_units = pdata.get("starting_units", units_alive)
            starting_cities = pdata.get("starting_cities", cities_alive)
            total_enemy_assets += starting_units + starting_cities
            destroyed_enemy_assets += (starting_units - units_alive) + (
                starting_cities - cities_alive
            )
        if total_enemy_assets > 0:
            progress["elimination"] = min(
                1.0, max(0.0, destroyed_enemy_assets / total_enemy_assets)
            )
        else:
            progress["elimination"] = 1.0

        return progress

    def encode_for_vector(
        self, player: int, game_state: dict[str, Any]
    ) -> dict[str, float]:
        """Return victory progress normalised for neural network input.

        Identical to :meth:`get_victory_progress` but guaranteed to
        contain stable keys suitable for concatenation into a feature
        vector.

        Args:
            player: Player index.
            game_state: Current game state dictionary.

        Returns:
            Mapping with keys ``"domination"``, ``"score"``, and
            ``"elimination"``, each a float in ``[0.0, 1.0]``.
        """
        return self.get_victory_progress(player, game_state)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the system state to a plain dictionary.

        Returns:
            Dictionary suitable for JSON serialisation.
        """
        return {
            "num_players": self.num_players,
            "config": {
                "enabled_victories": list(self.config.enabled_victories),
                "score_weights": dict(self.config.score_weights),
                "domination_threshold": self.config.domination_threshold,
                "elimination_requires_all": self.config.elimination_requires_all,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VictorySystem:
        """Restore a :class:`VictorySystem` from a dictionary.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            Reconstructed :class:`VictorySystem`.
        """
        config_data = data.get("config", {})
        config = VictoryConfig(
            enabled_victories=config_data.get(
                "enabled_victories", ["elimination", "score"]
            ),
            score_weights=config_data.get(
                "score_weights",
                {"unit": 1.0, "city": 5.0, "population": 2.0, "gold": 0.02, "territory": 0.5},
            ),
            domination_threshold=config_data.get("domination_threshold", 0.66),
            elimination_requires_all=config_data.get(
                "elimination_requires_all", True
            ),
        )
        return cls(num_players=data.get("num_players", 2), config=config)
