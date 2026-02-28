"""Observation model for predicting hidden game state.

Provides a simple initial model that predicts the positions and types
of hidden enemy units using uniform random sampling. This serves as a
placeholder for a future neural-network-guided observation model that
could learn opponent tendencies from self-play data.

The observation model is consumed by :class:`WorldSampler` in the
determinization module to produce plausible complete game states.

Example:
    >>> from alpha_zero.multiplayer.observation_model import ObservationModel
    >>> model = ObservationModel(board_size=16, num_players=2)
    >>> hidden = model.predict_hidden_units(
    ...     visible_state=state,
    ...     player=0,
    ...     num_hidden_units=3,
    ...     candidate_tiles=[(5, 5), (6, 6), (7, 7), (8, 8)],
    ... )
    >>> len(hidden)
    3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationModelConfig:
    """Configuration for the observation model.

    Attributes:
        method: Prediction method. Currently only ``"uniform"`` is
            supported. Future options: ``"learned"``, ``"heuristic"``.
        default_unit_types: Unit types to sample from when generating
            hidden units.
        default_unit_weights: Sampling weights for each unit type
            (must match ``default_unit_types`` in length).
    """

    method: str = "uniform"
    default_unit_types: tuple[str, ...] = ("warrior", "archer", "settler")
    default_unit_weights: tuple[float, ...] = (0.5, 0.35, 0.15)

    def __post_init__(self) -> None:
        if len(self.default_unit_types) != len(self.default_unit_weights):
            msg = (
                f"unit_types length ({len(self.default_unit_types)}) must match "
                f"unit_weights length ({len(self.default_unit_weights)})"
            )
            raise ValueError(msg)
        if self.method not in ("uniform", "learned", "heuristic"):
            msg = f"Unknown observation model method: {self.method!r}"
            raise ValueError(msg)


# Strength lookup for synthetic hidden units.
_UNIT_STRENGTH: dict[str, int] = {
    "warrior": 20,
    "settler": 0,
    "archer": 15,
}

# Default moves for synthetic units.
_UNIT_MOVES: dict[str, int] = {
    "warrior": 2,
    "settler": 2,
    "archer": 2,
}


class ObservationModel:
    """Predicts hidden state from partial observations.

    The initial (V1) implementation uses uniform random placement.
    A future learned model could be trained on self-play replay data.

    Args:
        board_size: Width/height of the square board.
        num_players: Number of players.
        config: Observation model configuration.
        rng: Optional numpy random generator for reproducibility.
    """

    def __init__(
        self,
        board_size: int,
        num_players: int,
        config: ObservationModelConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.board_size = board_size
        self.num_players = num_players
        self.config = config or ObservationModelConfig()
        self._rng = rng or np.random.default_rng()

    def predict_hidden_units(
        self,
        visible_state: dict[str, Any],
        player: int,
        num_hidden_units: int,
        candidate_tiles: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        """Predict positions and types of hidden enemy units.

        V1 implementation: randomly place *num_hidden_units* on
        *candidate_tiles* with unit types sampled according to
        configured weights.

        Args:
            visible_state: The masked game state visible to *player*.
            player: The observing player.
            num_hidden_units: How many hidden units to generate.
            candidate_tiles: Board positions eligible for placement
                (non-visible, non-mountain, unoccupied).

        Returns:
            List of synthetic unit dicts, each with ``player``,
            ``position``, ``type``, ``health``, etc.
        """
        if num_hidden_units <= 0 or not candidate_tiles:
            return []

        num_to_place = min(num_hidden_units, len(candidate_tiles))

        # Sample positions without replacement.
        chosen_indices = self._rng.choice(
            len(candidate_tiles), size=num_to_place, replace=False
        )

        # Sample unit types.
        weights = np.array(self.config.default_unit_weights, dtype=np.float64)
        weights = weights / weights.sum()
        unit_types = list(self.config.default_unit_types)
        type_indices = self._rng.choice(len(unit_types), size=num_to_place, p=weights)

        hidden_units: list[dict[str, Any]] = []
        for i in range(num_to_place):
            pos = candidate_tiles[chosen_indices[i]]
            unit_type = unit_types[type_indices[i]]
            hidden_units.append(
                {
                    "id": -1,
                    "player": self._pick_opponent(player, visible_state),
                    "type": unit_type,
                    "position": pos,
                    "health": 100,
                    "moves_remaining": _UNIT_MOVES.get(unit_type, 2),
                    "strength": _UNIT_STRENGTH.get(unit_type, 20),
                }
            )

        return hidden_units

    def _pick_opponent(
        self,
        player: int,
        visible_state: dict[str, Any],
    ) -> int:
        """Pick an opponent player ID to assign a hidden unit to.

        Distributes uniformly among non-player opponents.
        """
        num_players: int = visible_state.get("num_players", self.num_players)
        opponents = [p for p in range(num_players) if p != player]
        if not opponents:
            return int((player + 1) % max(num_players, 2))
        return int(self._rng.choice(opponents))

    def reset(self) -> None:
        """Reset model state (no-op for uniform model)."""
