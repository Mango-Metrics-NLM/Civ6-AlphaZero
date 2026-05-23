"""Fog of war visibility management for multiplayer games.

Manages per-player visibility on the game board using a sight-range model.
Each unit and city provides visibility within a configurable range.
Tiles transition through: UNEXPLORED -> SEEN -> VISIBLE as they enter
and leave a player's sight range.

Design:
  - FogOfWarManager is environment-agnostic; it operates on position lists.
  - Chebyshev distance (max of |dx|, |dy|) for sight range calculation.
  - Default disabled (fog_of_war=False) to preserve existing behavior.

Example:
    >>> from alpha_zero.multiplayer.fog_of_war import FogOfWarManager, FogOfWarConfig
    >>> config = FogOfWarConfig(enabled=True)
    >>> manager = FogOfWarManager(config, board_size=16, num_players=2)
    >>> manager.compute_visibility(
    ...     units=[{"player": 0, "position": (5, 5), "type": "warrior"}],
    ...     cities=[{"player": 0, "position": (3, 3)}],
    ... )
    >>> vis = manager.get_visibility(player=0)
    >>> vis[5, 5]  # tile under own unit is visible
    2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VisibilityState(IntEnum):
    """Tile visibility states for a given player.

    Values are ordered so that higher == more information.

    Attributes:
        UNEXPLORED: Never been within sight range.
        SEEN: Previously visible, now in fog. Terrain is remembered
              but unit positions are stale.
        VISIBLE: Currently within sight range of a friendly unit/city.
    """

    UNEXPLORED = 0
    SEEN = 1
    VISIBLE = 2


@dataclass(frozen=True)
class FogOfWarConfig:
    """Configuration for fog of war.

    Attributes:
        enabled: Whether fog of war is active. ``False`` disables all
            masking and makes every tile visible to every player.
        default_sight_range: Fallback sight range for unknown unit types
            (Chebyshev distance).
        city_sight_range: Sight range provided by a city.
        unit_sight_ranges: Per-unit-type sight range overrides.
    """

    enabled: bool = False
    default_sight_range: int = 2
    city_sight_range: int = 3
    unit_sight_ranges: dict[str, int] = field(
        default_factory=lambda: {
            "warrior": 2,
            "settler": 2,
            "archer": 3,
        },
    )


class FogOfWarManager:
    """Manages per-player visibility on the game board.

    The manager maintains a visibility grid for each player. When
    :meth:`compute_visibility` is called the grids are updated:

    1. All ``VISIBLE`` tiles are demoted to ``SEEN``.
    2. Tiles within sight range of a friendly unit or city are promoted
       to ``VISIBLE``.

    Tiles that have never been seen remain ``UNEXPLORED``.

    Args:
        config: Fog of war configuration.
        board_size: Width/height of the square board.
        num_players: Number of players (>= 2).
    """

    def __init__(
        self,
        config: FogOfWarConfig,
        board_size: int,
        num_players: int,
    ) -> None:
        self.config = config
        self.board_size = board_size
        self.num_players = num_players

        # Per-player visibility grids: values are VisibilityState ints.
        self._visibility: dict[int, np.ndarray] = {}
        self.reset(num_players)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, num_players: int | None = None) -> None:
        """Reset visibility for a new game.

        All tiles start as ``UNEXPLORED`` for every player.

        Args:
            num_players: Optionally update the player count.
        """
        if num_players is not None:
            self.num_players = num_players

        self._visibility = {
            p: np.full(
                (self.board_size, self.board_size),
                VisibilityState.UNEXPLORED,
                dtype=np.int8,
            )
            for p in range(self.num_players)
        }

    def compute_visibility(
        self,
        units: list[dict[str, Any]],
        cities: list[dict[str, Any]],
    ) -> None:
        """Update visibility grids based on current unit/city positions.

        Previous ``VISIBLE`` tiles become ``SEEN`` (fog). Tiles currently
        within sight range of a friendly entity become ``VISIBLE``.

        Args:
            units: List of unit dicts with ``player``, ``position``, ``type``.
            cities: List of city dicts with ``player``, ``position``.
        """
        if not self.config.enabled:
            # When fog is disabled every tile is visible for every player.
            for grid in self._visibility.values():
                grid[:] = VisibilityState.VISIBLE
            logger.debug("Fog disabled — all tiles visible for all players")
            return

        # Phase 1: demote all VISIBLE -> SEEN (retain memory of terrain).
        for grid in self._visibility.values():
            grid[grid == VisibilityState.VISIBLE] = VisibilityState.SEEN

        # Phase 2: promote tiles in sight range to VISIBLE.
        logger.debug(
            "Computing visibility for %d units, %d cities",
            len(units),
            len(cities),
        )
        for unit in units:
            player = unit.get("player", 0)
            pos = unit.get("position", (0, 0))
            unit_type = unit.get("type", "warrior")
            sight = self.config.unit_sight_ranges.get(unit_type, self.config.default_sight_range)
            self._reveal(player, pos, sight)

        for city in cities:
            player = city.get("player", 0)
            pos = city.get("position", (0, 0))
            self._reveal(player, pos, self.config.city_sight_range)

    def get_visibility(self, player: int) -> np.ndarray:
        """Get the visibility grid for *player*.

        Args:
            player: Player index.

        Returns:
            ``np.ndarray`` of shape ``(board_size, board_size)`` with
            ``VisibilityState`` integer values.
        """
        if player not in self._visibility:
            return np.full(
                (self.board_size, self.board_size),
                VisibilityState.UNEXPLORED,
                dtype=np.int8,
            )
        return self._visibility[player]

    def get_visibility_dict(self, player: int) -> dict[tuple[int, int], str]:
        """Get per-tile visibility as a string dict for encoder consumption.

        Returns a mapping ``{(x, y): "visible"|"seen"|"unexplored"}``
        suitable for the ``visibility`` key in a game-state dict.

        Args:
            player: Player index.

        Returns:
            Dict mapping ``(x, y)`` -> visibility string.
        """
        grid = self.get_visibility(player)
        state_names = {
            VisibilityState.UNEXPLORED: "unexplored",
            VisibilityState.SEEN: "seen",
            VisibilityState.VISIBLE: "visible",
        }
        result: dict[tuple[int, int], str] = {}
        for y in range(self.board_size):
            for x in range(self.board_size):
                result[(x, y)] = state_names[VisibilityState(grid[y, x])]
        return result

    def mask_observation(
        self,
        game_state: dict[str, Any],
        player: int,
    ) -> dict[str, Any]:
        """Return *game_state* with hidden information removed for *player*.

        * Enemy units on tiles that are not ``VISIBLE`` are removed.
        * Enemy cities on tiles that are not ``VISIBLE`` are removed.
        * Terrain information is not masked and is assumed to be globally known.
        * The ``visibility`` key is injected into the returned dict.

        The original *game_state* is **not** mutated.

        Args:
            game_state: Full game state dict.
            player: Player whose perspective to mask for.

        Returns:
            Masked copy of the game state.
        """
        if not self.config.enabled:
            return game_state

        grid = self.get_visibility(player)
        logger.debug("Masking observation for player %d", player)
        masked = dict(game_state)  # shallow copy

        # Filter units: keep own units always, enemy units only if VISIBLE.
        filtered_units: list[dict[str, Any]] = []
        for unit in game_state.get("units", []):
            if unit.get("player") == player:
                filtered_units.append(unit)
            else:
                pos = unit.get("position", (0, 0))
                x, y = pos
                if (
                    0 <= x < self.board_size
                    and 0 <= y < self.board_size
                    and grid[y, x] == VisibilityState.VISIBLE
                ):
                    filtered_units.append(unit)
        masked["units"] = filtered_units
        hidden_units = len(game_state.get("units", [])) - len(filtered_units)
        if hidden_units > 0:
            logger.debug("Player %d: %d enemy units hidden by fog", player, hidden_units)

        # Filter cities: keep own always, enemy only if VISIBLE.
        filtered_cities: list[dict[str, Any]] = []
        for city in game_state.get("cities", []):
            if city.get("player") == player:
                filtered_cities.append(city)
            else:
                pos = city.get("position", (0, 0))
                x, y = pos
                if (
                    0 <= x < self.board_size
                    and 0 <= y < self.board_size
                    and grid[y, x] == VisibilityState.VISIBLE
                ):
                    filtered_cities.append(city)
        masked["cities"] = filtered_cities

        # Inject visibility data for the encoder.
        masked["visibility"] = self.get_visibility_dict(player)

        return masked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reveal(
        self,
        player: int,
        center: tuple[int, int],
        sight_range: int,
    ) -> None:
        """Reveal tiles within *sight_range* of *center* for *player*.

        Uses Chebyshev distance: ``max(|dx|, |dy|) <= sight_range``.
        """
        if player not in self._visibility:
            return

        cx, cy = center
        grid = self._visibility[player]
        bs = self.board_size

        x_min = max(0, cx - sight_range)
        x_max = min(bs - 1, cx + sight_range)
        y_min = max(0, cy - sight_range)
        y_max = min(bs - 1, cy + sight_range)

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                # Chebyshev distance check
                if max(abs(x - cx), abs(y - cy)) <= sight_range:
                    grid[y, x] = VisibilityState.VISIBLE
