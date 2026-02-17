"""Fog of war system for the MicroCiv environment.

A composable visibility system that tracks what each player can see.
Designed to be composed into the environment rather than inherited from.

Visibility states:
- UNEXPLORED (0): Tile has never been seen by the player.
- SEEN (1): Tile was previously visible but is not in current line of sight.
- VISIBLE (2): Tile is currently within line of sight of a friendly unit or city.

Mountains block line of sight but are themselves visible if within range.
Each unit type and cities have configurable sight ranges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visibility state constants
# ---------------------------------------------------------------------------
UNEXPLORED: int = 0
SEEN: int = 1
VISIBLE: int = 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FogOfWarConfig:
    """Configuration for the fog of war system.

    Attributes:
        sight_range: Mapping from unit/structure type to sight radius.
        remember_seen: When ``True``, tiles that were previously visible
            remain in the SEEN state; otherwise they revert to UNEXPLORED
            when they leave line of sight.
        default_sight_range: Sight range used for any unit type not listed
            in ``sight_range``.
    """

    sight_range: dict[str, int] = field(
        default_factory=lambda: {
            "warrior": 2,
            "settler": 2,
            "archer": 3,
            "city": 3,
        },
    )
    remember_seen: bool = True
    default_sight_range: int = 2


# ---------------------------------------------------------------------------
# Core system
# ---------------------------------------------------------------------------
class FogOfWarSystem:
    """Per-player visibility tracking for a tile-based map.

    This system is designed to be **composed** into an environment.  It owns
    no game-state beyond the visibility grids and is updated externally each
    time the authoritative game state changes.

    Parameters:
        board_size: Width and height of the square game board.
        num_players: Number of players tracked.
        config: Optional configuration; uses defaults if not provided.
    """

    def __init__(
        self,
        board_size: int,
        num_players: int = 2,
        config: FogOfWarConfig | None = None,
    ) -> None:
        self.board_size = board_size
        self.num_players = num_players
        self.config = config if config is not None else FogOfWarConfig()

        # Shape: (num_players, board_size, board_size), dtype int8
        self._visibility: np.ndarray = np.full(
            (num_players, board_size, board_size),
            UNEXPLORED,
            dtype=np.int8,
        )

        logger.info(
            "FogOfWarSystem initialised: board_size=%d, num_players=%d, "
            "remember_seen=%s, default_sight_range=%d",
            board_size,
            num_players,
            self.config.remember_seen,
            self.config.default_sight_range,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all visibility grids to UNEXPLORED."""
        self._visibility[:] = UNEXPLORED
        logger.debug("Visibility grids reset to UNEXPLORED")

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update_visibility(
        self,
        player: int,
        units: list[dict[str, Any]],
        cities: list[dict[str, Any]],
        terrain: dict[tuple[int, int], Any],
    ) -> None:
        """Recalculate visibility for *player* from scratch.

        Steps:
        1. Downgrade all currently VISIBLE tiles to SEEN (if
           ``remember_seen``) or UNEXPLORED.
        2. Reveal tiles around every friendly unit and city.

        Args:
            player: Player index (0-based).
            units: List of unit dicts.  Each dict must contain at least
                ``"player"`` (int), ``"position"`` (tuple[int, int]), and
                ``"type"`` (str).
            cities: List of city dicts.  Each dict must contain at least
                ``"player"`` (int) and ``"position"`` (tuple[int, int]).
            terrain: Mapping from ``(x, y)`` to a terrain object or dict.
                A tile is treated as a mountain if its ``terrain_type``
                attribute or ``"type"`` key equals ``"mountain"``.
        """
        grid = self._visibility[player]

        # Step 1 -- clear current visibility --------------------------------
        previously_visible = grid == VISIBLE
        if self.config.remember_seen:
            grid[previously_visible] = SEEN
        else:
            grid[previously_visible] = UNEXPLORED

        logger.debug(
            "Player %d: cleared %d previously visible tiles",
            player,
            int(np.sum(previously_visible)),
        )

        # Step 2 -- reveal from each friendly unit --------------------------
        own_units = [u for u in units if u.get("player") == player]
        for unit in own_units:
            pos: tuple[int, int] = (int(unit["position"][0]), int(unit["position"][1]))
            unit_type: str = unit.get("type", "")
            sight = self.config.sight_range.get(unit_type, self.config.default_sight_range)
            visible_tiles = self._get_visible_tiles(pos, sight, terrain)
            for tx, ty in visible_tiles:
                grid[ty, tx] = VISIBLE

        # Step 3 -- reveal from each friendly city --------------------------
        own_cities = [c for c in cities if c.get("player") == player]
        for city in own_cities:
            pos = (int(city["position"][0]), int(city["position"][1]))
            sight = self.config.sight_range.get("city", self.config.default_sight_range)
            visible_tiles = self._get_visible_tiles(pos, sight, terrain)
            for tx, ty in visible_tiles:
                grid[ty, tx] = VISIBLE

        total_visible = int(np.sum(grid == VISIBLE))
        total_seen = int(np.sum(grid == SEEN))
        logger.debug(
            "Player %d visibility updated: %d visible, %d seen, " "%d units, %d cities",
            player,
            total_visible,
            total_seen,
            len(own_units),
            len(own_cities),
        )

    # ------------------------------------------------------------------
    # Line-of-sight calculation
    # ------------------------------------------------------------------

    def _get_visible_tiles(
        self,
        position: tuple[int, int],
        sight_range: int,
        terrain: dict[tuple[int, int], Any],
    ) -> set[tuple[int, int]]:
        """Return the set of tiles visible from *position*.

        Visibility uses **Chebyshev distance** (king-move distance) with
        mountain-blocking.  A tile is visible if:

        1. It is within ``sight_range`` Chebyshev distance of *position*.
        2. It is within the board boundaries.
        3. No mountain tile intervenes along the Bresenham line from the
           source to the target (the mountain tile itself *is* visible;
           tiles behind it are not).

        Args:
            position: ``(x, y)`` origin.
            sight_range: Maximum Chebyshev distance.
            terrain: Terrain mapping (see ``update_visibility``).

        Returns:
            Set of ``(x, y)`` positions visible from *position*.
        """
        ox, oy = position
        visible: set[tuple[int, int]] = set()

        # The source tile is always visible.
        if 0 <= ox < self.board_size and 0 <= oy < self.board_size:
            visible.add((ox, oy))

        for dx in range(-sight_range, sight_range + 1):
            for dy in range(-sight_range, sight_range + 1):
                if dx == 0 and dy == 0:
                    continue

                tx, ty = ox + dx, oy + dy

                # Bounds check
                if not (0 <= tx < self.board_size and 0 <= ty < self.board_size):
                    continue

                # Chebyshev distance check
                if max(abs(dx), abs(dy)) > sight_range:
                    continue

                # Line-of-sight check (Bresenham)
                if self._has_line_of_sight(ox, oy, tx, ty, terrain):
                    visible.add((tx, ty))

        return visible

    def _has_line_of_sight(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        terrain: dict[tuple[int, int], Any],
    ) -> bool:
        """Check whether the line from (x0, y0) to (x1, y1) is unblocked.

        Uses Bresenham's line algorithm to walk intervening tiles.  A
        mountain tile **blocks** everything behind it, but the mountain
        tile itself is visible (the caller still adds it).

        Returns:
            ``True`` if no mountain intervenes *between* source and target
            (exclusive of both endpoints).
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cx, cy = x0, y0

        while True:
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

            # Reached the target -- line of sight is clear.
            if cx == x1 and cy == y1:
                return True

            # Check for mountain blocking on intervening tiles.
            if self._is_mountain(cx, cy, terrain):
                return False

    # ------------------------------------------------------------------
    # Terrain helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_mountain(
        x: int,
        y: int,
        terrain: dict[tuple[int, int], Any],
    ) -> bool:
        """Return ``True`` if the tile at (x, y) is a mountain.

        Handles both attribute-style objects (``tile.terrain_type``) and
        plain dicts (``tile["type"]``).
        """
        tile = terrain.get((x, y))
        if tile is None:
            return False
        # Attribute access (e.g. Tile dataclass)
        terrain_type = getattr(tile, "terrain_type", None)
        if terrain_type is not None:
            return bool(terrain_type == "mountain")
        # Dict access
        if isinstance(tile, dict):
            return bool(tile.get("type") == "mountain")
        return False

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def is_visible(self, player: int, position: tuple[int, int]) -> bool:
        """Return ``True`` if *position* is currently VISIBLE for *player*."""
        x, y = position
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            logger.debug(
                "is_visible: position (%d, %d) out of bounds for player %d",
                x,
                y,
                player,
            )
            return False
        return bool(self._visibility[player, y, x] == VISIBLE)

    def is_explored(self, player: int, position: tuple[int, int]) -> bool:
        """Return ``True`` if *position* has been explored (SEEN or VISIBLE)."""
        x, y = position
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            logger.debug(
                "is_explored: position (%d, %d) out of bounds for player %d",
                x,
                y,
                player,
            )
            return False
        return bool(self._visibility[player, y, x] >= SEEN)

    def get_visibility_state(self, player: int, position: tuple[int, int]) -> int:
        """Return the raw visibility constant for a tile.

        Returns:
            One of ``UNEXPLORED``, ``SEEN``, or ``VISIBLE``.
        """
        x, y = position
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return UNEXPLORED
        return int(self._visibility[player, y, x])

    def get_visibility_mask(self, player: int) -> np.ndarray:
        """Return a copy of the full visibility grid for *player*.

        Returns:
            ``np.ndarray`` of shape ``(board_size, board_size)`` with
            values in {UNEXPLORED, SEEN, VISIBLE}.
        """
        return self._visibility[player].copy()

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def get_visible_units(
        self,
        player: int,
        all_units: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return only those units located on tiles VISIBLE to *player*.

        Args:
            player: Observing player index.
            all_units: List of unit dicts with ``"position"`` keys.

        Returns:
            Filtered list of unit dicts.
        """
        result = [
            u
            for u in all_units
            if self.is_visible(player, (int(u["position"][0]), int(u["position"][1])))
        ]
        logger.debug(
            "get_visible_units: player %d sees %d / %d units",
            player,
            len(result),
            len(all_units),
        )
        return result

    def get_visible_cities(
        self,
        player: int,
        all_cities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return only those cities located on tiles VISIBLE to *player*.

        Args:
            player: Observing player index.
            all_cities: List of city dicts with ``"position"`` keys.

        Returns:
            Filtered list of city dicts.
        """
        result = [
            c
            for c in all_cities
            if self.is_visible(player, (int(c["position"][0]), int(c["position"][1])))
        ]
        logger.debug(
            "get_visible_cities: player %d sees %d / %d cities",
            player,
            len(result),
            len(all_cities),
        )
        return result

    # ------------------------------------------------------------------
    # Neural-network encoding
    # ------------------------------------------------------------------

    def encode_for_spatial(self, player: int) -> np.ndarray:
        """Encode visibility into a 3-channel spatial tensor.

        Returns:
            ``np.ndarray`` of shape ``(3, board_size, board_size)`` with
            dtype ``float32``:

            - Channel 0: currently VISIBLE tiles (1.0 / 0.0).
            - Channel 1: previously SEEN tiles (1.0 / 0.0).
            - Channel 2: UNEXPLORED tiles (1.0 / 0.0).
        """
        grid = self._visibility[player]
        encoding = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        encoding[0] = (grid == VISIBLE).astype(np.float32)
        encoding[1] = (grid == SEEN).astype(np.float32)
        encoding[2] = (grid == UNEXPLORED).astype(np.float32)

        logger.debug(
            "encode_for_spatial: player %d -- visible=%d, seen=%d, " "unexplored=%d",
            player,
            int(encoding[0].sum()),
            int(encoding[1].sum()),
            int(encoding[2].sum()),
        )
        return encoding

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the system state to a plain dict.

        The visibility grid is stored as a nested list so it is
        JSON-compatible.
        """
        return {
            "board_size": self.board_size,
            "num_players": self.num_players,
            "config": {
                "sight_range": self.config.sight_range,
                "remember_seen": self.config.remember_seen,
                "default_sight_range": self.config.default_sight_range,
            },
            "visibility": self._visibility.tolist(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        board_size: int,
    ) -> FogOfWarSystem:
        """Deserialize from a dict produced by ``to_dict``.

        Args:
            data: Dictionary with serialized state.
            board_size: Board size (used as a cross-check).

        Returns:
            A fully-reconstituted ``FogOfWarSystem``.
        """
        stored_board_size = data.get("board_size", board_size)
        if stored_board_size != board_size:
            logger.warning(
                "from_dict: stored board_size %d differs from requested %d; " "using stored value",
                stored_board_size,
                board_size,
            )
            board_size = stored_board_size

        num_players = data.get("num_players", 2)
        config_data = data.get("config", {})
        defaults = FogOfWarConfig()
        config = FogOfWarConfig(
            sight_range=config_data.get("sight_range", defaults.sight_range),
            remember_seen=config_data.get("remember_seen", defaults.remember_seen),
            default_sight_range=config_data.get(
                "default_sight_range", defaults.default_sight_range
            ),
        )

        system = cls(
            board_size=board_size,
            num_players=num_players,
            config=config,
        )

        visibility_list = data.get("visibility")
        if visibility_list is not None:
            system._visibility = np.array(visibility_list, dtype=np.int8)
            logger.info(
                "from_dict: restored visibility grid with shape %s",
                system._visibility.shape,
            )
        else:
            logger.warning("from_dict: no visibility data found; grid left as UNEXPLORED")

        return system
