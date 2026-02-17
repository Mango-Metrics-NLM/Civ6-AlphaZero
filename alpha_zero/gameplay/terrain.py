"""Composable terrain system for Civ6 AlphaZero.

Provides movement cost calculation, yield computation, defense bonuses,
and BFS-based reachability queries. All values are driven by
:class:`TerrainConfig` -- nothing is hardcoded.

Typical usage::

    system = TerrainSystem()  # uses defaults
    cost = system.get_movement_cost("hills", features=["forest"])
    yields = system.get_yields("plains")
    reachable = system.get_tiles_in_range((5, 5), 3, terrain_map, 16)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TerrainConfig:
    """Data-driven terrain configuration.

    Attributes:
        movement_costs: Movement point cost per base terrain type.
            A value of ``-1`` means the terrain is impassable.
        terrain_yields: Base resource yields per terrain type.
        defense_bonuses: Fractional defense bonus per terrain type
            or feature.
        feature_yield_modifiers: Additive yield modifiers applied
            when a feature (e.g. forest) is present on a tile.
    """

    movement_costs: dict[str, int] = field(
        default_factory=lambda: {
            "plains": 1,
            "hills": 2,
            "mountain": -1,
            "forest": 2,
        }
    )
    terrain_yields: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "plains": {"food": 2, "production": 1, "gold": 0},
            "hills": {"food": 1, "production": 2, "gold": 0},
            "mountain": {"food": 0, "production": 0, "gold": 0},
        }
    )
    defense_bonuses: dict[str, float] = field(
        default_factory=lambda: {
            "hills": 0.25,
            "forest": 0.15,
            "mountain": 0.0,
        }
    )
    feature_yield_modifiers: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "forest": {"food": -1, "production": 1, "gold": 0},
        }
    )


class TerrainSystem:
    """Composable terrain system driven entirely by :class:`TerrainConfig`.

    All movement costs, yields, and defense bonuses are looked up from
    the supplied configuration.  Unknown terrain types or features are
    logged and treated with safe defaults (cost 1, zero yields / bonus).
    """

    def __init__(self, config: TerrainConfig | None = None) -> None:
        self.config = config or TerrainConfig()
        logger.debug(
            "TerrainSystem initialised with %d terrain types, %d features",
            len(self.config.movement_costs),
            len(self.config.feature_yield_modifiers),
        )

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def get_movement_cost(self, terrain_type: str, features: list[str] | None = None) -> int:
        """Return the movement cost for *terrain_type* with optional *features*.

        If the base terrain is impassable (cost ``-1``), the result is
        always ``-1`` regardless of features.  Features whose names
        appear as keys in ``movement_costs`` add their cost value on
        top of the base cost.  Unknown terrain types default to cost 1.

        Args:
            terrain_type: Base terrain identifier (e.g. ``"plains"``).
            features: Optional list of feature identifiers
                (e.g. ``["forest"]``).

        Returns:
            Total movement cost in movement points, or ``-1`` for
            impassable terrain.
        """
        base_cost = self.config.movement_costs.get(terrain_type, 1)
        if base_cost == -1:
            return -1

        if not features:
            return base_cost

        total = base_cost
        for feat in features:
            feat_cost = self.config.movement_costs.get(feat)
            if feat_cost is not None:
                if feat_cost == -1:
                    return -1
                total += feat_cost
            else:
                # Unknown feature -- add 1 as a conservative default.
                logger.debug("Unknown feature '%s'; adding +1 movement cost", feat)
                total += 1
        return total

    def is_passable(self, terrain_type: str) -> bool:
        """Return whether *terrain_type* can be entered.

        Args:
            terrain_type: Base terrain identifier.

        Returns:
            ``True`` if the movement cost is not ``-1``.
        """
        return self.config.movement_costs.get(terrain_type, 1) != -1

    # ------------------------------------------------------------------
    # Yields
    # ------------------------------------------------------------------

    def get_yields(self, terrain_type: str, features: list[str] | None = None) -> dict[str, int]:
        """Return the resource yields for *terrain_type* with *features*.

        Base yields come from ``terrain_yields``.  Each feature in
        *features* adds (or subtracts) its modifier from
        ``feature_yield_modifiers``.  Yields are never reduced below 0.

        Args:
            terrain_type: Base terrain identifier.
            features: Optional list of feature identifiers.

        Returns:
            Dictionary mapping resource names to integer yield values.
        """
        base = self.config.terrain_yields.get(terrain_type)
        if base is None:
            logger.debug("Unknown terrain '%s' for yields; returning zeros", terrain_type)
            base = {"food": 0, "production": 0, "gold": 0}

        result: dict[str, int] = dict(base)

        if features:
            for feat in features:
                modifier = self.config.feature_yield_modifiers.get(feat)
                if modifier is None:
                    continue
                for resource, delta in modifier.items():
                    result[resource] = max(0, result.get(resource, 0) + delta)

        return result

    # ------------------------------------------------------------------
    # Defense
    # ------------------------------------------------------------------

    def get_defense_bonus(self, terrain_type: str, features: list[str] | None = None) -> float:
        """Return the combined defense bonus for a tile.

        Defense bonuses from the base terrain and each feature are
        summed.

        Args:
            terrain_type: Base terrain identifier.
            features: Optional list of feature identifiers.

        Returns:
            Combined fractional defense bonus (e.g. ``0.40``).
        """
        bonus = self.config.defense_bonuses.get(terrain_type, 0.0)

        if features:
            for feat in features:
                bonus += self.config.defense_bonuses.get(feat, 0.0)

        return bonus

    # ------------------------------------------------------------------
    # BFS reachability
    # ------------------------------------------------------------------

    def get_tiles_in_range(
        self,
        position: tuple[int, int],
        move_range: int,
        terrain: dict[tuple[int, int], str],
        board_size: int,
    ) -> set[tuple[int, int]]:
        """Return tiles reachable from *position* within *move_range*.

        Uses breadth-first search that respects per-tile movement
        costs.  Impassable tiles are never entered.  The starting
        *position* is always included in the result.

        Args:
            position: ``(row, col)`` starting coordinate.
            move_range: Maximum movement points available.
            terrain: Mapping from ``(row, col)`` to terrain type string.
                Tiles absent from the map are treated as ``"plains"``.
            board_size: Width/height of the square board (tiles outside
                ``[0, board_size)`` are excluded).

        Returns:
            Set of ``(row, col)`` coordinates reachable within
            *move_range* movement points.
        """
        if move_range <= 0:
            return {position}

        # (row, col) -> best remaining movement when reaching that tile
        visited: dict[tuple[int, int], int] = {position: move_range}
        queue: deque[tuple[tuple[int, int], int]] = deque()
        queue.append((position, move_range))

        # 4-connected neighbours (N, S, E, W)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            (r, c), remaining = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= board_size or nc < 0 or nc >= board_size:
                    continue

                tile_terrain = terrain.get((nr, nc), "plains")
                cost = self.config.movement_costs.get(tile_terrain, 1)
                if cost == -1:
                    continue  # impassable

                new_remaining = remaining - cost
                if new_remaining < 0:
                    continue

                prev_remaining = visited.get((nr, nc))
                if prev_remaining is not None and prev_remaining >= new_remaining:
                    continue  # already reached with equal or more movement

                visited[(nr, nc)] = new_remaining
                queue.append(((nr, nc), new_remaining))

        return set(visited.keys())
