"""Determinized MCTS for partially observable games.

When fog of war is active a player cannot see the full game state.
Determinized MCTS bridges this gap by:

1. Sampling K *complete* game states ("worlds") consistent with the
   player's partial observations.
2. Running standard MCTS on each world independently.
3. Combining the resulting action probabilities across worlds.

This is a well-known technique from imperfect-information game-playing
(see *Information Set Monte Carlo Tree Search*).

Example:
    >>> from alpha_zero.multiplayer.determinization import (
    ...     DeterminizationConfig, DeterminizedMCTS, WorldSampler,
    ... )
    >>> config = DeterminizationConfig(num_worlds=8)
    >>> sampler = WorldSampler(board_size=16, num_players=2)
    >>> det_mcts = DeterminizedMCTS(base_mcts, config, sampler)
    >>> probs, values = det_mcts.search(env, game_state, player=0, fog_manager=fog)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from alpha_zero.multiplayer.fog_of_war import FogOfWarManager, VisibilityState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeterminizationConfig:
    """Configuration for determinized MCTS.

    Attributes:
        num_worlds: Number of sampled worlds per search. More worlds
            reduce variance but increase computation linearly.
        combine_method: How to aggregate action probabilities across
            worlds. ``"average"`` takes the mean; ``"max"`` takes the
            element-wise maximum (then renormalizes).
    """

    num_worlds: int = 8
    combine_method: str = "average"

    def __post_init__(self) -> None:
        if self.num_worlds < 1:
            msg = f"num_worlds must be >= 1, got {self.num_worlds}"
            raise ValueError(msg)
        if self.combine_method not in ("average", "max"):
            msg = f"combine_method must be 'average' or 'max', got {self.combine_method!r}"
            raise ValueError(msg)


class WorldSampler:
    """Samples complete game states consistent with partial observations.

    For tiles not visible to the querying player, hidden enemy units
    are redistributed uniformly at random onto non-visible, passable
    tiles. Terrain and the player's own units/cities are always kept.

    This is a simple *uniform* sampler. Future versions may use a
    learned observation model for more realistic sampling.

    Args:
        board_size: Width/height of the square board.
        num_players: Number of players.
        rng: Optional numpy random generator for reproducibility.
    """

    def __init__(
        self,
        board_size: int,
        num_players: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.board_size = board_size
        self.num_players = num_players
        self._rng = rng or np.random.default_rng()

    def sample_world(
        self,
        observed_state: dict[str, Any],
        player: int,
        fog_manager: FogOfWarManager,
    ) -> dict[str, Any]:
        """Generate a possible complete game state.

        The returned state is consistent with everything *player* can
        currently observe. Hidden enemy units are placed randomly on
        non-visible, non-mountain tiles.

        Args:
            observed_state: Masked game state (from ``fog_manager.mask_observation``).
            player: The observing player.
            fog_manager: Fog of war manager with up-to-date visibility.

        Returns:
            A plausible complete game state dict.
        """
        world = copy.deepcopy(observed_state)
        grid = fog_manager.get_visibility(player)

        # Collect non-visible, passable tiles for placing hidden units.
        candidate_tiles: list[tuple[int, int]] = []
        terrain = observed_state.get("terrain", {})
        for y in range(self.board_size):
            for x in range(self.board_size):
                if grid[y, x] == VisibilityState.VISIBLE:
                    continue
                tile = terrain.get((x, y), {})
                if tile.get("type", "plains") == "mountain":
                    continue
                candidate_tiles.append((x, y))

        if not candidate_tiles:
            return world

        # Remove positions already occupied by friendly units.
        own_positions = {
            tuple(u.get("position", (0, 0)))
            for u in world.get("units", [])
            if u.get("player") == player
        }
        candidate_tiles = [p for p in candidate_tiles if p not in own_positions]

        if not candidate_tiles:
            return world

        # Estimate how many hidden enemy units to place.
        # Use a simple heuristic: each non-player should have roughly as
        # many units as the player has (a naive prior).
        own_unit_count = sum(
            1 for u in world.get("units", []) if u.get("player") == player
        )
        num_players = observed_state.get("num_players", self.num_players)

        for opponent in range(num_players):
            if opponent == player:
                continue

            # Count already-visible opponent units in the observed state.
            visible_opponent_units = [
                u
                for u in world.get("units", [])
                if u.get("player") == opponent
            ]
            visible_count = len(visible_opponent_units)

            # Estimate hidden units: assume same total as player, minus visible.
            estimated_hidden = max(0, own_unit_count - visible_count)

            # Place hidden units.
            if estimated_hidden > 0 and candidate_tiles:
                num_to_place = min(estimated_hidden, len(candidate_tiles))
                chosen_indices = self._rng.choice(
                    len(candidate_tiles), size=num_to_place, replace=False
                )
                for idx in chosen_indices:
                    pos = candidate_tiles[idx]
                    unit_type = self._rng.choice(["warrior", "archer", "settler"])
                    world["units"].append(
                        {
                            "id": -1,  # synthetic ID
                            "player": opponent,
                            "type": unit_type,
                            "position": pos,
                            "health": 100,
                            "moves_remaining": 2,
                            "strength": {"warrior": 20, "archer": 15, "settler": 0}.get(
                                unit_type, 20
                            ),
                        }
                    )

                # Remove used tiles from candidates for next opponent.
                used = {candidate_tiles[i] for i in chosen_indices}
                candidate_tiles = [p for p in candidate_tiles if p not in used]

        return world


class DeterminizedMCTS:
    """Wraps an MCTS implementation with determinization for partial observability.

    Performs multiple MCTS searches on independently sampled worlds and
    combines the resulting action probabilities.

    Args:
        base_mcts: An MCTS instance with a ``search(env, ...)`` method
            that returns ``(action_probs, player_values)``.
        config: Determinization configuration.
        world_sampler: Sampler that generates complete worlds from
            partial observations.
    """

    def __init__(
        self,
        base_mcts: Any,
        config: DeterminizationConfig,
        world_sampler: WorldSampler,
    ) -> None:
        self.base_mcts = base_mcts
        self.config = config
        self.world_sampler = world_sampler

    def search(
        self,
        env: Any,
        game_state: dict[str, Any],
        player: int,
        fog_manager: FogOfWarManager,
    ) -> tuple[np.ndarray, dict[int, float]]:
        """Run determinized MCTS search.

        1. Mask the game state for *player*.
        2. Sample ``num_worlds`` complete worlds consistent with observations.
        3. For each world, inject it into a cloned environment and run
           the base MCTS search.
        4. Combine action probabilities across worlds.

        Args:
            env: Game environment (must support ``clone()``).
            game_state: Full (unmasked) game state dict.
            player: Player performing the search.
            fog_manager: Fog of war manager with current visibility.

        Returns:
            Tuple of (action_probs, combined_player_values).
        """
        observed = fog_manager.mask_observation(game_state, player)

        all_probs: list[np.ndarray] = []
        all_values: list[dict[int, float]] = []

        for world_idx in range(self.config.num_worlds):
            world = self.world_sampler.sample_world(observed, player, fog_manager)

            # Clone the environment and inject the sampled world.
            sim_env = env.clone()
            self._inject_world(sim_env, world)

            # Run base MCTS on the determinized world.
            try:
                probs, values = self.base_mcts.search(sim_env, add_noise=(world_idx == 0))
            except Exception:
                logger.warning(
                    "MCTS search failed on sampled world %d; skipping",
                    world_idx,
                    exc_info=True,
                )
                continue

            all_probs.append(probs)
            all_values.append(values)

        if not all_probs:
            # Fallback: uniform over valid actions
            action_size = env.action_space.n
            return np.ones(action_size) / action_size, dict.fromkeys(range(env.num_players), 0.0)

        # Combine action probabilities.
        combined_probs = self._combine_probs(all_probs)

        # Average player values across worlds.
        combined_values = self._average_values(all_values, env.num_players)

        return combined_probs, combined_values

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_world(env: Any, world: dict[str, Any]) -> None:
        """Inject a sampled world state into a cloned environment.

        This sets units, cities, and turn information on *env* so that
        the base MCTS operates on the sampled world.
        """
        if hasattr(env, "units") and hasattr(env, "cities"):
            # Direct attribute injection for MicroCivEnv-like envs.
            from alpha_zero.envs.microciv import City, Unit

            env.units = []
            for u in world.get("units", []):
                env.units.append(
                    Unit(
                        id=u.get("id", 0),
                        player=u.get("player", 0),
                        unit_type=u.get("type", "warrior"),
                        position=tuple(u.get("position", (0, 0))),
                        health=u.get("health", 100),
                        moves_remaining=u.get("moves_remaining", 2),
                        strength=u.get("strength", 20),
                    )
                )

            env.cities = []
            for c in world.get("cities", []):
                env.cities.append(
                    City(
                        id=c.get("id", 0),
                        player=c.get("player", 0),
                        name=c.get("name", "City"),
                        position=tuple(c.get("position", (0, 0))),
                        population=c.get("population", 1),
                        production=c.get("production", 0),
                        production_queue=c.get("producing"),
                        defense=c.get("defense", 20),
                    )
                )

    def _combine_probs(self, all_probs: list[np.ndarray]) -> np.ndarray:
        """Combine action probability arrays from multiple worlds."""
        stacked = np.stack(all_probs, axis=0)

        if self.config.combine_method == "max":
            combined = np.max(stacked, axis=0)
        else:
            combined = np.mean(stacked, axis=0)

        # Renormalize.
        total = combined.sum()
        if total > 0:
            combined = combined / total
        result: np.ndarray = combined
        return result

    @staticmethod
    def _average_values(
        all_values: list[dict[int, float]],
        num_players: int,
    ) -> dict[int, float]:
        """Average per-player values across multiple worlds."""
        if not all_values:
            return dict.fromkeys(range(num_players), 0.0)

        sums: dict[int, float] = dict.fromkeys(range(num_players), 0.0)
        for vals in all_values:
            for p in range(num_players):
                sums[p] += vals.get(p, 0.0)

        count = len(all_values)
        return {p: sums[p] / count for p in range(num_players)}
