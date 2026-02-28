"""MicroCiv: Simplified Civilization environment for Phase 1 training.

A minimal 4X game that captures core Civ mechanics:
- Explore: Fog of war, terrain discovery
- Expand: Settlers found new cities
- Exploit: Cities produce units
- Exterminate: Combat between units

Supports 2-8 players (default 2 for backward compatibility).
Victory: Eliminate all enemy cities/units, or reach turn limit (most cities wins).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame

from alpha_zero.core.civ6_actions import MicroCivActionSpace
from alpha_zero.envs.civ6_encoder import MicroCivEncoder
from alpha_zero.utils.constants import GAME, REWARDS, UNITS

logger = logging.getLogger(__name__)


@dataclass
class Unit:
    """A unit in the game."""

    id: int
    player: int
    unit_type: str
    position: tuple[int, int]
    health: int = GAME.DEFAULT_UNIT_HEALTH
    moves_remaining: int = UNITS.WARRIOR_MOVES
    strength: int = UNITS.WARRIOR_STRENGTH

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "player": self.player,
            "type": self.unit_type,
            "position": self.position,
            "health": self.health,
            "moves_remaining": self.moves_remaining,
            "strength": self.strength,
        }


@dataclass
class City:
    """A city in the game."""

    id: int
    player: int
    name: str
    position: tuple[int, int]
    population: int = 1
    production: int = 0
    production_queue: str | None = None
    defense: int = GAME.DEFAULT_CITY_DEFENSE

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "player": self.player,
            "name": self.name,
            "position": self.position,
            "population": self.population,
            "production": self.production,
            "producing": self.production_queue,
            "defense": self.defense,
        }


@dataclass
class Tile:
    """A map tile."""

    terrain_type: str = "plains"
    features: list[str] = field(default_factory=list)
    yields: dict[str, int] = field(default_factory=lambda: {"food": 2, "production": 1})

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.terrain_type,
            "features": self.features,
            "yields": self.yields,
        }


class MicroCivEnv(gym.Env):
    """MicroCiv environment for AlphaZero training.

    A simplified 4X game with:
    - 2-8 players (turn-based, default 2)
    - 16x16 tile map
    - 3 unit types: warrior, settler, archer
    - Simple combat and city founding

    Observation space:
    - Spatial: (13, board_size, board_size) tensor
    - Vector: (64,) tensor

    Action space:
    - Flat discrete space covering all possible actions
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["ansi"]}

    # Unit production costs (turns)
    UNIT_COSTS: ClassVar[dict[str, int]] = {
        "warrior": UNITS.WARRIOR_COST,
        "settler": UNITS.SETTLER_COST,
        "archer": UNITS.ARCHER_COST,
    }
    UNIT_STRENGTH: ClassVar[dict[str, int]] = {
        "warrior": UNITS.WARRIOR_STRENGTH,
        "settler": UNITS.SETTLER_STRENGTH,
        "archer": UNITS.ARCHER_STRENGTH,
    }
    UNIT_MOVES: ClassVar[dict[str, int]] = {
        "warrior": UNITS.WARRIOR_MOVES,
        "settler": UNITS.SETTLER_MOVES,
        "archer": UNITS.ARCHER_MOVES,
    }

    # Player symbols for rendering (indexed by player ID)
    PLAYER_SYMBOLS: ClassVar[dict[str, list[str]]] = {
        "warrior": ["W", "w", "2", "3", "4", "5", "6", "7"],
        "settler": ["S", "s", "T", "U", "V", "X", "Y", "Z"],
        "archer": ["A", "a", "b", "d", "e", "f", "g", "h"],
        "city": ["C", "c", "o", "p", "q", "r", "u", "x"],
    }

    # Explicit type hint for MCTS compatibility
    action_space: spaces.Discrete

    def __init__(
        self,
        board_size: int = 16,
        max_turns: int = 200,
        render_mode: str | None = None,
        seed: int | None = None,
        num_players: int = 2,
    ) -> None:
        """Initialize MicroCiv environment.

        Args:
            board_size: Size of the square game board.
            max_turns: Maximum turns before draw.
            render_mode: Rendering mode ('ansi' or None).
            seed: Random seed for reproducibility.
            num_players: Number of players (2-8, default 2).

        Raises:
            ValueError: If num_players is not in range [2, MAX_PLAYERS].
        """
        super().__init__()

        if not (2 <= num_players <= GAME.MAX_PLAYERS):
            raise ValueError(
                f"num_players must be between 2 and {GAME.MAX_PLAYERS}, "
                f"got {num_players}"
            )

        self.board_size = board_size
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.num_players = num_players

        # Initialize RNG
        self._rng = np.random.default_rng(seed)

        # Action and observation spaces
        self.action_space_handler = MicroCivActionSpace(
            board_size=board_size,
            max_units=20,
            max_cities=5,
        )
        self.encoder = MicroCivEncoder(board_size=board_size)

        # Gymnasium spaces
        self.action_space = spaces.Discrete(self.action_space_handler.total_size)
        self.observation_space = spaces.Dict(
            {
                "spatial": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(MicroCivEncoder.TOTAL_CHANNELS, board_size, board_size),
                    dtype=np.float32,
                ),
                "vector": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(64,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.action_space_handler.total_size,),
                    dtype=np.float32,
                ),
            },
        )

        # Game state
        self.terrain: dict[tuple[int, int], Tile] = {}
        self.units: list[Unit] = []
        self.cities: list[City] = []
        self.turn: int = 0
        self.current_player: int = 0
        self.done: bool = False
        self.winner: int | None = None
        self._next_unit_id: int = 0
        self._next_city_id: int = 0
        self.visited: set[tuple[int, int]] = set()  # Track visited tiles

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.visited = set()
        self._generate_map()
        self._place_starting_units()
        self.turn = 0
        self.current_player = 0
        self.done = False
        self.winner = None

        obs = self.get_observation()
        info = self.get_info()

        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute action and return new state.

        Args:
            action: Flat action index.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self.done:
            return self.get_observation(), 0.0, True, False, self.get_info()

        # Decode and execute action
        action_dict = self.action_space_handler.decode_action(action)
        reward, action_success, action_error = self._execute_action(action_dict)

        # Log failed actions for debugging
        if not action_success:
            logger.warning(
                f"Turn {self.turn}, Player {self.current_player}: "
                f"Action {action} ({action_dict.get('type')}) failed: {action_error}"
            )

        # Check victory conditions
        terminated = self._check_victory()
        truncated = self.turn >= self.max_turns

        if truncated and not terminated:
            # Count cities for each player and pick winner by most cities
            city_counts = [
                sum(1 for c in self.cities if c.player == p)
                for p in range(self.num_players)
            ]
            max_cities = max(city_counts)
            players_with_max = [
                p for p, count in enumerate(city_counts) if count == max_cities
            ]
            if len(players_with_max) == 1:
                self.winner = players_with_max[0]
            else:
                self.winner = None  # Draw (tied for most cities)
            terminated = True

        if terminated:
            self.done = True
            # Final reward based on winner
            if self.winner == self.current_player:
                reward += REWARDS.WIN_REWARD
            elif self.winner is not None:
                reward -= REWARDS.WIN_REWARD

        obs = self.get_observation()
        info = self.get_info()

        # Add action execution info for debugging
        info["action_success"] = action_success
        info["action_error"] = action_error

        return obs, reward, terminated, truncated, info

    def _generate_map(self) -> None:
        """Generate random terrain map."""
        self.terrain = {}
        for y in range(self.board_size):
            for x in range(self.board_size):
                # Random terrain type
                r = self._rng.random()
                if r < GAME.PLAINS_RATIO:
                    terrain_type = "plains"
                    yields = {"food": 2, "production": 1}
                elif r < GAME.PLAINS_RATIO + GAME.HILLS_RATIO:
                    terrain_type = "hills"
                    yields = {"food": 1, "production": 2}
                else:
                    terrain_type = "mountain"
                    yields = {"food": 0, "production": 0}

                self.terrain[(x, y)] = Tile(terrain_type=terrain_type, yields=yields)

    def _place_starting_units(self) -> None:
        """Place starting units for each player.

        Uses angular distribution to spread players evenly around the map.
        For 2 players, positions approximate the original bottom-left / top-right layout.
        """
        self.units = []
        self.cities = []
        self._next_unit_id = 0
        self._next_city_id = 0

        # Compute starting positions using angular distribution
        positions = self._compute_starting_positions()

        # Create starting units for each player
        for player, base_pos in enumerate(positions):
            # Find a valid (non-mountain) position near the computed position
            pos = self._find_valid_position(
                max(0, base_pos[0] - 2),
                max(0, base_pos[1] - 2),
                5,
            )

            # Settler
            self.units.append(
                Unit(
                    id=self._next_unit_id,
                    player=player,
                    unit_type="settler",
                    position=pos,
                    strength=UNITS.SETTLER_STRENGTH,
                    moves_remaining=UNITS.SETTLER_MOVES,
                ),
            )
            self.visited.add(pos)  # Mark start pos as visited
            self._next_unit_id += 1

            # Warrior nearby
            warrior_pos = self._find_adjacent_valid(pos)
            self.units.append(
                Unit(
                    id=self._next_unit_id,
                    player=player,
                    unit_type="warrior",
                    position=warrior_pos,
                    strength=UNITS.WARRIOR_STRENGTH,
                    moves_remaining=UNITS.WARRIOR_MOVES,
                ),
            )
            self.visited.add(warrior_pos)  # Mark warrior pos as visited
            self._next_unit_id += 1

    def _compute_starting_positions(self) -> list[tuple[int, int]]:
        """Compute starting positions for all players using angular distribution.

        Players are evenly spaced around a circle centered on the map.

        Returns:
            List of (x, y) positions, one per player.
        """
        positions: list[tuple[int, int]] = []
        center = self.board_size / 2
        radius = self.board_size * 0.35  # Keep away from edges

        for i in range(self.num_players):
            angle = 2 * math.pi * i / self.num_players
            x = int(center + radius * math.cos(angle))
            y = int(center + radius * math.sin(angle))
            # Clamp to ensure we stay within valid board range
            x = max(1, min(self.board_size - 2, x))
            y = max(1, min(self.board_size - 2, y))
            positions.append((x, y))

        return positions

    def _find_valid_position(
        self,
        start_x: int,
        start_y: int,
        range_: int,
    ) -> tuple[int, int]:
        """Find a valid (non-mountain) position in a range."""
        for _ in range(100):
            x = start_x + self._rng.integers(0, range_)
            y = start_y + self._rng.integers(0, range_)
            x = min(max(x, 0), self.board_size - 1)
            y = min(max(y, 0), self.board_size - 1)
            if self.terrain.get((x, y), Tile()).terrain_type != "mountain":
                return (x, y)
        return (start_x, start_y)

    def _find_adjacent_valid(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Find an adjacent valid position."""
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if self.terrain.get((nx, ny), Tile()).terrain_type != "mountain":
                    return (nx, ny)
        return pos

    def _execute_action(self, action: dict) -> tuple[float, bool, str]:
        """Execute an action and return immediate reward with success status.

        Args:
            action: Action dictionary.

        Returns:
            Tuple of (reward, success, error_message).
            - reward: Immediate shaped reward
            - success: Whether the action was successfully executed
            - error_message: Description of failure if success is False
        """
        action_type = action.get("type")
        reward = 0.0

        if action_type == "end_turn":
            self._end_turn()
            return reward, True, ""

        elif action_type == "move":
            unit_idx = action.get("unit_idx", 0)
            direction = action.get("direction", (0, 0))
            own_units = self._get_player_units(self.current_player)

            if unit_idx >= len(own_units):
                return 0.0, False, f"Invalid unit_idx {unit_idx} for {len(own_units)} units"

            unit = own_units[unit_idx]
            if unit.moves_remaining <= 0:
                return 0.0, False, f"Unit {unit_idx} has no moves remaining"

            new_pos = (
                unit.position[0] + direction[0],
                unit.position[1] + direction[1],
            )
            if self._is_valid_move(new_pos):
                unit.position = new_pos
                if unit.player == self.current_player:
                    self.visited.add(new_pos)  # Track visitation
                unit.moves_remaining -= 1
                return reward, True, ""
            else:
                # Move failed, treat as end of unit's actions to prevent lock
                unit.moves_remaining = 0
                return 0.0, False, f"Invalid move to {new_pos}"

        elif action_type == "attack":
            unit_idx = action.get("unit_idx", 0)
            direction = action.get("direction", (0, 0))
            own_units = self._get_player_units(self.current_player)

            if unit_idx >= len(own_units):
                return 0.0, False, f"Invalid unit_idx {unit_idx} for {len(own_units)} units"

            unit = own_units[unit_idx]
            if unit.moves_remaining <= 0:
                return 0.0, False, f"Unit {unit_idx} has no moves remaining"

            target_pos = (
                unit.position[0] + direction[0],
                unit.position[1] + direction[1],
            )

            # Check if there's a valid target before attempting attack
            has_defender = any(
                u.position == target_pos and u.player != unit.player for u in self.units
            )
            if not has_defender:
                return 0.0, False, f"No enemy unit to attack at {target_pos}"

            attack_reward = self._execute_attack(unit, target_pos)
            reward += attack_reward
            unit.moves_remaining -= 1
            return reward, True, ""

        elif action_type == "skip":
            unit_idx = action.get("unit_idx", 0)
            own_units = self._get_player_units(self.current_player)

            if unit_idx >= len(own_units):
                return 0.0, False, f"Invalid unit_idx {unit_idx} for {len(own_units)} units"

            own_units[unit_idx].moves_remaining = 0
            return reward, True, ""

        elif action_type == "found_city":
            unit_idx = action.get("unit_idx", 0)
            own_units = self._get_player_units(self.current_player)

            if unit_idx >= len(own_units):
                return 0.0, False, f"Invalid unit_idx {unit_idx} for {len(own_units)} units"

            unit = own_units[unit_idx]
            if unit.unit_type != "settler":
                return 0.0, False, f"Unit {unit_idx} is not a settler"

            # Check for nearby cities
            has_nearby_city = any(
                abs(c.position[0] - unit.position[0]) <= GAME.MIN_CITY_DISTANCE
                and abs(c.position[1] - unit.position[1]) <= GAME.MIN_CITY_DISTANCE
                for c in self.cities
            )
            if has_nearby_city:
                return 0.0, False, "Cannot found city near another city"

            # Found city
            city_name = f"City {self._next_city_id}"
            self.cities.append(
                City(
                    id=self._next_city_id,
                    player=unit.player,
                    name=city_name,
                    position=unit.position,
                )
            )
            self._next_city_id += 1
            unit.moves_remaining = 0  # Action used

            # Reward for founding city
            reward += GAME.CITY_FOUND_REWARD

            # Remove settler
            if unit in self.units:
                self.units.remove(unit)

            return reward, True, ""

        elif action_type == "produce":
            city_idx = action.get("city_idx", 0)
            unit_type = action.get("unit_type", "warrior")
            own_cities = self._get_player_cities(self.current_player)

            if city_idx >= len(own_cities):
                return 0.0, False, f"Invalid city_idx {city_idx} for {len(own_cities)} cities"

            city = own_cities[city_idx]
            if city.production_queue is not None:
                return 0.0, False, f"City {city_idx} already has production queued"

            city.production_queue = unit_type
            reward += GAME.PRODUCTION_START_REWARD
            return reward, True, ""

        return 0.0, False, f"Unknown action type: {action_type}"

    def _execute_attack(self, attacker: Unit, target_pos: tuple[int, int]) -> float:
        """Execute combat between units.

        Args:
            attacker: Attacking unit.
            target_pos: Position being attacked.

        Returns:
            Reward from combat.
        """
        reward = 0.0

        # Find defender at position
        defender = None
        for unit in self.units:
            if unit.position == target_pos and unit.player != attacker.player:
                defender = unit
                break

        if defender is None:
            return 0.0

        # Simple combat resolution
        attacker_str = attacker.strength * (attacker.health / float(GAME.DEFAULT_UNIT_HEALTH))
        defender_str = defender.strength * (defender.health / float(GAME.DEFAULT_UNIT_HEALTH))

        # Damage calculation
        if attacker_str + defender_str > 0:
            attacker_ratio = attacker_str / (
                attacker_str + defender_str + GAME.COMBAT_SMOOTHING_EPSILON
            )
            damage_to_defender = int(
                GAME.COMBAT_BASE_DAMAGE * attacker_ratio + GAME.COMBAT_MIN_DAMAGE_DEFENDER
            )
            damage_to_attacker = int(
                GAME.COMBAT_BASE_DAMAGE * (1 - attacker_ratio) + GAME.COMBAT_MIN_DAMAGE_ATTACKER
            )

            defender.health -= damage_to_defender
            attacker.health -= damage_to_attacker

            # Remove dead units
            if defender.health <= 0:
                self.units.remove(defender)
                reward += REWARDS.MICROCIV_UNIT_KILL_REWARD

            if attacker.health <= 0:
                self.units.remove(attacker)
                reward += REWARDS.MICROCIV_UNIT_LOST_PENALTY

        return reward

    def _end_turn(self) -> None:
        """End current player's turn."""
        # Process cities (production)
        for city in self._get_player_cities(self.current_player):
            if city.production_queue:
                city.production += city.population + GAME.CITY_BASE_PRODUCTION
                cost = (
                    self.UNIT_COSTS.get(city.production_queue, UNITS.SETTLER_COST)
                    * GAME.PRODUCTION_COST_MULTIPLIER
                )
                if city.production >= cost:
                    # Spawn unit
                    spawn_pos = self._find_adjacent_valid(city.position)
                    self.units.append(
                        Unit(
                            id=self._next_unit_id,
                            player=city.player,
                            unit_type=city.production_queue,
                            position=spawn_pos,
                            strength=self.UNIT_STRENGTH.get(
                        city.production_queue, UNITS.WARRIOR_STRENGTH
                    ),
                            moves_remaining=0,
                        ),
                    )
                    self._next_unit_id += 1
                    city.production = 0
                    city.production_queue = None

        # Switch to next player
        self.current_player = (self.current_player + 1) % self.num_players

        # If back to player 0, increment turn
        if self.current_player == 0:
            self.turn += 1
            # Reset all units' moves
            for unit in self.units:
                unit.moves_remaining = self.UNIT_MOVES.get(
                    unit.unit_type, UNITS.WARRIOR_MOVES
                )

            # Check for settler -> city conversion
            self._process_settlers()

    def _process_settlers(self) -> None:
        """Process settlers that should found cities."""
        settlers_to_remove = []
        for unit in self.units:
            if unit.unit_type == "settler":
                # Auto-found city if on good terrain and no city nearby
                has_nearby_city = any(
                    abs(c.position[0] - unit.position[0]) <= GAME.MIN_CITY_DISTANCE
                    and abs(c.position[1] - unit.position[1]) <= GAME.MIN_CITY_DISTANCE
                    for c in self.cities
                )
                if not has_nearby_city and self._rng.random() < GAME.SETTLER_AUTO_FOUND_PROBABILITY:
                    # Found city
                    city_name = f"City {self._next_city_id}"
                    self.cities.append(
                        City(
                            id=self._next_city_id,
                            player=unit.player,
                            name=city_name,
                            position=unit.position,
                        ),
                    )
                    self._next_city_id += 1
                    settlers_to_remove.append(unit)

        for settler in settlers_to_remove:
            self.units.remove(settler)

    def _check_victory(self) -> bool:
        """Check if game has ended.

        A player is eliminated when they have 0 cities AND 0 units.
        If exactly 1 player remains, they win.

        Returns:
            True if game is over.
        """
        # Find all surviving players (those with at least one city or unit)
        surviving_players: list[int] = []
        for p in range(self.num_players):
            p_cities = sum(1 for c in self.cities if c.player == p)
            p_units = sum(1 for u in self.units if u.player == p)
            if p_cities > 0 or p_units > 0:
                surviving_players.append(p)

        # If exactly one player survives, they win
        if len(surviving_players) == 1:
            self.winner = surviving_players[0]
            return True

        # If no players survive (edge case), it's a draw
        if len(surviving_players) == 0:
            self.winner = None
            return True

        return False

    def _get_player_units(self, player: int) -> list[Unit]:
        """Get units belonging to player."""
        return [u for u in self.units if u.player == player]

    def _get_player_cities(self, player: int) -> list[City]:
        """Get cities belonging to player."""
        return [c for c in self.cities if c.player == player]

    def _is_valid_move(self, pos: tuple[int, int]) -> bool:
        """Check if position is valid for movement."""
        x, y = pos
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return False
        if self.terrain.get(pos, Tile()).terrain_type == "mountain":
            return False
        # Check for unit collision
        for unit in self.units:
            if unit.position == pos:
                return False
        return True

    def _get_game_state(self) -> dict:
        """Get current game state as dictionary."""
        return {
            "terrain": {pos: tile.to_dict() for pos, tile in self.terrain.items()},
            "units": [u.to_dict() for u in self.units],
            "cities": [c.to_dict() for c in self.cities],
            "turn": self.turn,
            "current_player": self.current_player,
            "num_players": self.num_players,
        }

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get observation for current player."""
        game_state = self._get_game_state()
        spatial, vector = self.encoder.encode(game_state)
        action_mask = self.action_space_handler.get_valid_mask(game_state)

        return {
            "spatial": spatial.numpy(),
            "vector": vector.numpy(),
            "action_mask": action_mask,
        }

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Get observation for current player (deprecated).

        .. deprecated::
            Use :meth:`get_observation` instead. This method will be removed
            in a future version.

        Returns:
            Observation dictionary with spatial, vector, and action_mask.
        """
        import warnings

        warnings.warn(
            "_get_observation() is deprecated, use get_observation() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_observation()

    def get_info(self) -> dict[str, Any]:
        """Get info dictionary."""
        return {
            "turn": self.turn,
            "current_player": self.current_player,
            "winner": self.winner,
            "num_players": self.num_players,
            "num_units": [
                len(self._get_player_units(p)) for p in range(self.num_players)
            ],
            "num_cities": [
                len(self._get_player_cities(p)) for p in range(self.num_players)
            ],
            "visited_count": len(self.visited),
        }

    def _get_info(self) -> dict[str, Any]:
        """Get info dictionary (deprecated).

        .. deprecated::
            Use :meth:`get_info` instead. This method will be removed
            in a future version.

        Returns:
            Info dictionary with game state details.
        """
        import warnings

        warnings.warn(
            "_get_info() is deprecated, use get_info() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_info()

    def get_valid_actions(self) -> np.ndarray:
        """Get valid action mask.

        Returns:
            Boolean mask of shape (action_space_size,).
        """
        game_state = self._get_game_state()
        return self.action_space_handler.get_valid_mask(game_state)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Render the environment.

        Returns:
            String representation if render_mode is 'ansi'.
        """
        if self.render_mode != "ansi":
            return None

        lines = [f"Turn {self.turn}, Player {self.current_player}'s turn"]
        lines.append("=" * (self.board_size * 2 + 1))

        # Create grid
        grid = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]

        # Mark terrain
        for (x, y), tile in self.terrain.items():
            if tile.terrain_type == "mountain":
                grid[y][x] = "^"
            elif tile.terrain_type == "hills":
                grid[y][x] = "~"

        # Mark cities
        for city in self.cities:
            x, y = city.position
            symbols = self.PLAYER_SYMBOLS["city"]
            grid[y][x] = symbols[city.player % len(symbols)]

        # Mark units
        for unit in self.units:
            x, y = unit.position
            unit_symbols = self.PLAYER_SYMBOLS.get(unit.unit_type)
            if unit_symbols:
                grid[y][x] = unit_symbols[unit.player % len(unit_symbols)]

        for row in grid:
            lines.append(" ".join(row))

        lines.append("=" * (self.board_size * 2 + 1))
        for p in range(self.num_players):
            lines.append(
                f"P{p}: {len(self._get_player_units(p))} units, "
                f"{len(self._get_player_cities(p))} cities"
            )

        return "\n".join(lines)  # type: ignore[return-value]

    def clone(self) -> MicroCivEnv:
        """Create a deep copy of the environment.

        Returns:
            Cloned environment.
        """
        import copy

        new_env = MicroCivEnv(
            board_size=self.board_size,
            max_turns=self.max_turns,
            render_mode=self.render_mode,
            num_players=self.num_players,
        )
        new_env.terrain = copy.deepcopy(self.terrain)
        new_env.units = copy.deepcopy(self.units)
        new_env.cities = copy.deepcopy(self.cities)
        new_env.turn = self.turn
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        new_env._next_unit_id = self._next_unit_id
        new_env._next_city_id = self._next_city_id
        new_env.visited = copy.deepcopy(self.visited)
        return new_env
