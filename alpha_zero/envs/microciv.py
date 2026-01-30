"""MicroCiv: Simplified Civilization environment for Phase 1 training.

A minimal 4X game that captures core Civ mechanics:
- Explore: Fog of war, terrain discovery
- Expand: Settlers found new cities
- Exploit: Cities produce units
- Exterminate: Combat between units

Victory: Eliminate all enemy cities, or reach turn limit (most cities wins).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame

from alpha_zero.core.civ6_actions import MicroCivActionSpace
from alpha_zero.envs.civ6_encoder import MicroCivEncoder


@dataclass
class Unit:
    """A unit in the game."""

    id: int
    player: int
    unit_type: str
    position: tuple[int, int]
    health: int = 100
    moves_remaining: int = 2
    strength: int = 20

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
    defense: int = 20

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
    - 2 players (turn-based)
    - 16x16 tile map
    - 3 unit types: warrior, settler, archer
    - Simple combat and city founding

    Observation space:
    - Spatial: (13, board_size, board_size) tensor
    - Vector: (64,) tensor

    Action space:
    - Flat discrete space covering all possible actions
    """

    metadata = {"render_modes": ["ansi"]}

    # Unit production costs (turns)
    UNIT_COSTS = {"warrior": 3, "settler": 5, "archer": 4}
    UNIT_STRENGTH = {"warrior": 20, "settler": 0, "archer": 15}
    UNIT_MOVES = {"warrior": 2, "settler": 2, "archer": 2}

    def __init__(
        self,
        board_size: int = 16,
        max_turns: int = 200,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        """Initialize MicroCiv environment.

        Args:
            board_size: Size of the square game board.
            max_turns: Maximum turns before draw.
            render_mode: Rendering mode ('ansi' or None).
            seed: Random seed for reproducibility.
        """
        super().__init__()

        self.board_size = board_size
        self.max_turns = max_turns
        self.render_mode = render_mode

        # Initialize RNG
        self._rng = np.random.default_rng(seed)

        # Action and observation spaces
        self.action_space_handler = MicroCivActionSpace(
            board_size=board_size, max_units=20, max_cities=5,
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
                    low=-1.0, high=1.0, shape=(64,), dtype=np.float32,
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
        self, action: int,
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
        reward = self._execute_action(action_dict)

        # Check victory conditions
        terminated = self._check_victory()
        truncated = self.turn >= self.max_turns

        if truncated and not terminated:
            # Count cities for each player
            p0_cities = sum(1 for c in self.cities if c.player == 0)
            p1_cities = sum(1 for c in self.cities if c.player == 1)
            if p0_cities > p1_cities:
                self.winner = 0
            elif p1_cities > p0_cities:
                self.winner = 1
            else:
                self.winner = None  # Draw
            terminated = True

        if terminated:
            self.done = True
            # Final reward based on winner
            if self.winner == self.current_player:
                reward += 1.0
            elif self.winner is not None:
                reward -= 1.0

        obs = self.get_observation()
        info = self.get_info()

        return obs, reward, terminated, truncated, info

    def _generate_map(self) -> None:
        """Generate random terrain map."""
        self.terrain = {}
        for y in range(self.board_size):
            for x in range(self.board_size):
                # Random terrain type
                r = self._rng.random()
                if r < 0.7:
                    terrain_type = "plains"
                    yields = {"food": 2, "production": 1}
                elif r < 0.9:
                    terrain_type = "hills"
                    yields = {"food": 1, "production": 2}
                else:
                    terrain_type = "mountain"
                    yields = {"food": 0, "production": 0}

                self.terrain[(x, y)] = Tile(terrain_type=terrain_type, yields=yields)

    def _place_starting_units(self) -> None:
        """Place starting units for each player."""
        self.units = []
        self.cities = []
        self._next_unit_id = 0
        self._next_city_id = 0

        # Player 0 starts in bottom-left area
        p0_start = self._find_valid_position(2, 2, 5)
        # Player 1 starts in top-right area
        p1_start = self._find_valid_position(
            self.board_size - 5, self.board_size - 5, 5,
        )

        # Create starting units
        for player, pos in enumerate([p0_start, p1_start]):
            # Settler
            self.units.append(
                Unit(
                    id=self._next_unit_id,
                    player=player,
                    unit_type="settler",
                    position=pos,
                    strength=0,
                    moves_remaining=2,
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
                    strength=20,
                    moves_remaining=2,
                ),
            )
            self.visited.add(warrior_pos)  # Mark warrior pos as visited
            self._next_unit_id += 1

    def _find_valid_position(
        self, start_x: int, start_y: int, range_: int,
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

    def _execute_action(self, action: dict) -> float:
        """Execute an action and return immediate reward.

        Args:
            action: Action dictionary.

        Returns:
            Immediate reward (shaped).
        """
        action_type = action.get("type")
        reward = 0.0

        if action_type == "end_turn":
            self._end_turn()

        elif action_type == "move":
            unit_idx = action.get("unit_idx", 0)
            direction = action.get("direction", (0, 0))
            own_units = self._get_player_units(self.current_player)
            if unit_idx < len(own_units):
                unit = own_units[unit_idx]
                if unit.moves_remaining > 0:
                    new_pos = (
                        unit.position[0] + direction[0],
                        unit.position[1] + direction[1],
                    )
                    if self._is_valid_move(new_pos):
                        unit.position = new_pos
                        if unit.player == self.current_player:
                            self.visited.add(new_pos)  # Track visitation
                        unit.moves_remaining -= 1
                    else:
                        # Move failed, treat as end of unit's actions to prevent lock
                        unit.moves_remaining = 0

        elif action_type == "attack":
            unit_idx = action.get("unit_idx", 0)
            direction = action.get("direction", (0, 0))
            own_units = self._get_player_units(self.current_player)
            if unit_idx < len(own_units):
                unit = own_units[unit_idx]
                if unit.moves_remaining > 0:
                    target_pos = (
                        unit.position[0] + direction[0],
                        unit.position[1] + direction[1],
                    )
                    reward += self._execute_attack(unit, target_pos)
                    unit.moves_remaining -= 1

        elif action_type == "skip":
            unit_idx = action.get("unit_idx", 0)
            own_units = self._get_player_units(self.current_player)
            if unit_idx < len(own_units):
                own_units[unit_idx].moves_remaining = 0

        elif action_type == "found_city":
            unit_idx = action.get("unit_idx", 0)
            own_units = self._get_player_units(self.current_player)
            if unit_idx < len(own_units):
                unit = own_units[unit_idx]
                if unit.unit_type == "settler":
                    # Check for nearby cities
                    has_nearby_city = any(
                        abs(c.position[0] - unit.position[0]) <= 3
                        and abs(c.position[1] - unit.position[1]) <= 3
                        for c in self.cities
                    )
                    if not has_nearby_city:
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
                        reward += 1.0

                        # Remove settler
                        if unit in self.units:
                            self.units.remove(unit)

        elif action_type == "produce":
            city_idx = action.get("city_idx", 0)
            unit_type = action.get("unit_type", "warrior")
            own_cities = self._get_player_cities(self.current_player)
            if city_idx < len(own_cities):
                city = own_cities[city_idx]
                if city.production_queue is None:
                    city.production_queue = unit_type
                    reward += 0.01  # Small reward for starting production

        return reward

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
        attacker_str = attacker.strength * (attacker.health / 100.0)
        defender_str = defender.strength * (defender.health / 100.0)

        # Damage calculation
        if attacker_str + defender_str > 0:
            attacker_ratio = attacker_str / (attacker_str + defender_str + 0.1)
            damage_to_defender = int(30 * attacker_ratio + 10)
            damage_to_attacker = int(30 * (1 - attacker_ratio) + 5)

            defender.health -= damage_to_defender
            attacker.health -= damage_to_attacker

            # Remove dead units
            if defender.health <= 0:
                self.units.remove(defender)
                reward += 0.1  # Reward for killing enemy unit

            if attacker.health <= 0:
                self.units.remove(attacker)
                reward -= 0.1  # Penalty for losing unit

        return reward

    def _end_turn(self) -> None:
        """End current player's turn."""
        # Process cities (production)
        for city in self._get_player_cities(self.current_player):
            if city.production_queue:
                city.production += city.population + 2  # Base production
                cost = self.UNIT_COSTS.get(city.production_queue, 5) * 10
                if city.production >= cost:
                    # Spawn unit
                    spawn_pos = self._find_adjacent_valid(city.position)
                    self.units.append(
                        Unit(
                            id=self._next_unit_id,
                            player=city.player,
                            unit_type=city.production_queue,
                            position=spawn_pos,
                            strength=self.UNIT_STRENGTH.get(city.production_queue, 20),
                            moves_remaining=0,
                        ),
                    )
                    self._next_unit_id += 1
                    city.production = 0
                    city.production_queue = None

        # Switch player
        self.current_player = 1 - self.current_player

        # If back to player 0, increment turn
        if self.current_player == 0:
            self.turn += 1
            # Reset all units' moves
            for unit in self.units:
                unit.moves_remaining = self.UNIT_MOVES.get(unit.unit_type, 2)

            # Check for settler -> city conversion
            self._process_settlers()

    def _process_settlers(self) -> None:
        """Process settlers that should found cities."""
        settlers_to_remove = []
        for unit in self.units:
            if unit.unit_type == "settler":
                # Auto-found city if on good terrain and no city nearby
                has_nearby_city = any(
                    abs(c.position[0] - unit.position[0]) <= 3
                    and abs(c.position[1] - unit.position[1]) <= 3
                    for c in self.cities
                )
                if not has_nearby_city and self._rng.random() < 0.1:
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

        Returns:
            True if game is over.
        """
        # Check for elimination
        p0_cities = sum(1 for c in self.cities if c.player == 0)
        p1_cities = sum(1 for c in self.cities if c.player == 1)
        p0_units = sum(1 for u in self.units if u.player == 0)
        p1_units = sum(1 for u in self.units if u.player == 1)

        # Player 0 eliminated
        if p0_cities == 0 and p0_units == 0:
            self.winner = 1
            return True

        # Player 1 eliminated
        if p1_cities == 0 and p1_units == 0:
            self.winner = 0
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
            "num_players": 2,
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
            "num_units": [
                len(self._get_player_units(0)),
                len(self._get_player_units(1)),
            ],
            "num_cities": [
                len(self._get_player_cities(0)),
                len(self._get_player_cities(1)),
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
            grid[y][x] = "C" if city.player == 0 else "c"

        # Mark units
        for unit in self.units:
            x, y = unit.position
            if unit.unit_type == "warrior":
                grid[y][x] = "W" if unit.player == 0 else "w"
            elif unit.unit_type == "settler":
                grid[y][x] = "S" if unit.player == 0 else "s"
            elif unit.unit_type == "archer":
                grid[y][x] = "A" if unit.player == 0 else "a"

        for row in grid:
            lines.append(" ".join(row))

        lines.append("=" * (self.board_size * 2 + 1))
        lines.append(f"P0: {len(self._get_player_units(0))} units, "
                    f"{len(self._get_player_cities(0))} cities")
        lines.append(f"P1: {len(self._get_player_units(1))} units, "
                    f"{len(self._get_player_cities(1))} cities")

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
