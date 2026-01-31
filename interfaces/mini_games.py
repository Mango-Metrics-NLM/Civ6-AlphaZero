"""Mini-Game Curriculum Training Scenarios.

Provides simplified game scenarios for curriculum learning.
Starting with simple scenarios and progressively increasing complexity
helps the neural network learn fundamental concepts before tackling
the full game.

Curriculum Stages:
    1. Combat: Simple 1v1 and small-scale combat
    2. City Development: Resource management and production
    3. Expansion: Settler placement and territory control
    4. Full Game: Complete game with all mechanics
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .base import (
    ActionType,
    City,
    GameAction,
    GameState,
    Position,
    Tile,
    Unit,
)

logger = logging.getLogger(__name__)


class MiniGameType(Enum):
    """Types of mini-games."""
    COMBAT_1V1 = "combat_1v1"
    COMBAT_SMALL = "combat_small"
    COMBAT_SIEGE = "combat_siege"
    CITY_GROWTH = "city_growth"
    CITY_PRODUCTION = "city_production"
    CITY_DEFENSE = "city_defense"
    EXPANSION_EARLY = "expansion_early"
    EXPANSION_CONTEST = "expansion_contest"
    EXPLORATION = "exploration"


@dataclass
class MiniGameConfig:
    """Configuration for mini-game scenarios."""
    game_type: MiniGameType
    board_size: int = 8
    max_turns: int = 50
    num_players: int = 2
    difficulty: float = 0.5  # 0.0 = easy, 1.0 = hard
    random_seed: int | None = None
    custom_params: dict[str, Any] = field(default_factory=dict)

    def with_difficulty(self, difficulty: float) -> MiniGameConfig:
        """Create copy with different difficulty."""
        return MiniGameConfig(
            game_type=self.game_type,
            board_size=self.board_size,
            max_turns=self.max_turns,
            num_players=self.num_players,
            difficulty=difficulty,
            random_seed=self.random_seed,
            custom_params=self.custom_params.copy(),
        )


class MiniGame(ABC):
    """Abstract base class for mini-game scenarios."""

    def __init__(self, config: MiniGameConfig):
        """Initialize mini-game.

        Args:
            config: Game configuration.
        """
        self.config = config
        self._rng = random.Random(config.random_seed)
        self._state: GameState | None = None
        self._turn = 0
        self._done = False
        self._winner: int | None = None

    @property
    def state(self) -> GameState:
        """Current game state."""
        if self._state is None:
            self._state = self.generate_initial_state()
        return self._state

    @property
    def is_done(self) -> bool:
        """Check if game is finished."""
        return self._done

    @property
    def winner(self) -> int | None:
        """Get winner player ID."""
        return self._winner

    @abstractmethod
    def generate_initial_state(self) -> GameState:
        """Generate the initial game state for this scenario.

        Returns:
            Initial GameState.
        """
        pass

    @abstractmethod
    def get_reward(self, player_id: int) -> float:
        """Get current reward for player.

        Args:
            player_id: Player to get reward for.

        Returns:
            Reward value (typically -1 to 1).
        """
        pass

    @abstractmethod
    def check_victory(self) -> int | None:
        """Check if a player has won.

        Returns:
            Winner player ID or None.
        """
        pass

    def step(self, action: GameAction) -> tuple[GameState, float, bool]:
        """Apply action and advance game state.

        Args:
            action: Action to apply.

        Returns:
            Tuple of (new_state, reward, done).
        """
        if self._done:
            return self.state, 0.0, True

        # Apply action
        self._apply_action(action)

        # Check turn end
        if action.action_type == ActionType.END_TURN:
            self._advance_turn()

        # Check victory
        winner = self.check_victory()
        if winner is not None:
            self._done = True
            self._winner = winner

        # Check turn limit
        if self._turn >= self.config.max_turns:
            self._done = True

        # Get reward
        reward = self.get_reward(self.state.current_player)

        return self.state, reward, self._done

    def reset(self, seed: int | None = None) -> GameState:
        """Reset the game to initial state.

        Args:
            seed: Optional random seed.

        Returns:
            Initial state.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        self._state = self.generate_initial_state()
        self._turn = 0
        self._done = False
        self._winner = None
        return self._state

    def get_valid_actions(self) -> list[GameAction]:
        """Get list of valid actions in current state.

        Returns:
            List of valid actions.
        """
        actions: list[GameAction] = []

        # End turn
        actions.append(GameAction.end_turn())

        # Unit actions
        for unit in self.state.get_own_units():
            actions.append(GameAction.skip(unit.id))

            if unit.can_move:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        target = Position(
                            x=unit.position.x + dx,
                            y=unit.position.y + dy
                        )
                        if self._is_valid_move(unit, target):
                            actions.append(GameAction.move(unit.id, target))

            if unit.can_attack:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        target = Position(
                            x=unit.position.x + dx,
                            y=unit.position.y + dy
                        )
                        if self._is_valid_attack(unit, target):
                            actions.append(GameAction.attack(unit.id, target))

        # City actions
        for city in self.state.get_own_cities():
            if not city.producing:
                for unit_type in self._get_producible_units():
                    actions.append(GameAction.produce(city.id, unit_type))

        return actions

    def _apply_action(self, action: GameAction) -> None:
        """Apply action to current state."""
        if action.action_type == ActionType.MOVE:
            self._apply_move(action)
        elif action.action_type == ActionType.ATTACK:
            self._apply_attack(action)
        elif action.action_type == ActionType.SKIP:
            self._apply_skip(action)
        elif action.action_type == ActionType.PRODUCE:
            self._apply_produce(action)

    def _apply_move(self, action: GameAction) -> None:
        """Apply move action."""
        unit = self.state.get_unit_by_id(action.unit_id or "")
        if unit and action.target_position:
            unit.position = action.target_position
            unit.movement = max(0, unit.movement - 1)
            unit.can_move = unit.movement > 0

    def _apply_attack(self, action: GameAction) -> None:
        """Apply attack action."""
        attacker = self.state.get_unit_by_id(action.unit_id or "")
        if not attacker or not action.target_position:
            return

        # Find defender
        defenders = self.state.get_units_at(action.target_position)
        defender = next(
            (u for u in defenders if u.player != attacker.player),
            None
        )

        if defender:
            # Calculate damage
            attack_strength = attacker.strength
            defense_strength = defender.strength

            # Simple combat formula
            damage_to_defender = int(30 * attack_strength / max(defense_strength, 1))
            damage_to_attacker = int(20 * defense_strength / max(attack_strength, 1))

            defender.health -= damage_to_defender
            attacker.health -= damage_to_attacker

            # Remove destroyed units
            if defender.health <= 0:
                self.state.units.remove(defender)
            if attacker.health <= 0:
                self.state.units.remove(attacker)

        attacker.can_attack = False
        attacker.can_move = False

    def _apply_skip(self, action: GameAction) -> None:
        """Apply skip action."""
        unit = self.state.get_unit_by_id(action.unit_id or "")
        if unit:
            unit.movement = 0
            unit.can_move = False

    def _apply_produce(self, action: GameAction) -> None:
        """Apply produce action."""
        city = self.state.get_city_by_id(action.city_id or "")
        if city:
            city.producing = action.production_type

    def _advance_turn(self) -> None:
        """Advance to next turn."""
        # Switch player
        self.state.current_player = (
            self.state.current_player + 1
        ) % self.state.num_players

        if self.state.current_player == 0:
            self._turn += 1
            self.state.turn = self._turn

            # Process production
            self._process_production()

            # Reset unit movement
            for unit in self.state.units:
                unit.movement = unit.extra.get("maxMoves", 2)
                unit.can_move = True
                unit.can_attack = True

    def _process_production(self) -> None:
        """Process city production."""
        for city in self.state.cities:
            if city.producing:
                city.production += city.population + 2

                # Check if complete (simplified)
                production_cost = self._get_production_cost(city.producing)
                if city.production >= production_cost:
                    # Create unit
                    unit_id = f"unit_{len(self.state.units)}"
                    new_unit = Unit(
                        id=unit_id,
                        type=city.producing,
                        player=city.player,
                        position=Position(city.position.x, city.position.y),
                        health=100,
                        movement=2,
                        strength=self._get_unit_strength(city.producing),
                        can_attack=True,
                        can_move=True,
                        extra={"maxMoves": 2},
                    )
                    self.state.units.append(new_unit)
                    city.production = 0
                    city.producing = None

    def _is_valid_move(self, unit: Unit, target: Position) -> bool:
        """Check if move is valid."""
        # Check bounds
        if not (0 <= target.x < self.config.board_size and
                0 <= target.y < self.config.board_size):
            return False

        # Check terrain
        tile = self.state.terrain.get(target.to_tuple())
        if tile and tile.terrain == "TERRAIN_MOUNTAIN":
            return False

        # Check for friendly units
        units_at_target = self.state.get_units_at(target)
        for u in units_at_target:
            if u.player == unit.player:
                return False

        return True

    def _is_valid_attack(self, unit: Unit, target: Position) -> bool:
        """Check if attack is valid."""
        # Check for enemy units
        units_at_target = self.state.get_units_at(target)
        for u in units_at_target:
            if u.player != unit.player:
                return True
        return False

    def _get_producible_units(self) -> list[str]:
        """Get list of units that can be produced."""
        return ["warrior", "archer", "settler"]

    def _get_production_cost(self, unit_type: str) -> int:
        """Get production cost for unit type."""
        costs = {
            "warrior": 40,
            "archer": 50,
            "settler": 80,
        }
        return costs.get(unit_type, 40)

    def _get_unit_strength(self, unit_type: str) -> int:
        """Get combat strength for unit type."""
        strengths = {
            "warrior": 20,
            "archer": 15,
            "settler": 0,
        }
        return strengths.get(unit_type, 20)

    def _generate_terrain(self) -> dict[tuple[int, int], Tile]:
        """Generate terrain for the mini-game."""
        terrain = {}
        for x in range(self.config.board_size):
            for y in range(self.config.board_size):
                terrain_type = "TERRAIN_PLAINS"
                if self._rng.random() < 0.15:
                    terrain_type = "TERRAIN_HILLS"
                elif self._rng.random() < 0.05:
                    terrain_type = "TERRAIN_MOUNTAIN"

                terrain[(x, y)] = Tile(
                    position=Position(x=x, y=y),
                    terrain=terrain_type,
                    yields={
                        "food": 2 if terrain_type == "TERRAIN_PLAINS" else 1,
                        "production": 2 if terrain_type == "TERRAIN_HILLS" else 1,
                    }
                )
        return terrain


class CombatMiniGame(MiniGame):
    """Combat-focused mini-game scenarios.

    Trains the network on combat tactics including:
    - Unit positioning and movement
    - Attack/defense trade-offs
    - Focus fire and target priority
    """

    def generate_initial_state(self) -> GameState:
        """Generate combat scenario."""
        state = GameState(
            turn=0,
            current_player=0,
            num_players=2,
            map_width=self.config.board_size,
            map_height=self.config.board_size,
        )

        state.terrain = self._generate_terrain()

        # Determine unit count based on difficulty
        if self.config.game_type == MiniGameType.COMBAT_1V1:
            p0_units = 1
            p1_units = 1
        elif self.config.game_type == MiniGameType.COMBAT_SMALL:
            p0_units = 2 + int(self.config.difficulty * 2)
            p1_units = 2 + int(self.config.difficulty * 2)
        else:
            p0_units = 3 + int(self.config.difficulty * 3)
            p1_units = 3 + int(self.config.difficulty * 3)

        # Place player 0 units on left
        for i in range(p0_units):
            pos = Position(
                x=1,
                y=self.config.board_size // 2 - p0_units // 2 + i
            )
            state.units.append(Unit(
                id=f"p0_unit_{i}",
                type="warrior",
                player=0,
                position=pos,
                health=100,
                movement=2,
                strength=20,
                extra={"maxMoves": 2},
            ))

        # Place player 1 units on right
        for i in range(p1_units):
            pos = Position(
                x=self.config.board_size - 2,
                y=self.config.board_size // 2 - p1_units // 2 + i
            )
            state.units.append(Unit(
                id=f"p1_unit_{i}",
                type="warrior",
                player=1,
                position=pos,
                health=100,
                movement=2,
                strength=20,
                extra={"maxMoves": 2},
            ))

        return state

    def get_reward(self, player_id: int) -> float:
        """Get reward based on combat state."""
        own_units = [u for u in self.state.units if u.player == player_id]
        enemy_units = [u for u in self.state.units if u.player != player_id]

        if not enemy_units:
            return 1.0  # Victory
        if not own_units:
            return -1.0  # Defeat

        # Intermediate reward based on health advantage
        own_health = sum(u.health for u in own_units)
        enemy_health = sum(u.health for u in enemy_units)
        total_health = own_health + enemy_health

        if total_health > 0:
            return (own_health - enemy_health) / total_health * 0.1

        return 0.0

    def check_victory(self) -> int | None:
        """Check if one side eliminated."""
        p0_units = [u for u in self.state.units if u.player == 0]
        p1_units = [u for u in self.state.units if u.player == 1]

        if not p1_units:
            return 0
        if not p0_units:
            return 1

        return None


class CityDevelopmentMiniGame(MiniGame):
    """City development mini-game.

    Trains the network on:
    - Production choices
    - Growth vs military trade-offs
    - City placement
    """

    def generate_initial_state(self) -> GameState:
        """Generate city development scenario."""
        state = GameState(
            turn=0,
            current_player=0,
            num_players=2,
            map_width=self.config.board_size,
            map_height=self.config.board_size,
        )

        state.terrain = self._generate_terrain()

        # Each player gets a city
        for player_id in range(2):
            city_x = 2 if player_id == 0 else self.config.board_size - 3
            city_y = self.config.board_size // 2

            state.cities.append(City(
                id=f"city_{player_id}",
                name=f"Player {player_id} Capital",
                player=player_id,
                position=Position(x=city_x, y=city_y),
                population=1,
                production=0,
                defense=20,
            ))

            # Starting warrior
            state.units.append(Unit(
                id=f"warrior_{player_id}",
                type="warrior",
                player=player_id,
                position=Position(x=city_x + 1, y=city_y),
                health=100,
                movement=2,
                strength=20,
                extra={"maxMoves": 2},
            ))

        return state

    def get_reward(self, player_id: int) -> float:
        """Get reward based on development."""
        own_cities = [c for c in self.state.cities if c.player == player_id]
        own_units = [u for u in self.state.units if u.player == player_id]

        if not own_cities:
            return -1.0  # Lost capital

        enemy_cities = [c for c in self.state.cities if c.player != player_id]
        if not enemy_cities:
            return 1.0  # Conquered enemy

        # Score based on development
        own_pop = sum(c.population for c in own_cities)
        own_military = len(own_units)

        enemy_pop = sum(c.population for c in enemy_cities)
        enemy_military = len([u for u in self.state.units if u.player != player_id])

        score = (own_pop + own_military) - (enemy_pop + enemy_military)
        return max(-1.0, min(1.0, score * 0.1))

    def check_victory(self) -> int | None:
        """Check if one side lost all cities."""
        p0_cities = [c for c in self.state.cities if c.player == 0]
        p1_cities = [c for c in self.state.cities if c.player == 1]

        if not p1_cities:
            return 0
        if not p0_cities:
            return 1

        return None


class ExpansionMiniGame(MiniGame):
    """Expansion and territory control mini-game.

    Trains the network on:
    - Settler placement
    - Territory control
    - Resource acquisition
    """

    def generate_initial_state(self) -> GameState:
        """Generate expansion scenario."""
        state = GameState(
            turn=0,
            current_player=0,
            num_players=2,
            map_width=self.config.board_size,
            map_height=self.config.board_size,
        )

        state.terrain = self._generate_terrain()

        # Place starting settlers
        for player_id in range(2):
            settler_x = 1 if player_id == 0 else self.config.board_size - 2
            settler_y = self.config.board_size // 2

            state.units.append(Unit(
                id=f"settler_{player_id}",
                type="settler",
                player=player_id,
                position=Position(x=settler_x, y=settler_y),
                health=100,
                movement=2,
                strength=0,
                extra={"maxMoves": 2},
            ))

            # Escort warrior
            state.units.append(Unit(
                id=f"warrior_{player_id}",
                type="warrior",
                player=player_id,
                position=Position(x=settler_x, y=settler_y + 1),
                health=100,
                movement=2,
                strength=20,
                extra={"maxMoves": 2},
            ))

        return state

    def get_valid_actions(self) -> list[GameAction]:
        """Get valid actions including settle."""
        actions = super().get_valid_actions()

        # Add settle actions for settlers
        for unit in self.state.get_own_units():
            if "settler" in unit.type.lower():
                # Can settle if not on or adjacent to existing city
                can_settle = True
                for city in self.state.cities:
                    if unit.position.distance_to(city.position) < 4:
                        can_settle = False
                        break

                if can_settle:
                    actions.append(GameAction(
                        action_type=ActionType.SETTLE,
                        unit_id=unit.id,
                    ))

        return actions

    def _apply_action(self, action: GameAction) -> None:
        """Apply action including settle."""
        if action.action_type == ActionType.SETTLE:
            self._apply_settle(action)
        else:
            super()._apply_action(action)

    def _apply_settle(self, action: GameAction) -> None:
        """Apply settle action."""
        settler = self.state.get_unit_by_id(action.unit_id or "")
        if not settler:
            return

        # Create city
        city_id = f"city_{len(self.state.cities)}"
        new_city = City(
            id=city_id,
            name=f"City {len(self.state.cities) + 1}",
            player=settler.player,
            position=Position(settler.position.x, settler.position.y),
            population=1,
            production=0,
            defense=20,
        )
        self.state.cities.append(new_city)

        # Remove settler
        self.state.units.remove(settler)

    def get_reward(self, player_id: int) -> float:
        """Get reward based on expansion."""
        own_cities = [c for c in self.state.cities if c.player == player_id]
        enemy_cities = [c for c in self.state.cities if c.player != player_id]

        # Reward for more cities
        city_diff = len(own_cities) - len(enemy_cities)

        # Bonus for total cities
        total_bonus = len(own_cities) * 0.1

        return max(-1.0, min(1.0, city_diff * 0.3 + total_bonus))

    def check_victory(self) -> int | None:
        """Check victory by city count or elimination."""
        p0_cities = [c for c in self.state.cities if c.player == 0]
        p1_cities = [c for c in self.state.cities if c.player == 1]

        # Win by having more cities after turn limit
        if self._turn >= self.config.max_turns:
            if len(p0_cities) > len(p1_cities):
                return 0
            elif len(p1_cities) > len(p0_cities):
                return 1
            # Tie goes to player 0 (first player disadvantage compensation)
            return 0

        return None


class MiniGameGenerator:
    """Factory for creating mini-game scenarios.

    Example:
        generator = MiniGameGenerator()

        # Generate curriculum of increasing difficulty
        for stage in generator.generate_curriculum():
            for game in stage:
                # Train on game...
                state = game.reset()
                while not game.is_done:
                    action = agent.select_action(state)
                    state, reward, done = game.step(action)
    """

    def __init__(
        self,
        base_config: MiniGameConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize generator.

        Args:
            base_config: Base configuration for generated games.
            seed: Random seed for reproducibility.
        """
        self.base_config = base_config or MiniGameConfig(
            game_type=MiniGameType.COMBAT_1V1,
            board_size=8,
            max_turns=50,
        )
        self._rng = random.Random(seed)

    def create_game(
        self,
        game_type: MiniGameType,
        difficulty: float = 0.5,
        **kwargs: Any,
    ) -> MiniGame:
        """Create a mini-game of the specified type.

        Args:
            game_type: Type of mini-game to create.
            difficulty: Difficulty level (0.0 to 1.0).
            **kwargs: Additional configuration parameters.

        Returns:
            MiniGame instance.
        """
        config = MiniGameConfig(
            game_type=game_type,
            board_size=kwargs.get("board_size", self.base_config.board_size),
            max_turns=kwargs.get("max_turns", self.base_config.max_turns),
            num_players=kwargs.get("num_players", 2),
            difficulty=difficulty,
            random_seed=kwargs.get("seed", self._rng.randint(0, 2**32 - 1)),
            custom_params=kwargs,
        )

        if game_type in [
            MiniGameType.COMBAT_1V1,
            MiniGameType.COMBAT_SMALL,
            MiniGameType.COMBAT_SIEGE,
        ]:
            return CombatMiniGame(config)
        elif game_type in [
            MiniGameType.CITY_GROWTH,
            MiniGameType.CITY_PRODUCTION,
            MiniGameType.CITY_DEFENSE,
        ]:
            return CityDevelopmentMiniGame(config)
        elif game_type in [
            MiniGameType.EXPANSION_EARLY,
            MiniGameType.EXPANSION_CONTEST,
            MiniGameType.EXPLORATION,
        ]:
            return ExpansionMiniGame(config)
        else:
            raise ValueError(f"Unknown game type: {game_type}")

    def generate_curriculum(
        self,
        stages: int = 4,
        games_per_stage: int = 100,
    ) -> list[list[MiniGame]]:
        """Generate a curriculum of mini-games.

        Args:
            stages: Number of curriculum stages.
            games_per_stage: Number of games per stage.

        Returns:
            List of game lists, one per stage.
        """
        curriculum: list[list[MiniGame]] = []

        # Stage 1: Simple 1v1 combat
        stage1 = [
            self.create_game(
                MiniGameType.COMBAT_1V1,
                difficulty=i / games_per_stage * 0.5,
            )
            for i in range(games_per_stage)
        ]
        curriculum.append(stage1)

        if stages >= 2:
            # Stage 2: Small combat scenarios
            stage2 = [
                self.create_game(
                    MiniGameType.COMBAT_SMALL,
                    difficulty=0.3 + i / games_per_stage * 0.4,
                )
                for i in range(games_per_stage)
            ]
            curriculum.append(stage2)

        if stages >= 3:
            # Stage 3: City development
            stage3 = [
                self.create_game(
                    MiniGameType.CITY_PRODUCTION,
                    difficulty=0.4 + i / games_per_stage * 0.3,
                )
                for i in range(games_per_stage)
            ]
            curriculum.append(stage3)

        if stages >= 4:
            # Stage 4: Expansion
            stage4 = [
                self.create_game(
                    MiniGameType.EXPANSION_EARLY,
                    difficulty=0.5 + i / games_per_stage * 0.3,
                )
                for i in range(games_per_stage)
            ]
            curriculum.append(stage4)

        return curriculum

    def create_random_game(self, difficulty: float = 0.5) -> MiniGame:
        """Create a random mini-game.

        Args:
            difficulty: Difficulty level.

        Returns:
            Random MiniGame instance.
        """
        game_type = self._rng.choice(list(MiniGameType))
        return self.create_game(game_type, difficulty)

    def create_batch(
        self,
        game_types: list[MiniGameType],
        batch_size: int,
        difficulty: float = 0.5,
    ) -> list[MiniGame]:
        """Create a batch of mini-games.

        Args:
            game_types: Types of games to include.
            batch_size: Number of games to create.
            difficulty: Difficulty level.

        Returns:
            List of MiniGame instances.
        """
        games = []
        for i in range(batch_size):
            game_type = game_types[i % len(game_types)]
            # Vary difficulty slightly
            d = difficulty + self._rng.uniform(-0.1, 0.1)
            d = max(0.0, min(1.0, d))
            games.append(self.create_game(game_type, d))
        return games
