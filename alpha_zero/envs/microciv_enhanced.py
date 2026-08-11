"""Enhanced MicroCiv environment with composable gameplay systems.

This module provides ``EnhancedMicroCivEnv``, a drop-in replacement for
:class:`~alpha_zero.envs.microciv.MicroCivEnv` that integrates the gameplay
systems from :mod:`alpha_zero.gameplay`:

- **Fog of War** -- per-player visibility with line-of-sight blocking.
- **Enhanced Combat** -- terrain defense bonuses, ranged attacks, flanking.
- **Economy** -- gold income, unit maintenance, purchasing.
- **City Growth** -- food surplus, population growth, starvation.
- **Terrain Effects** -- movement costs, tile yields.
- **Victory Conditions** -- elimination, domination, and score-based victories.

All features are controlled through :class:`~alpha_zero.gameplay.config.GameplayConfig`.
When all feature flags are ``False`` (the default), behaviour is identical to the
original ``MicroCivEnv``.

Example::

    from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv
    from alpha_zero.gameplay.config import GameplayConfig

    # Backward-compatible (same as original MicroCivEnv)
    env = EnhancedMicroCivEnv()

    # All features enabled
    env = EnhancedMicroCivEnv(gameplay_config=GameplayConfig.enhanced())
"""

from __future__ import annotations

import copy
import logging
from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import RenderFrame

from alpha_zero.envs.microciv import MicroCivEnv, Tile, Unit
from alpha_zero.gameplay.city_growth import CityGrowthConfig, CityGrowthSystem
from alpha_zero.gameplay.combat import CombatConfig, CombatResult, CombatSystem
from alpha_zero.gameplay.config import GameplayConfig
from alpha_zero.gameplay.economy import EconomyConfig, EconomySystem
from alpha_zero.gameplay.fog_of_war import FogOfWarConfig, FogOfWarSystem
from alpha_zero.gameplay.terrain import TerrainSystem
from alpha_zero.gameplay.victory import VictoryCondition, VictorySystem

logger = logging.getLogger(__name__)


class EnhancedMicroCivEnv(MicroCivEnv):
    """MicroCiv environment enhanced with composable gameplay systems.

    When *gameplay_config* is ``None`` or ``GameplayConfig.default()``, all
    feature flags are ``False`` and the environment behaves identically to
    :class:`MicroCivEnv`. When features are enabled, the corresponding
    systems are activated and integrated into the game loop.

    Args:
        board_size: Size of the square game board.
        max_turns: Maximum turns before the game ends.
        render_mode: Rendering mode (``'ansi'`` or ``None``).
        seed: Random seed for reproducibility.
        gameplay_config: Master configuration for all gameplay systems.
            If ``None``, uses ``GameplayConfig.default()`` (backward compat).
    """

    def __init__(
        self,
        board_size: int = 16,
        max_turns: int = 200,
        render_mode: str | None = None,
        seed: int | None = None,
        gameplay_config: GameplayConfig | None = None,
    ) -> None:
        super().__init__(
            board_size=board_size,
            max_turns=max_turns,
            render_mode=render_mode,
            seed=seed,
        )

        self.gameplay_config = gameplay_config or GameplayConfig.default()
        self._init_gameplay_systems()

        logger.info(
            "EnhancedMicroCivEnv created: fog=%s, combat=%s, economy=%s, growth=%s, terrain=%s",
            self.gameplay_config.enable_fog_of_war,
            self.gameplay_config.enable_enhanced_combat,
            self.gameplay_config.enable_economy,
            self.gameplay_config.enable_city_growth,
            self.gameplay_config.enable_terrain_effects,
        )

    # ------------------------------------------------------------------
    # System initialization
    # ------------------------------------------------------------------

    def _init_gameplay_systems(self) -> None:
        """Initialize all gameplay subsystems based on configuration."""
        cfg = self.gameplay_config
        num_players = cfg.num_players

        # Fog of war
        self._fog: FogOfWarSystem | None = None
        if cfg.enable_fog_of_war:
            fow_cfg = FogOfWarConfig(
                sight_range={
                    "warrior": cfg.fog_of_war.sight_range_warrior,
                    "settler": cfg.fog_of_war.sight_range_settler,
                    "archer": cfg.fog_of_war.sight_range_archer,
                    "city": cfg.fog_of_war.sight_range_city,
                },
                remember_seen=cfg.fog_of_war.remember_seen_tiles,
            )
            self._fog = FogOfWarSystem(
                board_size=self.board_size,
                num_players=num_players,
                config=fow_cfg,
            )

        # Combat
        self._combat: CombatSystem | None = None
        if cfg.enable_enhanced_combat:
            combat_cfg = CombatConfig(
                base_damage=cfg.combat.base_damage,
                base_retaliation_damage=cfg.combat.base_retaliation,
                terrain_defense_bonuses={
                    "hills": cfg.combat.terrain_defense_bonus_hills,
                    "forest": cfg.combat.terrain_defense_bonus_forest,
                },
                flanking_bonus_per_unit=cfg.combat.flanking_bonus_per_unit,
                max_flanking_bonus=cfg.combat.max_flanking_bonus,
                ranged_attack_range=cfg.combat.ranged_attack_range,
                ranged_no_retaliation=cfg.combat.ranged_no_retaliation,
                city_defense_strength=cfg.combat.city_defense_strength,
                health_damage_scaling=cfg.combat.health_damage_scaling,
            )
            self._combat = CombatSystem(config=combat_cfg)

        # Economy
        self._economy: EconomySystem | None = None
        if cfg.enable_economy:
            econ_cfg = EconomyConfig(
                starting_gold=cfg.economy.starting_gold,
                gold_per_city_base=cfg.economy.gold_per_city_base,
                gold_per_population=cfg.economy.gold_per_population,
                unit_maintenance={
                    "warrior": cfg.economy.unit_maintenance_warrior,
                    "archer": cfg.economy.unit_maintenance_archer,
                    "settler": cfg.economy.unit_maintenance_settler,
                },
            )
            self._economy = EconomySystem(
                num_players=num_players,
                config=econ_cfg,
            )

        # City growth
        self._city_growth: CityGrowthSystem | None = None
        if cfg.enable_city_growth:
            growth_cfg = CityGrowthConfig(
                food_consumption_per_pop=cfg.city_growth.food_per_population,
                growth_threshold_base=cfg.city_growth.growth_threshold_base,
                growth_threshold_scaling=cfg.city_growth.growth_threshold_scaling,
                max_population=cfg.city_growth.max_population,
                base_food_yield=cfg.city_growth.food_per_turn_base,
                production_per_population=cfg.city_growth.production_per_population,
            )
            self._city_growth = CityGrowthSystem(config=growth_cfg)

        # Terrain
        self._terrain_system: TerrainSystem | None = None
        if cfg.enable_terrain_effects:
            self._terrain_system = TerrainSystem()

        # Victory
        self._victory = VictorySystem(
            num_players=num_players,
            config=None,  # uses defaults (elimination + score)
        )

    # ------------------------------------------------------------------
    # Lifecycle overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment and all gameplay subsystems."""
        obs, info = super().reset(seed=seed, options=options)

        # Reset subsystems
        if self._fog is not None:
            self._fog.reset()
            self._update_fog_of_war()

        if self._economy is not None:
            self._economy.reset()

        if self._city_growth is not None:
            self._city_growth.reset()

        if self._victory is not None:
            self._victory.reset()

        # Re-encode with updated fog of war
        obs = self.get_observation()
        info = self.get_info()

        return obs, info

    # ------------------------------------------------------------------
    # Step override
    # ------------------------------------------------------------------

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute action with enhanced gameplay systems."""
        if self.done:
            return self.get_observation(), 0.0, True, False, self.get_info()

        # Decode and execute action
        action_dict = self.action_space_handler.decode_action(action)
        reward, action_success, action_error = self._execute_action(action_dict)

        if not action_success:
            logger.warning(
                "Turn %d, Player %d: Action %d (%s) failed: %s",
                self.turn,
                self.current_player,
                action,
                action_dict.get("type"),
                action_error,
            )

        # Update fog of war after every action
        if self._fog is not None:
            self._update_fog_of_war()

        # Check victory conditions
        terminated = self._check_victory_enhanced()
        truncated = self.turn >= self.max_turns

        if truncated and not terminated:
            terminated = self._handle_turn_limit()

        if terminated:
            self.done = True
            if self.winner == self.current_player:
                reward += 1.0
            elif self.winner is not None:
                reward -= 1.0

        obs = self.get_observation()
        info = self.get_info()
        info["action_success"] = action_success
        info["action_error"] = action_error

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Enhanced combat
    # ------------------------------------------------------------------

    def _execute_attack(self, attacker: Unit, target_pos: tuple[int, int]) -> float:
        """Execute combat, using enhanced system if enabled."""
        if self._combat is None:
            return super()._execute_attack(attacker, target_pos)

        # Find defender
        defender = None
        for unit in self.units:
            if unit.position == target_pos and unit.player != attacker.player:
                defender = unit
                break

        if defender is None:
            return 0.0

        # Get terrain info for defender's tile
        tile = self.terrain.get(target_pos, Tile())
        terrain_type = tile.terrain_type
        terrain_features = tile.features if tile.features else []

        # Count flanking units (own units adjacent to defender, excluding attacker)
        flanking = 0
        for unit in self._get_player_units(attacker.player):
            if unit.id == attacker.id:
                continue
            dx = abs(unit.position[0] - defender.position[0])
            dy = abs(unit.position[1] - defender.position[1])
            if dx <= 1 and dy <= 1:
                flanking += 1

        # Resolve combat
        result: CombatResult = self._combat.resolve_combat(
            attacker={
                "strength": attacker.strength,
                "health": attacker.health,
                "unit_type": attacker.unit_type,
                "position": attacker.position,
            },
            defender={
                "strength": defender.strength,
                "health": defender.health,
                "unit_type": defender.unit_type,
                "position": defender.position,
            },
            terrain_type=terrain_type,
            terrain_features=terrain_features,
            flanking_units=flanking,
        )

        # Apply damage
        defender.health -= int(result.attacker_damage_dealt)
        attacker.health -= int(result.defender_damage_dealt)

        logger.debug(
            "Combat: %s(%d hp) vs %s(%d hp) -> %d/%d damage, terrain=%s, flanking=%d",
            attacker.unit_type,
            attacker.health,
            defender.unit_type,
            defender.health,
            int(result.attacker_damage_dealt),
            int(result.defender_damage_dealt),
            terrain_type,
            flanking,
        )

        # Use the fully-shaped reward from CombatSystem (includes damage
        # scaling, kill/death bonuses, and city-capture reward) so the env
        # stays consistent with CombatConfig tuning.
        reward = float(result.reward)

        # Remove dead units
        if defender.health <= 0:
            self.units.remove(defender)
            logger.debug("Defender %s killed", defender.unit_type)

        if attacker.health <= 0:
            self.units.remove(attacker)
            logger.debug("Attacker %s killed", attacker.unit_type)

        return reward

    # ------------------------------------------------------------------
    # Ranged attack support
    # ------------------------------------------------------------------

    def _execute_action(self, action: dict[str, Any]) -> tuple[float, bool, str]:
        """Override to add ranged attack support."""
        action_type = action.get("type")

        # Ranged attack: archer can attack at range without retaliation
        if (
            action_type == "attack"
            and self._combat is not None
            and self.gameplay_config.enable_enhanced_combat
        ):
            unit_idx = action.get("unit_idx", 0)
            direction = action.get("direction", (0, 0))
            own_units = self._get_player_units(self.current_player)

            if unit_idx >= len(own_units):
                return 0.0, False, f"Invalid unit_idx {unit_idx}"

            unit = own_units[unit_idx]
            if unit.moves_remaining <= 0:
                return 0.0, False, f"Unit {unit_idx} has no moves"

            target_pos = (
                unit.position[0] + direction[0],
                unit.position[1] + direction[1],
            )

            has_defender = any(
                u.position == target_pos and u.player != unit.player for u in self.units
            )
            if not has_defender:
                return 0.0, False, f"No enemy at {target_pos}"

            attack_reward = self._execute_attack(unit, target_pos)
            unit.moves_remaining -= 1
            return attack_reward, True, ""

        # Delegate all other actions to parent
        return super()._execute_action(action)

    # ------------------------------------------------------------------
    # End turn enhancements
    # ------------------------------------------------------------------

    def _end_turn(self) -> None:
        """End turn with economy and city growth processing."""
        # Process cities (production) -- base behavior
        for city in self._get_player_cities(self.current_player):
            production_rate = city.population + 2  # base production

            # Enhanced: add production bonus from city growth system
            if self._city_growth is not None:
                production_rate += self._city_growth.get_production_bonus(city.population)

            if city.production_queue:
                city.production += production_rate
                cost = self.UNIT_COSTS.get(city.production_queue, 5) * 10
                if city.production >= cost:
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

            # Process economy for both players
            if self._economy is not None:
                for player in range(self.gameplay_config.num_players):
                    own_cities = [c.to_dict() for c in self._get_player_cities(player)]
                    own_units = [u.to_dict() for u in self._get_player_units(player)]
                    self._economy.process_turn(player, own_cities, own_units)
                    logger.debug(
                        "Economy P%d: gold=%d, gpt=%d",
                        player,
                        self._economy.get_gold(player),
                        self._economy.gold_per_turn.get(player, 0),
                    )

            # Process city growth for both players
            if self._city_growth is not None:
                for player in range(self.gameplay_config.num_players):
                    for city in self._get_player_cities(player):
                        # Get tile yields around city
                        tile = self.terrain.get(city.position, Tile())
                        tile_yields = [tile.yields]  # simplified: just city center
                        surplus = self._city_growth.calculate_food_surplus(
                            city.to_dict(), tile_yields
                        )
                        result = self._city_growth.process_city_growth(
                            city.id, city.population, surplus
                        )
                        if result["population_change"] != 0:
                            city.population = max(1, city.population + result["population_change"])
                            logger.debug(
                                "City '%s' pop %d->%d (surplus=%d)",
                                city.name,
                                city.population - result["population_change"],
                                city.population,
                                surplus,
                            )

            # Process settlers (legacy behavior)
            self._process_settlers()

        # Update fog of war
        if self._fog is not None:
            self._update_fog_of_war()

    # ------------------------------------------------------------------
    # Victory condition enhancements
    # ------------------------------------------------------------------

    def _check_victory_enhanced(self) -> bool:
        """Check victory using enhanced system or fall back to base."""
        # Build game state for victory system
        game_state = self._build_victory_game_state()
        result: VictoryCondition | None = self._victory.check_victory(game_state)

        if result is not None:
            self.winner = result.winner
            logger.info(
                "Victory: %s, winner=%s, reason=%s",
                result.victory_type.name,
                result.winner,
                result.reason,
            )
            return True

        return False

    def _handle_turn_limit(self) -> bool:
        """Handle turn limit with score-based victory if available."""
        game_state = self._build_victory_game_state()
        game_state["turn_limit_reached"] = True

        result = self._victory.check_victory(game_state)
        if result is not None:
            self.winner = result.winner
            return True

        # Fallback: most cities wins (backward compat)
        p0_cities = len(self._get_player_cities(0))
        p1_cities = len(self._get_player_cities(1))
        if p0_cities > p1_cities:
            self.winner = 0
        elif p1_cities > p0_cities:
            self.winner = 1
        else:
            self.winner = None
        return True

    def _build_victory_game_state(self) -> dict[str, Any]:
        """Build game state dict in the format expected by VictorySystem."""
        players: dict[int, dict[str, Any]] = {}
        for player in range(self.gameplay_config.num_players):
            own_units = [u.to_dict() for u in self._get_player_units(player)]
            own_cities = [c.to_dict() for c in self._get_player_cities(player)]
            gold = 0
            if self._economy is not None:
                gold = self._economy.get_gold(player)
            players[player] = {
                "units": own_units,
                "cities": own_cities,
                "gold": gold,
                "territory": len(own_cities) * 4,  # rough estimate
            }

        return {
            "players": players,
            "turn_limit_reached": self.turn >= self.max_turns,
        }

    # ------------------------------------------------------------------
    # Fog of war helpers
    # ------------------------------------------------------------------

    def _update_fog_of_war(self) -> None:
        """Update fog of war for all players."""
        if self._fog is None:
            return

        units_dicts = [u.to_dict() for u in self.units]
        cities_dicts = [c.to_dict() for c in self.cities]

        for player in range(self.gameplay_config.num_players):
            self._fog.update_visibility(
                player=player,
                units=units_dicts,
                cities=cities_dicts,
                terrain=self.terrain,
            )

    # ------------------------------------------------------------------
    # Observation override
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get observation with fog of war applied if enabled."""
        game_state = self._get_game_state()

        # Add fog of war visibility to game state
        if self._fog is not None:
            visibility = {}
            for y in range(self.board_size):
                for x in range(self.board_size):
                    state = self._fog.get_visibility_state(self.current_player, (x, y))
                    if state == 2:  # VISIBLE
                        visibility[(x, y)] = "visible"
                    elif state == 1:  # SEEN
                        visibility[(x, y)] = "seen"
                    else:
                        visibility[(x, y)] = "unexplored"
            game_state["visibility"] = visibility

        # Add economy data to game state
        if self._economy is not None:
            game_state["gold"] = self._economy.get_gold(self.current_player)
            game_state["gold_per_turn"] = self._economy.gold_per_turn.get(self.current_player, 0)

        spatial, vector = self.encoder.encode(game_state)
        action_mask = self.action_space_handler.get_valid_mask(game_state)

        return {
            "spatial": spatial.cpu().numpy(),
            "vector": vector.cpu().numpy(),
            "action_mask": action_mask,
        }

    def get_info(self) -> dict[str, Any]:
        """Get info dict with enhanced gameplay data."""
        info = super().get_info()

        # Add economy info
        if self._economy is not None:
            info["gold"] = {
                p: self._economy.get_gold(p) for p in range(self.gameplay_config.num_players)
            }
            info["gold_per_turn"] = {
                p: self._economy.gold_per_turn.get(p, 0)
                for p in range(self.gameplay_config.num_players)
            }

        # Add city growth info
        if self._city_growth is not None:
            info["populations"] = {
                p: [c.population for c in self._get_player_cities(p)]
                for p in range(self.gameplay_config.num_players)
            }

        # Add fog of war stats
        if self._fog is not None:
            mask = self._fog.get_visibility_mask(self.current_player)
            info["visibility"] = {
                "visible_tiles": int((mask == 2).sum()),
                "seen_tiles": int((mask == 1).sum()),
                "unexplored_tiles": int((mask == 0).sum()),
            }

        return info

    # ------------------------------------------------------------------
    # Rendering override
    # ------------------------------------------------------------------

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Render with fog of war if enabled."""
        if self.render_mode != "ansi":
            return None

        lines = [f"Turn {self.turn}, Player {self.current_player}'s turn"]
        lines.append("=" * (self.board_size * 2 + 1))

        grid = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]

        # Mark terrain
        for (x, y), tile in self.terrain.items():
            # Apply fog of war
            if self._fog is not None:
                vis = self._fog.get_visibility_state(self.current_player, (x, y))
                if vis == 0:  # UNEXPLORED
                    grid[y][x] = "?"
                    continue
                elif vis == 1:  # SEEN
                    grid[y][x] = "~" if tile.terrain_type == "hills" else "."
                    continue

            if tile.terrain_type == "mountain":
                grid[y][x] = "^"
            elif tile.terrain_type == "hills":
                grid[y][x] = "~"

        # Mark cities (only if visible)
        for city in self.cities:
            x, y = city.position
            if self._fog is not None and not self._fog.is_visible(self.current_player, (x, y)):
                continue
            grid[y][x] = "C" if city.player == 0 else "c"

        # Mark units (only if visible)
        for unit in self.units:
            x, y = unit.position
            if self._fog is not None and not self._fog.is_visible(self.current_player, (x, y)):
                continue
            if unit.unit_type == "warrior":
                grid[y][x] = "W" if unit.player == 0 else "w"
            elif unit.unit_type == "settler":
                grid[y][x] = "S" if unit.player == 0 else "s"
            elif unit.unit_type == "archer":
                grid[y][x] = "A" if unit.player == 0 else "a"

        for row in grid:
            lines.append(" ".join(row))

        lines.append("=" * (self.board_size * 2 + 1))

        # Player summary
        p0_units = len(self._get_player_units(0))
        p0_cities = len(self._get_player_cities(0))
        p1_units = len(self._get_player_units(1))
        p1_cities = len(self._get_player_cities(1))
        lines.append(f"P0: {p0_units} units, {p0_cities} cities")
        lines.append(f"P1: {p1_units} units, {p1_cities} cities")

        # Economy summary
        if self._economy is not None:
            for p in range(self.gameplay_config.num_players):
                gold = self._economy.get_gold(p)
                gpt = self._economy.gold_per_turn.get(p, 0)
                lines.append(f"P{p} Gold: {gold} (GPT: {gpt:+d})")

        # Fog summary
        if self._fog is not None:
            mask = self._fog.get_visibility_mask(self.current_player)
            visible = int((mask == 2).sum())
            total = self.board_size * self.board_size
            lines.append(f"Visibility: {visible}/{total} tiles ({visible * 100 // total}%)")

        result: RenderFrame = "\n".join(lines)  # type: ignore[assignment]
        return result

    # ------------------------------------------------------------------
    # Clone override
    # ------------------------------------------------------------------

    def clone(self) -> EnhancedMicroCivEnv:
        """Deep copy including gameplay systems."""
        new_env = EnhancedMicroCivEnv(
            board_size=self.board_size,
            max_turns=self.max_turns,
            render_mode=self.render_mode,
            gameplay_config=self.gameplay_config,
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

        # Deep copy subsystems
        if self._fog is not None:
            new_env._fog = FogOfWarSystem.from_dict(self._fog.to_dict(), self.board_size)
        if self._economy is not None:
            new_env._economy = EconomySystem.from_dict(self._economy.to_dict())
        if self._city_growth is not None:
            new_env._city_growth = CityGrowthSystem.from_dict(self._city_growth.to_dict())

        return new_env

    # ------------------------------------------------------------------
    # Properties for external access
    # ------------------------------------------------------------------

    @property
    def fog_of_war(self) -> FogOfWarSystem | None:
        """Access the fog of war system (None if disabled)."""
        return self._fog

    @property
    def economy(self) -> EconomySystem | None:
        """Access the economy system (None if disabled)."""
        return self._economy

    @property
    def city_growth(self) -> CityGrowthSystem | None:
        """Access the city growth system (None if disabled)."""
        return self._city_growth

    @property
    def combat_system(self) -> CombatSystem | None:
        """Access the combat system (None if disabled)."""
        return self._combat

    @property
    def terrain_system(self) -> TerrainSystem | None:
        """Access the terrain system (None if disabled)."""
        return self._terrain_system

    @property
    def victory_system(self) -> VictorySystem:
        """Access the victory system (always present)."""
        return self._victory
