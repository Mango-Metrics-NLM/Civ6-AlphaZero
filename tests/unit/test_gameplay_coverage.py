"""Targeted tests to cover missing lines in alpha_zero.gameplay modules.

Coverage gaps addressed:
- economy.py: lines 63, 135, 154-155, 190
- combat.py: lines 199, 240-241, 299-303, 313, 372, 388-391
- city_growth.py: lines 93-94, 280-286
- victory.py: lines 127-128, 146-161, 194-195, 247, 334, 376-380, 412, 433
"""

import pytest

from alpha_zero.gameplay.city_growth import CityGrowthConfig, CityGrowthSystem
from alpha_zero.gameplay.combat import CombatConfig, CombatResult, CombatSystem
from alpha_zero.gameplay.economy import EconomyConfig, EconomySystem
from alpha_zero.gameplay.victory import (
    VictoryConfig,
    VictorySystem,
    VictoryType,
)

# ======================================================================
# Economy coverage
# ======================================================================


class TestEconomyCoverage:
    """Tests targeting uncovered lines in economy.py."""

    def test_reset_with_num_players_change(self):
        """Line 63: reset(num_players=...) should change the player count."""
        economy = EconomySystem(num_players=2, config=EconomyConfig(starting_gold=10))
        assert economy.get_gold(0) == 10
        assert economy.get_gold(1) == 10

        # Reset with a new player count
        economy.reset(num_players=3)

        assert economy._num_players == 3
        assert economy.get_gold(0) == 10
        assert economy.get_gold(1) == 10
        assert economy.get_gold(2) == 10
        assert economy.gold_per_turn[0] == 0
        assert economy.gold_per_turn[1] == 0
        assert economy.gold_per_turn[2] == 0

    def test_process_turn_triggers_bankrupt_logging(self):
        """Line 135: process_turn logs when the player goes bankrupt."""
        economy = EconomySystem(
            num_players=2,
            config=EconomyConfig(starting_gold=0),
        )
        # Give player 0 no cities (no income) but lots of units (high maintenance)
        cities: list[dict] = []
        units = [{"type": "warrior"}, {"type": "warrior"}, {"type": "warrior"}]

        result = economy.process_turn(0, cities, units)

        # Net income is negative (0 city income - 3 maintenance = -3)
        assert result["net"] < 0
        assert result["is_bankrupt"] is True
        assert result["new_balance"] < 0

    def test_can_purchase_returns_true_when_affordable(self):
        """Lines 154-155: can_purchase calculates effective cost and compares."""
        config = EconomyConfig(purchase_cost_multiplier=2.0)
        economy = EconomySystem(num_players=1, config=config)
        economy.gold[0] = 100

        # base_cost=30, multiplier=2.0 => effective=60; 100 >= 60 => True
        assert economy.can_purchase(0, "warrior", 30) is True

    def test_can_purchase_returns_false_when_too_expensive(self):
        """Lines 154-155: can_purchase returns False when gold < effective cost."""
        config = EconomyConfig(purchase_cost_multiplier=4.0)
        economy = EconomySystem(num_players=1, config=config)
        economy.gold[0] = 10

        # base_cost=10, multiplier=4.0 => effective=40; 10 < 40 => False
        assert economy.can_purchase(0, "warrior", 10) is False

    def test_add_gold_positive(self):
        """Line 190: add_gold adds a positive amount to the player's balance."""
        economy = EconomySystem(num_players=2, config=EconomyConfig(starting_gold=0))
        economy.add_gold(0, 50)
        assert economy.get_gold(0) == 50

    def test_add_gold_negative(self):
        """Line 190: add_gold supports negative amounts."""
        economy = EconomySystem(num_players=2, config=EconomyConfig(starting_gold=100))
        economy.add_gold(0, -30)
        assert economy.get_gold(0) == 70

    def test_add_gold_for_unknown_player(self):
        """Line 190: add_gold creates an entry for a player not yet tracked."""
        economy = EconomySystem(num_players=1, config=EconomyConfig(starting_gold=0))
        # Player 5 doesn't exist in the initial dict
        economy.add_gold(5, 25)
        assert economy.get_gold(5) == 25


# ======================================================================
# Combat coverage
# ======================================================================


class TestCombatCoverage:
    """Tests targeting uncovered lines in combat.py."""

    def test_can_attack_with_none_position(self):
        """Line 199: can_attack returns False when attacker has no position."""
        combat = CombatSystem()
        attacker = {"unit_type": "warrior"}  # no 'position' key
        assert combat.can_attack(attacker, (3, 5)) is False

    def test_can_attack_with_explicit_none_position(self):
        """Line 199: can_attack returns False when position is explicitly None."""
        combat = CombatSystem()
        attacker = {"unit_type": "warrior", "position": None}
        assert combat.can_attack(attacker, (3, 5)) is False

    def test_calculate_terrain_modifier_with_features(self):
        """Lines 240-241: terrain modifier accumulates feature bonuses."""
        config = CombatConfig(
            terrain_defense_bonuses={"hills": 0.25, "forest": 0.15, "river": 0.10},
        )
        combat = CombatSystem(config)

        # hills base + forest feature
        mod = combat.calculate_terrain_modifier("hills", features=["forest"])
        assert mod == pytest.approx(0.40)

    def test_calculate_terrain_modifier_multiple_features(self):
        """Lines 240-241: multiple features stack additively."""
        config = CombatConfig(
            terrain_defense_bonuses={"hills": 0.25, "forest": 0.15, "river": 0.10},
        )
        combat = CombatSystem(config)

        mod = combat.calculate_terrain_modifier("hills", features=["forest", "river"])
        assert mod == pytest.approx(0.50)

    def test_calculate_terrain_modifier_unknown_feature(self):
        """Lines 240-241: unknown features contribute 0.0."""
        combat = CombatSystem()
        mod = combat.calculate_terrain_modifier("plains", features=["lava"])
        assert mod == pytest.approx(0.0)

    def test_city_attack_health_scaling(self):
        """Lines 299-303: city attack applies health_damage_scaling to city_eff."""
        config = CombatConfig(health_damage_scaling=True, base_damage=30.0)
        combat = CombatSystem(config)

        attacker = {
            "strength": 40,
            "health": 100,
            "unit_type": "warrior",
            "position": (3, 4),
        }

        # Full-health city
        city_full = {"defense_strength": 25, "health": 100, "position": (3, 5)}
        # Half-health city
        city_half = {"defense_strength": 25, "health": 50, "position": (3, 5)}

        # Run many trials to average out random variance
        n = 200
        damage_full = sum(
            combat.resolve_city_attack(attacker, city_full).attacker_damage_dealt for _ in range(n)
        )
        damage_half = sum(
            combat.resolve_city_attack(attacker, city_half).attacker_damage_dealt for _ in range(n)
        )

        # A half-health city has weaker effective defense, so attacker should
        # deal MORE average damage to it.
        assert damage_half / n > damage_full / n

    def test_city_attack_ranged_no_retaliation(self):
        """Line 313: ranged attacker receives no retaliation in city attack."""
        config = CombatConfig(ranged_no_retaliation=True)
        combat = CombatSystem(config)

        archer = {
            "strength": 20,
            "health": 100,
            "unit_type": "archer",
            "position": (3, 4),
        }
        city = {"defense_strength": 25, "health": 100, "position": (3, 5)}

        result = combat.resolve_city_attack(archer, city)
        assert result.defender_damage_dealt == 0.0

    def test_compute_reward_death_penalty(self):
        """Line 372: attacker death subtracts death_penalty from reward."""
        config = CombatConfig(
            kill_reward=0.1,
            death_penalty=0.2,
            damage_reward_scaling=0.0,
            city_capture_reward=0.0,
        )
        combat = CombatSystem(config)

        result = CombatResult(
            attacker_damage_dealt=0.0,
            defender_damage_dealt=0.0,
            attacker_killed=True,
            defender_killed=False,
            city_captured=False,
        )
        reward = combat.compute_reward(result)
        assert reward == pytest.approx(-0.2)

    def test_compute_reward_both_kill_and_death(self):
        """Lines 370-372: both kill and death contribute to reward."""
        config = CombatConfig(
            kill_reward=0.1,
            death_penalty=0.1,
            damage_reward_scaling=0.0,
            city_capture_reward=0.0,
        )
        combat = CombatSystem(config)

        result = CombatResult(
            attacker_damage_dealt=0.0,
            defender_damage_dealt=0.0,
            attacker_killed=True,
            defender_killed=True,
            city_captured=False,
        )
        reward = combat.compute_reward(result)
        # +0.1 (kill) - 0.1 (death) = 0.0
        assert reward == pytest.approx(0.0)

    def test_effective_strength_health_scaling(self):
        """Lines 388-391: _effective_strength scales by health/100 when enabled."""
        config = CombatConfig(health_damage_scaling=True)
        combat = CombatSystem(config)

        full_health = {"strength": 40, "health": 100}
        half_health = {"strength": 40, "health": 50}
        quarter_health = {"strength": 40, "health": 25}

        assert combat._effective_strength(full_health) == pytest.approx(40.0)
        assert combat._effective_strength(half_health) == pytest.approx(20.0)
        assert combat._effective_strength(quarter_health) == pytest.approx(10.0)

    def test_effective_strength_without_health_scaling(self):
        """Lines 388-391: when health_damage_scaling=False, health is ignored."""
        config = CombatConfig(health_damage_scaling=False)
        combat = CombatSystem(config)

        injured = {"strength": 40, "health": 10}
        assert combat._effective_strength(injured) == pytest.approx(40.0)

    def test_resolve_combat_with_terrain_features(self):
        """Lines 240-241 via resolve_combat: features affect combat outcome."""
        config = CombatConfig(
            terrain_defense_bonuses={"hills": 0.25, "forest": 0.15},
            health_damage_scaling=False,
            base_damage=30.0,
            base_retaliation_damage=20.0,
        )
        combat = CombatSystem(config)

        attacker = {
            "strength": 30,
            "health": 100,
            "unit_type": "warrior",
            "position": (3, 4),
        }
        defender = {
            "strength": 30,
            "health": 100,
            "unit_type": "warrior",
            "position": (3, 5),
        }

        result = combat.resolve_combat(
            attacker, defender, terrain_type="hills", terrain_features=["forest"]
        )
        assert isinstance(result, CombatResult)
        assert result.attacker_damage_dealt >= 0


# ======================================================================
# CityGrowth coverage
# ======================================================================


class TestCityGrowthCoverage:
    """Tests targeting uncovered lines in city_growth.py."""

    def test_reset_clears_stockpiles(self):
        """Lines 93-94: reset() clears all food stockpiles."""
        system = CityGrowthSystem()
        system.set_food_stockpile(0, 50)
        system.set_food_stockpile(1, 30)

        system.reset()

        assert system.food_stockpile == {}
        assert system.get_food_stockpile(0) == 0
        assert system.get_food_stockpile(1) == 0

    def test_encode_for_vector_basic(self):
        """Lines 280-286: encode_for_vector returns food_progress and population_normalized."""
        config = CityGrowthConfig(
            growth_threshold_base=15,
            growth_threshold_scaling=8,
            max_population=10,
        )
        system = CityGrowthSystem(config)
        system.set_food_stockpile(0, 10)

        encoded = system.encode_for_vector(city_id=0, population=3)

        # threshold = 15 + (3-1)*8 = 31
        # food_progress = 10 / 31
        assert encoded["food_progress"] == pytest.approx(10 / 31)
        # population_normalized = 3 / 10
        assert encoded["population_normalized"] == pytest.approx(0.3)

    def test_encode_for_vector_zero_stockpile(self):
        """Lines 280-286: encode_for_vector with zero stockpile."""
        system = CityGrowthSystem(CityGrowthConfig(max_population=15))

        encoded = system.encode_for_vector(city_id=99, population=1)

        assert encoded["food_progress"] == pytest.approx(0.0)
        assert encoded["population_normalized"] == pytest.approx(1 / 15)

    def test_encode_for_vector_at_max_population(self):
        """Lines 280-286: encode_for_vector at max population normalizes to 1.0."""
        config = CityGrowthConfig(max_population=10)
        system = CityGrowthSystem(config)
        system.set_food_stockpile(0, 100)

        encoded = system.encode_for_vector(city_id=0, population=10)

        assert encoded["population_normalized"] == pytest.approx(1.0)
        # threshold = 15 + (10-1)*8 = 87; progress = 100/87 > 1.0
        assert encoded["food_progress"] > 1.0


# ======================================================================
# Victory coverage
# ======================================================================


class TestVictoryCoverage:
    """Tests targeting uncovered lines in victory.py."""

    @staticmethod
    def _make_state(
        players: dict[int, dict] | None = None,
        turn_limit_reached: bool = False,
        total_cities: int | None = None,
    ) -> dict:
        """Helper to build a game_state dict."""
        state: dict = {
            "players": players if players is not None else {},
            "turn_limit_reached": turn_limit_reached,
        }
        if total_cities is not None:
            state["total_cities"] = total_cities
        return state

    def test_reset_restores_enabled(self):
        """Lines 127-128: reset() re-sets _enabled from config."""
        config = VictoryConfig(enabled_victories=["elimination", "domination"])
        victory = VictorySystem(num_players=2, config=config)

        # Manually tamper with _enabled
        victory._enabled = set()

        victory.reset()

        assert victory._enabled == {"elimination", "domination"}

    def test_check_victory_elimination_path(self):
        """Lines 146-149: check_victory delegates to check_elimination."""
        config = VictoryConfig(enabled_victories=["elimination"])
        victory = VictorySystem(num_players=2, config=config)

        state = self._make_state(
            players={
                0: {"units": ["w"], "cities": [{"population": 1}], "gold": 0, "territory": 0},
                1: {"units": [], "cities": [], "gold": 0, "territory": 0},
            }
        )
        result = victory.check_victory(state)
        assert result is not None
        assert result.victory_type == VictoryType.ELIMINATION
        assert result.winner == 0

    def test_check_victory_domination_path(self):
        """Lines 151-154: check_victory reaches domination when elimination fails."""
        config = VictoryConfig(enabled_victories=["elimination", "domination"])
        victory = VictorySystem(num_players=2, config=config)

        # Both players have units, so no elimination. Player 0 has 3/4 cities.
        state = self._make_state(
            players={
                0: {
                    "units": ["w"],
                    "cities": [{"population": 1}] * 3,
                    "gold": 0,
                    "territory": 0,
                },
                1: {
                    "units": ["w"],
                    "cities": [{"population": 1}],
                    "gold": 0,
                    "territory": 0,
                },
            }
        )
        result = victory.check_victory(state)
        assert result is not None
        assert result.victory_type == VictoryType.DOMINATION
        assert result.winner == 0

    def test_check_victory_score_path_at_turn_limit(self):
        """Lines 156-159: check_victory reaches score branch at turn limit."""
        config = VictoryConfig(enabled_victories=["score"])
        victory = VictorySystem(num_players=2, config=config)

        state = self._make_state(
            players={
                0: {
                    "units": ["w1", "w2"],
                    "cities": [{"population": 3}],
                    "gold": 100,
                    "territory": 10,
                },
                1: {
                    "units": [],
                    "cities": [],
                    "gold": 0,
                    "territory": 0,
                },
            },
            turn_limit_reached=True,
        )
        result = victory.check_victory(state)
        assert result is not None
        assert result.victory_type == VictoryType.SCORE
        assert result.winner == 0

    def test_check_victory_score_not_triggered_without_turn_limit(self):
        """Lines 156-159: score is NOT checked when turn_limit_reached=False."""
        config = VictoryConfig(enabled_victories=["score"])
        victory = VictorySystem(num_players=2, config=config)

        state = self._make_state(
            players={
                0: {"units": ["w"], "cities": [{"population": 1}], "gold": 0, "territory": 0},
                1: {"units": ["w"], "cities": [{"population": 1}], "gold": 0, "territory": 0},
            },
            turn_limit_reached=False,
        )
        result = victory.check_victory(state)
        assert result is None

    def test_check_victory_returns_none_when_no_conditions_met(self):
        """Line 161: check_victory returns None at the end when nothing triggers."""
        config = VictoryConfig(enabled_victories=["elimination", "domination", "score"])
        victory = VictorySystem(num_players=2, config=config)

        state = self._make_state(
            players={
                0: {"units": ["w"], "cities": [{"population": 1}], "gold": 0, "territory": 0},
                1: {"units": ["w"], "cities": [{"population": 1}], "gold": 0, "territory": 0},
            },
            turn_limit_reached=False,
        )
        result = victory.check_victory(state)
        assert result is None

    def test_check_elimination_all_players_eliminated(self):
        """Lines 194-195 (208-216): when all players are eliminated, it is a draw."""
        config = VictoryConfig(elimination_requires_all=True)
        victory = VictorySystem(num_players=2, config=config)

        state = self._make_state(
            players={
                0: {"units": [], "cities": [], "gold": 0, "territory": 0},
                1: {"units": [], "cities": [], "gold": 0, "territory": 0},
            }
        )
        result = victory.check_elimination(state)
        assert result is not None
        assert result.winner is None
        assert result.victory_type == VictoryType.ELIMINATION
        assert "All players eliminated" in result.reason

    def test_check_elimination_not_requires_all(self):
        """Lines 193-195: elimination_requires_all=False checks only cities."""
        config = VictoryConfig(
            enabled_victories=["elimination"],
            elimination_requires_all=False,
        )
        victory = VictorySystem(num_players=2, config=config)

        # Player 1 has units but no cities => eliminated
        state = self._make_state(
            players={
                0: {"units": ["w"], "cities": [{"population": 1}], "gold": 0, "territory": 0},
                1: {"units": ["w", "a"], "cities": [], "gold": 0, "territory": 0},
            }
        )
        result = victory.check_elimination(state)
        assert result is not None
        assert result.winner == 0

    def test_domination_zero_total_cities(self):
        """Line 247: check_domination returns None when total_cities is 0."""
        config = VictoryConfig(enabled_victories=["domination"])
        victory = VictorySystem(num_players=2, config=config)

        state = self._make_state(
            players={
                0: {"units": ["w"], "cities": [], "gold": 0, "territory": 0},
                1: {"units": ["w"], "cities": [], "gold": 0, "territory": 0},
            }
        )
        result = victory.check_domination(state)
        assert result is None

    def test_check_score_victory_no_players(self):
        """Line 334: check_score_victory returns None if there are no players."""
        victory = VictorySystem(num_players=2)
        state = self._make_state(players={})
        result = victory.check_score_victory(state)
        assert result is None

    def test_check_score_victory_selects_highest(self):
        """Lines 332-345: check_score_victory picks the player with highest score."""
        config = VictoryConfig(
            score_weights={
                "unit": 1.0,
                "city": 5.0,
                "population": 2.0,
                "gold": 0.02,
                "territory": 0.5,
            },
        )
        victory = VictorySystem(num_players=2, config=config)

        state = self._make_state(
            players={
                0: {
                    "units": [],
                    "cities": [{"population": 1}],
                    "gold": 10,
                    "territory": 2,
                },
                1: {
                    "units": ["w1", "w2", "w3"],
                    "cities": [{"population": 5}, {"population": 3}],
                    "gold": 200,
                    "territory": 20,
                },
            }
        )
        result = victory.check_score_victory(state)
        assert result is not None
        assert result.winner == 1
        assert result.victory_type == VictoryType.SCORE

    def test_get_victory_progress_total_cities_computed(self):
        """Lines 376-380: get_victory_progress computes total_cities from players."""
        config = VictoryConfig(domination_threshold=0.66)
        victory = VictorySystem(num_players=2, config=config)

        state = self._make_state(
            players={
                0: {
                    "units": ["w1"],
                    "cities": [{"population": 2}],
                    "gold": 50,
                    "territory": 5,
                },
                1: {
                    "units": ["w1", "w2"],
                    "cities": [{"population": 1}, {"population": 1}],
                    "gold": 30,
                    "territory": 3,
                },
            }
        )
        progress = victory.get_victory_progress(0, state)

        # total_cities = 1 + 2 = 3, own_cities = 1
        # dom_target = max(3 * 0.66, 1.0) = 1.98
        # domination progress = 1 / 1.98
        assert progress["domination"] == pytest.approx(1.0 / 1.98, rel=1e-4)
        assert 0.0 <= progress["score"] <= 1.0
        assert 0.0 <= progress["elimination"] <= 1.0

    def test_get_victory_progress_with_total_cities_key(self):
        """Lines 375-376: uses explicit total_cities from game_state if provided."""
        victory = VictorySystem(num_players=2)

        state = self._make_state(
            players={
                0: {
                    "units": ["w"],
                    "cities": [{"population": 2}],
                    "gold": 0,
                    "territory": 0,
                },
                1: {
                    "units": ["w"],
                    "cities": [{"population": 1}],
                    "gold": 0,
                    "territory": 0,
                },
            },
            total_cities=10,  # explicit override
        )
        progress = victory.get_victory_progress(0, state)

        # dom_target = max(10 * 0.66, 1.0) = 6.6
        # own_cities = 1 => progress = 1/6.6
        assert progress["domination"] == pytest.approx(1.0 / 6.6, rel=1e-4)

    def test_get_victory_progress_zero_enemy_assets(self):
        """Line 412: when total_enemy_assets=0, elimination progress is 1.0."""
        victory = VictorySystem(num_players=2)

        state = self._make_state(
            players={
                0: {
                    "units": ["w"],
                    "cities": [{"population": 1}],
                    "gold": 0,
                    "territory": 0,
                },
                1: {
                    "units": [],
                    "cities": [],
                    "gold": 0,
                    "territory": 0,
                },
            }
        )
        progress = victory.get_victory_progress(0, state)
        assert progress["elimination"] == pytest.approx(1.0)

    def test_encode_for_vector_delegates_to_progress(self):
        """Line 433: encode_for_vector returns same result as get_victory_progress."""
        victory = VictorySystem(num_players=2)

        state = self._make_state(
            players={
                0: {
                    "units": ["w"],
                    "cities": [{"population": 2}],
                    "gold": 50,
                    "territory": 5,
                },
                1: {
                    "units": ["w", "a"],
                    "cities": [{"population": 1}],
                    "gold": 20,
                    "territory": 3,
                },
            }
        )

        progress = victory.get_victory_progress(0, state)
        encoded = victory.encode_for_vector(0, state)

        assert encoded == progress
        assert "domination" in encoded
        assert "score" in encoded
        assert "elimination" in encoded

    def test_check_victory_full_pipeline_domination_over_elimination(self):
        """Integration: domination is checked when elimination does not trigger."""
        config = VictoryConfig(
            enabled_victories=["elimination", "domination"],
            domination_threshold=0.60,
        )
        victory = VictorySystem(num_players=2, config=config)

        # Both players alive (no elimination), but player 0 has 4/5 = 80% cities
        state = self._make_state(
            players={
                0: {
                    "units": ["w"],
                    "cities": [{"population": 1}] * 4,
                    "gold": 0,
                    "territory": 0,
                },
                1: {
                    "units": ["w"],
                    "cities": [{"population": 1}],
                    "gold": 0,
                    "territory": 0,
                },
            }
        )
        result = victory.check_victory(state)
        assert result is not None
        assert result.victory_type == VictoryType.DOMINATION

    def test_get_victory_progress_with_starting_assets(self):
        """Lines 401-409: elimination progress uses starting asset metadata."""
        victory = VictorySystem(num_players=2)

        state = self._make_state(
            players={
                0: {
                    "units": ["w"],
                    "cities": [{"population": 1}],
                    "gold": 0,
                    "territory": 0,
                },
                1: {
                    "units": ["w"],
                    "cities": [{"population": 1}],
                    "gold": 0,
                    "territory": 0,
                    "starting_units": 3,
                    "starting_cities": 2,
                },
            }
        )
        progress = victory.get_victory_progress(0, state)

        # Enemy started with 3 units + 2 cities = 5 assets
        # Enemy currently has 1 unit + 1 city = 2 assets
        # Destroyed = (3-1) + (2-1) = 3
        # elimination = 3/5 = 0.6
        assert progress["elimination"] == pytest.approx(0.6)
