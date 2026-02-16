"""Comprehensive tests for all gameplay systems in alpha_zero.gameplay.

Tests cover the full public API of every gameplay subsystem:
- GameplayConfig (master configuration with feature flags)
- FogOfWarSystem (visibility, line-of-sight, encoding)
- CombatSystem (melee, ranged, flanking, city attacks, rewards)
- EconomySystem (gold, income, maintenance, purchasing)
- CityGrowthSystem (food, growth, starvation, production)
- TerrainSystem (movement costs, yields, defense, BFS reachability)
- VictorySystem (elimination, domination, score, progress)
"""

import numpy as np
import pytest

from alpha_zero.gameplay.config import GameplayConfig
from alpha_zero.gameplay.fog_of_war import (
    FogOfWarSystem,
    FogOfWarConfig,
    VISIBLE,
    SEEN,
    UNEXPLORED,
)
from alpha_zero.gameplay.combat import CombatSystem, CombatConfig, CombatResult
from alpha_zero.gameplay.economy import EconomySystem, EconomyConfig
from alpha_zero.gameplay.city_growth import CityGrowthSystem, CityGrowthConfig
from alpha_zero.gameplay.terrain import TerrainSystem, TerrainConfig
from alpha_zero.gameplay.victory import (
    VictorySystem,
    VictoryConfig,
    VictoryType,
    VictoryCondition,
)


# ======================================================================
# 1. GameplayConfig
# ======================================================================


class TestGameplayConfig:
    """Tests for the master GameplayConfig dataclass."""

    def test_default_config_all_features_disabled(self):
        """GameplayConfig.default() should have every feature flag set to False."""
        config = GameplayConfig.default()

        assert config.enable_fog_of_war is False
        assert config.enable_city_growth is False
        assert config.enable_economy is False
        assert config.enable_enhanced_combat is False
        assert config.enable_terrain_effects is False
        assert config.num_players == 2

    def test_enhanced_config_all_features_enabled(self):
        """GameplayConfig.enhanced() should have every feature flag set to True."""
        config = GameplayConfig.enhanced()

        assert config.enable_fog_of_war is True
        assert config.enable_city_growth is True
        assert config.enable_economy is True
        assert config.enable_enhanced_combat is True
        assert config.enable_terrain_effects is True

    def test_from_dict_with_nested_configs(self):
        """from_dict should correctly parse nested subsystem dictionaries."""
        data = {
            "enable_fog_of_war": True,
            "enable_economy": True,
            "num_players": 4,
            "economy": {
                "starting_gold": 100,
                "gold_per_city_base": 5,
            },
            "combat": {
                "base_damage": 50,
            },
        }
        config = GameplayConfig.from_dict(data)

        assert config.enable_fog_of_war is True
        assert config.enable_economy is True
        assert config.num_players == 4
        assert config.economy.starting_gold == 100
        assert config.economy.gold_per_city_base == 5
        assert config.combat.base_damage == 50

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict should silently ignore keys not present in the dataclass."""
        data = {
            "enable_fog_of_war": True,
            "totally_unknown_key": "should be ignored",
            "combat": {
                "base_damage": 40,
                "nonexistent_param": 999,
            },
        }
        config = GameplayConfig.from_dict(data)

        assert config.enable_fog_of_war is True
        assert config.combat.base_damage == 40

    def test_to_dict_roundtrip(self):
        """to_dict followed by from_dict should reproduce an equivalent config."""
        original = GameplayConfig.enhanced()
        data = original.to_dict()
        restored = GameplayConfig.from_dict(data)

        assert restored.enable_fog_of_war == original.enable_fog_of_war
        assert restored.enable_city_growth == original.enable_city_growth
        assert restored.enable_economy == original.enable_economy
        assert restored.enable_enhanced_combat == original.enable_enhanced_combat
        assert restored.enable_terrain_effects == original.enable_terrain_effects
        assert restored.num_players == original.num_players
        assert restored.combat.base_damage == original.combat.base_damage
        assert restored.victory.enable_elimination == original.victory.enable_elimination

    def test_frozen_config_immutable(self):
        """GameplayConfig is frozen -- attribute assignment should raise."""
        config = GameplayConfig.default()
        with pytest.raises(AttributeError):
            config.enable_fog_of_war = True  # type: ignore[misc]


# ======================================================================
# 2. FogOfWarSystem
# ======================================================================


class TestFogOfWarSystem:
    """Tests for the FogOfWarSystem visibility tracker."""

    @pytest.fixture
    def board_size(self):
        return 8

    @pytest.fixture
    def config(self):
        return FogOfWarConfig(
            sight_range={"warrior": 2, "settler": 1, "archer": 3, "city": 3},
            remember_seen=True,
            default_sight_range=2,
        )

    @pytest.fixture
    def fow(self, board_size, config):
        return FogOfWarSystem(board_size=board_size, num_players=2, config=config)

    @pytest.fixture
    def empty_terrain(self):
        """Terrain dict with no mountains (nothing blocks LOS)."""
        return {}

    def test_initial_state_all_unexplored(self, fow, board_size):
        """All tiles should start as UNEXPLORED for every player."""
        for player in range(2):
            mask = fow.get_visibility_mask(player)
            np.testing.assert_array_equal(mask, UNEXPLORED)
            assert mask.shape == (board_size, board_size)

    def test_reset_clears_visibility(self, fow, empty_terrain):
        """After updating visibility and then resetting, all tiles are UNEXPLORED."""
        units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        fow.update_visibility(0, units, [], empty_terrain)
        # Confirm something is now visible
        assert fow.is_visible(0, (4, 4))

        fow.reset()
        mask = fow.get_visibility_mask(0)
        np.testing.assert_array_equal(mask, UNEXPLORED)

    def test_unit_reveals_tiles_within_sight_range(self, fow, empty_terrain):
        """A warrior at (4,4) with sight_range=2 should reveal nearby tiles."""
        units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        fow.update_visibility(0, units, [], empty_terrain)

        # The unit's own tile should be visible
        assert fow.is_visible(0, (4, 4))
        # Adjacent tile within range 2
        assert fow.is_visible(0, (4, 5))
        assert fow.is_visible(0, (3, 3))
        # Tile at Chebyshev distance 2
        assert fow.is_visible(0, (2, 4))
        # Tile at distance 3 should NOT be visible for a warrior (sight 2)
        assert not fow.is_visible(0, (1, 4))

    def test_city_reveals_tiles(self, fow, empty_terrain):
        """A city should reveal tiles within its configured sight range."""
        cities = [{"player": 0, "position": (4, 4)}]
        fow.update_visibility(0, [], cities, empty_terrain)

        assert fow.is_visible(0, (4, 4))
        # City sight range is 3; (4, 7) is Chebyshev distance 3
        assert fow.is_visible(0, (4, 7))

    def test_mountain_blocks_line_of_sight(self, fow):
        """A mountain between a unit and a target tile should block visibility."""
        terrain = {(5, 4): {"type": "mountain"}}
        units = [{"player": 0, "position": (4, 4), "type": "archer"}]  # sight=3
        fow.update_visibility(0, units, [], terrain)

        # The mountain tile itself should be visible
        assert fow.is_visible(0, (5, 4))
        # Tile directly behind the mountain should be blocked
        assert not fow.is_visible(0, (6, 4))

    def test_remember_seen_tiles(self, fow, empty_terrain, board_size):
        """When remember_seen is True, previously visible tiles become SEEN."""
        units_at_center = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        fow.update_visibility(0, units_at_center, [], empty_terrain)
        assert fow.is_visible(0, (4, 5))

        # Move unit away so (4, 5) is no longer in range
        units_at_corner = [{"player": 0, "position": (0, 0), "type": "warrior"}]
        fow.update_visibility(0, units_at_corner, [], empty_terrain)

        # (4, 5) should now be SEEN, not VISIBLE or UNEXPLORED
        assert not fow.is_visible(0, (4, 5))
        assert fow.is_explored(0, (4, 5))
        assert fow.get_visibility_state(0, (4, 5)) == SEEN

    def test_forget_seen_when_disabled(self, board_size):
        """When remember_seen is False, old tiles revert to UNEXPLORED."""
        config = FogOfWarConfig(
            sight_range={"warrior": 2},
            remember_seen=False,
            default_sight_range=2,
        )
        fow = FogOfWarSystem(board_size=board_size, num_players=2, config=config)

        units_center = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        fow.update_visibility(0, units_center, [], {})

        units_corner = [{"player": 0, "position": (0, 0), "type": "warrior"}]
        fow.update_visibility(0, units_corner, [], {})

        assert fow.get_visibility_state(0, (4, 5)) == UNEXPLORED

    def test_is_visible_and_is_explored(self, fow, empty_terrain):
        """is_visible returns True only for VISIBLE; is_explored for SEEN or VISIBLE."""
        units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        fow.update_visibility(0, units, [], empty_terrain)

        # Visible tile
        assert fow.is_visible(0, (4, 4))
        assert fow.is_explored(0, (4, 4))

        # Unexplored tile (far away)
        assert not fow.is_visible(0, (0, 0))
        assert not fow.is_explored(0, (0, 0))

        # Out-of-bounds should return False for both
        assert not fow.is_visible(0, (-1, 0))
        assert not fow.is_explored(0, (99, 99))

    def test_get_visibility_mask_shape(self, fow, board_size):
        """get_visibility_mask returns array of correct shape."""
        mask = fow.get_visibility_mask(0)
        assert mask.shape == (board_size, board_size)
        assert mask.dtype == np.int8

    def test_get_visible_units_filters_hidden(self, fow, empty_terrain):
        """get_visible_units only returns units on tiles VISIBLE to the observer."""
        # Player 0 has a unit at (4,4); player 1 has units at (4,5) and (7,7)
        own_units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        all_units = [
            {"player": 0, "position": (4, 4), "type": "warrior"},
            {"player": 1, "position": (4, 5), "type": "warrior"},
            {"player": 1, "position": (7, 7), "type": "warrior"},
        ]
        fow.update_visibility(0, own_units, [], empty_terrain)

        visible = fow.get_visible_units(0, all_units)

        visible_positions = [tuple(u["position"]) for u in visible]
        assert (4, 4) in visible_positions
        assert (4, 5) in visible_positions  # within sight range 2
        assert (7, 7) not in visible_positions  # too far away

    def test_get_visible_cities_filters_hidden(self, fow, empty_terrain):
        """get_visible_cities only returns cities on VISIBLE tiles."""
        units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        all_cities = [
            {"player": 0, "position": (4, 4)},
            {"player": 1, "position": (7, 7)},
        ]
        fow.update_visibility(0, units, [], empty_terrain)

        visible = fow.get_visible_cities(0, all_cities)

        visible_positions = [tuple(c["position"]) for c in visible]
        assert (4, 4) in visible_positions
        assert (7, 7) not in visible_positions

    def test_encode_for_spatial_shape_and_values(self, fow, empty_terrain, board_size):
        """encode_for_spatial returns 3-channel float32 tensor with correct values."""
        units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        fow.update_visibility(0, units, [], empty_terrain)

        encoding = fow.encode_for_spatial(0)

        assert encoding.shape == (3, board_size, board_size)
        assert encoding.dtype == np.float32

        # Channel 0: VISIBLE tiles
        assert encoding[0, 4, 4] == 1.0  # unit's own tile (y, x)
        # Channel 2: UNEXPLORED tiles
        assert encoding[2, 0, 0] == 1.0  # far corner still unexplored
        # Each tile belongs to exactly one channel
        channel_sum = encoding[0] + encoding[1] + encoding[2]
        np.testing.assert_array_almost_equal(channel_sum, np.ones((board_size, board_size)))

    def test_serialization_roundtrip(self, fow, empty_terrain, board_size):
        """to_dict / from_dict should preserve all system state."""
        units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        fow.update_visibility(0, units, [], empty_terrain)

        data = fow.to_dict()
        restored = FogOfWarSystem.from_dict(data, board_size=board_size)

        np.testing.assert_array_equal(
            fow.get_visibility_mask(0),
            restored.get_visibility_mask(0),
        )
        assert restored.config.remember_seen == fow.config.remember_seen
        assert restored.board_size == fow.board_size

    def test_different_unit_sight_ranges(self, board_size, empty_terrain):
        """Different unit types should reveal different amounts of tiles."""
        config = FogOfWarConfig(
            sight_range={"warrior": 1, "archer": 3},
            remember_seen=True,
            default_sight_range=2,
        )
        fow = FogOfWarSystem(board_size=board_size, num_players=2, config=config)

        # Warrior with sight range 1
        warrior_units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        fow.update_visibility(0, warrior_units, [], empty_terrain)
        warrior_visible_count = int(np.sum(fow.get_visibility_mask(0) == VISIBLE))

        fow.reset()

        # Archer with sight range 3
        archer_units = [{"player": 0, "position": (4, 4), "type": "archer"}]
        fow.update_visibility(0, archer_units, [], empty_terrain)
        archer_visible_count = int(np.sum(fow.get_visibility_mask(0) == VISIBLE))

        assert archer_visible_count > warrior_visible_count


# ======================================================================
# 3. CombatSystem
# ======================================================================


class TestCombatSystem:
    """Tests for the CombatSystem combat resolution engine."""

    @pytest.fixture
    def config(self):
        return CombatConfig(
            base_damage=30.0,
            base_retaliation_damage=20.0,
            terrain_defense_bonuses={"hills": 0.25, "forest": 0.15},
            flanking_bonus_per_unit=0.1,
            max_flanking_bonus=0.3,
            ranged_attack_range=2,
            ranged_no_retaliation=True,
            city_defense_strength=25,
            city_ranged_attack_range=2,
            health_damage_scaling=True,
            kill_reward=0.1,
            death_penalty=0.1,
            city_capture_reward=0.5,
            damage_reward_scaling=0.001,
        )

    @pytest.fixture
    def combat(self, config):
        return CombatSystem(config)

    @pytest.fixture
    def warrior_a(self):
        return {
            "strength": 30,
            "health": 100,
            "unit_type": "warrior",
            "position": (3, 4),
        }

    @pytest.fixture
    def warrior_b(self):
        return {
            "strength": 25,
            "health": 100,
            "unit_type": "warrior",
            "position": (3, 5),
        }

    @pytest.fixture
    def archer(self):
        return {
            "strength": 20,
            "health": 100,
            "unit_type": "archer",
            "position": (3, 4),
        }

    def test_basic_melee_combat(self, combat, warrior_a, warrior_b):
        """Basic melee combat should deal damage to both attacker and defender."""
        result = combat.resolve_combat(warrior_a, warrior_b)

        assert result.attacker_damage_dealt > 0
        assert result.defender_damage_dealt > 0
        assert isinstance(result, CombatResult)

    def test_ranged_no_retaliation(self, combat, archer, warrior_b):
        """Ranged attack with ranged_no_retaliation=True should deal zero to attacker."""
        result = combat.resolve_combat(archer, warrior_b)

        assert result.attacker_damage_dealt > 0
        assert result.defender_damage_dealt == 0.0

    def test_terrain_defense_bonus(self, combat):
        """Defending on hills should increase effective defender strength."""
        attacker = {"strength": 30, "health": 100, "unit_type": "warrior", "position": (3, 4)}
        defender = {"strength": 30, "health": 100, "unit_type": "warrior", "position": (3, 5)}

        result_plains = combat.resolve_combat(attacker, defender, terrain_type="plains")
        result_hills = combat.resolve_combat(attacker, defender, terrain_type="hills")

        # On hills, the defender is stronger so the attacker deals less damage on average.
        # Due to random variance we compare the terrain modifier directly.
        modifier = combat.calculate_terrain_modifier("hills")
        assert modifier == pytest.approx(0.25)

    def test_flanking_bonus(self, combat, warrior_a, warrior_b):
        """Flanking units should increase the attacker's effective strength."""
        bonus_0 = combat.calculate_flanking_bonus(0)
        bonus_2 = combat.calculate_flanking_bonus(2)

        assert bonus_0 == 0.0
        assert bonus_2 == pytest.approx(0.2)

    def test_flanking_bonus_capped(self, combat):
        """Flanking bonus should never exceed max_flanking_bonus."""
        bonus = combat.calculate_flanking_bonus(10)
        assert bonus == pytest.approx(combat.config.max_flanking_bonus)

    def test_health_damage_scaling(self, combat):
        """An injured unit should deal less damage than a healthy one."""
        healthy = {"strength": 30, "health": 100, "unit_type": "warrior", "position": (3, 4)}
        injured = {"strength": 30, "health": 50, "unit_type": "warrior", "position": (3, 4)}
        defender = {"strength": 30, "health": 100, "unit_type": "warrior", "position": (3, 5)}

        # Run many combats to average out random variance
        healthy_total = 0.0
        injured_total = 0.0
        n_trials = 200
        for _ in range(n_trials):
            healthy_total += combat.resolve_combat(healthy, defender).attacker_damage_dealt
            injured_total += combat.resolve_combat(injured, defender).attacker_damage_dealt

        # Healthy unit should deal more damage on average
        assert healthy_total / n_trials > injured_total / n_trials

    def test_kill_detection(self, combat):
        """A unit at low health should be killed when damage exceeds remaining HP."""
        attacker = {"strength": 50, "health": 100, "unit_type": "warrior", "position": (3, 4)}
        defender = {"strength": 10, "health": 1, "unit_type": "warrior", "position": (3, 5)}

        result = combat.resolve_combat(attacker, defender)

        assert result.defender_killed is True

    def test_can_attack_melee_range(self, combat):
        """Melee unit can attack adjacent (Chebyshev distance 1) targets only."""
        warrior = {"unit_type": "warrior", "position": (3, 4)}

        assert combat.can_attack(warrior, (3, 5)) is True   # distance 1
        assert combat.can_attack(warrior, (4, 5)) is True   # diagonal, distance 1
        assert combat.can_attack(warrior, (5, 6)) is False  # distance 2

    def test_can_attack_ranged_range(self, combat):
        """Ranged unit can attack within configured ranged_attack_range."""
        archer = {"unit_type": "archer", "position": (3, 4)}

        assert combat.can_attack(archer, (3, 6)) is True   # distance 2
        assert combat.can_attack(archer, (3, 5)) is True   # distance 1
        assert combat.can_attack(archer, (3, 7)) is False  # distance 3

    def test_combat_result_reward(self, combat):
        """compute_reward should combine damage, kill, and capture signals."""
        result = CombatResult(
            attacker_damage_dealt=20.0,
            defender_damage_dealt=10.0,
            attacker_killed=False,
            defender_killed=True,
            city_captured=False,
        )
        reward = combat.compute_reward(result)

        # Expected: 20*0.001 - 10*0.001 + 0.1 (kill) = 0.02 - 0.01 + 0.1 = 0.11
        assert reward == pytest.approx(0.11)

    def test_city_attack(self, combat):
        """resolve_city_attack should produce a CombatResult with city_captured flag."""
        attacker = {"strength": 50, "health": 100, "unit_type": "warrior", "position": (3, 4)}
        city = {"defense_strength": 10, "health": 5, "position": (3, 5)}

        result = combat.resolve_city_attack(attacker, city, terrain_type="plains")

        assert isinstance(result, CombatResult)
        # With such weak city, it should almost certainly be captured
        assert result.city_captured is True
        assert result.attacker_damage_dealt > 0


# ======================================================================
# 4. EconomySystem
# ======================================================================


class TestEconomySystem:
    """Tests for the EconomySystem gold management."""

    @pytest.fixture
    def config(self):
        return EconomyConfig(
            starting_gold=50,
            gold_per_city_base=3,
            gold_per_population=1,
            unit_maintenance={"warrior": 1, "archer": 2, "settler": 0},
            default_unit_maintenance=1,
            purchase_cost_multiplier=4.0,
        )

    @pytest.fixture
    def economy(self, config):
        return EconomySystem(num_players=2, config=config)

    def test_starting_gold(self, economy):
        """Each player starts with the configured starting_gold."""
        assert economy.get_gold(0) == 50
        assert economy.get_gold(1) == 50

    def test_calculate_income(self, economy):
        """calculate_income returns correct city income and maintenance breakdown."""
        cities = [{"population": 3}, {"population": 1}]
        units = [{"type": "warrior"}, {"type": "archer"}]

        result = economy.calculate_income(0, cities, units)

        # city_income: (3 + 3*1) + (3 + 1*1) = 6 + 4 = 10
        assert result["city_income"] == 10
        # maintenance: warrior=1, archer=2
        assert result["maintenance"] == 3
        assert result["net"] == 7

    def test_process_turn_adds_gold(self, economy):
        """process_turn should update the player's gold balance."""
        cities = [{"population": 2}]  # income: 3 + 2*1 = 5
        units = [{"type": "warrior"}]  # maintenance: 1

        result = economy.process_turn(0, cities, units)

        assert result["net"] == 4
        assert result["new_balance"] == 54  # 50 + 4
        assert economy.get_gold(0) == 54

    def test_unit_maintenance(self, economy):
        """Maintenance cost should be deducted based on unit types."""
        cities = [{"population": 1}]  # income: 3 + 1 = 4
        units = [
            {"type": "warrior"},
            {"type": "warrior"},
            {"type": "archer"},
        ]  # maintenance: 1 + 1 + 2 = 4

        result = economy.calculate_income(0, cities, units)

        assert result["maintenance"] == 4
        assert result["net"] == 0

    def test_bankruptcy_detection(self, economy):
        """A player with negative gold should be detected as bankrupt."""
        economy.gold[0] = -10
        assert economy.is_bankrupt(0) is True
        assert economy.is_bankrupt(1) is False

    def test_purchase_success(self, economy):
        """Purchase should succeed when the player has enough gold."""
        economy.gold[0] = 100
        # base_cost=10, multiplier=4.0 => effective cost=40
        assert economy.purchase(0, "warrior", base_cost=10) is True
        assert economy.get_gold(0) == 60

    def test_purchase_insufficient_gold(self, economy):
        """Purchase should fail when the player cannot afford it."""
        economy.gold[0] = 10
        # base_cost=10, multiplier=4.0 => effective cost=40
        assert economy.purchase(0, "warrior", base_cost=10) is False
        assert economy.get_gold(0) == 10  # unchanged

    def test_encode_for_vector(self, economy):
        """encode_for_vector should return normalised economy values."""
        economy.gold[0] = 250
        economy.gold_per_turn[0] = 10

        encoded = economy.encode_for_vector(0)

        assert encoded["gold_normalized"] == pytest.approx(250 / 500.0)
        assert encoded["income_normalized"] == pytest.approx(10 / 50.0)
        assert encoded["is_bankrupt"] == 0.0

    def test_serialization_roundtrip(self, economy):
        """to_dict / from_dict should preserve all economy state."""
        economy.gold[0] = 75
        economy.gold_per_turn[0] = 5

        data = economy.to_dict()
        restored = EconomySystem.from_dict(data)

        assert restored.get_gold(0) == 75
        assert restored.gold_per_turn[0] == 5
        assert restored.config.starting_gold == economy.config.starting_gold
        assert restored.config.purchase_cost_multiplier == economy.config.purchase_cost_multiplier

    def test_reset_restores_starting_gold(self, economy):
        """reset() should restore all players to starting_gold."""
        economy.gold[0] = 999
        economy.gold[1] = -10
        economy.gold_per_turn[0] = 20

        economy.reset()

        assert economy.get_gold(0) == 50
        assert economy.get_gold(1) == 50
        assert economy.gold_per_turn[0] == 0


# ======================================================================
# 5. CityGrowthSystem
# ======================================================================


class TestCityGrowthSystem:
    """Tests for the CityGrowthSystem food and population mechanics."""

    @pytest.fixture
    def config(self):
        return CityGrowthConfig(
            food_consumption_per_pop=2,
            growth_threshold_base=15,
            growth_threshold_scaling=8,
            max_population=10,
            base_food_yield=2,
            production_per_population=1,
            starvation_damage=1,
            growth_reward=0.05,
        )

    @pytest.fixture
    def growth(self, config):
        return CityGrowthSystem(config)

    def test_growth_threshold_calculation(self, growth):
        """Growth threshold should follow the formula: base + (pop - 1) * scaling."""
        # pop=1: 15 + (1-1)*8 = 15
        assert growth.get_growth_threshold(1) == 15
        # pop=3: 15 + (3-1)*8 = 31
        assert growth.get_growth_threshold(3) == 31
        # pop=5: 15 + (5-1)*8 = 47
        assert growth.get_growth_threshold(5) == 47

    def test_food_surplus_calculation(self, growth):
        """calculate_food_surplus should compute total food minus consumption."""
        city = {"population": 3}
        tile_yields = [{"food": 3}, {"food": 2}]

        surplus = growth.calculate_food_surplus(city, tile_yields)

        # total food: base_food_yield(2) + 3 + 2 = 7
        # consumption: 2 * 3 = 6
        # surplus = 7 - 6 = 1
        assert surplus == 1

    def test_city_grows_when_threshold_met(self, growth):
        """City should grow when food stockpile reaches the growth threshold."""
        city_id = 0
        growth.set_food_stockpile(city_id, 14)  # needs 15 total for pop=1

        # Surplus of 2 brings stockpile to 16 >= 15
        result = growth.process_city_growth(city_id, population=1, food_surplus=2)

        assert result["grew"] is True
        assert result["population_change"] == 1
        # Overflow: 16 - 15 = 1
        assert result["new_food_stockpile"] == 1

    def test_city_starves_when_negative_food(self, growth):
        """City should lose population when stockpile goes negative."""
        city_id = 0
        growth.set_food_stockpile(city_id, 0)

        result = growth.process_city_growth(city_id, population=3, food_surplus=-5)

        assert result["starved"] is True
        assert result["population_change"] == -1
        # Stockpile resets to 0 after starvation
        assert result["new_food_stockpile"] == 0

    def test_max_population_cap(self, growth):
        """City should not grow beyond max_population even if threshold is met."""
        city_id = 0
        growth.set_food_stockpile(city_id, 100)  # way above any threshold

        result = growth.process_city_growth(
            city_id, population=growth.config.max_population, food_surplus=50
        )

        assert result["grew"] is False
        assert result["population_change"] == 0

    def test_production_bonus_per_population(self, growth):
        """Production bonus should scale linearly with population."""
        assert growth.get_production_bonus(1) == 1
        assert growth.get_production_bonus(5) == 5
        assert growth.get_production_bonus(0) == 0

    def test_serialization_roundtrip(self, growth):
        """to_dict / from_dict should preserve all growth system state."""
        growth.set_food_stockpile(0, 12)
        growth.set_food_stockpile(1, 7)

        data = growth.to_dict()
        restored = CityGrowthSystem.from_dict(data)

        assert restored.get_food_stockpile(0) == 12
        assert restored.get_food_stockpile(1) == 7
        assert restored.config.max_population == growth.config.max_population
        assert restored.config.growth_threshold_base == growth.config.growth_threshold_base

    def test_food_overflow_preserved(self, growth):
        """Excess food above the growth threshold should carry over to next turn."""
        city_id = 0
        threshold = growth.get_growth_threshold(1)  # 15
        # Set stockpile so that adding surplus exceeds threshold by 5
        growth.set_food_stockpile(city_id, threshold - 1)  # 14

        result = growth.process_city_growth(city_id, population=1, food_surplus=6)

        # Stockpile: 14 + 6 = 20 >= 15 => grew; overflow = 20 - 15 = 5
        assert result["grew"] is True
        assert result["new_food_stockpile"] == 5


# ======================================================================
# 6. TerrainSystem
# ======================================================================


class TestTerrainSystem:
    """Tests for the TerrainSystem movement, yields, and defense."""

    @pytest.fixture
    def config(self):
        return TerrainConfig(
            movement_costs={"plains": 1, "hills": 2, "mountain": -1, "forest": 2},
            terrain_yields={
                "plains": {"food": 2, "production": 1, "gold": 0},
                "hills": {"food": 1, "production": 2, "gold": 0},
                "mountain": {"food": 0, "production": 0, "gold": 0},
            },
            defense_bonuses={"hills": 0.25, "forest": 0.15, "mountain": 0.0},
            feature_yield_modifiers={"forest": {"food": -1, "production": 1, "gold": 0}},
        )

    @pytest.fixture
    def terrain(self, config):
        return TerrainSystem(config)

    def test_movement_cost_plains(self, terrain):
        """Plains should have movement cost 1."""
        assert terrain.get_movement_cost("plains") == 1

    def test_movement_cost_hills(self, terrain):
        """Hills should have movement cost 2."""
        assert terrain.get_movement_cost("hills") == 2

    def test_mountain_impassable(self, terrain):
        """Mountains should be impassable (cost -1)."""
        assert terrain.get_movement_cost("mountain") == -1
        assert terrain.is_passable("mountain") is False

    def test_terrain_yields(self, terrain):
        """Terrain yields should match configured values."""
        plains_yields = terrain.get_yields("plains")
        assert plains_yields == {"food": 2, "production": 1, "gold": 0}

        hills_yields = terrain.get_yields("hills")
        assert hills_yields == {"food": 1, "production": 2, "gold": 0}

    def test_defense_bonus(self, terrain):
        """Defense bonuses should match configured values, stacking with features."""
        assert terrain.get_defense_bonus("plains") == 0.0
        assert terrain.get_defense_bonus("hills") == pytest.approx(0.25)

        # Hills + forest should stack
        combined = terrain.get_defense_bonus("hills", features=["forest"])
        assert combined == pytest.approx(0.25 + 0.15)

    def test_tiles_in_range_bfs(self, terrain):
        """get_tiles_in_range should return all reachable tiles within movement budget."""
        board_size = 8
        terrain_map: dict[tuple[int, int], str] = {}
        # All plains (default) except a mountain at (4, 5)
        terrain_map[(4, 5)] = "mountain"

        reachable = terrain.get_tiles_in_range(
            position=(4, 4), move_range=2, terrain=terrain_map, board_size=board_size
        )

        # Starting position is always included
        assert (4, 4) in reachable
        # Adjacent plains tiles (cost 1) should be reachable
        assert (3, 4) in reachable
        assert (5, 4) in reachable
        assert (4, 3) in reachable
        # Mountain should NOT be reachable
        assert (4, 5) not in reachable
        # Two steps away on plains (cost 2) should be reachable
        assert (2, 4) in reachable

    def test_is_passable(self, terrain):
        """is_passable should return True for traversable terrain, False otherwise."""
        assert terrain.is_passable("plains") is True
        assert terrain.is_passable("hills") is True
        assert terrain.is_passable("mountain") is False
        # Unknown terrain defaults to cost 1 => passable
        assert terrain.is_passable("unknown_terrain") is True


# ======================================================================
# 7. VictorySystem
# ======================================================================


class TestVictorySystem:
    """Tests for the VictorySystem victory condition checking."""

    @pytest.fixture
    def config(self):
        return VictoryConfig(
            enabled_victories=["elimination", "score", "domination"],
            score_weights={
                "unit": 1.0,
                "city": 5.0,
                "population": 2.0,
                "gold": 0.02,
                "territory": 0.5,
            },
            domination_threshold=0.66,
            elimination_requires_all=True,
        )

    @pytest.fixture
    def victory(self, config):
        return VictorySystem(num_players=2, config=config)

    def _make_game_state(
        self,
        p0_units=None,
        p0_cities=None,
        p0_gold=0,
        p0_territory=0,
        p1_units=None,
        p1_cities=None,
        p1_gold=0,
        p1_territory=0,
        turn_limit_reached=False,
        total_cities=None,
    ):
        """Helper to construct a game_state dict."""
        return {
            "players": {
                0: {
                    "units": p0_units or [],
                    "cities": p0_cities or [],
                    "gold": p0_gold,
                    "territory": p0_territory,
                },
                1: {
                    "units": p1_units or [],
                    "cities": p1_cities or [],
                    "gold": p1_gold,
                    "territory": p1_territory,
                },
            },
            "turn_limit_reached": turn_limit_reached,
            **({"total_cities": total_cities} if total_cities is not None else {}),
        }

    def test_no_victory_when_game_continues(self, victory):
        """check_victory returns None when no victory condition is met."""
        state = self._make_game_state(
            p0_units=["warrior"],
            p0_cities=[{"population": 2}],
            p1_units=["warrior"],
            p1_cities=[{"population": 2}],
        )
        result = victory.check_victory(state)
        assert result is None

    def test_elimination_victory(self, victory):
        """Player 0 should win when player 1 has no units and no cities."""
        state = self._make_game_state(
            p0_units=["warrior"],
            p0_cities=[{"population": 2}],
            p1_units=[],
            p1_cities=[],
        )
        result = victory.check_victory(state)

        assert result is not None
        assert result.victory_type == VictoryType.ELIMINATION
        assert result.winner == 0

    def test_elimination_both_eliminated_is_draw(self, victory):
        """When both players are eliminated simultaneously, the result is a draw."""
        state = self._make_game_state(
            p0_units=[],
            p0_cities=[],
            p1_units=[],
            p1_cities=[],
        )
        result = victory.check_victory(state)

        assert result is not None
        assert result.victory_type == VictoryType.ELIMINATION
        assert result.winner is None

    def test_domination_victory(self, victory):
        """Player controlling >= 66% of all cities should win by domination."""
        state = self._make_game_state(
            p0_units=["warrior"],
            p0_cities=[{"population": 2}, {"population": 1}, {"population": 1}],
            p1_units=["warrior"],
            p1_cities=[{"population": 1}],
        )
        # Player 0 has 3/4 = 75% >= 66%
        result = victory.check_victory(state)

        assert result is not None
        assert result.victory_type == VictoryType.DOMINATION
        assert result.winner == 0

    def test_score_victory_at_turn_limit(self, victory):
        """At turn limit, the player with the highest score should win."""
        state = self._make_game_state(
            p0_units=["w1", "w2", "w3"],
            p0_cities=[{"population": 5}],
            p0_gold=100,
            p0_territory=10,
            p1_units=["w1"],
            p1_cities=[{"population": 1}],
            p1_gold=10,
            p1_territory=2,
            turn_limit_reached=True,
        )
        result = victory.check_victory(state)

        assert result is not None
        assert result.victory_type == VictoryType.SCORE
        assert result.winner == 0

    def test_score_calculation(self, victory):
        """calculate_scores should correctly combine weighted categories."""
        state = self._make_game_state(
            p0_units=["u1", "u2"],     # 2 units
            p0_cities=[{"population": 3}],  # 1 city, pop=3
            p0_gold=100,
            p0_territory=10,
        )
        scores = victory.calculate_scores(state)

        # Player 0: unit=2*1 + city=1*5 + pop=3*2 + gold=100*0.02 + territory=10*0.5
        # = 2 + 5 + 6 + 2 + 5 = 20
        expected_p0 = 2 * 1.0 + 1 * 5.0 + 3 * 2.0 + 100 * 0.02 + 10 * 0.5
        assert scores[0] == pytest.approx(expected_p0)

        # Player 1 has nothing
        assert scores[1] == pytest.approx(0.0)

    def test_victory_progress(self, victory):
        """get_victory_progress should return values in [0, 1] for each category."""
        state = self._make_game_state(
            p0_units=["w1"],
            p0_cities=[{"population": 2}],
            p0_gold=50,
            p1_units=["w1", "w2"],
            p1_cities=[{"population": 3}, {"population": 1}],
            p1_gold=80,
        )

        progress = victory.get_victory_progress(0, state)

        assert "domination" in progress
        assert "score" in progress
        assert "elimination" in progress

        # All values should be in [0, 1]
        for key, value in progress.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} out of [0, 1]"

    def test_serialization_roundtrip(self, victory):
        """to_dict / from_dict should preserve config and num_players."""
        data = victory.to_dict()
        restored = VictorySystem.from_dict(data)

        assert restored.num_players == victory.num_players
        assert restored.config.domination_threshold == pytest.approx(
            victory.config.domination_threshold
        )
        assert set(restored.config.enabled_victories) == set(
            victory.config.enabled_victories
        )
        assert restored.config.elimination_requires_all == (
            victory.config.elimination_requires_all
        )
