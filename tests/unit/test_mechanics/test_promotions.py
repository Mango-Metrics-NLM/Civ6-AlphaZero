"""Unit tests for alpha_zero/mechanics/promotions.py."""

from __future__ import annotations

import pytest

from alpha_zero.mechanics.promotions import (
    DEFAULT_PROMOTIONS,
    Ability,
    AbilityType,
    PromotionConfig,
    PromotionManager,
    PromotionTier,
    StatModifier,
    StatModifierType,
    UnitClass,
    UnitExperience,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> PromotionConfig:
    return PromotionConfig(
        xp_per_combat=5,
        xp_from_kill=10,
        xp_from_city_capture=20,
        xp_from_pillage=3,
        xp_per_level=[15, 30, 45, 60, 75, 90, 105],
        max_promotions=7,
        heal_on_promotion=50,
    )


@pytest.fixture()
def manager(config: PromotionConfig) -> PromotionManager:
    return PromotionManager(config)


@pytest.fixture()
def warrior(manager: PromotionManager) -> tuple[PromotionManager, str]:
    manager.register_unit("warrior_1", UnitClass.MELEE)
    return manager, "warrior_1"


@pytest.fixture()
def archer(manager: PromotionManager) -> tuple[PromotionManager, str]:
    manager.register_unit("archer_1", UnitClass.RANGED)
    return manager, "archer_1"


# ---------------------------------------------------------------------------
# PromotionConfig
# ---------------------------------------------------------------------------


class TestPromotionConfig:
    def test_defaults(self) -> None:
        cfg = PromotionConfig()
        assert cfg.xp_per_combat == 5
        assert cfg.max_promotions == 7
        assert cfg.heal_on_promotion == 50

    def test_xp_per_level_list(self) -> None:
        cfg = PromotionConfig()
        assert len(cfg.xp_per_level) == 7
        assert cfg.xp_per_level[0] == 15


# ---------------------------------------------------------------------------
# StatModifier / Ability
# ---------------------------------------------------------------------------


class TestStatModifierAndAbility:
    def test_stat_modifier_defaults(self) -> None:
        mod = StatModifier(StatModifierType.COMBAT_STRENGTH, 5.0)
        assert mod.value == pytest.approx(5.0)
        assert mod.condition is None

    def test_ability_defaults(self) -> None:
        ab = Ability(AbilityType.FLANKING_BONUS)
        assert ab.value == pytest.approx(1.0)
        assert ab.conditions == []


# ---------------------------------------------------------------------------
# Default promotions
# ---------------------------------------------------------------------------


class TestDefaultPromotions:
    def test_battlecry_exists(self) -> None:
        assert "battlecry" in DEFAULT_PROMOTIONS

    def test_battlecry_tier_1(self) -> None:
        assert DEFAULT_PROMOTIONS["battlecry"].tier == PromotionTier.TIER_1

    def test_commando_requires_battlecry(self) -> None:
        assert "battlecry" in DEFAULT_PROMOTIONS["commando"].prerequisites

    def test_volley_for_ranged(self) -> None:
        prom = DEFAULT_PROMOTIONS["volley"]
        assert UnitClass.RANGED in prom.unit_classes
        assert UnitClass.MELEE not in prom.unit_classes

    def test_arrow_storm_tier_2(self) -> None:
        assert DEFAULT_PROMOTIONS["arrow_storm"].tier == PromotionTier.TIER_2


# ---------------------------------------------------------------------------
# PromotionManager – register_unit
# ---------------------------------------------------------------------------


class TestRegisterUnit:
    def test_register_returns_unit_experience(self, manager: PromotionManager) -> None:
        ue = manager.register_unit("w1", UnitClass.MELEE)
        assert isinstance(ue, UnitExperience)
        assert ue.unit_id == "w1"
        assert ue.unit_class == UnitClass.MELEE

    def test_register_with_starting_xp(self, manager: PromotionManager) -> None:
        ue = manager.register_unit("w1", UnitClass.MELEE, starting_xp=15)
        assert ue.experience == 15

    def test_starting_xp_at_threshold_sets_pending(self, manager: PromotionManager) -> None:
        ue = manager.register_unit("w1", UnitClass.MELEE, starting_xp=15)
        assert ue.pending_promotion is True

    def test_register_below_threshold_not_pending(self, manager: PromotionManager) -> None:
        ue = manager.register_unit("w1", UnitClass.MELEE, starting_xp=5)
        assert ue.pending_promotion is False


# ---------------------------------------------------------------------------
# PromotionManager – add_experience
# ---------------------------------------------------------------------------


class TestAddExperience:
    def test_add_experience_basic(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        total = manager.add_experience(unit_id, 10)
        assert total == 10

    def test_add_experience_with_multiplier(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        total = manager.add_experience(unit_id, 10, xp_multiplier=2.0)
        assert total == 20

    def test_add_experience_triggers_pending(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        unit = manager._units[unit_id]
        assert unit.pending_promotion is True

    def test_add_experience_unknown_unit(self, manager: PromotionManager) -> None:
        result = manager.add_experience("ghost", 10)
        assert result == 0

    def test_add_experience_accumulates(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 5)
        manager.add_experience(unit_id, 7)
        assert manager._units[unit_id].experience == 12


# ---------------------------------------------------------------------------
# PromotionManager – get_available_promotions
# ---------------------------------------------------------------------------


class TestGetAvailablePromotions:
    def test_no_promotions_when_not_pending(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        available = manager.get_available_promotions(unit_id)
        assert available == []

    def test_tier1_available_when_pending(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        available = manager.get_available_promotions(unit_id)
        promo_ids = [p.promotion_id for p in available]
        assert "battlecry" in promo_ids
        assert "tortoise" in promo_ids

    def test_ranged_promotions_only_for_ranged(self, archer: tuple) -> None:
        manager, unit_id = archer
        manager.add_experience(unit_id, 15)
        available = manager.get_available_promotions(unit_id)
        promo_ids = [p.promotion_id for p in available]
        assert "volley" in promo_ids
        assert "battlecry" not in promo_ids

    def test_tier2_unavailable_without_tier1(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        available = manager.get_available_promotions(unit_id)
        promo_ids = [p.promotion_id for p in available]
        assert "commando" not in promo_ids

    def test_tier2_available_after_tier1(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        manager.apply_promotion(unit_id, "battlecry")
        manager.add_experience(unit_id, 30)  # enough for next level
        available = manager.get_available_promotions(unit_id)
        promo_ids = [p.promotion_id for p in available]
        assert "commando" in promo_ids

    def test_unknown_unit_returns_empty(self, manager: PromotionManager) -> None:
        assert manager.get_available_promotions("ghost") == []


# ---------------------------------------------------------------------------
# PromotionManager – apply_promotion
# ---------------------------------------------------------------------------


class TestApplyPromotion:
    def test_apply_promotion_success(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        result = manager.apply_promotion(unit_id, "battlecry")
        assert result is not None
        assert result["promotion"] == "battlecry"
        assert result["new_level"] == 1

    def test_apply_promotion_adds_to_list(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        manager.apply_promotion(unit_id, "battlecry")
        unit = manager._units[unit_id]
        assert "battlecry" in unit.promotions

    def test_apply_promotion_clears_pending(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        manager.apply_promotion(unit_id, "battlecry")
        unit = manager._units[unit_id]
        assert unit.pending_promotion is False

    def test_apply_promotion_returns_heal_amount(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        result = manager.apply_promotion(unit_id, "battlecry")
        assert result is not None
        assert result["heal_amount"] == manager.config.heal_on_promotion

    def test_apply_promotion_returns_stat_modifiers(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        result = manager.apply_promotion(unit_id, "battlecry")
        assert result is not None
        assert len(result["stat_modifiers"]) > 0

    def test_apply_promotion_not_pending_fails(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        result = manager.apply_promotion(unit_id, "battlecry")
        assert result is None

    def test_apply_promotion_unknown_unit_fails(self, manager: PromotionManager) -> None:
        result = manager.apply_promotion("ghost", "battlecry")
        assert result is None

    def test_apply_promotion_unknown_promotion_fails(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        result = manager.apply_promotion(unit_id, "nonexistent_promo")
        assert result is None

    def test_apply_unavailable_promotion_fails(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        # commando requires battlecry first
        result = manager.apply_promotion(unit_id, "commando")
        assert result is None


# ---------------------------------------------------------------------------
# PromotionManager – get_unit_stat_modifiers
# ---------------------------------------------------------------------------


class TestGetUnitStatModifiers:
    def test_no_modifiers_before_promotion(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        mods = manager.get_unit_stat_modifiers(unit_id)
        assert mods == {}

    def test_modifiers_after_battlecry(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        manager.apply_promotion(unit_id, "battlecry")
        mods = manager.get_unit_stat_modifiers(unit_id, context={"vs_melee": True})
        assert StatModifierType.COMBAT_STRENGTH in mods
        assert mods[StatModifierType.COMBAT_STRENGTH] == pytest.approx(7.0)

    def test_conditional_modifier_excluded_without_context(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        manager.apply_promotion(unit_id, "battlecry")
        # No context → conditional modifiers excluded
        mods = manager.get_unit_stat_modifiers(unit_id, context={})
        assert mods.get(StatModifierType.COMBAT_STRENGTH, 0.0) == pytest.approx(0.0)

    def test_unconditional_modifier_always_applied(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        manager.apply_promotion(unit_id, "battlecry")
        manager.add_experience(unit_id, 30)
        manager.apply_promotion(unit_id, "commando")
        # commando grants +1 MOVEMENT unconditionally
        mods = manager.get_unit_stat_modifiers(unit_id, context={})
        assert StatModifierType.MOVEMENT in mods
        assert mods[StatModifierType.MOVEMENT] == pytest.approx(1.0)

    def test_unknown_unit_returns_empty(self, manager: PromotionManager) -> None:
        assert manager.get_unit_stat_modifiers("ghost") == {}


# ---------------------------------------------------------------------------
# PromotionManager – get_unit_abilities / has_ability
# ---------------------------------------------------------------------------


class TestAbilities:
    def _get_recon_with_ranger(self, manager: PromotionManager) -> str:
        manager.register_unit("scout_1", UnitClass.RECON)
        manager.add_experience("scout_1", 15)
        manager.apply_promotion("scout_1", "ranger")
        return "scout_1"

    def test_abilities_empty_before_promotion(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        assert manager.get_unit_abilities(unit_id) == []

    def test_ability_returned_after_promotion(self, manager: PromotionManager) -> None:
        unit_id = self._get_recon_with_ranger(manager)
        abilities = manager.get_unit_abilities(unit_id)
        assert len(abilities) > 0

    def test_has_ability_true(self, manager: PromotionManager) -> None:
        unit_id = self._get_recon_with_ranger(manager)
        assert manager.has_ability(unit_id, AbilityType.TERRAIN_BONUS, context={"forest": True})

    def test_has_ability_false_wrong_context(self, manager: PromotionManager) -> None:
        unit_id = self._get_recon_with_ranger(manager)
        # ranger requires forest or jungle context
        assert not manager.has_ability(unit_id, AbilityType.TERRAIN_BONUS, context={})

    def test_has_ability_false_no_promotion(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        assert not manager.has_ability(unit_id, AbilityType.FLANKING_BONUS)

    def test_unknown_unit_abilities_empty(self, manager: PromotionManager) -> None:
        assert manager.get_unit_abilities("ghost") == []


# ---------------------------------------------------------------------------
# PromotionManager – get_unit_info
# ---------------------------------------------------------------------------


class TestGetUnitInfo:
    def test_unit_info_fields(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        info = manager.get_unit_info(unit_id)
        assert info is not None
        for key in ("unit_id", "unit_class", "experience", "level", "promotions"):
            assert key in info

    def test_unit_info_unknown_returns_none(self, manager: PromotionManager) -> None:
        assert manager.get_unit_info("ghost") is None

    def test_unit_info_next_level_xp(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        info = manager.get_unit_info(unit_id)
        assert info is not None
        assert info["next_level_xp"] == manager.config.xp_per_level[0]

    def test_unit_info_next_level_none_at_max(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        unit = manager._units[unit_id]
        unit.level = len(manager.config.xp_per_level)
        info = manager.get_unit_info(unit_id)
        assert info is not None
        assert info["next_level_xp"] is None


# ---------------------------------------------------------------------------
# PromotionManager – remove_unit
# ---------------------------------------------------------------------------


class TestRemoveUnit:
    def test_remove_unit_success(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        result = manager.remove_unit(unit_id)
        assert result is True
        assert manager.get_unit_info(unit_id) is None

    def test_remove_unknown_unit_false(self, manager: PromotionManager) -> None:
        assert manager.remove_unit("ghost") is False


# ---------------------------------------------------------------------------
# PromotionManager – encode_for_state
# ---------------------------------------------------------------------------


class TestEncodeForState:
    def test_encode_no_units(self, manager: PromotionManager) -> None:
        enc = manager.encode_for_state(0, unit_ids=[])
        assert enc["total_promotions"] == 0
        assert enc["avg_unit_level"] == pytest.approx(0.0)

    def test_encode_with_units(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        enc = manager.encode_for_state(0, unit_ids=[unit_id])
        assert enc["units_tracked"] == 1

    def test_encode_pending_promotions_counted(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        enc = manager.encode_for_state(0, unit_ids=[unit_id])
        assert enc["pending_promotions"] == 1

    def test_encode_after_promotion(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.add_experience(unit_id, 15)
        manager.apply_promotion(unit_id, "battlecry")
        enc = manager.encode_for_state(0, unit_ids=[unit_id])
        assert enc["total_promotions"] == 1
        assert enc["avg_unit_level"] == pytest.approx(1.0)

    def test_encode_unknown_unit_skipped(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        enc = manager.encode_for_state(0, unit_ids=[unit_id, "ghost"])
        # ghost is skipped gracefully
        assert enc["units_tracked"] == 2  # list length counts, but ghost has no data


# ---------------------------------------------------------------------------
# PromotionManager – reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_units(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.reset()
        assert manager.get_unit_info(unit_id) is None

    def test_can_register_after_reset(self, warrior: tuple) -> None:
        manager, unit_id = warrior
        manager.reset()
        ue = manager.register_unit(unit_id, UnitClass.RANGED)
        assert ue.unit_class == UnitClass.RANGED
