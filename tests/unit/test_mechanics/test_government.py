"""Unit tests for alpha_zero/mechanics/government.py."""

from __future__ import annotations

import pytest

from alpha_zero.mechanics.government import (
    DEFAULT_GOVERNMENTS,
    GovernmentConfig,
    GovernmentManager,
    GovernmentTier,
    PolicySlots,
    PolicySlotType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> GovernmentConfig:
    return GovernmentConfig(
        change_cooldown_turns=5,
        anarchy_turns=1,
        legacy_bonus_per_turn=0.1,
        max_legacy_bonus=15.0,
    )


@pytest.fixture()
def manager(config: GovernmentConfig) -> GovernmentManager:
    return GovernmentManager(config)


@pytest.fixture()
def manager_with_autocracy(manager: GovernmentManager) -> GovernmentManager:
    manager.set_government(0, "autocracy")
    return manager


# ---------------------------------------------------------------------------
# GovernmentConfig
# ---------------------------------------------------------------------------


class TestGovernmentConfig:
    def test_defaults(self) -> None:
        cfg = GovernmentConfig()
        assert cfg.change_cooldown_turns == 10
        assert cfg.legacy_bonus_per_turn == pytest.approx(0.1)

    def test_custom_values(self) -> None:
        cfg = GovernmentConfig(change_cooldown_turns=3, max_legacy_bonus=10.0)
        assert cfg.change_cooldown_turns == 3
        assert cfg.max_legacy_bonus == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# PolicySlots
# ---------------------------------------------------------------------------


class TestPolicySlots:
    def test_total_slots(self) -> None:
        slots = PolicySlots(military=2, economic=2, diplomatic=1, wildcard=1)
        assert slots.total() == 6

    def test_get_slot_count(self) -> None:
        slots = PolicySlots(military=3, economic=2, diplomatic=0, wildcard=1)
        assert slots.get_slot_count(PolicySlotType.MILITARY) == 3
        assert slots.get_slot_count(PolicySlotType.DIPLOMATIC) == 0
        assert slots.get_slot_count(PolicySlotType.WILDCARD) == 1

    def test_default_slots(self) -> None:
        slots = PolicySlots()
        assert slots.total() == 2


# ---------------------------------------------------------------------------
# GovernmentType (default governments)
# ---------------------------------------------------------------------------


class TestDefaultGovernments:
    def test_chiefdom_exists(self) -> None:
        assert "chiefdom" in DEFAULT_GOVERNMENTS

    def test_democracy_has_eight_slots(self) -> None:
        gov = DEFAULT_GOVERNMENTS["democracy"]
        assert gov.slots.total() == 8

    def test_communism_has_no_diplomatic(self) -> None:
        gov = DEFAULT_GOVERNMENTS["communism"]
        assert gov.slots.diplomatic == 0

    def test_fascism_is_modern_tier(self) -> None:
        gov = DEFAULT_GOVERNMENTS["fascism"]
        assert gov.tier == GovernmentTier.MODERN

    def test_autocracy_has_inherent_effects(self) -> None:
        gov = DEFAULT_GOVERNMENTS["autocracy"]
        assert "capital_yield_bonus" in gov.inherent_effects


# ---------------------------------------------------------------------------
# GovernmentManager – set_government
# ---------------------------------------------------------------------------


class TestSetGovernment:
    def test_set_government_success(self, manager: GovernmentManager) -> None:
        result = manager.set_government(0, "chiefdom")
        assert result is True
        gov = manager.get_player_government(0)
        assert gov is not None
        assert gov.government_id == "chiefdom"

    def test_set_unknown_government_fails(self, manager: GovernmentManager) -> None:
        result = manager.set_government(0, "utopia")
        assert result is False

    def test_cooldown_prevents_change(self, manager: GovernmentManager) -> None:
        manager.set_government(0, "chiefdom")
        result = manager.set_government(0, "autocracy")
        assert result is False

    def test_force_bypasses_cooldown(self, manager: GovernmentManager) -> None:
        manager.set_government(0, "chiefdom")
        result = manager.set_government(0, "autocracy", force=True)
        assert result is True

    def test_legacy_bonus_recorded_on_change(self, manager: GovernmentManager) -> None:
        manager.set_government(0, "chiefdom")
        # Simulate 10 turns in government
        for _ in range(10):
            manager.process_turn()
        manager.set_government(0, "autocracy", force=True)
        legacy = manager._player_legacy.get(0, {})
        assert "chiefdom" in legacy
        assert legacy["chiefdom"] > 0

    def test_legacy_bonus_capped_at_max(self, manager: GovernmentManager) -> None:
        cfg = GovernmentConfig(legacy_bonus_per_turn=10.0, max_legacy_bonus=5.0)
        mgr = GovernmentManager(cfg)
        mgr.set_government(0, "chiefdom")
        for _ in range(5):
            mgr.process_turn()
        mgr.set_government(0, "autocracy", force=True)
        assert mgr._player_legacy[0]["chiefdom"] <= 5.0

    def test_cooldown_decrements_each_turn(self, manager: GovernmentManager) -> None:
        cfg = GovernmentConfig(change_cooldown_turns=2)
        mgr = GovernmentManager(cfg)
        mgr.set_government(0, "chiefdom")
        mgr.process_turn()
        mgr.process_turn()
        # Cooldown expired after 2 turns
        result = mgr.set_government(0, "autocracy")
        assert result is True


# ---------------------------------------------------------------------------
# GovernmentManager – slot_policy / unslot_policy
# ---------------------------------------------------------------------------


class TestSlotPolicy:
    def test_slot_military_policy(self, manager_with_autocracy: GovernmentManager) -> None:
        result = manager_with_autocracy.slot_policy(0, "discipline", slot_idx=0)
        assert result is True

    def test_slot_unknown_policy_fails(self, manager_with_autocracy: GovernmentManager) -> None:
        result = manager_with_autocracy.slot_policy(0, "nonexistent", slot_idx=0)
        assert result is False

    def test_slot_invalid_index_fails(self, manager_with_autocracy: GovernmentManager) -> None:
        # autocracy has 3 slots (military=2, economic=1)
        result = manager_with_autocracy.slot_policy(0, "discipline", slot_idx=99)
        assert result is False

    def test_slot_wrong_type_fails(self, manager_with_autocracy: GovernmentManager) -> None:
        # slot_idx=0 is MILITARY, god_king is ECONOMIC
        result = manager_with_autocracy.slot_policy(0, "god_king", slot_idx=0)
        assert result is False

    def test_slot_economic_in_correct_slot(self, manager_with_autocracy: GovernmentManager) -> None:
        # autocracy: military=2 (slots 0,1), economic=1 (slot 2)
        result = manager_with_autocracy.slot_policy(0, "god_king", slot_idx=2)
        assert result is True

    def test_slot_requires_government(self, manager: GovernmentManager) -> None:
        result = manager.slot_policy(0, "discipline", slot_idx=0)
        assert result is False

    def test_unslot_policy(self, manager_with_autocracy: GovernmentManager) -> None:
        manager_with_autocracy.slot_policy(0, "discipline", slot_idx=0)
        result = manager_with_autocracy.unslot_policy(0, slot_idx=0)
        assert result is True
        policies = manager_with_autocracy.get_player_policies(0)
        assert len(policies) == 0

    def test_unslot_empty_slot(self, manager_with_autocracy: GovernmentManager) -> None:
        result = manager_with_autocracy.unslot_policy(0, slot_idx=0)
        assert result is False


# ---------------------------------------------------------------------------
# GovernmentManager – get_slot_type
# ---------------------------------------------------------------------------


class TestGetSlotType:
    def test_slot_type_military(self, manager: GovernmentManager) -> None:
        slots = PolicySlots(military=2, economic=1, diplomatic=0, wildcard=1)
        assert manager._get_slot_type(slots, 0) == PolicySlotType.MILITARY
        assert manager._get_slot_type(slots, 1) == PolicySlotType.MILITARY

    def test_slot_type_economic(self, manager: GovernmentManager) -> None:
        slots = PolicySlots(military=2, economic=1, diplomatic=0, wildcard=1)
        assert manager._get_slot_type(slots, 2) == PolicySlotType.ECONOMIC

    def test_slot_type_wildcard(self, manager: GovernmentManager) -> None:
        slots = PolicySlots(military=2, economic=1, diplomatic=0, wildcard=1)
        assert manager._get_slot_type(slots, 3) == PolicySlotType.WILDCARD


# ---------------------------------------------------------------------------
# GovernmentManager – get_total_effects
# ---------------------------------------------------------------------------


class TestGetTotalEffects:
    def test_inherent_effects_included(self, manager_with_autocracy: GovernmentManager) -> None:
        effects = manager_with_autocracy.get_total_effects(0)
        assert "capital_yield_bonus" in effects
        assert effects["capital_yield_bonus"] == pytest.approx(1.0)

    def test_policy_effects_added(self, manager_with_autocracy: GovernmentManager) -> None:
        manager_with_autocracy.slot_policy(0, "discipline", slot_idx=0)
        effects = manager_with_autocracy.get_total_effects(0)
        assert "barbarian_combat_bonus" in effects
        assert effects["barbarian_combat_bonus"] == pytest.approx(5.0)

    def test_multiple_policies_stack(self, manager_with_autocracy: GovernmentManager) -> None:
        manager_with_autocracy.slot_policy(0, "discipline", slot_idx=0)
        manager_with_autocracy.slot_policy(0, "god_king", slot_idx=2)
        effects = manager_with_autocracy.get_total_effects(0)
        assert "barbarian_combat_bonus" in effects
        assert "capital_faith_bonus" in effects

    def test_no_government_returns_empty(self, manager: GovernmentManager) -> None:
        effects = manager.get_total_effects(0)
        assert effects == {}

    def test_legacy_effects_included(self, manager: GovernmentManager) -> None:
        # Set chiefdom, accumulate turns, switch
        manager.set_government(0, "chiefdom")
        for _ in range(5):
            manager.process_turn()
        manager.set_government(0, "autocracy", force=True)
        # Autocracy has legacy_effects; small value from chiefdom (which has none)
        effects = manager.get_total_effects(0)
        # Main check: no error, returns dict
        assert isinstance(effects, dict)


# ---------------------------------------------------------------------------
# GovernmentManager – get_available_governments / policies
# ---------------------------------------------------------------------------


class TestAvailableLists:
    def test_chiefdom_available_no_civics(self, manager: GovernmentManager) -> None:
        available = manager.get_available_governments(0, unlocked_civics=set())
        assert "chiefdom" in available

    def test_autocracy_requires_civic(self, manager: GovernmentManager) -> None:
        available = manager.get_available_governments(0, unlocked_civics=set())
        assert "autocracy" not in available

    def test_autocracy_available_with_civic(self, manager: GovernmentManager) -> None:
        available = manager.get_available_governments(0, unlocked_civics={"Political Philosophy"})
        assert "autocracy" in available

    def test_available_policies_filtered_by_civic(self, manager: GovernmentManager) -> None:
        policies = manager.get_available_policies(0, unlocked_civics={"Code of Laws"})
        assert "discipline" in policies

    def test_available_policies_blocked_by_missing_civic(self, manager: GovernmentManager) -> None:
        policies = manager.get_available_policies(0, unlocked_civics=set())
        assert "discipline" not in policies

    def test_available_policies_filter_by_slot_type(self, manager: GovernmentManager) -> None:
        policies = manager.get_available_policies(
            0,
            unlocked_civics={"Code of Laws"},
            slot_type=PolicySlotType.ECONOMIC,
        )
        assert "discipline" not in policies  # discipline is MILITARY
        assert "god_king" in policies  # god_king is ECONOMIC


# ---------------------------------------------------------------------------
# GovernmentManager – process_turn
# ---------------------------------------------------------------------------


class TestProcessTurn:
    def test_turns_in_government_increments(
        self, manager_with_autocracy: GovernmentManager
    ) -> None:
        manager_with_autocracy.process_turn()
        assert manager_with_autocracy._turns_in_government.get(0, 0) == 1

    def test_policy_turns_active_increment(self, manager_with_autocracy: GovernmentManager) -> None:
        manager_with_autocracy.slot_policy(0, "discipline", slot_idx=0)
        manager_with_autocracy.process_turn()
        slot = manager_with_autocracy._player_policies[0][0]
        assert slot.turns_active == 1


# ---------------------------------------------------------------------------
# GovernmentManager – encode_for_state
# ---------------------------------------------------------------------------


class TestEncodeForState:
    def test_encode_no_government(self, manager: GovernmentManager) -> None:
        enc = manager.encode_for_state(0)
        assert enc["government_tier"] == 0
        assert enc["num_policy_slots"] == 0

    def test_encode_with_government(self, manager_with_autocracy: GovernmentManager) -> None:
        enc = manager_with_autocracy.encode_for_state(0)
        assert enc["government_tier"] > 0
        assert enc["num_policy_slots"] == 3  # autocracy has 3 slots

    def test_encode_tier_encoding_one_hot(self, manager_with_autocracy: GovernmentManager) -> None:
        enc = manager_with_autocracy.encode_for_state(0)
        tier_enc = enc["government_tier_encoding"]
        assert len(tier_enc) == len(GovernmentTier)
        assert sum(tier_enc) == pytest.approx(1.0)

    def test_encode_military_policies_counted(
        self, manager_with_autocracy: GovernmentManager
    ) -> None:
        manager_with_autocracy.slot_policy(0, "discipline", slot_idx=0)
        enc = manager_with_autocracy.encode_for_state(0)
        assert enc["military_policies"] == 1
        assert enc["economic_policies"] == 0


# ---------------------------------------------------------------------------
# GovernmentManager – reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_governments(self, manager_with_autocracy: GovernmentManager) -> None:
        manager_with_autocracy.reset()
        assert manager_with_autocracy.get_player_government(0) is None

    def test_reset_clears_policies(self, manager_with_autocracy: GovernmentManager) -> None:
        manager_with_autocracy.slot_policy(0, "discipline", slot_idx=0)
        manager_with_autocracy.reset()
        assert manager_with_autocracy.get_player_policies(0) == []

    def test_reset_clears_legacy(self, manager: GovernmentManager) -> None:
        manager.set_government(0, "chiefdom")
        for _ in range(5):
            manager.process_turn()
        manager.set_government(0, "autocracy", force=True)
        manager.reset()
        assert manager._player_legacy == {}
