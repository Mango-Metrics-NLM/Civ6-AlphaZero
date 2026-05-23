"""Unit tests for alpha_zero/mechanics/great_people.py."""

from __future__ import annotations

import pytest

from alpha_zero.mechanics.great_people import (
    DEFAULT_GREAT_PEOPLE,
    ActivationType,
    GreatPeopleConfig,
    GreatPeopleManager,
    GreatPerson,
    GreatPersonAbility,
    GreatPersonDefinition,
    GreatPersonEra,
    GreatPersonType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> GreatPeopleConfig:
    return GreatPeopleConfig(
        base_points_per_district=2.0,
        points_per_building=1.0,
        recruit_cost_gold_multiplier=0.5,
        patronage_bonus=2.0,
        max_active_per_type=10,
        prophet_faith_cost_base=25,
        prophet_faith_cost_increase=15,
    )


@pytest.fixture()
def manager(config: GreatPeopleConfig) -> GreatPeopleManager:
    return GreatPeopleManager(config)


@pytest.fixture()
def minimal_definition() -> GreatPersonDefinition:
    """A single scientist definition for isolated testing."""
    ability = GreatPersonAbility(
        ability_id="test_ability",
        name="Test Ability",
        description="Does something",
        activation_type=ActivationType.RETIRE,
        effects={"science_boost": 50.0},
        charges=1,
        target_required=False,
    )
    return GreatPersonDefinition(
        person_id="test_scientist",
        name="Test Scientist",
        person_type=GreatPersonType.GREAT_SCIENTIST,
        era=GreatPersonEra.ANCIENT,
        passive_bonus="+1 Science",
        passive_effects={"science_bonus": 1.0},
        active_ability=ability,
        recruitment_cost=60,
    )


@pytest.fixture()
def minimal_manager(
    config: GreatPeopleConfig, minimal_definition: GreatPersonDefinition
) -> GreatPeopleManager:
    return GreatPeopleManager(config, definitions=[minimal_definition])


# ---------------------------------------------------------------------------
# GreatPeopleConfig
# ---------------------------------------------------------------------------


class TestGreatPeopleConfig:
    def test_defaults(self) -> None:
        cfg = GreatPeopleConfig()
        assert cfg.base_points_per_district == pytest.approx(2.0)
        assert cfg.max_active_per_type == 10

    def test_custom_values(self) -> None:
        cfg = GreatPeopleConfig(base_points_per_district=3.0, max_active_per_type=5)
        assert cfg.base_points_per_district == pytest.approx(3.0)
        assert cfg.max_active_per_type == 5


# ---------------------------------------------------------------------------
# GreatPersonDefinition / GreatPersonAbility
# ---------------------------------------------------------------------------


class TestDefinitionDataclasses:
    def test_definition_defaults(self, minimal_definition: GreatPersonDefinition) -> None:
        assert minimal_definition.recruitment_cost == 60
        assert minimal_definition.person_type == GreatPersonType.GREAT_SCIENTIST

    def test_ability_fields(self, minimal_definition: GreatPersonDefinition) -> None:
        ability = minimal_definition.active_ability
        assert ability is not None
        assert ability.charges == 1
        assert ability.target_required is False

    def test_default_great_people_populated(self) -> None:
        types = {d.person_type for d in DEFAULT_GREAT_PEOPLE}
        assert GreatPersonType.GREAT_SCIENTIST in types
        assert GreatPersonType.GREAT_ENGINEER in types

    def test_default_great_people_have_abilities(self) -> None:
        for d in DEFAULT_GREAT_PEOPLE:
            # All default definitions have an active ability
            assert d.active_ability is not None


# ---------------------------------------------------------------------------
# GreatPerson instance
# ---------------------------------------------------------------------------


class TestGreatPersonInstance:
    def test_post_init_sets_charges(self, minimal_definition: GreatPersonDefinition) -> None:
        person = GreatPerson(
            instance_id="gp_1",
            definition=minimal_definition,
            owner_id=0,
        )
        assert person.charges_remaining == 1

    def test_name_property(self, minimal_definition: GreatPersonDefinition) -> None:
        person = GreatPerson(
            instance_id="gp_1",
            definition=minimal_definition,
            owner_id=0,
        )
        assert person.name == "Test Scientist"

    def test_type_property(self, minimal_definition: GreatPersonDefinition) -> None:
        person = GreatPerson(
            instance_id="gp_1",
            definition=minimal_definition,
            owner_id=0,
        )
        assert person.person_type == GreatPersonType.GREAT_SCIENTIST

    def test_initial_active_state(self, minimal_definition: GreatPersonDefinition) -> None:
        person = GreatPerson(
            instance_id="gp_1",
            definition=minimal_definition,
            owner_id=0,
        )
        assert person.is_active is True
        assert person.turns_alive == 0


# ---------------------------------------------------------------------------
# GreatPeopleManager – initialization
# ---------------------------------------------------------------------------


class TestManagerInit:
    def test_manager_initializes_available(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        defn = minimal_manager.get_available_for_recruitment(GreatPersonType.GREAT_SCIENTIST)
        assert defn is not None
        assert defn.person_id == "test_scientist"

    def test_manager_loads_default_definitions(
        self, manager: GreatPeopleManager
    ) -> None:
        assert len(manager._definitions) == len(DEFAULT_GREAT_PEOPLE)

    def test_unavailable_type_returns_none(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        # Only scientist defined, no general available
        defn = minimal_manager.get_available_for_recruitment(GreatPersonType.GREAT_GENERAL)
        assert defn is None


# ---------------------------------------------------------------------------
# GreatPeopleManager – add_points / get_points
# ---------------------------------------------------------------------------


class TestPoints:
    def test_add_points(self, manager: GreatPeopleManager) -> None:
        total = manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 10.0)
        assert total == pytest.approx(10.0)

    def test_add_points_accumulates(self, manager: GreatPeopleManager) -> None:
        manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 10.0)
        manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 15.0)
        assert manager.get_points(0, GreatPersonType.GREAT_SCIENTIST) == pytest.approx(25.0)

    def test_get_points_new_player_zero(self, manager: GreatPeopleManager) -> None:
        assert manager.get_points(99, GreatPersonType.GREAT_SCIENTIST) == pytest.approx(0.0)

    def test_points_independent_per_type(self, manager: GreatPeopleManager) -> None:
        manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 20.0)
        manager.add_points(0, GreatPersonType.GREAT_ENGINEER, 5.0)
        assert manager.get_points(0, GreatPersonType.GREAT_SCIENTIST) == pytest.approx(20.0)
        assert manager.get_points(0, GreatPersonType.GREAT_ENGINEER) == pytest.approx(5.0)

    def test_points_also_added_to_recruitment_progress(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 30.0)
        progress = minimal_manager._available[GreatPersonType.GREAT_SCIENTIST]
        assert progress.player_points.get(0, 0.0) == pytest.approx(30.0)

    def test_points_independent_per_player(self, manager: GreatPeopleManager) -> None:
        manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 10.0)
        manager.add_points(1, GreatPersonType.GREAT_SCIENTIST, 20.0)
        assert manager.get_points(0, GreatPersonType.GREAT_SCIENTIST) == pytest.approx(10.0)
        assert manager.get_points(1, GreatPersonType.GREAT_SCIENTIST) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# GreatPeopleManager – can_recruit
# ---------------------------------------------------------------------------


class TestCanRecruit:
    def test_can_recruit_with_enough_points(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        can, reason = minimal_manager.can_recruit(0, "test_scientist")
        assert can is True

    def test_cannot_recruit_insufficient_points(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 30.0)
        can, reason = minimal_manager.can_recruit(0, "test_scientist")
        assert can is False
        assert "Insufficient" in reason

    def test_cannot_recruit_unknown_person(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        can, reason = minimal_manager.can_recruit(0, "nobody")
        assert can is False

    def test_cannot_recruit_already_taken(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        minimal_manager.recruit(0, "test_scientist", (0, 0))
        can, reason = minimal_manager.can_recruit(1, "test_scientist")
        assert can is False
        assert "Already recruited" in reason


# ---------------------------------------------------------------------------
# GreatPeopleManager – recruit
# ---------------------------------------------------------------------------


class TestRecruit:
    def test_recruit_returns_great_person(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        person = minimal_manager.recruit(0, "test_scientist", (5, 5))
        assert person is not None
        assert person.owner_id == 0
        assert person.position == (5, 5)

    def test_recruit_marks_definition_recruited(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        minimal_manager.recruit(0, "test_scientist", (0, 0))
        assert "test_scientist" in minimal_manager._recruited_definitions

    def test_recruit_deducts_points(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 100.0)
        minimal_manager.recruit(0, "test_scientist", (0, 0))
        remaining = minimal_manager.get_points(0, GreatPersonType.GREAT_SCIENTIST)
        assert remaining == pytest.approx(40.0)  # 100 - 60

    def test_recruit_fails_without_points(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        person = minimal_manager.recruit(0, "test_scientist", (0, 0))
        assert person is None

    def test_recruit_advances_available(
        self, manager: GreatPeopleManager
    ) -> None:
        manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 200.0)
        first = manager.get_available_for_recruitment(GreatPersonType.GREAT_SCIENTIST)
        assert first is not None
        manager.recruit(0, first.person_id, (0, 0))
        second = manager.get_available_for_recruitment(GreatPersonType.GREAT_SCIENTIST)
        # A different scientist should now be available
        if second is not None:
            assert second.person_id != first.person_id


# ---------------------------------------------------------------------------
# GreatPeopleManager – purchase_with_gold
# ---------------------------------------------------------------------------


class TestPurchaseWithGold:
    def test_purchase_with_gold_success(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        # No points → cost = 60 * 0.5 = 30 gold
        person, cost = minimal_manager.purchase_with_gold(0, "test_scientist", (0, 0), 100)
        assert person is not None
        assert cost == 30

    def test_purchase_cost_reduces_with_points(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 30.0)
        # Needs 30 more points → 30 * 0.5 = 15 gold
        person, cost = minimal_manager.purchase_with_gold(0, "test_scientist", (0, 0), 50)
        assert person is not None
        assert cost == 15

    def test_purchase_insufficient_gold_fails(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        person, cost = minimal_manager.purchase_with_gold(0, "test_scientist", (0, 0), 5)
        assert person is None
        assert cost == 0

    def test_purchase_unknown_person_fails(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        person, cost = minimal_manager.purchase_with_gold(0, "nobody", (0, 0), 1000)
        assert person is None

    def test_purchase_unavailable_type_fails(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        # No general defined in minimal_manager
        person, cost = minimal_manager.purchase_with_gold(
            0, "sun_tzu", (0, 0), 1000
        )
        assert person is None


# ---------------------------------------------------------------------------
# GreatPeopleManager – activate_ability
# ---------------------------------------------------------------------------


class TestActivateAbility:
    def _get_person(self, minimal_manager: GreatPeopleManager) -> GreatPerson:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        person = minimal_manager.recruit(0, "test_scientist", (0, 0))
        assert person is not None
        return person

    def test_activate_returns_effects(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        person = self._get_person(minimal_manager)
        effects = minimal_manager.activate_ability(person.instance_id)
        assert effects is not None
        assert "science_boost" in effects

    def test_activate_retire_deactivates_person(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        person = self._get_person(minimal_manager)
        minimal_manager.activate_ability(person.instance_id)
        assert person.is_active is False

    def test_activate_consumes_charge(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        person = self._get_person(minimal_manager)
        assert person.charges_remaining == 1
        minimal_manager.activate_ability(person.instance_id)
        assert person.charges_remaining == 0

    def test_activate_inactive_person_fails(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        person = self._get_person(minimal_manager)
        person.is_active = False
        result = minimal_manager.activate_ability(person.instance_id)
        assert result is None

    def test_activate_unknown_instance_fails(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        result = minimal_manager.activate_ability("ghost_gp")
        assert result is None

    def test_activate_requires_target_when_needed(
        self, config: GreatPeopleConfig
    ) -> None:
        ability = GreatPersonAbility(
            ability_id="target_ability",
            name="Targeted",
            description="Needs target",
            activation_type=ActivationType.RETIRE,
            effects={"foo": 1.0},
            charges=1,
            target_required=True,
        )
        defn = GreatPersonDefinition(
            person_id="targeted_person",
            name="Targeted Person",
            person_type=GreatPersonType.GREAT_SCIENTIST,
            era=GreatPersonEra.ANCIENT,
            active_ability=ability,
            recruitment_cost=60,
        )
        mgr = GreatPeopleManager(config, definitions=[defn])
        mgr.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        person = mgr.recruit(0, "targeted_person", (0, 0))
        assert person is not None
        # Without target → fails
        result = mgr.activate_ability(person.instance_id, target=None)
        assert result is None
        # With target → succeeds
        result = mgr.activate_ability(person.instance_id, target=(3, 3))
        assert result is not None


# ---------------------------------------------------------------------------
# GreatPeopleManager – get_passive_effects
# ---------------------------------------------------------------------------


class TestPassiveEffects:
    def test_passive_effects_from_recruited(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        minimal_manager.recruit(0, "test_scientist", (0, 0))
        effects = minimal_manager.get_passive_effects(0)
        assert "science_bonus" in effects
        assert effects["science_bonus"] == pytest.approx(1.0)

    def test_passive_effects_empty_before_recruitment(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        effects = minimal_manager.get_passive_effects(0)
        assert effects == {}

    def test_inactive_person_excluded_from_passive(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        person = minimal_manager.recruit(0, "test_scientist", (0, 0))
        assert person is not None
        person.is_active = False
        effects = minimal_manager.get_passive_effects(0)
        assert effects == {}

    def test_passive_effects_independent_per_player(
        self, manager: GreatPeopleManager
    ) -> None:
        effects_p1 = manager.get_passive_effects(1)
        assert effects_p1 == {}


# ---------------------------------------------------------------------------
# GreatPeopleManager – get_player_great_people
# ---------------------------------------------------------------------------


class TestGetPlayerGreatPeople:
    def test_get_active_people(self, minimal_manager: GreatPeopleManager) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        minimal_manager.recruit(0, "test_scientist", (0, 0))
        people = minimal_manager.get_player_great_people(0)
        assert len(people) == 1

    def test_inactive_excluded_by_default(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        person = minimal_manager.recruit(0, "test_scientist", (0, 0))
        assert person is not None
        person.is_active = False
        people = minimal_manager.get_player_great_people(0)
        assert len(people) == 0

    def test_inactive_included_when_flag_false(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        person = minimal_manager.recruit(0, "test_scientist", (0, 0))
        assert person is not None
        person.is_active = False
        people = minimal_manager.get_player_great_people(0, active_only=False)
        assert len(people) == 1


# ---------------------------------------------------------------------------
# GreatPeopleManager – get_recruitment_progress
# ---------------------------------------------------------------------------


class TestRecruitmentProgress:
    def test_returns_progress_dict(self, minimal_manager: GreatPeopleManager) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 30.0)
        progress = minimal_manager.get_recruitment_progress(
            GreatPersonType.GREAT_SCIENTIST, 0
        )
        assert progress is not None
        assert progress["player_points"] == pytest.approx(30.0)
        assert progress["cost"] == 60

    def test_returns_none_for_unavailable_type(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        progress = minimal_manager.get_recruitment_progress(
            GreatPersonType.GREAT_GENERAL, 0
        )
        assert progress is None

    def test_leader_tracking(self, minimal_manager: GreatPeopleManager) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 40.0)
        minimal_manager.add_points(1, GreatPersonType.GREAT_SCIENTIST, 50.0)
        progress = minimal_manager.get_recruitment_progress(
            GreatPersonType.GREAT_SCIENTIST, 0
        )
        assert progress is not None
        assert progress["leader_player_id"] == 1
        assert progress["leader_points"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# GreatPeopleManager – process_turn
# ---------------------------------------------------------------------------


class TestProcessTurn:
    def test_turns_available_increments(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.process_turn()
        progress = minimal_manager._available.get(GreatPersonType.GREAT_SCIENTIST)
        assert progress is not None
        assert progress.turns_available == 1

    def test_turns_alive_increments(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        person = minimal_manager.recruit(0, "test_scientist", (0, 0))
        assert person is not None
        minimal_manager.process_turn()
        assert person.turns_alive == 1


# ---------------------------------------------------------------------------
# GreatPeopleManager – encode_for_state
# ---------------------------------------------------------------------------


class TestEncodeForState:
    def test_encode_no_great_people(self, manager: GreatPeopleManager) -> None:
        enc = manager.encode_for_state(0)
        assert enc["num_great_people"] == 0
        assert enc["scientists"] == 0

    def test_encode_counts_scientists(self, minimal_manager: GreatPeopleManager) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        minimal_manager.recruit(0, "test_scientist", (0, 0))
        enc = minimal_manager.encode_for_state(0)
        assert enc["scientists"] == 1
        assert enc["num_great_people"] == 1

    def test_encode_points_tracked(self, manager: GreatPeopleManager) -> None:
        manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 25.0)
        enc = manager.encode_for_state(0)
        assert enc["points_scientist"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# GreatPeopleManager – reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_recruited(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        minimal_manager.recruit(0, "test_scientist", (0, 0))
        minimal_manager.reset()
        assert minimal_manager.get_player_great_people(0) == []

    def test_reset_clears_points(self, minimal_manager: GreatPeopleManager) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        minimal_manager.reset()
        assert minimal_manager.get_points(0, GreatPersonType.GREAT_SCIENTIST) == pytest.approx(
            0.0
        )

    def test_reset_reinitializes_available(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        minimal_manager.recruit(0, "test_scientist", (0, 0))
        minimal_manager.reset()
        # After reset, test_scientist should be available again
        defn = minimal_manager.get_available_for_recruitment(GreatPersonType.GREAT_SCIENTIST)
        assert defn is not None

    def test_reset_clears_recruited_definitions(
        self, minimal_manager: GreatPeopleManager
    ) -> None:
        minimal_manager.add_points(0, GreatPersonType.GREAT_SCIENTIST, 60.0)
        minimal_manager.recruit(0, "test_scientist", (0, 0))
        minimal_manager.reset()
        assert len(minimal_manager._recruited_definitions) == 0
