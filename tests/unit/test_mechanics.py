"""
Comprehensive Tests for Game Mechanics Systems

Tests diplomacy, religion, and culture mechanics with full coverage.
"""

from __future__ import annotations

import pytest

from alpha_zero.mechanics.diplomacy import (
    DiplomacyConfig,
    DiplomacyManager,
    DiplomacyType,
    DiplomaticAction,
    DiplomaticActionType,
    DiplomaticModifier,
    DiplomaticRelation,
    ModifierType,
)
from alpha_zero.mechanics.religion import (
    BeliefType,
    Religion,
    ReligionConfig,
    ReligionManager,
    ReligiousBelief,
    ReligiousPressure,
    ReligiousUnit,
)
from alpha_zero.mechanics.culture import (
    CultureConfig,
    CultureManager,
    CulturalInfluence,
    CulturalVictoryProgress,
    GreatWork,
    GreatWorkType,
    TourismSource,
)


# =============================================================================
# Diplomacy Tests
# =============================================================================


class TestDiplomacyConfig:
    """Tests for DiplomacyConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DiplomacyConfig()
        assert config.war_threshold == -80
        assert config.peace_treaty_duration == 10
        assert config.alliance_duration == 50

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = DiplomacyConfig(
            war_threshold=-100,
            peace_treaty_duration=15,
            alliance_duration=40,
        )
        assert config.war_threshold == -100
        assert config.peace_treaty_duration == 15
        assert config.alliance_duration == 40


class TestDiplomaticModifier:
    """Tests for DiplomaticModifier."""

    def test_creation(self) -> None:
        """Test modifier creation."""
        modifier = DiplomaticModifier(
            modifier_type=ModifierType.TRADE_PARTNER,
            value=15,
            turns_remaining=10,
        )
        assert modifier.modifier_type == ModifierType.TRADE_PARTNER
        assert modifier.value == 15
        assert modifier.turns_remaining == 10

    def test_decay(self) -> None:
        """Test modifier decay."""
        modifier = DiplomaticModifier(
            modifier_type=ModifierType.TRADE_PARTNER,
            value=15,
            turns_remaining=5,
        )
        expired = modifier.decay(1)
        assert modifier.turns_remaining == 4
        assert not expired

    def test_permanent_modifier(self) -> None:
        """Test permanent modifier (turns_remaining = -1)."""
        modifier = DiplomaticModifier(
            modifier_type=ModifierType.SAME_RELIGION,
            value=10,
            turns_remaining=-1,
        )
        assert modifier.turns_remaining == -1

    def test_to_dict(self) -> None:
        """Test serialization."""
        modifier = DiplomaticModifier(
            modifier_type=ModifierType.ATTACKED_US,
            value=-50,
            turns_remaining=10,
        )
        data = modifier.to_dict()
        assert data["modifier_type"] == "ATTACKED_US"
        assert data["value"] == -50

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "modifier_type": "TRADE_PARTNER",
            "value": 15,
            "turns_remaining": 5,
        }
        modifier = DiplomaticModifier.from_dict(data)
        assert modifier.modifier_type == ModifierType.TRADE_PARTNER
        assert modifier.value == 15


class TestDiplomaticRelation:
    """Tests for DiplomaticRelation."""

    def test_creation(self) -> None:
        """Test relation creation."""
        relation = DiplomaticRelation(player_a=0, player_b=1)
        assert relation.player_a == 0
        assert relation.player_b == 1
        assert relation.relation_type == DiplomacyType.NEUTRAL

    def test_relationship_score(self) -> None:
        """Test relationship score calculation."""
        relation = DiplomaticRelation(player_a=0, player_b=1)
        relation.add_modifier(DiplomaticModifier(
            modifier_type=ModifierType.TRADE_PARTNER,
            value=15,
        ))
        relation.add_modifier(DiplomaticModifier(
            modifier_type=ModifierType.SAME_RELIGION,
            value=10,
        ))
        assert relation.relationship_score == 25

    def test_add_modifier(self) -> None:
        """Test adding modifiers."""
        relation = DiplomaticRelation(player_a=0, player_b=1)
        modifier = DiplomaticModifier(
            modifier_type=ModifierType.TRADE_PARTNER,
            value=15,
        )
        relation.add_modifier(modifier)
        assert len(relation.modifiers) == 1

    def test_has_treaty(self) -> None:
        """Test treaty checking."""
        relation = DiplomaticRelation(player_a=0, player_b=1)
        assert not relation.has_treaty("peace")
        relation.add_treaty("peace", 10, 0)
        assert relation.has_treaty("peace")

    def test_to_dict(self) -> None:
        """Test serialization."""
        relation = DiplomaticRelation(
            player_a=0,
            player_b=1,
            relation_type=DiplomacyType.FRIENDLY,
        )
        data = relation.to_dict()
        assert data["player_a"] == 0
        assert data["player_b"] == 1
        assert data["relation_type"] == "FRIENDLY"


class TestDiplomaticAction:
    """Tests for DiplomaticAction."""

    def test_creation(self) -> None:
        """Test action creation."""
        action = DiplomaticAction(
            action_type=DiplomaticActionType.DECLARE_WAR,
            source_player=0,
            target_player=1,
        )
        assert action.action_type == DiplomaticActionType.DECLARE_WAR
        assert action.source_player == 0
        assert action.target_player == 1

    def test_to_dict(self) -> None:
        """Test serialization."""
        action = DiplomaticAction(
            action_type=DiplomaticActionType.MAKE_PEACE,
            source_player=1,
            target_player=2,
        )
        data = action.to_dict()
        assert data["action_type"] == "MAKE_PEACE"


class TestDiplomacyManager:
    """Tests for DiplomacyManager."""

    @pytest.fixture
    def manager(self) -> DiplomacyManager:
        """Create a diplomacy manager."""
        return DiplomacyManager()

    def test_initialization(self, manager: DiplomacyManager) -> None:
        """Test manager initialization."""
        assert manager.config is not None

    def test_get_relation(self, manager: DiplomacyManager) -> None:
        """Test getting relations."""
        relation = manager.get_relation(0, 1)
        assert relation.player_a == 0
        assert relation.player_b == 1

    def test_get_relation_canonical(self, manager: DiplomacyManager) -> None:
        """Test that relation keys are canonical (lower ID first)."""
        relation1 = manager.get_relation(0, 1)
        relation2 = manager.get_relation(1, 0)
        assert relation1 is relation2

    def test_are_at_war(self, manager: DiplomacyManager) -> None:
        """Test war status check."""
        assert not manager.are_at_war(0, 1)
        relation = manager.get_relation(0, 1)
        relation.relation_type = DiplomacyType.WAR
        assert manager.are_at_war(0, 1)

    def test_are_allied(self, manager: DiplomacyManager) -> None:
        """Test alliance status check."""
        assert not manager.are_allied(0, 1)
        relation = manager.get_relation(0, 1)
        relation.relation_type = DiplomacyType.ALLIANCE
        assert manager.are_allied(0, 1)

    def test_can_declare_war(self, manager: DiplomacyManager) -> None:
        """Test war declaration validation."""
        can_war, reason = manager.can_declare_war(0, 1, current_turn=0)
        assert can_war

        # Can't declare war if already at war
        relation = manager.get_relation(0, 1)
        relation.relation_type = DiplomacyType.WAR
        can_war, reason = manager.can_declare_war(0, 1, current_turn=0)
        assert not can_war
        assert "Already at war" in reason

    def test_can_make_peace(self, manager: DiplomacyManager) -> None:
        """Test peace validation."""
        # Can't make peace if not at war
        can_peace, reason = manager.can_make_peace(0, 1)
        assert not can_peace

        # Can make peace if at war
        relation = manager.get_relation(0, 1)
        relation.relation_type = DiplomacyType.WAR
        can_peace, reason = manager.can_make_peace(0, 1)
        assert can_peace

    def test_apply_action_war(self, manager: DiplomacyManager) -> None:
        """Test applying war declaration."""
        action = DiplomaticAction(
            action_type=DiplomaticActionType.DECLARE_WAR,
            source_player=0,
            target_player=1,
        )
        result = manager.apply_action(action, current_turn=0)
        assert result is True
        assert manager.are_at_war(0, 1)

    def test_apply_action_peace(self, manager: DiplomacyManager) -> None:
        """Test applying peace treaty."""
        # First go to war
        relation = manager.get_relation(0, 1)
        relation.relation_type = DiplomacyType.WAR

        action = DiplomaticAction(
            action_type=DiplomaticActionType.MAKE_PEACE,
            source_player=0,
            target_player=1,
        )
        result = manager.apply_action(action, current_turn=0)
        assert result is True
        assert relation.relation_type == DiplomacyType.PEACE_TREATY

    def test_process_turn(self, manager: DiplomacyManager) -> None:
        """Test turn processing."""
        relation = manager.get_relation(0, 1)
        relation.add_modifier(DiplomaticModifier(
            modifier_type=ModifierType.TRADE_PARTNER,
            value=15,
            turns_remaining=2,
        ))
        manager.process_turn(current_turn=0)
        assert relation.modifiers[0].turns_remaining == 1

    def test_diplomatic_vector(self, manager: DiplomacyManager) -> None:
        """Test feature vector generation."""
        vector = manager.get_diplomatic_vector(0, num_players=4)
        assert isinstance(vector, list)
        assert len(vector) > 0
        # Should have 5 features per other player (3 others)
        assert len(vector) == 4 * 5

    def test_serialization(self, manager: DiplomacyManager) -> None:
        """Test serialization/deserialization."""
        relation = manager.get_relation(0, 1)
        relation.relation_type = DiplomacyType.WAR

        data = manager.to_dict()
        assert "relations" in data

        restored = DiplomacyManager.from_dict(data)
        assert restored.are_at_war(0, 1)


# =============================================================================
# Religion Tests
# =============================================================================


class TestReligionConfig:
    """Tests for ReligionConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ReligionConfig()
        assert config.max_religions > 0
        assert config.great_prophet_faith_cost > 0
        assert config.missionary_faith_cost > 0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ReligionConfig(
            max_religions=8,
            great_prophet_faith_cost=300,
            missionary_charges=5,
        )
        assert config.max_religions == 8
        assert config.great_prophet_faith_cost == 300
        assert config.missionary_charges == 5


class TestReligiousBelief:
    """Tests for ReligiousBelief."""

    def test_creation(self) -> None:
        """Test belief creation."""
        belief = ReligiousBelief(
            name="Test Belief",
            belief_type=BeliefType.FOUNDER,
            effect_type="faith_per_city",
            effect_value=2.0,
        )
        assert belief.name == "Test Belief"
        assert belief.belief_type == BeliefType.FOUNDER
        assert belief.effect_type == "faith_per_city"
        assert belief.effect_value == 2.0

    def test_to_dict(self) -> None:
        """Test serialization."""
        belief = ReligiousBelief(
            name="Test",
            belief_type=BeliefType.FOLLOWER,
            effect_type="food_bonus",
            effect_value=1.5,
        )
        data = belief.to_dict()
        assert data["name"] == "Test"
        assert data["belief_type"] == "FOLLOWER"
        assert data["effect_type"] == "food_bonus"

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "name": "Test Belief",
            "belief_type": "FOUNDER",
            "effect_type": "faith_per_city",
            "effect_value": 2.0,
        }
        belief = ReligiousBelief.from_dict(data)
        assert belief.name == "Test Belief"
        assert belief.belief_type == BeliefType.FOUNDER


class TestReligion:
    """Tests for Religion."""

    def test_creation(self) -> None:
        """Test religion creation."""
        religion = Religion(
            id=1,
            name="Test Religion",
            founder_player=0,
            holy_city=(5, 5),
        )
        assert religion.id == 1
        assert religion.name == "Test Religion"
        assert religion.founder_player == 0
        assert religion.holy_city == (5, 5)

    def test_add_belief(self) -> None:
        """Test adding beliefs."""
        religion = Religion(
            id=1,
            name="Test",
            founder_player=0,
            holy_city=(5, 5),
        )
        belief = ReligiousBelief(
            name="Test Belief",
            belief_type=BeliefType.FOUNDER,
            effect_type="faith_bonus",
            effect_value=5.0,
        )
        result = religion.add_belief(belief)
        assert result is True
        assert len(religion.beliefs) == 1

    def test_add_duplicate_belief_type(self) -> None:
        """Test that duplicate belief types are rejected."""
        religion = Religion(
            id=1,
            name="Test",
            founder_player=0,
            holy_city=(5, 5),
        )
        belief1 = ReligiousBelief(
            name="Belief 1",
            belief_type=BeliefType.FOUNDER,
            effect_type="bonus1",
            effect_value=1.0,
        )
        belief2 = ReligiousBelief(
            name="Belief 2",
            belief_type=BeliefType.FOUNDER,
            effect_type="bonus2",
            effect_value=2.0,
        )
        religion.add_belief(belief1)
        result = religion.add_belief(belief2)
        assert result is False
        assert len(religion.beliefs) == 1

    def test_get_belief(self) -> None:
        """Test getting beliefs by type."""
        religion = Religion(
            id=1,
            name="Test",
            founder_player=0,
            holy_city=(5, 5),
        )
        belief = ReligiousBelief(
            name="Test Belief",
            belief_type=BeliefType.WORSHIP,
            effect_type="building",
            effect_value=1.0,
        )
        religion.add_belief(belief)
        found = religion.get_belief(BeliefType.WORSHIP)
        assert found is not None
        assert found.name == "Test Belief"

    def test_to_dict(self) -> None:
        """Test serialization."""
        religion = Religion(
            id=1,
            name="Test",
            founder_player=0,
            holy_city=(5, 5),
            founded_turn=10,
        )
        data = religion.to_dict()
        assert data["id"] == 1
        assert data["name"] == "Test"
        assert data["holy_city"] == (5, 5)

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "id": 1,
            "name": "Test",
            "founder_player": 0,
            "holy_city": [5, 5],
            "beliefs": [],
            "founded_turn": 10,
        }
        religion = Religion.from_dict(data)
        assert religion.id == 1
        assert religion.holy_city == (5, 5)


class TestReligiousUnit:
    """Tests for ReligiousUnit."""

    def test_creation(self) -> None:
        """Test unit creation."""
        from alpha_zero.mechanics.religion import ReligiousUnitType

        unit = ReligiousUnit(
            id=1,
            unit_type=ReligiousUnitType.MISSIONARY,
            player=0,
            position=(5, 5),
            religion_id=1,
            charges_remaining=3,
        )
        assert unit.id == 1
        assert unit.charges_remaining == 3
        assert unit.unit_type == ReligiousUnitType.MISSIONARY

    def test_use_charge(self) -> None:
        """Test using charges."""
        from alpha_zero.mechanics.religion import ReligiousUnitType

        unit = ReligiousUnit(
            id=1,
            unit_type=ReligiousUnitType.MISSIONARY,
            player=0,
            position=(5, 5),
            religion_id=1,
            charges_remaining=3,
        )
        result = unit.use_charge()
        assert result is True
        assert unit.charges_remaining == 2

    def test_use_charge_empty(self) -> None:
        """Test using charge when empty."""
        from alpha_zero.mechanics.religion import ReligiousUnitType

        unit = ReligiousUnit(
            id=1,
            unit_type=ReligiousUnitType.MISSIONARY,
            player=0,
            position=(5, 5),
            religion_id=1,
            charges_remaining=0,
        )
        result = unit.use_charge()
        assert result is False
        assert unit.charges_remaining == 0


class TestReligiousPressure:
    """Tests for ReligiousPressure."""

    def test_creation(self) -> None:
        """Test pressure creation."""
        pressure = ReligiousPressure(city_position=(5, 5))
        assert pressure.city_position == (5, 5)
        assert pressure.total_citizens == 1

    def test_add_pressure(self) -> None:
        """Test adding pressure."""
        pressure = ReligiousPressure(city_position=(5, 5))
        pressure.add_pressure(religion_id=1, amount=50.0)
        assert pressure.pressure_by_religion[1] == 50.0

    def test_majority_religion(self) -> None:
        """Test majority religion calculation."""
        pressure = ReligiousPressure(
            city_position=(5, 5),
            total_citizens=10,
        )
        pressure.citizens_by_religion = {1: 6, 2: 4}
        assert pressure.majority_religion == 1

    def test_no_majority(self) -> None:
        """Test when no religion has majority."""
        pressure = ReligiousPressure(
            city_position=(5, 5),
            total_citizens=10,
        )
        pressure.citizens_by_religion = {1: 4, 2: 4}
        assert pressure.majority_religion is None

    def test_religion_percentage(self) -> None:
        """Test religion percentage calculation."""
        pressure = ReligiousPressure(
            city_position=(5, 5),
            total_citizens=10,
        )
        pressure.citizens_by_religion = {1: 6}
        pct = pressure.get_religion_percentage(1)
        assert pct == 0.6

    def test_convert_citizens(self) -> None:
        """Test converting citizens based on pressure."""
        pressure = ReligiousPressure(
            city_position=(5, 5),
            total_citizens=10,
        )
        pressure.add_pressure(1, 75.0)
        pressure.add_pressure(2, 25.0)
        pressure.convert_citizens(threshold=50.0)
        # Should have ~7.5 followers of religion 1, ~2.5 of religion 2
        assert pressure.citizens_by_religion[1] == 7
        assert pressure.citizens_by_religion[2] == 2


class TestReligionManager:
    """Tests for ReligionManager."""

    @pytest.fixture
    def manager(self) -> ReligionManager:
        """Create a religion manager."""
        return ReligionManager()

    def test_initialization(self, manager: ReligionManager) -> None:
        """Test manager initialization."""
        assert manager.config is not None

    def test_add_faith(self, manager: ReligionManager) -> None:
        """Test adding faith."""
        manager.add_faith(player=0, amount=50)
        faith = manager.get_faith(0)
        assert faith == 50

    def test_can_found_religion(self, manager: ReligionManager) -> None:
        """Test checking if religion can be founded."""
        # No faith - cannot found
        can_found, reason = manager.can_found_religion(0)
        assert not can_found
        assert "Not enough faith" in reason

        # With enough faith - can found
        manager.add_faith(0, manager.config.great_prophet_faith_cost)
        can_found, reason = manager.can_found_religion(0)
        assert can_found

    def test_found_religion(self, manager: ReligionManager) -> None:
        """Test founding a religion."""
        # Add faith first
        manager.add_faith(0, manager.config.great_prophet_faith_cost)

        religion = manager.found_religion(
            player=0,
            name="Test Religion",
            holy_city_position=(5, 5),
        )
        assert religion is not None
        assert religion.name == "Test Religion"
        assert religion.founder_player == 0
        assert religion.holy_city == (5, 5)

    def test_found_religion_deducts_faith(self, manager: ReligionManager) -> None:
        """Test that founding religion deducts faith cost."""
        initial_faith = 500
        manager.add_faith(0, initial_faith)
        manager.found_religion(
            player=0,
            name="Test",
            holy_city_position=(5, 5),
        )
        expected_faith = initial_faith - manager.config.great_prophet_faith_cost
        assert manager.get_faith(0) == expected_faith

    def test_found_religion_without_faith(self, manager: ReligionManager) -> None:
        """Test founding religion without enough faith fails."""
        religion = manager.found_religion(
            player=0,
            name="Test",
            holy_city_position=(5, 5),
        )
        assert religion is None

    def test_get_player_religion(self, manager: ReligionManager) -> None:
        """Test getting a player's founded religion."""
        manager.add_faith(0, manager.config.great_prophet_faith_cost)
        manager.found_religion(0, "Test", (5, 5))

        religion = manager.get_player_religion(0)
        assert religion is not None
        assert religion.name == "Test"

        # Player 1 has no religion
        assert manager.get_player_religion(1) is None

    def test_create_religious_unit(self, manager: ReligionManager) -> None:
        """Test creating a religious unit."""
        from alpha_zero.mechanics.religion import ReligiousUnitType

        # Found religion first
        manager.add_faith(0, 500)
        religion = manager.found_religion(0, "Test", (5, 5))

        # Create missionary
        manager.add_faith(0, manager.config.missionary_faith_cost)
        unit = manager.create_religious_unit(
            unit_type=ReligiousUnitType.MISSIONARY,
            player=0,
            position=(5, 5),
            religion_id=religion.id,
        )
        assert unit is not None
        assert unit.unit_type == ReligiousUnitType.MISSIONARY
        assert unit.charges_remaining == manager.config.missionary_charges

    def test_create_religious_unit_without_faith(self, manager: ReligionManager) -> None:
        """Test creating unit without faith fails."""
        from alpha_zero.mechanics.religion import ReligiousUnitType

        unit = manager.create_religious_unit(
            unit_type=ReligiousUnitType.MISSIONARY,
            player=0,
            position=(5, 5),
            religion_id=1,
        )
        assert unit is None

    def test_get_religious_vector(self, manager: ReligionManager) -> None:
        """Test feature vector generation."""
        manager.add_faith(0, manager.config.great_prophet_faith_cost)
        manager.found_religion(player=0, name="Test", holy_city_position=(5, 5))
        vector = manager.get_religious_vector(player=0, max_religions=7)
        assert isinstance(vector, list)
        assert len(vector) > 0

    def test_get_city_religious_state(self, manager: ReligionManager) -> None:
        """Test getting city religious state."""
        state = manager.get_city_religious_state((5, 5), max_religions=7)
        assert isinstance(state, list)
        # Should have 7 religion percentages + 1 is_holy_city flag
        assert len(state) == 8

    def test_serialization(self, manager: ReligionManager) -> None:
        """Test serialization/deserialization."""
        manager.add_faith(0, manager.config.great_prophet_faith_cost + 100)
        manager.found_religion(player=0, name="Test", holy_city_position=(5, 5))

        data = manager.to_dict()
        assert "religions" in data
        assert "player_faith" in data

        restored = ReligionManager.from_dict(data)
        religion = restored.get_player_religion(0)
        assert religion is not None
        assert religion.name == "Test"

    def test_max_religions_limit(self) -> None:
        """Test that max religions limit is enforced."""
        config = ReligionConfig(max_religions=2, great_prophet_faith_cost=10)
        manager = ReligionManager(config)

        # Found two religions
        manager.add_faith(0, 100)
        manager.add_faith(1, 100)
        manager.add_faith(2, 100)

        r1 = manager.found_religion(0, "Religion 1", (0, 0))
        r2 = manager.found_religion(1, "Religion 2", (5, 5))
        assert r1 is not None
        assert r2 is not None

        # Third should fail
        can_found, reason = manager.can_found_religion(2)
        assert not can_found
        assert "Maximum religions" in reason


# =============================================================================
# Culture Tests
# =============================================================================


class TestCultureConfig:
    """Tests for CultureConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = CultureConfig()
        assert config.base_tourism_per_great_work == 2
        assert config.cultural_victory_percentage == 1.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CultureConfig(
            base_tourism_per_great_work=3,
            artifact_tourism=5,
        )
        assert config.base_tourism_per_great_work == 3
        assert config.artifact_tourism == 5


class TestGreatWork:
    """Tests for GreatWork."""

    def test_creation(self) -> None:
        """Test great work creation."""
        work = GreatWork(
            id=1,
            work_type=GreatWorkType.WRITING,
            name="Test Writing",
            creator_player=0,
            era=1,
        )
        assert work.id == 1
        assert work.name == "Test Writing"
        assert work.work_type == GreatWorkType.WRITING

    def test_tourism_value(self) -> None:
        """Test tourism value."""
        work = GreatWork(
            id=1,
            work_type=GreatWorkType.ART,
            name="Test Art",
            creator_player=0,
            tourism_value=5,
        )
        assert work.tourism_value == 5

    def test_to_dict(self) -> None:
        """Test serialization."""
        work = GreatWork(
            id=1,
            work_type=GreatWorkType.MUSIC,
            name="Test Music",
            creator_player=0,
        )
        data = work.to_dict()
        assert data["id"] == 1
        assert data["work_type"] == "MUSIC"

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "id": 1,
            "work_type": "RELIC",
            "name": "Test Relic",
            "creator_player": 0,
        }
        work = GreatWork.from_dict(data)
        assert work.work_type == GreatWorkType.RELIC


class TestTourismSource:
    """Tests for TourismSource."""

    def test_creation(self) -> None:
        """Test source creation."""
        source = TourismSource(
            source_type="great_work",
            source_id=1,
            tourism_per_turn=5,
        )
        assert source.tourism_per_turn == 5

    def test_modified_tourism(self) -> None:
        """Test tourism with modifiers."""
        source = TourismSource(
            source_type="great_work",
            source_id=1,
            tourism_per_turn=10,
            modifiers={"open_borders": 0.25},
        )
        modified = source.modified_tourism
        assert modified == 12.5  # 10 * 1.25


class TestCulturalInfluence:
    """Tests for CulturalInfluence."""

    def test_creation(self) -> None:
        """Test influence creation."""
        influence = CulturalInfluence(
            source_player=0,
            target_player=1,
        )
        assert influence.cumulative_tourism == 0.0
        assert influence.influence_level == 0

    def test_add_tourism(self) -> None:
        """Test adding tourism."""
        influence = CulturalInfluence(
            source_player=0,
            target_player=1,
        )
        leveled_up = influence.add_tourism(150)
        assert influence.cumulative_tourism == 150
        assert influence.influence_level >= 1

    def test_level_thresholds(self) -> None:
        """Test influence level thresholds."""
        influence = CulturalInfluence(
            source_player=0,
            target_player=1,
        )

        # Unknown -> Exotic (100)
        influence.add_tourism(100)
        assert influence.influence_level == 1

        # Exotic -> Familiar (200)
        influence.add_tourism(100)
        assert influence.influence_level == 2

        # Familiar -> Influential (400)
        influence.add_tourism(200)
        assert influence.influence_level == 3

        # Influential -> Dominant (600)
        influence.add_tourism(200)
        assert influence.influence_level == 4


class TestCulturalVictoryProgress:
    """Tests for CulturalVictoryProgress."""

    def test_creation(self) -> None:
        """Test progress creation."""
        progress = CulturalVictoryProgress(player=0)
        assert progress.domestic_tourists == 0.0
        assert not progress.is_victory_achieved

    def test_total_foreign_tourists(self) -> None:
        """Test foreign tourist calculation."""
        progress = CulturalVictoryProgress(
            player=0,
            foreign_tourists={1: 50.0, 2: 30.0, 3: 20.0},
        )
        total = progress.get_total_foreign_tourists()
        assert total == 100.0

    def test_check_victory(self) -> None:
        """Test victory check."""
        progress = CulturalVictoryProgress(
            player=0,
            foreign_tourists={1: 100.0, 2: 100.0},
        )

        # Not enough to beat other domestic tourists
        other_domestic = {1: 200.0, 2: 150.0}
        result = progress.check_victory(other_domestic)
        assert result is False

        # Enough to beat all
        progress.foreign_tourists = {1: 300.0, 2: 300.0}
        result = progress.check_victory(other_domestic)
        assert result is True


class TestCultureManager:
    """Tests for CultureManager."""

    @pytest.fixture
    def manager(self) -> CultureManager:
        """Create a culture manager."""
        return CultureManager()

    def test_initialization(self, manager: CultureManager) -> None:
        """Test manager initialization."""
        assert manager.config is not None

    def test_create_great_work(self, manager: CultureManager) -> None:
        """Test creating great works."""
        work = manager.create_great_work(
            work_type=GreatWorkType.WRITING,
            name="Epic Poem",
            creator_player=0,
            era=2,
        )
        assert work is not None
        assert work.name == "Epic Poem"

    def test_get_player_great_works(self, manager: CultureManager) -> None:
        """Test getting player's great works."""
        manager.create_great_work(GreatWorkType.WRITING, "Work 1", 0)
        manager.create_great_work(GreatWorkType.ART, "Work 2", 0)
        manager.create_great_work(GreatWorkType.MUSIC, "Work 3", 1)

        works = manager.get_player_great_works(0)
        assert len(works) == 2

    def test_add_culture(self, manager: CultureManager) -> None:
        """Test adding culture."""
        manager.add_culture(0, 100)
        culture = manager.get_culture(0)
        assert culture == 100

    def test_calculate_tourism(self, manager: CultureManager) -> None:
        """Test tourism calculation."""
        manager.create_great_work(GreatWorkType.WRITING, "Work 1", 0)
        manager.create_great_work(GreatWorkType.ART, "Work 2", 0)

        tourism = manager.calculate_tourism_per_turn(0)
        assert tourism > 0

    def test_apply_tourism(self, manager: CultureManager) -> None:
        """Test applying tourism."""
        manager.create_great_work(GreatWorkType.WRITING, "Work", 0)

        results = manager.apply_tourism(0, [1, 2])
        assert 1 in results
        assert 2 in results

    def test_get_influence(self, manager: CultureManager) -> None:
        """Test getting influence."""
        influence = manager.get_influence(0, 1)
        assert influence.source_player == 0
        assert influence.target_player == 1

    def test_get_victory_progress(self, manager: CultureManager) -> None:
        """Test victory progress."""
        manager.add_culture(0, 1000)
        manager.create_great_work(GreatWorkType.WRITING, "Work", 0)

        progress = manager.get_victory_progress(0)
        assert progress.domestic_tourists > 0

    def test_check_cultural_victory(self, manager: CultureManager) -> None:
        """Test cultural victory check."""
        result = manager.check_cultural_victory(0, [0, 1, 2])
        assert isinstance(result, bool)

    def test_process_turn(self, manager: CultureManager) -> None:
        """Test turn processing."""
        manager.create_great_work(GreatWorkType.WRITING, "Work", 0)
        manager.process_turn([0, 1])

        # Tourism should have been applied
        influence = manager.get_influence(0, 1)
        assert influence.cumulative_tourism > 0

    def test_cultural_vector(self, manager: CultureManager) -> None:
        """Test feature vector generation."""
        manager.add_culture(0, 500)
        manager.create_great_work(GreatWorkType.WRITING, "Work", 0)

        vector = manager.get_cultural_vector(0, num_players=4)
        assert isinstance(vector, list)
        assert len(vector) > 0

    def test_serialization(self, manager: CultureManager) -> None:
        """Test serialization/deserialization."""
        manager.create_great_work(GreatWorkType.WRITING, "Work", 0)
        manager.add_culture(0, 500)

        data = manager.to_dict()
        assert "great_works" in data

        restored = CultureManager.from_dict(data)
        assert restored.get_culture(0) == 500
        assert len(restored.get_player_great_works(0)) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestMechanicsIntegration:
    """Integration tests for game mechanics."""

    def test_diplomacy_basic_relations(self) -> None:
        """Test basic diplomacy relations."""
        diplomacy = DiplomacyManager()

        # Get relations and verify
        relation = diplomacy.get_relation(0, 1)
        assert relation is not None
        assert relation.relation_type == DiplomacyType.NEUTRAL

    def test_religion_with_culture(self) -> None:
        """Test religion interactions with culture."""
        culture = CultureManager()
        religion = ReligionManager()

        # Add faith and found religion
        religion.add_faith(0, religion.config.great_prophet_faith_cost)
        religion.found_religion(0, "Test Religion", (5, 5))

        # Create a relic great work (relics come from religion)
        work = culture.create_great_work(
            GreatWorkType.RELIC,
            name="Sacred Relic",
            creator_player=0,
        )
        assert work.work_type == GreatWorkType.RELIC
        assert work.tourism_value == culture.config.relic_tourism

    def test_full_game_turn(self) -> None:
        """Test processing a full game turn with all mechanics."""
        diplomacy = DiplomacyManager()
        religion = ReligionManager()
        culture = CultureManager()

        all_players = [0, 1, 2, 3]

        # Set up initial state
        religion.add_faith(0, religion.config.great_prophet_faith_cost)
        religion.found_religion(0, "Religion A", (5, 5))
        culture.create_great_work(GreatWorkType.WRITING, "Epic", 0)

        # Process turns
        diplomacy.process_turn(current_turn=0)
        religion.process_turn()
        culture.process_turn(all_players)

        # Verify state after turn
        assert culture.get_influence(0, 2).cumulative_tourism > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_diplomacy_self_relation(self) -> None:
        """Test diplomacy with self (should handle gracefully)."""
        manager = DiplomacyManager()
        # Getting relation with self should work
        relation = manager.get_relation(0, 0)
        assert relation is not None

    def test_religion_no_faith(self) -> None:
        """Test religion operations with no faith."""
        manager = ReligionManager()
        faith = manager.get_faith(0)
        assert faith == 0

    def test_culture_no_works(self) -> None:
        """Test culture with no great works."""
        manager = CultureManager()
        tourism = manager.calculate_tourism_per_turn(0)
        assert tourism == 0

        works = manager.get_player_great_works(0)
        assert len(works) == 0

    def test_empty_player_list(self) -> None:
        """Test mechanics with empty player list."""
        culture = CultureManager()
        culture.process_turn([])  # Should not crash

    def test_large_player_count(self) -> None:
        """Test mechanics with large player count."""
        diplomacy = DiplomacyManager()
        num_players = 20

        # Should handle large player counts efficiently
        for i in range(num_players - 1):
            relation = diplomacy.get_relation(0, i + 1)
            assert relation is not None
