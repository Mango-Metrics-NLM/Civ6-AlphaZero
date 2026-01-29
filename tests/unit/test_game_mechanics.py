"""Unit tests for game mechanics modules.

Tests verify basic imports, data structures, and core functionality
of game mechanics modules.
"""

from __future__ import annotations

import pytest
from typing import Any


# =============================================================================
# Trade Module Tests
# =============================================================================


class TestTradeModule:
    """Tests for trade mechanics."""

    def test_imports(self):
        """Test trade module imports successfully."""
        from alpha_zero.mechanics.trade import (
            TradeManager, TradeRouteConfig, TradeRouteType,
            TradeRoute, Trader, TradeYields, TradeAgreement,
        )
        # All imports should work
        assert TradeManager is not None
        assert TradeRouteConfig is not None
        assert TradeRouteType is not None

    def test_trade_route_types(self):
        """Test trade route type enumeration."""
        from alpha_zero.mechanics.trade import TradeRouteType

        assert hasattr(TradeRouteType, 'DOMESTIC')
        assert hasattr(TradeRouteType, 'INTERNATIONAL')

    def test_trade_yields(self):
        """Test TradeYields dataclass."""
        from alpha_zero.mechanics.trade import TradeYields

        yields = TradeYields(food=2.0, production=1.0, gold=3.0)
        assert yields.food == 2.0
        assert yields.production == 1.0
        assert yields.gold == 3.0

    def test_trade_manager_creation(self):
        """Test TradeManager can be created."""
        from alpha_zero.mechanics.trade import TradeManager, TradeRouteConfig

        config = TradeRouteConfig()
        manager = TradeManager(config)
        assert manager is not None

    def test_trade_manager_reset(self):
        """Test TradeManager reset method."""
        from alpha_zero.mechanics.trade import TradeManager, TradeRouteConfig

        manager = TradeManager(TradeRouteConfig())
        manager.reset()
        # Should complete without error


# =============================================================================
# Espionage Module Tests
# =============================================================================


class TestEspionageModule:
    """Tests for espionage mechanics."""

    def test_imports(self):
        """Test espionage module imports successfully."""
        from alpha_zero.mechanics.espionage import (
            EspionageManager, EspionageConfig, Spy, Mission, MissionType,
        )
        assert EspionageManager is not None
        assert MissionType is not None

    def test_mission_types(self):
        """Test mission type enumeration."""
        from alpha_zero.mechanics.espionage import MissionType

        assert hasattr(MissionType, 'GATHER_INTELLIGENCE')
        assert hasattr(MissionType, 'STEAL_TECHNOLOGY')
        assert hasattr(MissionType, 'SABOTAGE_PRODUCTION')

    def test_espionage_manager_creation(self):
        """Test EspionageManager can be created."""
        from alpha_zero.mechanics.espionage import EspionageManager, EspionageConfig

        config = EspionageConfig()
        manager = EspionageManager(config)
        assert manager is not None


# =============================================================================
# Government Module Tests
# =============================================================================


class TestGovernmentModule:
    """Tests for government mechanics."""

    def test_imports(self):
        """Test government module imports successfully."""
        from alpha_zero.mechanics.government import (
            GovernmentManager, GovernmentConfig, PolicyCard,
            PolicySlots, PolicySlotType,
        )
        assert GovernmentManager is not None
        assert PolicySlotType is not None

    def test_policy_slot_types(self):
        """Test policy slot type enumeration."""
        from alpha_zero.mechanics.government import PolicySlotType

        assert hasattr(PolicySlotType, 'MILITARY')
        assert hasattr(PolicySlotType, 'ECONOMIC')
        assert hasattr(PolicySlotType, 'DIPLOMATIC')
        assert hasattr(PolicySlotType, 'WILDCARD')

    def test_policy_slots(self):
        """Test PolicySlots dataclass."""
        from alpha_zero.mechanics.government import PolicySlots

        slots = PolicySlots(military=2, economic=2, diplomatic=1, wildcard=1)
        assert slots.military == 2
        assert slots.economic == 2
        assert slots.total() == 6

    def test_government_manager_creation(self):
        """Test GovernmentManager can be created."""
        from alpha_zero.mechanics.government import GovernmentManager, GovernmentConfig

        config = GovernmentConfig()
        manager = GovernmentManager(config)
        assert manager is not None


# =============================================================================
# Great People Module Tests
# =============================================================================


class TestGreatPeopleModule:
    """Tests for great people mechanics."""

    def test_imports(self):
        """Test great people module imports successfully."""
        from alpha_zero.mechanics.great_people import (
            GreatPeopleManager, GreatPeopleConfig, GreatPerson,
            GreatPersonType, GreatPersonDefinition,
        )
        assert GreatPeopleManager is not None
        assert GreatPersonType is not None

    def test_great_person_types(self):
        """Test great person type enumeration."""
        from alpha_zero.mechanics.great_people import GreatPersonType

        assert hasattr(GreatPersonType, 'GREAT_GENERAL')
        assert hasattr(GreatPersonType, 'GREAT_SCIENTIST')
        assert hasattr(GreatPersonType, 'GREAT_ENGINEER')
        assert hasattr(GreatPersonType, 'GREAT_MERCHANT')
        assert hasattr(GreatPersonType, 'GREAT_ARTIST')

    def test_great_people_manager_creation(self):
        """Test GreatPeopleManager can be created."""
        from alpha_zero.mechanics.great_people import GreatPeopleManager, GreatPeopleConfig

        config = GreatPeopleConfig()
        manager = GreatPeopleManager(config)
        assert manager is not None


# =============================================================================
# Eras Module Tests
# =============================================================================


class TestErasModule:
    """Tests for era mechanics."""

    def test_imports(self):
        """Test eras module imports successfully."""
        from alpha_zero.mechanics.eras import (
            EraManager, EraConfig, HistoricMoment,
            AgeType, HistoricMomentType,
        )
        assert EraManager is not None
        assert AgeType is not None

    def test_age_types(self):
        """Test age type enumeration."""
        from alpha_zero.mechanics.eras import AgeType

        assert hasattr(AgeType, 'NORMAL')
        assert hasattr(AgeType, 'GOLDEN')
        assert hasattr(AgeType, 'DARK')
        assert hasattr(AgeType, 'HEROIC')

    def test_era_manager_creation(self):
        """Test EraManager can be created."""
        from alpha_zero.mechanics.eras import EraManager, EraConfig

        config = EraConfig()
        manager = EraManager(config)
        assert manager is not None


# =============================================================================
# Climate Module Tests
# =============================================================================


class TestClimateModule:
    """Tests for climate mechanics."""

    def test_imports(self):
        """Test climate module imports successfully."""
        from alpha_zero.mechanics.climate import (
            ClimateManager, ClimateConfig, DisasterEvent,
            DisasterType, ClimatePhase,
        )
        assert ClimateManager is not None
        assert DisasterType is not None

    def test_disaster_types(self):
        """Test disaster type enumeration."""
        from alpha_zero.mechanics.climate import DisasterType

        assert hasattr(DisasterType, 'FLOOD')
        assert hasattr(DisasterType, 'VOLCANIC_ERUPTION')
        assert hasattr(DisasterType, 'HURRICANE')
        assert hasattr(DisasterType, 'TORNADO')

    def test_climate_phases(self):
        """Test climate phase enumeration."""
        from alpha_zero.mechanics.climate import ClimatePhase

        # Check that phases are defined
        phases = list(ClimatePhase)
        assert len(phases) > 0

    def test_climate_manager_creation(self):
        """Test ClimateManager can be created."""
        from alpha_zero.mechanics.climate import ClimateManager, ClimateConfig

        config = ClimateConfig()
        manager = ClimateManager(config)
        assert manager is not None


# =============================================================================
# World Congress Module Tests
# =============================================================================


class TestWorldCongressModule:
    """Tests for world congress mechanics."""

    def test_imports(self):
        """Test world congress module imports successfully."""
        from alpha_zero.mechanics.world_congress import (
            WorldCongressManager, CongressConfig, Resolution,
            ResolutionCategory, Vote,
        )
        assert WorldCongressManager is not None
        assert ResolutionCategory is not None

    def test_resolution_categories(self):
        """Test resolution category enumeration exists."""
        from alpha_zero.mechanics.world_congress import ResolutionCategory

        # Check that resolution categories are defined
        categories = list(ResolutionCategory)
        assert len(categories) > 0

    def test_world_congress_manager_creation(self):
        """Test WorldCongressManager can be created."""
        from alpha_zero.mechanics.world_congress import (
            WorldCongressManager, CongressConfig
        )

        config = CongressConfig()
        manager = WorldCongressManager(config)
        assert manager is not None


# =============================================================================
# Promotions Module Tests
# =============================================================================


class TestPromotionsModule:
    """Tests for unit promotions mechanics."""

    def test_imports(self):
        """Test promotions module imports successfully."""
        from alpha_zero.mechanics.promotions import (
            PromotionManager, PromotionConfig, Promotion,
            UnitClass, StatModifier,
        )
        assert PromotionManager is not None
        assert UnitClass is not None

    def test_unit_classes(self):
        """Test unit class enumeration."""
        from alpha_zero.mechanics.promotions import UnitClass

        assert hasattr(UnitClass, 'MELEE')
        assert hasattr(UnitClass, 'RANGED')
        assert hasattr(UnitClass, 'CAVALRY')
        assert hasattr(UnitClass, 'SIEGE')

    def test_promotion_manager_creation(self):
        """Test PromotionManager can be created."""
        from alpha_zero.mechanics.promotions import PromotionManager, PromotionConfig

        config = PromotionConfig()
        manager = PromotionManager(config)
        assert manager is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestMechanicsIntegration:
    """Integration tests for mechanics modules."""

    def test_all_managers_can_be_created(self):
        """Test all managers can be instantiated."""
        from alpha_zero.mechanics import (
            TradeManager, EspionageManager, GovernmentManager,
            GreatPeopleManager, EraManager, ClimateManager,
            WorldCongressManager, PromotionManager,
        )

        managers = [
            TradeManager(),
            EspionageManager(),
            GovernmentManager(),
            GreatPeopleManager(),
            EraManager(),
            ClimateManager(),
            WorldCongressManager(),
            PromotionManager(),
        ]

        for manager in managers:
            assert manager is not None

    def test_all_managers_have_reset(self):
        """Test all managers have reset method."""
        from alpha_zero.mechanics import (
            TradeManager, EspionageManager, GovernmentManager,
            GreatPeopleManager, EraManager, ClimateManager,
            WorldCongressManager, PromotionManager,
        )

        managers = [
            TradeManager(),
            EspionageManager(),
            GovernmentManager(),
            GreatPeopleManager(),
            EraManager(),
            ClimateManager(),
            WorldCongressManager(),
            PromotionManager(),
        ]

        for manager in managers:
            assert hasattr(manager, 'reset')
            manager.reset()  # Should not raise

    def test_all_managers_have_encode_for_state(self):
        """Test all managers have encode_for_state method."""
        from alpha_zero.mechanics import (
            TradeManager, EspionageManager, GovernmentManager,
            GreatPeopleManager, EraManager, ClimateManager,
            WorldCongressManager, PromotionManager,
        )

        managers = [
            TradeManager(),
            EspionageManager(),
            GovernmentManager(),
            GreatPeopleManager(),
            EraManager(),
            ClimateManager(),
            WorldCongressManager(),
            PromotionManager(),
        ]

        for manager in managers:
            assert hasattr(manager, 'encode_for_state')
            # encode_for_state signatures vary - just verify method exists
            # Actual encoding tested in individual module tests
