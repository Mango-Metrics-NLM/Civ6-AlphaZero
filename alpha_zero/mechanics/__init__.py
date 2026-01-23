"""
Advanced Game Mechanics for Civ6-AlphaZero

This module implements advanced Civilization game mechanics including:
- Diplomacy system (treaties, wars, alliances)
- Religion system (founding, spreading, beliefs)
- Culture/Tourism system (great works, tourism pressure)
- Trade system (routes, traders, agreements)
- Espionage system (spies, missions, counterintelligence)
- Government system (types, policy cards, legacy bonuses)
- Great People system (recruitment, abilities)
- Era/Ages system (era score, golden/dark ages)
- Climate system (CO2, disasters, sea level)
- World Congress (resolutions, voting, emergencies)
- Unit Promotions (experience, promotion trees)

All mechanics are designed to be:
- Fully configurable via dataclasses
- Testable in isolation
- Compatible with the existing encoder system
"""

from __future__ import annotations

from .climate import (
    ClimateConfig,
    ClimateManager,
    ClimatePhase,
    DisasterConfig,
    DisasterEvent,
    DisasterSeverity,
    DisasterType,
    FloodedTile,
    TerrainFeature,
)
from .culture import (
    CulturalVictoryProgress,
    CultureManager,
    GreatWork,
    GreatWorkType,
    TourismSource,
)
from .diplomacy import (
    DiplomacyManager,
    DiplomacyType,
    DiplomaticAction,
    DiplomaticActionType,
    DiplomaticModifier,
    DiplomaticRelation,
)
from .eras import (
    AgeType,
    Dedication,
    DedicationType,
    Era,
    EraConfig,
    EraManager,
    EraProgress,
    HistoricMoment,
    HistoricMomentType,
)
from .espionage import (
    EspionageConfig,
    EspionageManager,
    Mission,
    MissionConfig,
    MissionOutcome,
    MissionType,
    Spy,
    SpyRank,
    SpyStatus,
)
from .government import (
    ActivePolicy,
    GovernmentConfig,
    GovernmentManager,
    GovernmentTier,
    GovernmentType,
    PolicyCard,
    PolicySlots,
    PolicySlotType,
)
from .great_people import (
    ActivationType,
    GreatPeopleConfig,
    GreatPeopleManager,
    GreatPerson,
    GreatPersonAbility,
    GreatPersonDefinition,
    GreatPersonEra,
    GreatPersonType,
)
from .promotions import (
    Ability,
    AbilityType,
    Promotion,
    PromotionConfig,
    PromotionManager,
    PromotionTier,
    StatModifier,
    StatModifierType,
    UnitClass,
    UnitExperience,
)
from .religion import (
    BeliefType,
    Religion,
    ReligionManager,
    ReligiousBelief,
    ReligiousPressure,
    ReligiousUnit,
)
from .trade import (
    TradeAgreement,
    TradeGoodType,
    TradeManager,
    Trader,
    TradeRoute,
    TradeRouteConfig,
    TradeRouteType,
    TraderStatus,
    TradeYields,
)
from .world_congress import (
    ActiveResolution,
    CongressConfig,
    CongressPhase,
    Emergency,
    EmergencyType,
    Resolution,
    ResolutionCategory,
    ResolutionEffect,
    Vote,
    VoteType,
    WorldCongressManager,
)

__all__ = [
    # Diplomacy
    "DiplomacyManager",
    "DiplomaticRelation",
    "DiplomaticModifier",
    "DiplomacyType",
    "DiplomaticAction",
    "DiplomaticActionType",
    # Religion
    "ReligionManager",
    "Religion",
    "ReligiousBelief",
    "BeliefType",
    "ReligiousUnit",
    "ReligiousPressure",
    # Culture
    "CultureManager",
    "GreatWork",
    "GreatWorkType",
    "TourismSource",
    "CulturalVictoryProgress",
    # Trade
    "TradeManager",
    "TradeRoute",
    "TradeRouteType",
    "TradeYields",
    "TradeRouteConfig",
    "Trader",
    "TraderStatus",
    "TradeAgreement",
    "TradeGoodType",
    # Espionage
    "EspionageManager",
    "EspionageConfig",
    "Spy",
    "SpyRank",
    "SpyStatus",
    "Mission",
    "MissionType",
    "MissionOutcome",
    "MissionConfig",
    # Government
    "GovernmentManager",
    "GovernmentConfig",
    "GovernmentType",
    "GovernmentTier",
    "PolicyCard",
    "PolicySlotType",
    "PolicySlots",
    "ActivePolicy",
    # Great People
    "GreatPeopleManager",
    "GreatPeopleConfig",
    "GreatPerson",
    "GreatPersonType",
    "GreatPersonEra",
    "GreatPersonDefinition",
    "GreatPersonAbility",
    "ActivationType",
    # Eras
    "EraManager",
    "EraConfig",
    "Era",
    "AgeType",
    "DedicationType",
    "Dedication",
    "HistoricMoment",
    "HistoricMomentType",
    "EraProgress",
    # Climate
    "ClimateManager",
    "ClimateConfig",
    "ClimatePhase",
    "DisasterType",
    "DisasterSeverity",
    "DisasterConfig",
    "DisasterEvent",
    "FloodedTile",
    "TerrainFeature",
    # World Congress
    "WorldCongressManager",
    "CongressConfig",
    "CongressPhase",
    "Resolution",
    "ResolutionCategory",
    "ResolutionEffect",
    "Vote",
    "VoteType",
    "ActiveResolution",
    "Emergency",
    "EmergencyType",
    # Promotions
    "PromotionManager",
    "PromotionConfig",
    "Promotion",
    "PromotionTier",
    "UnitClass",
    "StatModifier",
    "StatModifierType",
    "Ability",
    "AbilityType",
    "UnitExperience",
]
