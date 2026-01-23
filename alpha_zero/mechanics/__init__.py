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

from .diplomacy import (
    DiplomacyManager,
    DiplomaticRelation,
    DiplomaticModifier,
    DiplomacyType,
    DiplomaticAction,
    DiplomaticActionType,
)
from .religion import (
    ReligionManager,
    Religion,
    ReligiousBelief,
    BeliefType,
    ReligiousUnit,
    ReligiousPressure,
)
from .culture import (
    CultureManager,
    GreatWork,
    GreatWorkType,
    TourismSource,
    CulturalVictoryProgress,
)
from .trade import (
    TradeManager,
    TradeRoute,
    TradeRouteType,
    TradeYields,
    TradeRouteConfig,
    Trader,
    TraderStatus,
    TradeAgreement,
    TradeGoodType,
)
from .espionage import (
    EspionageManager,
    EspionageConfig,
    Spy,
    SpyRank,
    SpyStatus,
    Mission,
    MissionType,
    MissionOutcome,
    MissionConfig,
)
from .government import (
    GovernmentManager,
    GovernmentConfig,
    GovernmentType,
    GovernmentTier,
    PolicyCard,
    PolicySlotType,
    PolicySlots,
    ActivePolicy,
)
from .great_people import (
    GreatPeopleManager,
    GreatPeopleConfig,
    GreatPerson,
    GreatPersonType,
    GreatPersonEra,
    GreatPersonDefinition,
    GreatPersonAbility,
    ActivationType,
)
from .eras import (
    EraManager,
    EraConfig,
    Era,
    AgeType,
    DedicationType,
    Dedication,
    HistoricMoment,
    HistoricMomentType,
    EraProgress,
)
from .climate import (
    ClimateManager,
    ClimateConfig,
    ClimatePhase,
    DisasterType,
    DisasterSeverity,
    DisasterConfig,
    DisasterEvent,
    FloodedTile,
    TerrainFeature,
)
from .world_congress import (
    WorldCongressManager,
    CongressConfig,
    CongressPhase,
    Resolution,
    ResolutionCategory,
    ResolutionEffect,
    Vote,
    VoteType,
    ActiveResolution,
    Emergency,
    EmergencyType,
)
from .promotions import (
    PromotionManager,
    PromotionConfig,
    Promotion,
    PromotionTier,
    UnitClass,
    StatModifier,
    StatModifierType,
    Ability,
    AbilityType,
    UnitExperience,
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
