"""
Advanced Game Mechanics for Civ6-AlphaZero

This module implements advanced Civilization game mechanics including:
- Diplomacy system (treaties, wars, alliances)
- Religion system (founding, spreading, beliefs)
- Culture/Tourism system (great works, tourism pressure)

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
]
