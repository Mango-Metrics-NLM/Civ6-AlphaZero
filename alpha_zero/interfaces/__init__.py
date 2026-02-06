"""Interfaces for external game state formats.

This module provides interfaces to convert between external game formats
(such as Unciv JSON) and the internal format used by the Civ6-AlphaZero
encoder and action space.

Available interfaces:
- UncivInterface: Interface for Unciv (open-source Civ V clone) JSON saves
"""

from alpha_zero.interfaces.unciv_interface import (
    UncivActionError,
    UncivCity,
    UncivConversionError,
    # Data classes
    UncivGameState,
    UncivImprovementType,
    # Main interface class
    UncivInterface,
    # Exceptions
    UncivInterfaceError,
    UncivLoadError,
    # Mapping utilities
    UncivMappings,
    UncivPlayer,
    UncivResource,
    UncivSaveError,
    UncivTerrainFeature,
    # Enums
    UncivTerrainType,
    UncivTile,
    UncivUnit,
    UncivUnitType,
    create_unciv_interface,
)

__all__ = [
    "UncivActionError",
    "UncivCity",
    "UncivConversionError",
    # Data classes
    "UncivGameState",
    "UncivImprovementType",
    # Main interface
    "UncivInterface",
    # Exceptions
    "UncivInterfaceError",
    "UncivLoadError",
    # Utilities
    "UncivMappings",
    "UncivPlayer",
    "UncivResource",
    "UncivSaveError",
    "UncivTerrainFeature",
    # Enums
    "UncivTerrainType",
    "UncivTile",
    "UncivUnit",
    "UncivUnitType",
    "create_unciv_interface",
]
