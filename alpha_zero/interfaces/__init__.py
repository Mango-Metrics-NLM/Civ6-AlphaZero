"""Interfaces for external game state formats.

This module provides interfaces to convert between external game formats
(such as Unciv JSON) and the internal format used by the Civ6-AlphaZero
encoder and action space.

Available interfaces:
- UncivInterface: Interface for Unciv (open-source Civ V clone) JSON saves
"""

from alpha_zero.interfaces.unciv_interface import (
    # Main interface class
    UncivInterface,
    create_unciv_interface,
    # Data classes
    UncivGameState,
    UncivTile,
    UncivUnit,
    UncivCity,
    UncivPlayer,
    # Enums
    UncivTerrainType,
    UncivTerrainFeature,
    UncivResource,
    UncivUnitType,
    UncivImprovementType,
    # Exceptions
    UncivInterfaceError,
    UncivLoadError,
    UncivSaveError,
    UncivConversionError,
    UncivActionError,
    # Mapping utilities
    UncivMappings,
)

__all__ = [
    # Main interface
    "UncivInterface",
    "create_unciv_interface",
    # Data classes
    "UncivGameState",
    "UncivTile",
    "UncivUnit",
    "UncivCity",
    "UncivPlayer",
    # Enums
    "UncivTerrainType",
    "UncivTerrainFeature",
    "UncivResource",
    "UncivUnitType",
    "UncivImprovementType",
    # Exceptions
    "UncivInterfaceError",
    "UncivLoadError",
    "UncivSaveError",
    "UncivConversionError",
    "UncivActionError",
    # Utilities
    "UncivMappings",
]
