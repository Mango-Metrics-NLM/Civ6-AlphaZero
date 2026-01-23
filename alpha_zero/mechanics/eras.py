"""Era and Ages System for Civ6-AlphaZero.

This module implements the Era Score and Ages mechanics including:
- Era score tracking and thresholds
- Historic Moments
- Golden/Dark/Heroic Ages
- Dedications and their bonuses

All mechanics are designed to be:
- Fully configurable via dataclasses
- Testable in isolation
- Compatible with the existing encoder system
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Era(Enum):
    """Game eras."""

    ANCIENT = auto()
    CLASSICAL = auto()
    MEDIEVAL = auto()
    RENAISSANCE = auto()
    INDUSTRIAL = auto()
    MODERN = auto()
    ATOMIC = auto()
    INFORMATION = auto()
    FUTURE = auto()


class AgeType(Enum):
    """Types of ages."""

    NORMAL = auto()
    DARK = auto()
    GOLDEN = auto()
    HEROIC = auto()  # Golden Age after Dark Age


class DedicationType(Enum):
    """Types of dedications (Golden Age bonuses)."""

    MONUMENTALITY = auto()  # Builder/Settler bonus
    PEN_BRUSH_VOICE = auto()  # Culture/Tourism bonus
    FREE_INQUIRY = auto()  # Science bonus
    EXODUS_OF_EVANGELISTS = auto()  # Faith/Religion bonus
    REFORM_THE_COINAGE = auto()  # Gold bonus
    HICS_SUNT_DRACONES = auto()  # Exploration bonus
    WISH_YOU_WERE_HERE = auto()  # Tourism bonus


class HistoricMomentType(Enum):
    """Types of Historic Moments."""

    # Exploration
    FIRST_CONTACT = auto()
    CIRCUMNAVIGATION = auto()
    WORLD_MAP_COMPLETED = auto()

    # Culture
    GREAT_PERSON_RECRUITED = auto()
    WONDER_BUILT = auto()
    CIVIC_RESEARCHED = auto()

    # Science
    TECH_RESEARCHED = auto()
    EUREKA_TRIGGERED = auto()

    # Military
    CITY_CAPTURED = auto()
    UNIT_PROMOTED = auto()
    BARBARIAN_CAMP_CLEARED = auto()

    # Religion
    RELIGION_FOUNDED = auto()
    CITY_CONVERTED = auto()

    # Diplomacy
    ALLIANCE_FORMED = auto()
    CITY_STATE_SUZERAIN = auto()

    # Economics
    TRADE_ROUTE_ESTABLISHED = auto()
    LUXURY_RESOURCE_IMPROVED = auto()

    # General
    CITY_FOUNDED = auto()
    DISTRICT_BUILT = auto()


@dataclass
class EraConfig:
    """Configuration for Era mechanics.

    Attributes:
        dark_age_threshold_ratio: Ratio of score needed to avoid dark age
        golden_age_threshold_ratio: Ratio of score needed for golden age
        base_era_score_threshold: Base threshold that scales per era
        era_scaling_factor: Multiplier per era advancement
        dark_age_loyalty_penalty: Loyalty penalty during dark ages
        golden_age_loyalty_bonus: Loyalty bonus during golden ages
        max_dedications_golden: Dedications available in golden age
        max_dedications_heroic: Dedications available in heroic age
    """

    dark_age_threshold_ratio: float = 0.5
    golden_age_threshold_ratio: float = 1.0
    base_era_score_threshold: int = 12
    era_scaling_factor: float = 1.5
    dark_age_loyalty_penalty: int = 3
    golden_age_loyalty_bonus: int = 3
    max_dedications_golden: int = 1
    max_dedications_heroic: int = 3


@dataclass
class HistoricMomentConfig:
    """Configuration for a Historic Moment type.

    Attributes:
        moment_type: Type of moment
        base_score: Base era score granted
        first_bonus: Additional score if first in world
        era_scaling: Whether score scales with era
        once_per_era: Whether can only trigger once per era
        once_per_game: Whether can only trigger once per game
    """

    moment_type: HistoricMomentType
    base_score: int = 1
    first_bonus: int = 1
    era_scaling: bool = False
    once_per_era: bool = False
    once_per_game: bool = False


# Default historic moment configurations
DEFAULT_MOMENT_CONFIGS: dict[HistoricMomentType, HistoricMomentConfig] = {
    HistoricMomentType.FIRST_CONTACT: HistoricMomentConfig(
        moment_type=HistoricMomentType.FIRST_CONTACT,
        base_score=1,
        first_bonus=0,
        once_per_era=False,
    ),
    HistoricMomentType.CIRCUMNAVIGATION: HistoricMomentConfig(
        moment_type=HistoricMomentType.CIRCUMNAVIGATION,
        base_score=5,
        first_bonus=5,
        once_per_game=True,
    ),
    HistoricMomentType.GREAT_PERSON_RECRUITED: HistoricMomentConfig(
        moment_type=HistoricMomentType.GREAT_PERSON_RECRUITED,
        base_score=2,
        first_bonus=1,
    ),
    HistoricMomentType.WONDER_BUILT: HistoricMomentConfig(
        moment_type=HistoricMomentType.WONDER_BUILT,
        base_score=4,
        first_bonus=0,
    ),
    HistoricMomentType.TECH_RESEARCHED: HistoricMomentConfig(
        moment_type=HistoricMomentType.TECH_RESEARCHED,
        base_score=1,
        first_bonus=2,
        era_scaling=True,
    ),
    HistoricMomentType.CIVIC_RESEARCHED: HistoricMomentConfig(
        moment_type=HistoricMomentType.CIVIC_RESEARCHED,
        base_score=1,
        first_bonus=2,
        era_scaling=True,
    ),
    HistoricMomentType.CITY_CAPTURED: HistoricMomentConfig(
        moment_type=HistoricMomentType.CITY_CAPTURED,
        base_score=3,
        first_bonus=0,
    ),
    HistoricMomentType.RELIGION_FOUNDED: HistoricMomentConfig(
        moment_type=HistoricMomentType.RELIGION_FOUNDED,
        base_score=4,
        first_bonus=2,
        once_per_game=True,
    ),
    HistoricMomentType.CITY_FOUNDED: HistoricMomentConfig(
        moment_type=HistoricMomentType.CITY_FOUNDED,
        base_score=1,
        first_bonus=0,
    ),
    HistoricMomentType.DISTRICT_BUILT: HistoricMomentConfig(
        moment_type=HistoricMomentType.DISTRICT_BUILT,
        base_score=1,
        first_bonus=1,
    ),
}


@dataclass
class Dedication:
    """Definition of a dedication.

    Attributes:
        dedication_type: Type of dedication
        name: Display name
        description: Effect description
        normal_effect: Effect during normal age
        golden_effect: Effect during golden/heroic age
        effects: Dictionary of effect values
    """

    dedication_type: DedicationType
    name: str
    description: str
    normal_effect: str
    golden_effect: str
    effects: dict[str, float] = field(default_factory=dict)
    golden_effects: dict[str, float] = field(default_factory=dict)


DEFAULT_DEDICATIONS: dict[DedicationType, Dedication] = {
    DedicationType.MONUMENTALITY: Dedication(
        dedication_type=DedicationType.MONUMENTALITY,
        name="Monumentality",
        description="Bonus to Builders and Settlers",
        normal_effect="+2 Movement for Builders and Settlers",
        golden_effect="Can purchase Builders and Settlers with Faith",
        effects={"builder_settler_movement": 2.0},
        golden_effects={"faith_purchase_builders_settlers": 1.0},
    ),
    DedicationType.FREE_INQUIRY: Dedication(
        dedication_type=DedicationType.FREE_INQUIRY,
        name="Free Inquiry",
        description="Bonus to Science",
        normal_effect="+1 Science per Commercial Hub and Harbor building",
        golden_effect="+10% Science in cities with a Campus",
        effects={"commercial_harbor_science": 1.0},
        golden_effects={"campus_science_bonus": 0.1},
    ),
    DedicationType.PEN_BRUSH_VOICE: Dedication(
        dedication_type=DedicationType.PEN_BRUSH_VOICE,
        name="Pen, Brush and Voice",
        description="Bonus to Culture and Tourism",
        normal_effect="+2 Culture per Great Work",
        golden_effect="+100% Theater Square adjacency",
        effects={"great_work_culture": 2.0},
        golden_effects={"theater_adjacency_bonus": 1.0},
    ),
    DedicationType.REFORM_THE_COINAGE: Dedication(
        dedication_type=DedicationType.REFORM_THE_COINAGE,
        name="Reform the Coinage",
        description="Bonus to Gold",
        normal_effect="+3 Gold per city",
        golden_effect="Double adjacency from Commercial Hub",
        effects={"city_gold_bonus": 3.0},
        golden_effects={"commercial_adjacency_bonus": 1.0},
    ),
}


@dataclass
class HistoricMoment:
    """A recorded historic moment.

    Attributes:
        moment_type: Type of moment
        player_id: Player who achieved it
        turn: Turn it occurred
        era: Era it occurred in
        score: Era score granted
        details: Additional details
        is_first: Whether first in world
    """

    moment_type: HistoricMomentType
    player_id: int
    turn: int
    era: Era
    score: int
    details: str = ""
    is_first: bool = False


@dataclass
class EraProgress:
    """Era progress for a player.

    Attributes:
        current_era: Current era
        era_score: Accumulated era score
        age_type: Current age type
        dedications: Active dedications
        moments: Historic moments this era
        was_dark_age: Whether previous age was dark
        era_turns: Turns in current era
    """

    current_era: Era = Era.ANCIENT
    era_score: int = 0
    age_type: AgeType = AgeType.NORMAL
    dedications: list[DedicationType] = field(default_factory=list)
    moments: list[HistoricMoment] = field(default_factory=list)
    was_dark_age: bool = False
    era_turns: int = 0


class EraManager:
    """Manages Era Score and Ages mechanics.

    This class handles:
    - Era score accumulation
    - Historic moment tracking
    - Age type determination
    - Dedication bonuses

    Example:
        >>> config = EraConfig(dark_age_threshold_ratio=0.5)
        >>> manager = EraManager(config)
        >>> manager.record_moment(
        ...     player_id=0,
        ...     moment_type=HistoricMomentType.WONDER_BUILT,
        ...     turn=50,
        ...     details="Built Stonehenge"
        ... )
        >>> threshold = manager.get_era_threshold(player_id=0)
        >>> age = manager.calculate_next_age(player_id=0)
    """

    def __init__(
        self,
        config: EraConfig | None = None,
        moment_configs: dict[HistoricMomentType, HistoricMomentConfig] | None = None,
        dedications: dict[DedicationType, Dedication] | None = None,
    ) -> None:
        """Initialize Era manager.

        Args:
            config: Era configuration
            moment_configs: Historic moment configurations
            dedications: Dedication definitions
        """
        self.config = config or EraConfig()
        self._moment_configs = moment_configs or DEFAULT_MOMENT_CONFIGS
        self._dedications = dedications or DEFAULT_DEDICATIONS

        self._player_progress: dict[int, EraProgress] = {}
        self._global_firsts: dict[HistoricMomentType, int] = {}  # moment -> player_id
        self._game_turn = 0

        logger.debug(f"EraManager initialized with {len(self._moment_configs)} moment types")

    def _get_progress(self, player_id: int) -> EraProgress:
        """Get or create progress for a player."""
        if player_id not in self._player_progress:
            self._player_progress[player_id] = EraProgress()
        return self._player_progress[player_id]

    def record_moment(
        self,
        player_id: int,
        moment_type: HistoricMomentType,
        turn: int,
        details: str = "",
    ) -> HistoricMoment | None:
        """Record a historic moment.

        Args:
            player_id: Player achieving the moment
            moment_type: Type of moment
            turn: Turn number
            details: Additional details

        Returns:
            Created moment, or None if invalid
        """
        config = self._moment_configs.get(moment_type)
        if not config:
            logger.warning(f"Unknown moment type: {moment_type}")
            return None

        progress = self._get_progress(player_id)

        # Check once-per-era restriction
        if config.once_per_era:
            for existing in progress.moments:
                if existing.moment_type == moment_type:
                    logger.debug(f"Moment {moment_type.name} already triggered this era")
                    return None

        # Check once-per-game restriction
        if config.once_per_game and moment_type in self._global_firsts:
            if self._global_firsts[moment_type] != player_id:
                logger.debug(f"Moment {moment_type.name} already triggered globally")
                return None

        # Calculate score
        score = config.base_score

        # Check for first-in-world bonus
        is_first = moment_type not in self._global_firsts
        if is_first:
            score += config.first_bonus
            self._global_firsts[moment_type] = player_id

        # Era scaling
        if config.era_scaling:
            era_multiplier = progress.current_era.value
            score = int(score * (1 + 0.1 * era_multiplier))

        moment = HistoricMoment(
            moment_type=moment_type,
            player_id=player_id,
            turn=turn,
            era=progress.current_era,
            score=score,
            details=details,
            is_first=is_first,
        )

        progress.moments.append(moment)
        progress.era_score += score

        logger.info(
            f"Player {player_id} achieved {moment_type.name}: +{score} era score "
            f"(total: {progress.era_score})"
        )

        return moment

    def get_era_threshold(self, player_id: int) -> dict[str, int]:
        """Get era score thresholds for a player.

        Args:
            player_id: Player to check

        Returns:
            Dictionary with dark and golden thresholds
        """
        progress = self._get_progress(player_id)
        era_level = progress.current_era.value

        # Base threshold scales with era
        base_threshold = self.config.base_era_score_threshold
        scaled = int(base_threshold * (self.config.era_scaling_factor ** (era_level - 1)))

        dark_threshold = int(scaled * self.config.dark_age_threshold_ratio)
        golden_threshold = int(scaled * self.config.golden_age_threshold_ratio)

        return {
            "dark_threshold": dark_threshold,
            "golden_threshold": golden_threshold,
            "current_score": progress.era_score,
        }

    def calculate_next_age(self, player_id: int) -> AgeType:
        """Calculate what age the player will enter.

        Args:
            player_id: Player to check

        Returns:
            Predicted age type
        """
        progress = self._get_progress(player_id)
        thresholds = self.get_era_threshold(player_id)

        score = progress.era_score
        dark_threshold = thresholds["dark_threshold"]
        golden_threshold = thresholds["golden_threshold"]

        if score >= golden_threshold:
            # Check for heroic age (golden after dark)
            if progress.was_dark_age:
                return AgeType.HEROIC
            return AgeType.GOLDEN
        elif score >= dark_threshold:
            return AgeType.NORMAL
        else:
            return AgeType.DARK

    def advance_era(self, player_id: int) -> dict[str, Any]:
        """Advance a player to the next era.

        Args:
            player_id: Player advancing

        Returns:
            Dictionary with era transition info
        """
        progress = self._get_progress(player_id)
        old_era = progress.current_era
        old_age = progress.age_type

        # Calculate new age
        new_age = self.calculate_next_age(player_id)

        # Advance era
        era_values = list(Era)
        current_idx = era_values.index(progress.current_era)
        if current_idx < len(era_values) - 1:
            progress.current_era = era_values[current_idx + 1]

        # Update age
        progress.was_dark_age = progress.age_type == AgeType.DARK
        progress.age_type = new_age

        # Reset era progress
        final_score = progress.era_score
        progress.era_score = 0
        progress.moments = []
        progress.dedications = []
        progress.era_turns = 0

        result = {
            "old_era": old_era.name,
            "new_era": progress.current_era.name,
            "old_age": old_age.name,
            "new_age": new_age.name,
            "final_score": final_score,
            "is_heroic": new_age == AgeType.HEROIC,
        }

        logger.info(
            f"Player {player_id} advanced to {progress.current_era.name} "
            f"({new_age.name} Age, score: {final_score})"
        )

        return result

    def select_dedication(
        self,
        player_id: int,
        dedication_type: DedicationType,
    ) -> bool:
        """Select a dedication for the current age.

        Args:
            player_id: Player selecting
            dedication_type: Dedication to select

        Returns:
            True if dedication was selected
        """
        progress = self._get_progress(player_id)

        if dedication_type not in self._dedications:
            logger.warning(f"Unknown dedication: {dedication_type}")
            return False

        # Check capacity
        if progress.age_type == AgeType.HEROIC:
            max_dedications = self.config.max_dedications_heroic
        elif progress.age_type == AgeType.GOLDEN:
            max_dedications = self.config.max_dedications_golden
        else:
            max_dedications = 0

        if len(progress.dedications) >= max_dedications:
            logger.warning(f"Player {player_id} at dedication capacity")
            return False

        if dedication_type in progress.dedications:
            logger.warning(f"Dedication {dedication_type.name} already selected")
            return False

        progress.dedications.append(dedication_type)
        logger.info(f"Player {player_id} selected dedication: {dedication_type.name}")
        return True

    def get_dedication_effects(self, player_id: int) -> dict[str, float]:
        """Get total effects from dedications.

        Args:
            player_id: Player to check

        Returns:
            Dictionary of effect values
        """
        progress = self._get_progress(player_id)
        effects: dict[str, float] = {}

        is_golden = progress.age_type in (AgeType.GOLDEN, AgeType.HEROIC)

        for dedication_type in progress.dedications:
            dedication = self._dedications.get(dedication_type)
            if not dedication:
                continue

            # Apply appropriate effects
            source = dedication.golden_effects if is_golden else dedication.effects
            for effect, value in source.items():
                effects[effect] = effects.get(effect, 0.0) + value

        return effects

    def get_age_effects(self, player_id: int) -> dict[str, float]:
        """Get effects from current age type.

        Args:
            player_id: Player to check

        Returns:
            Dictionary of effects
        """
        progress = self._get_progress(player_id)
        effects: dict[str, float] = {}

        if progress.age_type == AgeType.DARK:
            effects["loyalty_penalty"] = float(self.config.dark_age_loyalty_penalty)
            effects["dark_age_policies"] = 1.0  # Can use dark age policies
        elif progress.age_type in (AgeType.GOLDEN, AgeType.HEROIC):
            effects["loyalty_bonus"] = float(self.config.golden_age_loyalty_bonus)
            if progress.age_type == AgeType.HEROIC:
                effects["heroic_bonus"] = 1.0  # Extra dedication slot

        return effects

    def process_turn(self) -> None:
        """Process turn for era mechanics."""
        self._game_turn += 1

        for progress in self._player_progress.values():
            progress.era_turns += 1

    def get_player_era_info(self, player_id: int) -> dict[str, Any]:
        """Get complete era information for a player.

        Args:
            player_id: Player to check

        Returns:
            Dictionary of era information
        """
        progress = self._get_progress(player_id)
        thresholds = self.get_era_threshold(player_id)

        return {
            "era": progress.current_era.name,
            "era_number": progress.current_era.value,
            "age_type": progress.age_type.name,
            "era_score": progress.era_score,
            "dark_threshold": thresholds["dark_threshold"],
            "golden_threshold": thresholds["golden_threshold"],
            "dedications": [d.name for d in progress.dedications],
            "moments_count": len(progress.moments),
            "era_turns": progress.era_turns,
            "was_dark_age": progress.was_dark_age,
        }

    def encode_for_state(self, player_id: int) -> dict[str, Any]:
        """Encode era state for neural network input.

        Args:
            player_id: Player perspective

        Returns:
            Dictionary of encoded features
        """
        progress = self._get_progress(player_id)
        thresholds = self.get_era_threshold(player_id)

        # Normalized progress toward thresholds
        score = progress.era_score
        golden_threshold = max(1, thresholds["golden_threshold"])
        dark_threshold = max(1, thresholds["dark_threshold"])

        return {
            "era_number": progress.current_era.value,
            "age_type": progress.age_type.value,
            "is_dark_age": float(progress.age_type == AgeType.DARK),
            "is_golden_age": float(progress.age_type in (AgeType.GOLDEN, AgeType.HEROIC)),
            "is_heroic_age": float(progress.age_type == AgeType.HEROIC),
            "era_score": score,
            "score_toward_golden": min(1.0, score / golden_threshold),
            "score_toward_dark": min(1.0, score / dark_threshold),
            "dedications_count": len(progress.dedications),
            "moments_this_era": len(progress.moments),
        }

    def reset(self) -> None:
        """Reset all era state."""
        self._player_progress.clear()
        self._global_firsts.clear()
        self._game_turn = 0
        logger.debug("EraManager reset")
