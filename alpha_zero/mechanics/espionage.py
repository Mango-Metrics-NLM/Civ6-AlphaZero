"""Espionage System for Civ6-AlphaZero.

This module implements a comprehensive espionage system including:
- Spy recruitment and management
- Mission types (gathering intel, stealing tech, sabotage)
- Counterintelligence
- Success probability calculations

All mechanics are designed to be:
- Fully configurable via dataclasses
- Testable in isolation
- Compatible with the existing encoder system
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class SpyRank(Enum):
    """Spy experience ranks."""

    RECRUIT = auto()
    AGENT = auto()
    SECRET_AGENT = auto()
    MASTER_SPY = auto()


class SpyStatus(Enum):
    """Current status of a spy."""

    AVAILABLE = auto()  # Ready to deploy
    TRAVELING = auto()  # Moving to destination
    ESTABLISHING = auto()  # Building network
    ACTIVE = auto()  # On mission
    CAPTURED = auto()  # In enemy custody
    DEAD = auto()


class MissionType(Enum):
    """Types of spy missions."""

    ESTABLISH_SURVEILLANCE = auto()  # Gain visibility
    GATHER_INTELLIGENCE = auto()  # See city production/tech
    STEAL_TECHNOLOGY = auto()  # Steal tech from campus
    SABOTAGE_PRODUCTION = auto()  # Damage industrial zone
    SIPHON_FUNDS = auto()  # Steal gold from commercial hub
    RECRUIT_PARTISANS = auto()  # Spawn units
    FOMENT_UNREST = auto()  # Reduce loyalty
    NEUTRALIZE_GOVERNOR = auto()  # Disable governor
    BREACH_DAM = auto()  # Cause flooding
    FABRICATE_SCANDAL = auto()  # Reduce diplomatic favor
    COUNTERINTELLIGENCE = auto()  # Defend city


class MissionOutcome(Enum):
    """Possible mission outcomes."""

    CRITICAL_SUCCESS = auto()  # Extra rewards
    SUCCESS = auto()  # Mission accomplished
    PARTIAL_SUCCESS = auto()  # Reduced effect
    FAILURE = auto()  # No effect
    CAPTURED = auto()  # Spy captured
    KILLED = auto()  # Spy killed


@dataclass
class EspionageConfig:
    """Configuration for espionage mechanics.

    Attributes:
        base_mission_success_rate: Base chance of success (0-1)
        rank_success_bonus: Bonus per rank level
        counterspy_penalty: Penalty when counterspy present
        establish_turns: Turns to establish network
        mission_turns: Turns to complete mission
        travel_turns_per_distance: Turns per tile distance
        capture_base_chance: Base chance of capture on failure
        death_chance_on_capture: Chance of death if captured
        recruit_cost: Gold cost to recruit spy
        max_spies_base: Base number of spies allowed
        max_spies_per_gov_tier: Additional spies per government tier
    """

    base_mission_success_rate: float = 0.65
    rank_success_bonus: float = 0.10
    counterspy_penalty: float = 0.25
    establish_turns: int = 3
    mission_turns: int = 5
    travel_turns_per_distance: float = 0.5
    capture_base_chance: float = 0.30
    death_chance_on_capture: float = 0.25
    recruit_cost: int = 200
    max_spies_base: int = 1
    max_spies_per_gov_tier: int = 1


@dataclass
class MissionConfig:
    """Configuration for a specific mission type.

    Attributes:
        mission_type: Type of mission
        base_difficulty: Base difficulty modifier
        required_rank: Minimum rank required
        turns_required: Turns to complete
        target_district: Required district at target
        effect_value: Base effect value
        can_be_lethal: Whether spy can die on failure
    """

    mission_type: MissionType
    base_difficulty: float = 0.0
    required_rank: SpyRank = SpyRank.RECRUIT
    turns_required: int = 5
    target_district: str | None = None
    effect_value: float = 1.0
    can_be_lethal: bool = False


# Default mission configurations
DEFAULT_MISSION_CONFIGS: dict[MissionType, MissionConfig] = {
    MissionType.ESTABLISH_SURVEILLANCE: MissionConfig(
        mission_type=MissionType.ESTABLISH_SURVEILLANCE,
        base_difficulty=-0.10,
        required_rank=SpyRank.RECRUIT,
        turns_required=3,
        effect_value=1.0,
    ),
    MissionType.GATHER_INTELLIGENCE: MissionConfig(
        mission_type=MissionType.GATHER_INTELLIGENCE,
        base_difficulty=0.0,
        required_rank=SpyRank.RECRUIT,
        turns_required=4,
        effect_value=1.0,
    ),
    MissionType.STEAL_TECHNOLOGY: MissionConfig(
        mission_type=MissionType.STEAL_TECHNOLOGY,
        base_difficulty=0.15,
        required_rank=SpyRank.AGENT,
        turns_required=6,
        target_district="Campus",
        effect_value=1.0,
        can_be_lethal=True,
    ),
    MissionType.SABOTAGE_PRODUCTION: MissionConfig(
        mission_type=MissionType.SABOTAGE_PRODUCTION,
        base_difficulty=0.10,
        required_rank=SpyRank.AGENT,
        turns_required=5,
        target_district="Industrial Zone",
        effect_value=50.0,  # Production damage
        can_be_lethal=True,
    ),
    MissionType.SIPHON_FUNDS: MissionConfig(
        mission_type=MissionType.SIPHON_FUNDS,
        base_difficulty=0.05,
        required_rank=SpyRank.AGENT,
        turns_required=5,
        target_district="Commercial Hub",
        effect_value=100.0,  # Gold stolen
    ),
    MissionType.RECRUIT_PARTISANS: MissionConfig(
        mission_type=MissionType.RECRUIT_PARTISANS,
        base_difficulty=0.20,
        required_rank=SpyRank.SECRET_AGENT,
        turns_required=7,
        effect_value=2.0,  # Units spawned
        can_be_lethal=True,
    ),
    MissionType.FOMENT_UNREST: MissionConfig(
        mission_type=MissionType.FOMENT_UNREST,
        base_difficulty=0.15,
        required_rank=SpyRank.SECRET_AGENT,
        turns_required=6,
        effect_value=20.0,  # Loyalty reduction
    ),
    MissionType.COUNTERINTELLIGENCE: MissionConfig(
        mission_type=MissionType.COUNTERINTELLIGENCE,
        base_difficulty=-0.20,  # Easier, defensive
        required_rank=SpyRank.RECRUIT,
        turns_required=0,  # Passive
        effect_value=0.25,  # Detection bonus
    ),
}


@dataclass
class Spy:
    """Represents a spy unit.

    Attributes:
        spy_id: Unique identifier
        name: Spy's code name
        player_id: Owner player ID
        rank: Current experience rank
        status: Current status
        experience: Total experience points
        location: Current city location (None if at home)
        home_city: Spy's home city
        current_mission: Active mission type
        mission_target: Target city for mission
        turns_remaining: Turns until action completes
        times_captured: Number of times captured
        missions_completed: Total missions completed
        kills: Enemy spies killed
    """

    spy_id: str
    name: str
    player_id: int
    rank: SpyRank = SpyRank.RECRUIT
    status: SpyStatus = SpyStatus.AVAILABLE
    experience: int = 0
    location: str | None = None
    home_city: str | None = None
    current_mission: MissionType | None = None
    mission_target: str | None = None
    turns_remaining: int = 0
    times_captured: int = 0
    missions_completed: int = 0
    kills: int = 0

    def gain_experience(self, amount: int) -> bool:
        """Add experience and check for promotion.

        Args:
            amount: Experience to add

        Returns:
            True if spy was promoted
        """
        self.experience += amount
        old_rank = self.rank

        # Rank thresholds
        thresholds = {
            SpyRank.AGENT: 100,
            SpyRank.SECRET_AGENT: 300,
            SpyRank.MASTER_SPY: 600,
        }

        for rank, threshold in thresholds.items():
            if self.experience >= threshold and rank.value > self.rank.value:
                self.rank = rank

        promoted = self.rank != old_rank
        if promoted:
            logger.info(f"Spy {self.spy_id} promoted to {self.rank.name}")
        return promoted

    def get_rank_bonus(self) -> float:
        """Get success rate bonus from rank."""
        return (self.rank.value - 1) * 0.10


@dataclass
class Mission:
    """An active espionage mission.

    Attributes:
        mission_id: Unique identifier
        spy_id: Spy performing the mission
        mission_type: Type of mission
        target_city: Target city name
        target_player: Target player ID
        turns_remaining: Turns until completion
        config: Mission configuration
    """

    mission_id: str
    spy_id: str
    mission_type: MissionType
    target_city: str
    target_player: int
    turns_remaining: int
    config: MissionConfig


class EspionageManager:
    """Manages all espionage mechanics.

    This class handles:
    - Spy recruitment and deployment
    - Mission assignment and resolution
    - Counterintelligence
    - Success/failure calculations

    Example:
        >>> config = EspionageConfig(base_mission_success_rate=0.70)
        >>> manager = EspionageManager(config)
        >>> spy = manager.recruit_spy(player_id=0, name="Shadow", home_city="Rome")
        >>> manager.deploy_spy(spy.spy_id, "Alexandria", distance=10)
        >>> mission = manager.assign_mission(spy.spy_id, MissionType.GATHER_INTELLIGENCE)
    """

    def __init__(
        self,
        config: EspionageConfig | None = None,
        mission_configs: dict[MissionType, MissionConfig] | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Initialize espionage manager.

        Args:
            config: Espionage configuration
            mission_configs: Custom mission configurations
            random_seed: Seed for reproducibility
        """
        self.config = config or EspionageConfig()
        self._mission_configs = mission_configs or DEFAULT_MISSION_CONFIGS
        self._rng = random.Random(random_seed)

        self._spies: dict[str, Spy] = {}
        self._missions: dict[str, Mission] = {}
        self._player_spy_counts: dict[int, int] = {}
        self._counterspy_locations: dict[str, str] = {}  # city -> spy_id
        self._spy_counter = 0
        self._mission_counter = 0

        logger.debug(f"EspionageManager initialized with config: {self.config}")

    def recruit_spy(
        self,
        player_id: int,
        name: str,
        home_city: str,
    ) -> Spy | None:
        """Recruit a new spy.

        Args:
            player_id: Owner player ID
            name: Spy's code name
            home_city: Spy's home city

        Returns:
            Recruited spy, or None if at capacity
        """
        current_count = self._player_spy_counts.get(player_id, 0)
        max_spies = self.config.max_spies_base + self.config.max_spies_per_gov_tier

        if current_count >= max_spies:
            logger.warning(f"Player {player_id} at spy capacity ({max_spies})")
            return None

        self._spy_counter += 1
        spy_id = f"spy_{self._spy_counter}"

        spy = Spy(
            spy_id=spy_id,
            name=name,
            player_id=player_id,
            home_city=home_city,
        )

        self._spies[spy_id] = spy
        self._player_spy_counts[player_id] = current_count + 1

        logger.info(f"Recruited spy {spy_id} ({name}) for player {player_id}")
        return spy

    def deploy_spy(
        self,
        spy_id: str,
        destination: str,
        distance: int = 10,
    ) -> bool:
        """Deploy a spy to a city.

        Args:
            spy_id: ID of spy to deploy
            destination: Target city name
            distance: Distance in tiles

        Returns:
            True if deployment started
        """
        spy = self._spies.get(spy_id)
        if not spy:
            return False

        if spy.status not in (SpyStatus.AVAILABLE,):
            logger.warning(f"Spy {spy_id} not available (status: {spy.status})")
            return False

        travel_turns = max(1, int(distance * self.config.travel_turns_per_distance))

        spy.status = SpyStatus.TRAVELING
        spy.mission_target = destination
        spy.turns_remaining = travel_turns

        logger.info(f"Spy {spy_id} deploying to {destination} ({travel_turns} turns)")
        return True

    def assign_mission(
        self,
        spy_id: str,
        mission_type: MissionType,
        target_city: str | None = None,
    ) -> Mission | None:
        """Assign a mission to a spy.

        Args:
            spy_id: ID of spy
            mission_type: Type of mission
            target_city: Target city (uses spy location if None)

        Returns:
            Created mission, or None if invalid
        """
        spy = self._spies.get(spy_id)
        if not spy:
            return None

        if spy.status != SpyStatus.ACTIVE:
            logger.warning(f"Spy {spy_id} not active (status: {spy.status})")
            return None

        config = self._mission_configs.get(mission_type)
        if not config:
            logger.warning(f"Unknown mission type: {mission_type}")
            return None

        # Check rank requirement
        if spy.rank.value < config.required_rank.value:
            logger.warning(
                f"Spy {spy_id} rank {spy.rank.name} insufficient for {mission_type.name}"
            )
            return None

        target = target_city or spy.location
        if not target:
            return None

        self._mission_counter += 1
        mission_id = f"mission_{self._mission_counter}"

        mission = Mission(
            mission_id=mission_id,
            spy_id=spy_id,
            mission_type=mission_type,
            target_city=target,
            target_player=-1,  # Will be set by caller
            turns_remaining=config.turns_required,
            config=config,
        )

        self._missions[mission_id] = mission
        spy.current_mission = mission_type

        # Handle counterintelligence specially
        if mission_type == MissionType.COUNTERINTELLIGENCE:
            self._counterspy_locations[target] = spy_id

        logger.info(f"Assigned {mission_type.name} to spy {spy_id} in {target}")
        return mission

    def process_turn(self) -> list[dict[str, Any]]:
        """Process all espionage for a turn.

        Returns:
            List of mission results
        """
        results: list[dict[str, Any]] = []

        # Process traveling spies
        for spy in self._spies.values():
            if spy.status == SpyStatus.TRAVELING:
                spy.turns_remaining -= 1
                if spy.turns_remaining <= 0:
                    spy.status = SpyStatus.ESTABLISHING
                    spy.location = spy.mission_target
                    spy.turns_remaining = self.config.establish_turns
                    logger.debug(f"Spy {spy.spy_id} arrived at {spy.location}")

            elif spy.status == SpyStatus.ESTABLISHING:
                spy.turns_remaining -= 1
                if spy.turns_remaining <= 0:
                    spy.status = SpyStatus.ACTIVE
                    logger.debug(f"Spy {spy.spy_id} network established in {spy.location}")

        # Process active missions
        completed_missions: list[str] = []
        for mission_id, mission in self._missions.items():
            if mission.mission_type == MissionType.COUNTERINTELLIGENCE:
                continue  # Counterintelligence is passive

            mission.turns_remaining -= 1
            if mission.turns_remaining <= 0:
                result = self._resolve_mission(mission)
                results.append(result)
                completed_missions.append(mission_id)

        for mission_id in completed_missions:
            del self._missions[mission_id]

        return results

    def _resolve_mission(self, mission: Mission) -> dict[str, Any]:
        """Resolve a completed mission.

        Args:
            mission: Mission to resolve

        Returns:
            Result dictionary
        """
        spy = self._spies.get(mission.spy_id)
        if not spy:
            return {"error": "Spy not found"}

        # Calculate success chance
        base_chance = self.config.base_mission_success_rate
        rank_bonus = spy.get_rank_bonus()
        difficulty_mod = -mission.config.base_difficulty

        # Check for counterspy
        counterspy_id = self._counterspy_locations.get(mission.target_city)
        counterspy_penalty = 0.0
        if counterspy_id:
            counterspy = self._spies.get(counterspy_id)
            if counterspy and counterspy.player_id != spy.player_id:
                counterspy_penalty = self.config.counterspy_penalty
                counterspy_penalty += counterspy.get_rank_bonus() * 0.5

        success_chance = base_chance + rank_bonus + difficulty_mod - counterspy_penalty
        success_chance = max(0.1, min(0.95, success_chance))  # Clamp

        roll = self._rng.random()
        outcome = self._determine_outcome(roll, success_chance, mission.config)

        result: dict[str, Any] = {
            "spy_id": spy.spy_id,
            "spy_name": spy.name,
            "mission_type": mission.mission_type.name,
            "target_city": mission.target_city,
            "outcome": outcome.name,
            "success_chance": success_chance,
            "roll": roll,
            "effect_value": 0.0,
        }

        # Apply outcome effects
        if outcome == MissionOutcome.CRITICAL_SUCCESS:
            result["effect_value"] = mission.config.effect_value * 1.5
            spy.gain_experience(75)
            spy.missions_completed += 1

        elif outcome == MissionOutcome.SUCCESS:
            result["effect_value"] = mission.config.effect_value
            spy.gain_experience(50)
            spy.missions_completed += 1

        elif outcome == MissionOutcome.PARTIAL_SUCCESS:
            result["effect_value"] = mission.config.effect_value * 0.5
            spy.gain_experience(25)
            spy.missions_completed += 1

        elif outcome == MissionOutcome.FAILURE:
            spy.gain_experience(10)

        elif outcome == MissionOutcome.CAPTURED:
            spy.status = SpyStatus.CAPTURED
            spy.times_captured += 1
            spy.current_mission = None
            result["captured"] = True

            # Counterspy gets credit
            if counterspy_id:
                counterspy = self._spies.get(counterspy_id)
                if counterspy:
                    counterspy.gain_experience(40)

        elif outcome == MissionOutcome.KILLED:
            spy.status = SpyStatus.DEAD
            spy.current_mission = None
            result["killed"] = True

            if counterspy_id:
                counterspy = self._spies.get(counterspy_id)
                if counterspy:
                    counterspy.kills += 1
                    counterspy.gain_experience(60)

        # Reset spy mission state on completion
        if outcome not in (MissionOutcome.CAPTURED, MissionOutcome.KILLED):
            spy.current_mission = None

        logger.info(
            f"Mission {mission.mission_type.name} by {spy.name}: {outcome.name} "
            f"(chance: {success_chance:.1%}, roll: {roll:.2f})"
        )

        return result

    def _determine_outcome(
        self,
        roll: float,
        success_chance: float,
        config: MissionConfig,
    ) -> MissionOutcome:
        """Determine mission outcome from roll and success chance.

        Args:
            roll: Random roll (0-1)
            success_chance: Calculated success chance
            config: Mission configuration

        Returns:
            Mission outcome
        """
        # Critical success threshold
        if roll < success_chance * 0.2:
            return MissionOutcome.CRITICAL_SUCCESS

        if roll < success_chance:
            return MissionOutcome.SUCCESS

        # Failure zone
        if roll < success_chance + 0.15:
            return MissionOutcome.PARTIAL_SUCCESS

        # Check for capture/death
        capture_chance = self.config.capture_base_chance
        if roll > 1.0 - capture_chance:
            if config.can_be_lethal and self._rng.random() < self.config.death_chance_on_capture:
                return MissionOutcome.KILLED
            return MissionOutcome.CAPTURED

        return MissionOutcome.FAILURE

    def recall_spy(self, spy_id: str) -> bool:
        """Recall a spy back to home city.

        Args:
            spy_id: ID of spy to recall

        Returns:
            True if recall started
        """
        spy = self._spies.get(spy_id)
        if not spy:
            return False

        # Remove from counterspy locations
        for city, cs_id in list(self._counterspy_locations.items()):
            if cs_id == spy_id:
                del self._counterspy_locations[city]

        # Cancel active missions
        for mission_id, mission in list(self._missions.items()):
            if mission.spy_id == spy_id:
                del self._missions[mission_id]

        spy.status = SpyStatus.TRAVELING
        spy.current_mission = None
        spy.mission_target = spy.home_city
        spy.turns_remaining = 2

        logger.info(f"Recalling spy {spy_id} to {spy.home_city}")
        return True

    def exchange_captured_spy(
        self,
        spy_id: str,
        captor_player_id: int,
    ) -> bool:
        """Exchange a captured spy.

        Args:
            spy_id: ID of captured spy
            captor_player_id: Player who captured the spy

        Returns:
            True if exchange successful
        """
        spy = self._spies.get(spy_id)
        if not spy or spy.status != SpyStatus.CAPTURED:
            return False

        spy.status = SpyStatus.AVAILABLE
        spy.location = None
        logger.info(f"Spy {spy_id} exchanged and returned to player {spy.player_id}")
        return True

    def get_spy(self, spy_id: str) -> Spy | None:
        """Get a spy by ID."""
        return self._spies.get(spy_id)

    def get_player_spies(self, player_id: int) -> list[Spy]:
        """Get all spies for a player."""
        return [s for s in self._spies.values() if s.player_id == player_id]

    def get_available_missions(
        self,
        spy_id: str,
        city_districts: set[str] | None = None,
    ) -> list[MissionType]:
        """Get available missions for a spy.

        Args:
            spy_id: ID of spy
            city_districts: Districts present in target city

        Returns:
            List of available mission types
        """
        spy = self._spies.get(spy_id)
        if not spy or spy.status != SpyStatus.ACTIVE:
            return []

        available: list[MissionType] = []
        for mission_type, config in self._mission_configs.items():
            # Check rank
            if spy.rank.value < config.required_rank.value:
                continue

            # Check district requirement
            if config.target_district and city_districts:
                if config.target_district not in city_districts:
                    continue

            available.append(mission_type)

        return available

    def encode_for_state(self, player_id: int) -> dict[str, Any]:
        """Encode espionage state for neural network input.

        Args:
            player_id: Player perspective

        Returns:
            Dictionary of encoded espionage features
        """
        spies = self.get_player_spies(player_id)

        return {
            "num_spies": len(spies),
            "spies_available": len([s for s in spies if s.status == SpyStatus.AVAILABLE]),
            "spies_active": len([s for s in spies if s.status == SpyStatus.ACTIVE]),
            "spies_captured": len([s for s in spies if s.status == SpyStatus.CAPTURED]),
            "avg_spy_rank": sum(s.rank.value for s in spies) / max(1, len(spies)),
            "total_missions_completed": sum(s.missions_completed for s in spies),
            "counterspy_deployed": len([
                city for city, sid in self._counterspy_locations.items()
                if self._spies.get(sid, Spy("", "", -1)).player_id == player_id
            ]),
        }

    def reset(self) -> None:
        """Reset all espionage state."""
        self._spies.clear()
        self._missions.clear()
        self._player_spy_counts.clear()
        self._counterspy_locations.clear()
        self._spy_counter = 0
        self._mission_counter = 0
        logger.debug("EspionageManager reset")
