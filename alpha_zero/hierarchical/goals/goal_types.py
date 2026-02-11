"""
Goal Type Definitions for Hierarchical Reasoning.

This module defines the goal types at each hierarchy level, along with
the unified Goal dataclass and supporting types.

The goal types are designed to be:
- Exhaustive: Cover all meaningful decisions at each level
- Mutually exclusive within level: Each goal type represents a distinct objective
- Hierarchically decomposable: Higher-level goals map to lower-level subgoals

Example:
    >>> from alpha_zero.hierarchical.goals.goal_types import Goal, StrategicGoalType
    >>> goal = Goal.strategic(StrategicGoalType.DOMINATION_VICTORY)
    >>> print(goal.level, goal.goal_type)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, ClassVar

import numpy as np


class HierarchyLevel(Enum):
    """Levels in the hierarchical decision system."""

    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"

    @property
    def horizon(self) -> int:
        """Get typical planning horizon in turns."""
        horizons = {
            HierarchyLevel.STRATEGIC: 50,
            HierarchyLevel.TACTICAL: 15,
            HierarchyLevel.OPERATIONAL: 3,
        }
        return horizons[self]

    @property
    def update_frequency(self) -> int:
        """Get typical update frequency in turns."""
        frequencies = {
            HierarchyLevel.STRATEGIC: 10,
            HierarchyLevel.TACTICAL: 3,
            HierarchyLevel.OPERATIONAL: 1,
        }
        return frequencies[self]


class GoalStatus(Enum):
    """Status of a goal in the hierarchy."""

    PENDING = auto()  # Not yet started
    ACTIVE = auto()  # Currently being pursued
    PAUSED = auto()  # Temporarily suspended
    COMPLETED = auto()  # Successfully achieved
    FAILED = auto()  # Failed to achieve
    ABANDONED = auto()  # Intentionally dropped


# =============================================================================
# Strategic Goal Types (30-100 turn horizon)
# =============================================================================


class StrategicGoalType(Enum):
    """Strategic-level goal types representing victory paths and long-term objectives."""

    # Victory conditions
    DOMINATION_VICTORY = auto()  # Conquer all capitals
    SCIENCE_VICTORY = auto()  # Launch Mars mission
    CULTURE_VICTORY = auto()  # Achieve cultural hegemony
    RELIGIOUS_VICTORY = auto()  # Convert all civs
    DIPLOMATIC_VICTORY = auto()  # Win world congress
    SCORE_VICTORY = auto()  # Highest score at turn limit

    # Strategic postures
    MILITARY_SUPREMACY = auto()  # Build strongest military
    ECONOMIC_DOMINANCE = auto()  # Achieve highest GDP
    TECH_LEADERSHIP = auto()  # Lead in technology
    CULTURAL_INFLUENCE = auto()  # Spread culture globally
    RELIGIOUS_SPREAD = auto()  # Expand religion
    TERRITORIAL_EXPANSION = auto()  # Maximize territory
    DEFENSIVE_CONSOLIDATION = auto()  # Fortify and defend

    # Diplomatic stances
    ALLIANCE_BUILDING = auto()  # Form strong alliances
    ISOLATION_POLICY = auto()  # Avoid entanglements
    OPPORTUNISTIC_WARFARE = auto()  # Strike when advantageous

    # Aliases for backwards compatibility
    EXPAND_TERRITORY = auto()  # Alias for TERRITORIAL_EXPANSION
    DEFEND_EMPIRE = auto()  # Alias for DEFENSIVE_CONSOLIDATION
    ESTABLISH_ALLIANCE = auto()  # Form strategic alliance
    SABOTAGE_ENEMY = auto()  # Disrupt enemy operations
    CONTROL_CONTINENT = auto()  # Dominate geographic region
    TECHNOLOGICAL_LEAD = auto()  # Lead in technology
    RESOURCE_MONOPOLY = auto()  # Control key resources

    @property
    def category(self) -> str:
        """Get the category of this strategic goal."""
        victory_goals = {
            self.DOMINATION_VICTORY,
            self.SCIENCE_VICTORY,
            self.CULTURE_VICTORY,
            self.RELIGIOUS_VICTORY,
            self.DIPLOMATIC_VICTORY,
            self.SCORE_VICTORY,
        }
        posture_goals = {
            self.MILITARY_SUPREMACY,
            self.ECONOMIC_DOMINANCE,
            self.TECH_LEADERSHIP,
            self.CULTURAL_INFLUENCE,
            self.RELIGIOUS_SPREAD,
            self.TERRITORIAL_EXPANSION,
            self.DEFENSIVE_CONSOLIDATION,
            self.EXPAND_TERRITORY,
            self.DEFEND_EMPIRE,
        }
        if self in victory_goals:
            return "victory"
        if self in posture_goals:
            return "posture"
        return "diplomatic"


# =============================================================================
# Tactical Goal Types (5-15 turn horizon)
# =============================================================================


class TacticalGoalType(Enum):
    """Tactical-level goal types representing medium-term objectives."""

    # Military operations
    CONQUER_CITY = auto()  # Capture specific city
    DEFEND_CITY = auto()  # Protect city from attack
    BUILD_ARMY = auto()  # Raise military forces
    DESTROY_ARMY = auto()  # Eliminate enemy forces
    SIEGE_OPERATION = auto()  # Set up city siege
    RAID_TERRITORY = auto()  # Pillage enemy improvements
    NAVAL_BLOCKADE = auto()  # Block enemy sea routes

    # Infrastructure development
    DEVELOP_CITY = auto()  # Grow and improve city
    BUILD_WONDER = auto()  # Construct specific wonder
    IMPROVE_REGION = auto()  # Build improvements in area
    CONNECT_CITIES = auto()  # Build roads/trade routes
    ESTABLISH_OUTPOST = auto()  # Settle new city

    # Resource and economy
    SECURE_RESOURCE = auto()  # Control strategic resource
    MAXIMIZE_INCOME = auto()  # Boost gold generation
    BOOST_PRODUCTION = auto()  # Increase city production
    GROW_POPULATION = auto()  # Increase city population

    # Technology and culture
    RESEARCH_TECH = auto()  # Complete specific tech
    RESEARCH_CIVIC = auto()  # Complete specific civic
    BUILD_CAMPUS = auto()  # Establish science district
    BUILD_THEATER = auto()  # Establish culture district
    RECRUIT_GREAT_PERSON = auto()  # Acquire great person

    # Diplomacy
    FORM_ALLIANCE = auto()  # Establish alliance
    NEGOTIATE_PEACE = auto()  # End conflict
    DECLARE_WAR = auto()  # Start conflict
    TRADE_AGREEMENT = auto()  # Establish trade deal

    # Aliases for backwards compatibility
    CAPTURE_CITY = auto()  # Alias for CONQUER_CITY
    SETTLE_CITY = auto()  # Alias for ESTABLISH_OUTPOST
    BUILD_ENCAMPMENT = auto()  # Build encampment district
    GENERATE_TOURISM = auto()  # Generate tourism output
    BUILD_COMMERCIAL = auto()  # Build commercial hub
    FORTIFY_POSITION = auto()  # Fortify and hold position
    ELIMINATE_UNIT = auto()  # Destroy enemy unit
    SPREAD_RELIGION = auto()  # Spread religious beliefs
    BUILD_HOLY_SITE = auto()  # Build holy site district
    IMPROVE_TILE = auto()  # Build tile improvement
    ESTABLISH_TRADE = auto()  # Establish trade route
    UPGRADE_UNIT = auto()  # Upgrade military unit
    PILLAGE_IMPROVEMENT = auto()  # Pillage enemy improvements

    @property
    def category(self) -> str:
        """Get the category of this tactical goal."""
        military = {
            self.CONQUER_CITY,
            self.DEFEND_CITY,
            self.BUILD_ARMY,
            self.DESTROY_ARMY,
            self.SIEGE_OPERATION,
            self.RAID_TERRITORY,
            self.NAVAL_BLOCKADE,
            self.CAPTURE_CITY,
            self.FORTIFY_POSITION,
            self.ELIMINATE_UNIT,
            self.PILLAGE_IMPROVEMENT,
            self.UPGRADE_UNIT,
        }
        infrastructure = {
            self.DEVELOP_CITY,
            self.BUILD_WONDER,
            self.IMPROVE_REGION,
            self.CONNECT_CITIES,
            self.ESTABLISH_OUTPOST,
            self.SETTLE_CITY,
            self.BUILD_ENCAMPMENT,
            self.BUILD_COMMERCIAL,
            self.IMPROVE_TILE,
            self.BUILD_HOLY_SITE,
            self.ESTABLISH_TRADE,
        }
        economy = {
            self.SECURE_RESOURCE,
            self.MAXIMIZE_INCOME,
            self.BOOST_PRODUCTION,
            self.GROW_POPULATION,
        }
        research = {
            self.RESEARCH_TECH,
            self.RESEARCH_CIVIC,
            self.BUILD_CAMPUS,
            self.BUILD_THEATER,
            self.RECRUIT_GREAT_PERSON,
            self.GENERATE_TOURISM,
            self.SPREAD_RELIGION,
        }
        if self in military:
            return "military"
        if self in infrastructure:
            return "infrastructure"
        if self in economy:
            return "economy"
        if self in research:
            return "research"
        return "diplomacy"


# =============================================================================
# Operational Goal Types (1-3 turn horizon)
# =============================================================================


class OperationalGoalType(Enum):
    """Operational-level goal types representing immediate actions."""

    # Unit movement
    MOVE_TO_POSITION = auto()  # Move unit to location
    ATTACK_TARGET = auto()  # Attack specific target
    DEFEND_POSITION = auto()  # Hold current location
    RETREAT = auto()  # Fall back from danger
    ESCORT_UNIT = auto()  # Protect another unit
    SCOUT_AREA = auto()  # Explore territory
    FORTIFY = auto()  # Dig in for defense

    # City production
    PRODUCE_UNIT = auto()  # Build military unit
    PRODUCE_BUILDING = auto()  # Build city building
    PRODUCE_DISTRICT = auto()  # Build city district
    PRODUCE_WONDER = auto()  # Build world wonder
    PRODUCE_PROJECT = auto()  # Complete city project
    BUY_ITEM = auto()  # Purchase with gold

    # Worker actions
    BUILD_IMPROVEMENT = auto()  # Construct tile improvement
    REPAIR_IMPROVEMENT = auto()  # Fix pillaged improvement
    HARVEST_RESOURCE = auto()  # Remove and collect resource
    PLANT_FOREST = auto()  # Add forest to tile

    # Diplomatic actions
    SEND_DELEGATION = auto()  # Establish embassy
    TRADE_RESOURCES = auto()  # Resource exchange
    DENOUNCE_PLAYER = auto()  # Public condemnation
    MAKE_PROMISE = auto()  # Diplomatic promise

    # Turn management
    SKIP_TURN = auto()  # Pass action
    END_TURN = auto()  # Complete turn

    # Aliases for backwards compatibility
    MOVE_UNIT = auto()  # Alias for MOVE_TO_POSITION
    SELECT_RESEARCH = auto()  # Select research option
    BUILD_DISTRICT = auto()  # Alias for PRODUCE_DISTRICT
    SIEGE_CITY = auto()  # Siege a city
    FORTIFY_UNIT = auto()  # Fortify unit in place
    HEAL_UNIT = auto()  # Heal unit
    UPGRADE_UNIT = auto()  # Upgrade unit
    FOUND_CITY = auto()  # Found a new city
    BUILD_BUILDING = auto()  # Build city building
    ESTABLISH_TRADE_ROUTE = auto()  # Establish trade route
    SPREAD_RELIGION = auto()  # Spread religious influence
    PILLAGE = auto()  # Pillage tile
    CREATE_GREAT_WORK = auto()  # Create great work
    MAKE_DEAL = auto()  # Make diplomatic deal

    @property
    def category(self) -> str:
        """Get the category of this operational goal."""
        movement = {
            self.MOVE_TO_POSITION,
            self.ATTACK_TARGET,
            self.DEFEND_POSITION,
            self.RETREAT,
            self.ESCORT_UNIT,
            self.SCOUT_AREA,
            self.FORTIFY,
            self.MOVE_UNIT,
            self.SIEGE_CITY,
            self.FORTIFY_UNIT,
            self.HEAL_UNIT,
        }
        production = {
            self.PRODUCE_UNIT,
            self.PRODUCE_BUILDING,
            self.PRODUCE_DISTRICT,
            self.PRODUCE_WONDER,
            self.PRODUCE_PROJECT,
            self.BUY_ITEM,
            self.SELECT_RESEARCH,
            self.BUILD_DISTRICT,
            self.UPGRADE_UNIT,
            self.BUILD_BUILDING,
            self.FOUND_CITY,
            self.CREATE_GREAT_WORK,
        }
        improvement = {
            self.BUILD_IMPROVEMENT,
            self.REPAIR_IMPROVEMENT,
            self.HARVEST_RESOURCE,
            self.PLANT_FOREST,
            self.PILLAGE,
            self.ESTABLISH_TRADE_ROUTE,
        }
        diplomacy = {
            self.SEND_DELEGATION,
            self.TRADE_RESOURCES,
            self.DENOUNCE_PLAYER,
            self.MAKE_PROMISE,
            self.MAKE_DEAL,
            self.SPREAD_RELIGION,
        }
        if self in movement:
            return "movement"
        if self in production:
            return "production"
        if self in improvement:
            return "improvement"
        if self in diplomacy:
            return "diplomacy"
        if self in {self.SKIP_TURN, self.END_TURN}:
            return "turn_management"
        return "diplomacy"


# =============================================================================
# Goal Target
# =============================================================================


@dataclass
class GoalTarget:
    """Target specification for a goal.

    Attributes:
        target_type: Type of target (city, unit, position, etc.)
        target_id: Identifier for the target (alias: entity_id).
        position: Board position if applicable (x, y) (alias: location).
        quantity: Target quantity for numeric goals.
        metadata: Additional target-specific data.
    """

    target_type: str = "generic"
    target_id: str | None = None
    position: tuple[int, int] | None = None
    quantity: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Aliases for compatibility
    @property
    def entity_id(self) -> str | None:
        """Alias for target_id."""
        return self.target_id

    @entity_id.setter
    def entity_id(self, value: str | None) -> None:
        """Set target_id via entity_id alias."""
        self.target_id = value

    @property
    def location(self) -> tuple[int, int] | None:
        """Alias for position."""
        return self.position

    @location.setter
    def location(self, value: tuple[int, int] | None) -> None:
        """Set position via location alias."""
        self.position = value

    @classmethod
    def position_target(cls, x: int, y: int) -> GoalTarget:
        """Create a position-based target."""
        return cls(target_type="position", position=(x, y))

    @classmethod
    def city_target(cls, city_id: str, position: tuple[int, int] | None = None) -> GoalTarget:
        """Create a city target."""
        return cls(target_type="city", target_id=city_id, position=position)

    @classmethod
    def unit_target(cls, unit_id: str, position: tuple[int, int] | None = None) -> GoalTarget:
        """Create a unit target."""
        return cls(target_type="unit", target_id=unit_id, position=position)

    @classmethod
    def resource_target(cls, resource_type: str, position: tuple[int, int]) -> GoalTarget:
        """Create a resource target."""
        return cls(
            target_type="resource",
            target_id=resource_type,
            position=position,
        )

    @classmethod
    def player_target(cls, player_id: str) -> GoalTarget:
        """Create a player target."""
        return cls(target_type="player", target_id=player_id)

    @classmethod
    def tech_target(cls, tech_id: str) -> GoalTarget:
        """Create a technology target."""
        return cls(target_type="tech", target_id=tech_id)

    @classmethod
    def no_target(cls) -> GoalTarget:
        """Create a goal with no specific target."""
        return cls(target_type="none")


# =============================================================================
# Unified Goal Class
# =============================================================================

GoalType = StrategicGoalType | TacticalGoalType | OperationalGoalType


@dataclass
class Goal:
    """Unified goal representation across all hierarchy levels.

    This dataclass represents a goal at any level of the hierarchy,
    with support for progress tracking, temporal constraints, and
    neural network embedding.

    Attributes:
        id: Unique identifier for the goal.
        level: Hierarchy level (STRATEGIC, TACTICAL, OPERATIONAL).
        goal_type: Type of goal (level-specific enum).
        description: Human-readable description.
        target: Goal target specification.
        priority: Priority value (higher = more important).
        status: Current status of the goal.
        progress: Progress towards completion (0.0 to 1.0).
        created_turn: Turn when goal was created.
        deadline_turn: Optional deadline turn.
        estimated_duration: Estimated turns to complete.
        last_progress_turn: Turn when progress was last updated.
        parent_goal_id: ID of parent goal in hierarchy.
        child_goal_ids: IDs of child goals.
        strategic_value: Estimated contribution to winning.
        urgency: Time sensitivity factor.
        reward_modifier: Reward shaping multiplier for this goal.
        metadata: Additional goal-specific data.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: HierarchyLevel = HierarchyLevel.OPERATIONAL
    goal_type: GoalType = OperationalGoalType.END_TURN

    # Description
    description: str = ""
    target: GoalTarget = field(default_factory=GoalTarget.no_target)

    # Priority and status
    priority: float = 1.0
    status: GoalStatus = GoalStatus.PENDING

    # Progress tracking
    progress: float = 0.0
    created_turn: int = 0
    deadline_turn: int | None = None
    estimated_duration: int = 1
    last_progress_turn: int = 0

    # Hierarchy links
    parent_goal_id: str | None = None
    child_goal_ids: list[str] = field(default_factory=list)

    # Value estimates
    strategic_value: float = 0.0
    urgency: float = 0.0
    reward_modifier: float = 1.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Embedding configuration
    EMBED_DIM: ClassVar[int] = 32
    NUM_STRATEGIC_GOALS: ClassVar[int] = len(StrategicGoalType)
    NUM_TACTICAL_GOALS: ClassVar[int] = len(TacticalGoalType)
    NUM_OPERATIONAL_GOALS: ClassVar[int] = len(OperationalGoalType)

    def __post_init__(self) -> None:
        """Validate goal after initialization."""
        # Ensure goal_type matches level
        if self.level == HierarchyLevel.STRATEGIC:
            if not isinstance(self.goal_type, StrategicGoalType):
                raise ValueError(
                    f"Strategic level requires StrategicGoalType, got {type(self.goal_type)}"
                )
        elif self.level == HierarchyLevel.TACTICAL:
            if not isinstance(self.goal_type, TacticalGoalType):
                raise ValueError(
                    f"Tactical level requires TacticalGoalType, got {type(self.goal_type)}"
                )
        elif self.level == HierarchyLevel.OPERATIONAL:
            if not isinstance(self.goal_type, OperationalGoalType):
                raise ValueError(
                    f"Operational level requires OperationalGoalType, got {type(self.goal_type)}"
                )

    @classmethod
    def strategic(
        cls,
        goal_type: StrategicGoalType,
        target: GoalTarget | None = None,
        priority: float = 1.0,
        **kwargs: Any,
    ) -> Goal:
        """Create a strategic-level goal."""
        return cls(
            level=HierarchyLevel.STRATEGIC,
            goal_type=goal_type,
            target=target or GoalTarget.no_target(),
            priority=priority,
            estimated_duration=(goal_type.category == "victory" and 100) or 50,
            **kwargs,
        )

    @classmethod
    def tactical(
        cls,
        goal_type: TacticalGoalType,
        target: GoalTarget | None = None,
        priority: float = 1.0,
        **kwargs: Any,
    ) -> Goal:
        """Create a tactical-level goal."""
        return cls(
            level=HierarchyLevel.TACTICAL,
            goal_type=goal_type,
            target=target or GoalTarget.no_target(),
            priority=priority,
            estimated_duration=15,
            **kwargs,
        )

    @classmethod
    def operational(
        cls,
        goal_type: OperationalGoalType,
        target: GoalTarget | None = None,
        priority: float = 1.0,
        **kwargs: Any,
    ) -> Goal:
        """Create an operational-level goal."""
        return cls(
            level=HierarchyLevel.OPERATIONAL,
            goal_type=goal_type,
            target=target or GoalTarget.no_target(),
            priority=priority,
            estimated_duration=1,
            **kwargs,
        )

    @property
    def is_active(self) -> bool:
        """Check if goal is currently active."""
        return self.status == GoalStatus.ACTIVE

    @property
    def is_complete(self) -> bool:
        """Check if goal is completed."""
        return self.status == GoalStatus.COMPLETED

    @property
    def is_terminal(self) -> bool:
        """Check if goal is in a terminal state."""
        return self.status in {
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.ABANDONED,
        }

    @property
    def remaining_turns(self) -> int | None:
        """Get remaining turns until deadline."""
        if self.deadline_turn is None:
            return None
        # This would need current turn from context
        return None

    def activate(self) -> None:
        """Activate this goal."""
        if self.status == GoalStatus.PENDING:
            self.status = GoalStatus.ACTIVE

    def complete(self) -> None:
        """Mark this goal as completed."""
        self.status = GoalStatus.COMPLETED
        self.progress = 1.0

    def fail(self) -> None:
        """Mark this goal as failed."""
        self.status = GoalStatus.FAILED

    def abandon(self) -> None:
        """Abandon this goal."""
        self.status = GoalStatus.ABANDONED

    def update_progress(self, progress: float) -> None:
        """Update goal progress.

        Args:
            progress: New progress value (0.0 to 1.0).
        """
        self.progress = max(0.0, min(1.0, progress))
        if self.progress >= 1.0:
            self.complete()

    def to_embedding(self) -> np.ndarray:
        """Convert goal to embedding vector for neural network.

        Returns:
            Numpy array of shape (EMBED_DIM,).
        """
        # Compute goal type one-hot based on level
        if self.level == HierarchyLevel.STRATEGIC:
            type_dim = self.NUM_STRATEGIC_GOALS
            type_idx = list(StrategicGoalType).index(self.goal_type)
        elif self.level == HierarchyLevel.TACTICAL:
            type_dim = self.NUM_TACTICAL_GOALS
            type_idx = list(TacticalGoalType).index(self.goal_type)
        else:
            type_dim = self.NUM_OPERATIONAL_GOALS
            type_idx = list(OperationalGoalType).index(self.goal_type)

        # Build embedding components
        embed = np.zeros(self.EMBED_DIM, dtype=np.float32)

        # Level embedding (3 dims)
        level_idx = list(HierarchyLevel).index(self.level)
        embed[level_idx] = 1.0

        # Status embedding (6 dims)
        status_idx = list(GoalStatus).index(self.status)
        embed[3 + status_idx] = 1.0

        # Goal type hash (scaled to fit in embedding)
        embed[9] = type_idx / max(type_dim, 1)

        # Progress (1 dim)
        embed[10] = self.progress

        # Priority (1 dim)
        embed[11] = min(self.priority, 10.0) / 10.0

        # Strategic value (1 dim)
        embed[12] = (self.strategic_value + 1.0) / 2.0  # Normalize -1 to 1

        # Urgency (1 dim)
        embed[13] = min(self.urgency, 1.0)

        # Estimated duration (1 dim, log-scaled)
        embed[14] = min(np.log1p(self.estimated_duration) / 5.0, 1.0)

        # Target position if available (2 dims)
        if self.target.position:
            embed[15] = self.target.position[0] / 64.0  # Normalize assuming max 64
            embed[16] = self.target.position[1] / 64.0

        # Has parent/children flags (2 dims)
        embed[17] = 1.0 if self.parent_goal_id else 0.0
        embed[18] = min(len(self.child_goal_ids) / 5.0, 1.0)

        return embed

    def to_dict(self) -> dict[str, Any]:
        """Convert goal to dictionary for serialization."""
        return {
            "id": self.id,
            "level": self.level.value,
            "goal_type": self.goal_type.name,
            "description": self.description,
            "target": {
                "target_type": self.target.target_type,
                "target_id": self.target.target_id,
                "position": self.target.position,
            },
            "priority": self.priority,
            "status": self.status.name,
            "progress": self.progress,
            "created_turn": self.created_turn,
            "deadline_turn": self.deadline_turn,
            "parent_goal_id": self.parent_goal_id,
            "child_goal_ids": self.child_goal_ids,
            "strategic_value": self.strategic_value,
            "urgency": self.urgency,
        }
