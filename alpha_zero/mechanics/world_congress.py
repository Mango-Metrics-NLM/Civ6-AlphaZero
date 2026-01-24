"""World Congress System for Civ6-AlphaZero.

This module implements the World Congress mechanics including:
- Diplomatic favor accumulation
- Resolution proposals and voting
- Emergency sessions
- Special sessions

All mechanics are designed to be:
- Fully configurable via dataclasses
- Testable in isolation
- Compatible with the existing encoder system
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class CongressPhase(Enum):
    """Phases of a World Congress session."""

    INACTIVE = auto()  # Between sessions
    PROPOSAL = auto()  # Players propose resolutions
    DISCUSSION = auto()  # Discussion period
    VOTING = auto()  # Voting period
    RESOLUTION = auto()  # Applying results


class ResolutionCategory(Enum):
    """Categories of resolutions."""

    WORLD_IDEOLOGY = auto()  # Global effects
    TRADE_POLICY = auto()  # Trade effects
    MILITARY_ADVISORY = auto()  # Military effects
    CULTURAL_PATRONAGE = auto()  # Culture effects
    SCIENCE_FUNDING = auto()  # Science effects
    ENVIRONMENTAL = auto()  # Climate effects
    EMERGENCY = auto()  # Emergency resolutions


class VoteType(Enum):
    """Types of votes."""

    FOR = auto()
    AGAINST = auto()
    ABSTAIN = auto()


class EmergencyType(Enum):
    """Types of emergencies."""

    MILITARY_EMERGENCY = auto()  # Against warmonger
    HUMANITARIAN_EMERGENCY = auto()  # City lost to free
    SCORED_COMPETITION = auto()  # Competition (e.g., aid race)
    WORLD_GAMES = auto()  # Olympic-style competition
    NUCLEAR_EMERGENCY = auto()  # Nuclear weapon used
    CLIMATE_ACCORDS = auto()  # Climate change


@dataclass
class CongressConfig:
    """Configuration for World Congress mechanics.

    Attributes:
        favor_per_city: Diplomatic favor per city per turn
        favor_per_ally: Favor per alliance
        favor_per_suzerain: Favor per city-state suzerainty
        favor_per_wonder: Favor per diplomatic wonder
        session_interval_turns: Turns between regular sessions
        proposal_turns: Turns for proposal phase
        discussion_turns: Turns for discussion
        voting_turns: Turns for voting
        emergency_vote_threshold: Votes needed to pass emergency
        base_votes_per_player: Base votes each player gets
        favor_per_vote: Favor cost per additional vote
        max_proposals_per_player: Max resolutions player can propose
    """

    favor_per_city: float = 0.5
    favor_per_ally: float = 2.0
    favor_per_suzerain: float = 1.0
    favor_per_wonder: float = 1.0
    session_interval_turns: int = 30
    proposal_turns: int = 5
    discussion_turns: int = 3
    voting_turns: int = 5
    emergency_vote_threshold: float = 0.6
    base_votes_per_player: int = 1
    favor_per_vote: int = 20
    max_proposals_per_player: int = 1


@dataclass
class ResolutionEffect:
    """Effect of a resolution.

    Attributes:
        effect_type: Type of effect
        target: Target of effect (player_id, resource, etc.)
        value: Effect magnitude
        duration: Turns the effect lasts (-1 = permanent)
    """

    effect_type: str
    target: str | int | None = None
    value: float = 0.0
    duration: int = -1


@dataclass
class Resolution:
    """Definition of a resolution.

    Attributes:
        resolution_id: Unique identifier
        name: Display name
        category: Resolution category
        description: Human-readable description
        options: List of voting options (usually A or B)
        effects_a: Effects if option A passes
        effects_b: Effects if option B passes
        proposer_id: Player who proposed (None for standard)
        target_id: Targeted player/entity if applicable
    """

    resolution_id: str
    name: str
    category: ResolutionCategory
    description: str
    options: list[str] = field(default_factory=lambda: ["Option A", "Option B"])
    effects_a: list[ResolutionEffect] = field(default_factory=list)
    effects_b: list[ResolutionEffect] = field(default_factory=list)
    proposer_id: int | None = None
    target_id: int | None = None


# Default resolutions
DEFAULT_RESOLUTIONS: list[Resolution] = [
    Resolution(
        resolution_id="trade_policy",
        name="Global Trade Policy",
        category=ResolutionCategory.TRADE_POLICY,
        description="Determine global trade bonuses",
        options=["Boost Gold from Trade", "Boost Production from Trade"],
        effects_a=[ResolutionEffect("trade_gold_bonus", value=3.0)],
        effects_b=[ResolutionEffect("trade_production_bonus", value=2.0)],
    ),
    Resolution(
        resolution_id="science_funding",
        name="International Science Funding",
        category=ResolutionCategory.SCIENCE_FUNDING,
        description="Direct research priorities",
        options=["Campus Bonus", "Library Bonus"],
        effects_a=[ResolutionEffect("campus_adjacency_bonus", value=1.0)],
        effects_b=[ResolutionEffect("library_science_bonus", value=2.0)],
    ),
    Resolution(
        resolution_id="cultural_heritage",
        name="Cultural Heritage Sites",
        category=ResolutionCategory.CULTURAL_PATRONAGE,
        description="Protect and promote culture",
        options=["Tourism Bonus", "Culture Bonus"],
        effects_a=[ResolutionEffect("tourism_multiplier", value=0.25)],
        effects_b=[ResolutionEffect("culture_per_wonder", value=3.0)],
    ),
    Resolution(
        resolution_id="military_advisory",
        name="Military Advisory Council",
        category=ResolutionCategory.MILITARY_ADVISORY,
        description="Regulate military production",
        options=["Unit Production Bonus", "Unit Maintenance Reduction"],
        effects_a=[ResolutionEffect("unit_production_bonus", value=0.15)],
        effects_b=[ResolutionEffect("unit_maintenance_reduction", value=1.0)],
    ),
    Resolution(
        resolution_id="climate_accords",
        name="Global Climate Accords",
        category=ResolutionCategory.ENVIRONMENTAL,
        description="Address climate change",
        options=["CO2 Reduction Incentives", "Disaster Relief Fund"],
        effects_a=[ResolutionEffect("co2_per_industrial_reduction", value=0.5)],
        effects_b=[ResolutionEffect("disaster_recovery_bonus", value=0.5)],
    ),
]


@dataclass
class Vote:
    """A player's vote on a resolution.

    Attributes:
        player_id: Voting player
        resolution_id: Resolution being voted on
        choice: Vote choice (FOR option A, AGAINST for B, ABSTAIN)
        votes_cast: Number of votes cast
        favor_spent: Diplomatic favor spent
    """

    player_id: int
    resolution_id: str
    choice: VoteType
    votes_cast: int
    favor_spent: int


@dataclass
class ActiveResolution:
    """An active resolution with voting state.

    Attributes:
        resolution: The resolution definition
        votes_for: Total votes for option A
        votes_against: Total votes for option B
        player_votes: Dictionary of player votes
        turn_proposed: Turn when proposed
        winning_option: Winning option after resolution
    """

    resolution: Resolution
    votes_for: int = 0
    votes_against: int = 0
    player_votes: dict[int, Vote] = field(default_factory=dict)
    turn_proposed: int = 0
    winning_option: str | None = None


@dataclass
class Emergency:
    """An emergency session.

    Attributes:
        emergency_id: Unique identifier
        emergency_type: Type of emergency
        target_player: Player who triggered/is target
        target_city: Target city if applicable
        reward_pool: Reward for participants
        turns_remaining: Turns until emergency ends
        participants: Dictionary of player_id to contribution
        is_active: Whether emergency is ongoing
    """

    emergency_id: str
    emergency_type: EmergencyType
    target_player: int
    target_city: str | None = None
    reward_pool: dict[str, float] = field(default_factory=dict)
    turns_remaining: int = 30
    participants: dict[int, float] = field(default_factory=dict)
    is_active: bool = True


class WorldCongressManager:
    """Manages World Congress mechanics.

    This class handles:
    - Diplomatic favor tracking
    - Regular congress sessions
    - Resolution voting
    - Emergency sessions

    Example:
        >>> config = CongressConfig(session_interval_turns=30)
        >>> manager = WorldCongressManager(config, num_players=4)
        >>> manager.start_session(turn=60)
        >>> manager.propose_resolution(player_id=0, resolution_id="trade_policy")
        >>> manager.cast_vote(player_id=0, resolution_id="trade_policy", choice=VoteType.FOR, votes=2)
    """

    def __init__(
        self,
        config: CongressConfig | None = None,
        num_players: int = 2,
        resolutions: list[Resolution] | None = None,
    ) -> None:
        """Initialize World Congress manager.

        Args:
            config: Congress configuration
            num_players: Number of players in game
            resolutions: Available resolutions
        """
        self.config = config or CongressConfig()
        self._num_players = num_players
        self._resolutions = {r.resolution_id: r for r in (resolutions or DEFAULT_RESOLUTIONS)}

        self._phase = CongressPhase.INACTIVE
        self._favor: dict[int, float] = dict.fromkeys(range(num_players), 0.0)
        self._active_resolutions: dict[str, ActiveResolution] = {}
        self._passed_resolutions: list[ActiveResolution] = []
        self._active_effects: list[tuple[ResolutionEffect, int]] = []  # (effect, turns_remaining)
        self._emergencies: dict[str, Emergency] = {}
        self._session_turn = 0
        self._phase_turns_remaining = 0
        self._turn = 0
        self._emergency_counter = 0

        logger.debug(f"WorldCongressManager initialized with {len(self._resolutions)} resolutions")

    def add_favor(
        self,
        player_id: int,
        amount: float,
        source: str = "unknown",
    ) -> float:
        """Add diplomatic favor to a player.

        Args:
            player_id: Player receiving favor
            amount: Amount to add
            source: Source description

        Returns:
            New total favor
        """
        if player_id not in self._favor:
            self._favor[player_id] = 0.0

        self._favor[player_id] += amount
        logger.debug(f"Player {player_id} gained {amount:.1f} favor from {source}")
        return self._favor[player_id]

    def spend_favor(
        self,
        player_id: int,
        amount: float,
    ) -> bool:
        """Spend diplomatic favor.

        Args:
            player_id: Player spending
            amount: Amount to spend

        Returns:
            True if successful
        """
        if self._favor.get(player_id, 0) < amount:
            return False

        self._favor[player_id] -= amount
        return True

    def get_favor(self, player_id: int) -> float:
        """Get player's diplomatic favor."""
        return self._favor.get(player_id, 0.0)

    def start_session(self, turn: int) -> bool:
        """Start a new congress session.

        Args:
            turn: Current turn

        Returns:
            True if session started
        """
        if self._phase != CongressPhase.INACTIVE:
            logger.warning("Cannot start session: already in session")
            return False

        self._phase = CongressPhase.PROPOSAL
        self._session_turn = turn
        self._phase_turns_remaining = self.config.proposal_turns
        self._active_resolutions.clear()

        logger.info(f"World Congress session started on turn {turn}")
        return True

    def propose_resolution(
        self,
        player_id: int,
        resolution_id: str,
        target_id: int | None = None,
    ) -> bool:
        """Propose a resolution.

        Args:
            player_id: Proposing player
            resolution_id: Resolution to propose
            target_id: Target player if applicable

        Returns:
            True if proposal accepted
        """
        if self._phase != CongressPhase.PROPOSAL:
            logger.warning("Cannot propose: not in proposal phase")
            return False

        if resolution_id not in self._resolutions:
            logger.warning(f"Unknown resolution: {resolution_id}")
            return False

        if resolution_id in self._active_resolutions:
            logger.warning(f"Resolution already proposed: {resolution_id}")
            return False

        # Check proposal limit
        proposals_by_player = sum(
            1 for ar in self._active_resolutions.values()
            if ar.resolution.proposer_id == player_id
        )
        if proposals_by_player >= self.config.max_proposals_per_player:
            logger.warning(f"Player {player_id} at proposal limit")
            return False

        resolution = Resolution(
            resolution_id=resolution_id,
            name=self._resolutions[resolution_id].name,
            category=self._resolutions[resolution_id].category,
            description=self._resolutions[resolution_id].description,
            options=self._resolutions[resolution_id].options.copy(),
            effects_a=self._resolutions[resolution_id].effects_a.copy(),
            effects_b=self._resolutions[resolution_id].effects_b.copy(),
            proposer_id=player_id,
            target_id=target_id,
        )

        self._active_resolutions[resolution_id] = ActiveResolution(
            resolution=resolution,
            turn_proposed=self._turn,
        )

        logger.info(f"Player {player_id} proposed resolution: {resolution.name}")
        return True

    def cast_vote(
        self,
        player_id: int,
        resolution_id: str,
        choice: VoteType,
        extra_votes: int = 0,
    ) -> bool:
        """Cast votes on a resolution.

        Args:
            player_id: Voting player
            resolution_id: Resolution to vote on
            choice: Vote choice
            extra_votes: Additional votes to purchase

        Returns:
            True if vote cast
        """
        if self._phase != CongressPhase.VOTING:
            logger.warning("Cannot vote: not in voting phase")
            return False

        if resolution_id not in self._active_resolutions:
            logger.warning(f"Resolution not active: {resolution_id}")
            return False

        ar = self._active_resolutions[resolution_id]
        if player_id in ar.player_votes:
            logger.warning(f"Player {player_id} already voted on {resolution_id}")
            return False

        # Calculate favor cost
        favor_cost = extra_votes * self.config.favor_per_vote
        if favor_cost > 0 and not self.spend_favor(player_id, favor_cost):
            logger.warning(f"Insufficient favor for {extra_votes} extra votes")
            return False

        total_votes = self.config.base_votes_per_player + extra_votes

        vote = Vote(
            player_id=player_id,
            resolution_id=resolution_id,
            choice=choice,
            votes_cast=total_votes,
            favor_spent=int(favor_cost),
        )

        ar.player_votes[player_id] = vote

        if choice == VoteType.FOR:
            ar.votes_for += total_votes
        elif choice == VoteType.AGAINST:
            ar.votes_against += total_votes

        logger.info(
            f"Player {player_id} voted {choice.name} ({total_votes} votes) on {resolution_id}",
        )
        return True

    def process_turn(self, turn: int) -> list[dict[str, Any]]:
        """Process turn for congress mechanics.

        Args:
            turn: Current turn

        Returns:
            List of events that occurred
        """
        self._turn = turn
        events: list[dict[str, Any]] = []

        # Process phase transitions
        if self._phase != CongressPhase.INACTIVE:
            self._phase_turns_remaining -= 1

            if self._phase_turns_remaining <= 0:
                event = self._advance_phase()
                if event:
                    events.append(event)

        # Check for regular session
        elif (turn - self._session_turn) >= self.config.session_interval_turns:
            self.start_session(turn)
            events.append({"type": "session_started", "turn": turn})

        # Process active effects
        expired_effects: list[int] = []
        for i, (effect, remaining) in enumerate(self._active_effects):
            if remaining > 0:
                self._active_effects[i] = (effect, remaining - 1)
                if remaining - 1 <= 0:
                    expired_effects.append(i)

        for i in reversed(expired_effects):
            self._active_effects.pop(i)

        # Process emergencies
        completed_emergencies: list[str] = []
        for emergency_id, emergency in self._emergencies.items():
            if emergency.is_active:
                emergency.turns_remaining -= 1
                if emergency.turns_remaining <= 0:
                    completed_emergencies.append(emergency_id)
                    events.append({
                        "type": "emergency_ended",
                        "emergency_id": emergency_id,
                        "emergency_type": emergency.emergency_type.name,
                    })

        for emergency_id in completed_emergencies:
            self._emergencies[emergency_id].is_active = False

        return events

    def _advance_phase(self) -> dict[str, Any] | None:
        """Advance to next congress phase.

        Returns:
            Event dictionary if phase changed
        """
        old_phase = self._phase

        if self._phase == CongressPhase.PROPOSAL:
            self._phase = CongressPhase.DISCUSSION
            self._phase_turns_remaining = self.config.discussion_turns

        elif self._phase == CongressPhase.DISCUSSION:
            self._phase = CongressPhase.VOTING
            self._phase_turns_remaining = self.config.voting_turns

        elif self._phase == CongressPhase.VOTING:
            self._phase = CongressPhase.RESOLUTION
            self._resolve_votes()
            self._phase_turns_remaining = 1

        elif self._phase == CongressPhase.RESOLUTION:
            self._phase = CongressPhase.INACTIVE
            self._session_turn = self._turn

        logger.info(f"Congress phase: {old_phase.name} -> {self._phase.name}")
        return {
            "type": "phase_changed",
            "old_phase": old_phase.name,
            "new_phase": self._phase.name,
        }

    def _resolve_votes(self) -> None:
        """Resolve all active resolution votes."""
        for resolution_id, ar in self._active_resolutions.items():
            if ar.votes_for > ar.votes_against:
                ar.winning_option = ar.resolution.options[0]
                effects = ar.resolution.effects_a
            elif ar.votes_against > ar.votes_for:
                ar.winning_option = ar.resolution.options[1]
                effects = ar.resolution.effects_b
            else:
                ar.winning_option = None  # Tie - no effect
                effects = []

            # Apply effects
            for effect in effects:
                duration = effect.duration if effect.duration > 0 else 30
                self._active_effects.append((effect, duration))

            self._passed_resolutions.append(ar)

            logger.info(
                f"Resolution {resolution_id} resolved: {ar.winning_option} "
                f"(For: {ar.votes_for}, Against: {ar.votes_against})",
            )

    def call_emergency(
        self,
        emergency_type: EmergencyType,
        target_player: int,
        target_city: str | None = None,
        reward_pool: dict[str, float] | None = None,
    ) -> Emergency:
        """Call an emergency session.

        Args:
            emergency_type: Type of emergency
            target_player: Target player
            target_city: Target city if applicable
            reward_pool: Rewards for participants

        Returns:
            Created emergency
        """
        self._emergency_counter += 1
        emergency_id = f"emergency_{self._emergency_counter}"

        emergency = Emergency(
            emergency_id=emergency_id,
            emergency_type=emergency_type,
            target_player=target_player,
            target_city=target_city,
            reward_pool=reward_pool or {},
        )

        self._emergencies[emergency_id] = emergency
        logger.info(f"Emergency called: {emergency_type.name} targeting player {target_player}")
        return emergency

    def contribute_to_emergency(
        self,
        player_id: int,
        emergency_id: str,
        contribution: float,
    ) -> bool:
        """Contribute to an emergency.

        Args:
            player_id: Contributing player
            emergency_id: Emergency to contribute to
            contribution: Contribution amount

        Returns:
            True if successful
        """
        if emergency_id not in self._emergencies:
            return False

        emergency = self._emergencies[emergency_id]
        if not emergency.is_active:
            return False

        current = emergency.participants.get(player_id, 0.0)
        emergency.participants[player_id] = current + contribution

        logger.info(f"Player {player_id} contributed {contribution} to {emergency_id}")
        return True

    def get_active_effects(self) -> dict[str, float]:
        """Get all active resolution effects.

        Returns:
            Dictionary of effect type to total value
        """
        effects: dict[str, float] = {}
        for effect, _ in self._active_effects:
            effects[effect.effect_type] = effects.get(effect.effect_type, 0.0) + effect.value
        return effects

    def get_congress_status(self) -> dict[str, Any]:
        """Get current congress status.

        Returns:
            Dictionary of status information
        """
        return {
            "phase": self._phase.name,
            "phase_turns_remaining": self._phase_turns_remaining,
            "active_resolutions": len(self._active_resolutions),
            "passed_resolutions": len(self._passed_resolutions),
            "active_effects": len(self._active_effects),
            "active_emergencies": len([e for e in self._emergencies.values() if e.is_active]),
            "turns_until_next_session": max(
                0, self.config.session_interval_turns - (self._turn - self._session_turn),
            ) if self._phase == CongressPhase.INACTIVE else 0,
        }

    def encode_for_state(self, player_id: int) -> dict[str, Any]:
        """Encode congress state for neural network input.

        Args:
            player_id: Player perspective

        Returns:
            Dictionary of encoded features
        """
        status = self.get_congress_status()
        _ = self.get_active_effects()  # Called for potential side effects

        # Phase encoding
        phase_encoding = [0.0] * len(CongressPhase)
        phase_encoding[self._phase.value - 1] = 1.0

        return {
            "phase": self._phase.value,
            "phase_encoding": phase_encoding,
            "is_in_session": float(self._phase != CongressPhase.INACTIVE),
            "player_favor": self.get_favor(player_id),
            "active_resolutions": status["active_resolutions"],
            "active_effects_count": len(self._active_effects),
            "active_emergencies": status["active_emergencies"],
            "turns_until_session": status["turns_until_next_session"],
        }

    def reset(self) -> None:
        """Reset all congress state."""
        self._phase = CongressPhase.INACTIVE
        self._favor = dict.fromkeys(range(self._num_players), 0.0)
        self._active_resolutions.clear()
        self._passed_resolutions.clear()
        self._active_effects.clear()
        self._emergencies.clear()
        self._session_turn = 0
        self._phase_turns_remaining = 0
        self._turn = 0
        self._emergency_counter = 0
        logger.debug("WorldCongressManager reset")
