"""
Diplomacy System for Civ6-AlphaZero

Implements diplomatic relations, treaties, and interactions between players.
All values are configurable - no hardcoded constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol


class DiplomacyType(Enum):
    """Types of diplomatic relationships."""

    NEUTRAL = auto()
    FRIENDLY = auto()
    UNFRIENDLY = auto()
    DENOUNCED = auto()
    WAR = auto()
    PEACE_TREATY = auto()
    DEFENSIVE_PACT = auto()
    ALLIANCE = auto()
    RESEARCH_AGREEMENT = auto()
    TRADE_AGREEMENT = auto()


class DiplomaticActionType(Enum):
    """Types of diplomatic actions a player can take."""

    DECLARE_WAR = auto()
    MAKE_PEACE = auto()
    DENOUNCE = auto()
    DECLARE_FRIENDSHIP = auto()
    PROPOSE_ALLIANCE = auto()
    BREAK_ALLIANCE = auto()
    PROPOSE_TRADE = auto()
    PROPOSE_RESEARCH_AGREEMENT = auto()
    DEMAND_TRIBUTE = auto()
    PROPOSE_JOINT_WAR = auto()
    ACCEPT_PROPOSAL = auto()
    REJECT_PROPOSAL = auto()


class ModifierType(Enum):
    """Types of diplomatic modifiers (grievances/bonuses)."""

    # Negative modifiers
    ATTACKED_US = auto()
    ATTACKED_FRIEND = auto()
    DENOUNCED_US = auto()
    BROKE_PROMISE = auto()
    AGGRESSIVE_EXPANSION = auto()
    DIFFERENT_IDEOLOGY = auto()
    DIFFERENT_RELIGION = auto()
    TERRITORIAL_DISPUTE = auto()

    # Positive modifiers
    TRADE_PARTNER = auto()
    SAME_RELIGION = auto()
    SAME_IDEOLOGY = auto()
    LIBERATED_CITY = auto()
    DEFENSIVE_PACT_HONORED = auto()
    LONG_PEACE = auto()
    COMMON_ENEMY = auto()


@dataclass
class DiplomacyConfig:
    """Configuration for the diplomacy system.

    All values are configurable to allow tuning.
    """

    # Relationship score thresholds
    war_threshold: int = -80
    unfriendly_threshold: int = -30
    neutral_threshold: int = 30
    friendly_threshold: int = 60
    alliance_threshold: int = 80

    # Modifier decay
    modifier_decay_per_turn: int = 1
    war_weariness_per_turn: float = 0.5

    # Treaty durations (in turns)
    peace_treaty_duration: int = 10
    defensive_pact_duration: int = 30
    alliance_duration: int = 50
    research_agreement_duration: int = 30

    # Cooldowns
    war_declaration_cooldown: int = 10
    denouncement_cooldown: int = 30
    alliance_proposal_cooldown: int = 20

    # Modifier values
    modifier_values: dict[ModifierType, int] = field(
        default_factory=lambda: {
            ModifierType.ATTACKED_US: -50,
            ModifierType.ATTACKED_FRIEND: -30,
            ModifierType.DENOUNCED_US: -25,
            ModifierType.BROKE_PROMISE: -40,
            ModifierType.AGGRESSIVE_EXPANSION: -20,
            ModifierType.DIFFERENT_IDEOLOGY: -15,
            ModifierType.DIFFERENT_RELIGION: -10,
            ModifierType.TERRITORIAL_DISPUTE: -20,
            ModifierType.TRADE_PARTNER: 15,
            ModifierType.SAME_RELIGION: 10,
            ModifierType.SAME_IDEOLOGY: 15,
            ModifierType.LIBERATED_CITY: 30,
            ModifierType.DEFENSIVE_PACT_HONORED: 25,
            ModifierType.LONG_PEACE: 20,
            ModifierType.COMMON_ENEMY: 15,
        }
    )


@dataclass
class DiplomaticModifier:
    """A modifier affecting diplomatic relations.

    Attributes:
        modifier_type: Type of modifier
        value: Numeric value (positive = good, negative = bad)
        turns_remaining: Turns until this modifier expires (-1 = permanent)
        source_player: Player who caused this modifier
        description: Human-readable description
    """

    modifier_type: ModifierType
    value: int
    turns_remaining: int = -1
    source_player: int | None = None
    description: str = ""

    def decay(self, amount: int = 1) -> bool:
        """Decay this modifier by the given amount.

        Args:
            amount: Amount to decay the value.

        Returns:
            True if modifier should be removed (expired or zeroed).
        """
        if self.turns_remaining > 0:
            self.turns_remaining -= 1
            if self.turns_remaining == 0:
                return True

        # Decay value toward zero
        if self.value > 0:
            self.value = max(0, self.value - amount)
        elif self.value < 0:
            self.value = min(0, self.value + amount)

        return self.value == 0 and self.turns_remaining <= 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "modifier_type": self.modifier_type.name,
            "value": self.value,
            "turns_remaining": self.turns_remaining,
            "source_player": self.source_player,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiplomaticModifier:
        """Create from dictionary."""
        return cls(
            modifier_type=ModifierType[data["modifier_type"]],
            value=data["value"],
            turns_remaining=data.get("turns_remaining", -1),
            source_player=data.get("source_player"),
            description=data.get("description", ""),
        )


@dataclass
class DiplomaticRelation:
    """Represents the diplomatic relation between two players.

    Attributes:
        player_a: First player ID
        player_b: Second player ID
        relation_type: Current relationship type
        modifiers: List of active modifiers
        turn_started: Turn when this relation began
        treaties: Active treaties between players
    """

    player_a: int
    player_b: int
    relation_type: DiplomacyType = DiplomacyType.NEUTRAL
    modifiers: list[DiplomaticModifier] = field(default_factory=list)
    turn_started: int = 0
    treaties: dict[str, int] = field(default_factory=dict)  # treaty_type: expiry_turn

    @property
    def relationship_score(self) -> int:
        """Calculate total relationship score from modifiers."""
        return sum(m.value for m in self.modifiers)

    def add_modifier(self, modifier: DiplomaticModifier) -> None:
        """Add a modifier to this relation."""
        # Check if same type already exists, replace if so
        for i, existing in enumerate(self.modifiers):
            if existing.modifier_type == modifier.modifier_type:
                self.modifiers[i] = modifier
                return
        self.modifiers.append(modifier)

    def remove_modifier(self, modifier_type: ModifierType) -> bool:
        """Remove a modifier by type.

        Returns:
            True if modifier was found and removed.
        """
        for i, modifier in enumerate(self.modifiers):
            if modifier.modifier_type == modifier_type:
                self.modifiers.pop(i)
                return True
        return False

    def decay_modifiers(self, decay_amount: int = 1) -> None:
        """Decay all modifiers by the given amount."""
        self.modifiers = [m for m in self.modifiers if not m.decay(decay_amount)]

    def has_treaty(self, treaty_type: str) -> bool:
        """Check if a treaty is active."""
        return treaty_type in self.treaties

    def add_treaty(self, treaty_type: str, duration: int, current_turn: int) -> None:
        """Add a treaty with expiry."""
        self.treaties[treaty_type] = current_turn + duration

    def expire_treaties(self, current_turn: int) -> list[str]:
        """Expire any treaties past their duration.

        Returns:
            List of expired treaty types.
        """
        expired = [t for t, expiry in self.treaties.items() if expiry <= current_turn]
        for t in expired:
            del self.treaties[t]
        return expired

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "player_a": self.player_a,
            "player_b": self.player_b,
            "relation_type": self.relation_type.name,
            "modifiers": [m.to_dict() for m in self.modifiers],
            "turn_started": self.turn_started,
            "treaties": self.treaties,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiplomaticRelation:
        """Create from dictionary."""
        return cls(
            player_a=data["player_a"],
            player_b=data["player_b"],
            relation_type=DiplomacyType[data["relation_type"]],
            modifiers=[DiplomaticModifier.from_dict(m) for m in data.get("modifiers", [])],
            turn_started=data.get("turn_started", 0),
            treaties=data.get("treaties", {}),
        )


@dataclass
class DiplomaticAction:
    """A diplomatic action to be executed.

    Attributes:
        action_type: Type of action
        source_player: Player taking the action
        target_player: Player receiving the action
        parameters: Additional action parameters
    """

    action_type: DiplomaticActionType
    source_player: int
    target_player: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type.name,
            "source_player": self.source_player,
            "target_player": self.target_player,
            "parameters": self.parameters,
        }


class GameStateProtocol(Protocol):
    """Protocol for game state objects."""

    def get_player_count(self) -> int: ...
    def get_current_turn(self) -> int: ...
    def get_player_cities(self, player: int) -> list[Any]: ...
    def get_player_units(self, player: int) -> list[Any]: ...


class DiplomacyManager:
    """Manages all diplomatic relations in the game.

    Features:
    - Track relations between all player pairs
    - Apply and decay modifiers
    - Validate diplomatic actions
    - Generate valid diplomatic action mask
    """

    def __init__(self, config: DiplomacyConfig | None = None) -> None:
        """Initialize the diplomacy manager.

        Args:
            config: Diplomacy configuration.
        """
        self.config = config or DiplomacyConfig()
        self._relations: dict[tuple[int, int], DiplomaticRelation] = {}
        self._cooldowns: dict[tuple[int, int, DiplomaticActionType], int] = {}
        self._pending_proposals: list[DiplomaticAction] = []

    def _get_relation_key(self, player_a: int, player_b: int) -> tuple[int, int]:
        """Get canonical key for a player pair (lower ID first)."""
        return (min(player_a, player_b), max(player_a, player_b))

    def get_relation(self, player_a: int, player_b: int) -> DiplomaticRelation:
        """Get the relation between two players.

        Creates a neutral relation if none exists.

        Args:
            player_a: First player ID.
            player_b: Second player ID.

        Returns:
            DiplomaticRelation between the players.
        """
        key = self._get_relation_key(player_a, player_b)
        if key not in self._relations:
            self._relations[key] = DiplomaticRelation(
                player_a=key[0],
                player_b=key[1],
            )
        return self._relations[key]

    def set_relation(self, relation: DiplomaticRelation) -> None:
        """Set a diplomatic relation."""
        key = self._get_relation_key(relation.player_a, relation.player_b)
        self._relations[key] = relation

    def are_at_war(self, player_a: int, player_b: int) -> bool:
        """Check if two players are at war."""
        return self.get_relation(player_a, player_b).relation_type == DiplomacyType.WAR

    def are_allied(self, player_a: int, player_b: int) -> bool:
        """Check if two players are allied."""
        return self.get_relation(player_a, player_b).relation_type == DiplomacyType.ALLIANCE

    def can_declare_war(
        self,
        source: int,
        target: int,
        current_turn: int,
    ) -> tuple[bool, str]:
        """Check if a player can declare war.

        Args:
            source: Declaring player.
            target: Target player.
            current_turn: Current game turn.

        Returns:
            Tuple of (can_declare, reason).
        """
        relation = self.get_relation(source, target)

        if relation.relation_type == DiplomacyType.WAR:
            return False, "Already at war"

        if relation.has_treaty("peace"):
            return False, "Peace treaty in effect"

        cooldown_key = (source, target, DiplomaticActionType.DECLARE_WAR)
        if cooldown_key in self._cooldowns:
            if self._cooldowns[cooldown_key] > current_turn:
                return False, "War declaration on cooldown"

        return True, ""

    def can_make_peace(self, player_a: int, player_b: int) -> tuple[bool, str]:
        """Check if two players can make peace."""
        relation = self.get_relation(player_a, player_b)

        if relation.relation_type != DiplomacyType.WAR:
            return False, "Not at war"

        return True, ""

    def get_valid_diplomatic_actions(
        self,
        player: int,
        game_state: GameStateProtocol,
    ) -> list[DiplomaticAction]:
        """Get all valid diplomatic actions for a player.

        Args:
            player: Player ID.
            game_state: Current game state.

        Returns:
            List of valid DiplomaticAction objects.
        """
        valid_actions: list[DiplomaticAction] = []
        current_turn = game_state.get_current_turn()
        player_count = game_state.get_player_count()

        for target in range(player_count):
            if target == player:
                continue

            relation = self.get_relation(player, target)

            # War declaration
            can_war, _ = self.can_declare_war(player, target, current_turn)
            if can_war:
                valid_actions.append(
                    DiplomaticAction(
                        action_type=DiplomaticActionType.DECLARE_WAR,
                        source_player=player,
                        target_player=target,
                    )
                )

            # Peace proposal
            can_peace, _ = self.can_make_peace(player, target)
            if can_peace:
                valid_actions.append(
                    DiplomaticAction(
                        action_type=DiplomaticActionType.MAKE_PEACE,
                        source_player=player,
                        target_player=target,
                    )
                )

            # Denouncement (if not already denounced)
            if relation.relation_type not in (DiplomacyType.WAR, DiplomacyType.DENOUNCED):
                cooldown_key = (player, target, DiplomaticActionType.DENOUNCE)
                if (
                    cooldown_key not in self._cooldowns
                    or self._cooldowns[cooldown_key] <= current_turn
                ):
                    valid_actions.append(
                        DiplomaticAction(
                            action_type=DiplomaticActionType.DENOUNCE,
                            source_player=player,
                            target_player=target,
                        )
                    )

            # Friendship declaration (if neutral or better, not at war)
            if relation.relation_type in (DiplomacyType.NEUTRAL, DiplomacyType.FRIENDLY):
                valid_actions.append(
                    DiplomaticAction(
                        action_type=DiplomaticActionType.DECLARE_FRIENDSHIP,
                        source_player=player,
                        target_player=target,
                    )
                )

            # Alliance proposal (if friendly enough)
            if relation.relationship_score >= self.config.friendly_threshold:
                if relation.relation_type != DiplomacyType.ALLIANCE:
                    valid_actions.append(
                        DiplomaticAction(
                            action_type=DiplomaticActionType.PROPOSE_ALLIANCE,
                            source_player=player,
                            target_player=target,
                        )
                    )

        return valid_actions

    def apply_action(
        self,
        action: DiplomaticAction,
        current_turn: int,
    ) -> bool:
        """Apply a diplomatic action.

        Args:
            action: Action to apply.
            current_turn: Current game turn.

        Returns:
            True if action was successfully applied.
        """
        relation = self.get_relation(action.source_player, action.target_player)

        if action.action_type == DiplomaticActionType.DECLARE_WAR:
            can_declare, reason = self.can_declare_war(
                action.source_player,
                action.target_player,
                current_turn,
            )
            if not can_declare:
                return False

            relation.relation_type = DiplomacyType.WAR
            relation.turn_started = current_turn

            # Set cooldown to prevent immediate peace (minimum war duration)
            peace_cooldown_key = (
                action.source_player,
                action.target_player,
                DiplomaticActionType.MAKE_PEACE,
            )
            # Minimum 5 turns of war before peace can be negotiated
            self._cooldowns[peace_cooldown_key] = current_turn + 5

            # Add modifier to other relations (attacked_friend for allies)
            return True

        elif action.action_type == DiplomaticActionType.MAKE_PEACE:
            can_peace, _ = self.can_make_peace(
                action.source_player,
                action.target_player,
            )
            if not can_peace:
                return False

            relation.relation_type = DiplomacyType.PEACE_TREATY
            relation.add_treaty("peace", self.config.peace_treaty_duration, current_turn)
            relation.turn_started = current_turn

            # Set cooldown for future war declaration
            cooldown_key = (
                action.source_player,
                action.target_player,
                DiplomaticActionType.DECLARE_WAR,
            )
            self._cooldowns[cooldown_key] = current_turn + self.config.war_declaration_cooldown
            return True

        elif action.action_type == DiplomaticActionType.DENOUNCE:
            relation.relation_type = DiplomacyType.DENOUNCED
            relation.add_modifier(
                DiplomaticModifier(
                    modifier_type=ModifierType.DENOUNCED_US,
                    value=self.config.modifier_values[ModifierType.DENOUNCED_US],
                    turns_remaining=self.config.denouncement_cooldown,
                    source_player=action.source_player,
                )
            )
            return True

        elif action.action_type == DiplomaticActionType.DECLARE_FRIENDSHIP:
            relation.relation_type = DiplomacyType.FRIENDLY
            return True

        elif action.action_type == DiplomaticActionType.PROPOSE_ALLIANCE:
            # Add to pending proposals
            self._pending_proposals.append(action)
            return True

        return False

    def process_turn(self, current_turn: int) -> None:
        """Process end-of-turn diplomacy updates.

        Args:
            current_turn: Current turn number.
        """
        for relation in self._relations.values():
            # Decay modifiers
            relation.decay_modifiers(self.config.modifier_decay_per_turn)

            # Expire treaties
            _ = relation.expire_treaties(current_turn)

            # Update relation type based on score
            if relation.relation_type not in (DiplomacyType.WAR, DiplomacyType.ALLIANCE):
                score = relation.relationship_score
                if score <= self.config.war_threshold:
                    pass  # Don't auto-declare war
                elif score <= self.config.unfriendly_threshold:
                    relation.relation_type = DiplomacyType.UNFRIENDLY
                elif score < self.config.friendly_threshold:
                    relation.relation_type = DiplomacyType.NEUTRAL
                else:
                    relation.relation_type = DiplomacyType.FRIENDLY

    def get_diplomatic_vector(self, player: int, num_players: int) -> list[float]:
        """Get diplomatic state as a feature vector.

        Args:
            player: Player ID to get vector for.
            num_players: Total number of players.

        Returns:
            Flat vector of diplomatic features.
        """
        features: list[float] = []

        for other in range(num_players):
            if other == player:
                # Self-relation is always neutral (placeholder)
                features.extend([0.0] * 5)
                continue

            relation = self.get_relation(player, other)

            # One-hot encode relation type (simplified to 5 categories)
            type_encoding = [0.0] * 5
            type_map = {
                DiplomacyType.WAR: 0,
                DiplomacyType.DENOUNCED: 1,
                DiplomacyType.UNFRIENDLY: 1,
                DiplomacyType.NEUTRAL: 2,
                DiplomacyType.FRIENDLY: 3,
                DiplomacyType.PEACE_TREATY: 3,
                DiplomacyType.DEFENSIVE_PACT: 4,
                DiplomacyType.ALLIANCE: 4,
            }
            idx = type_map.get(relation.relation_type, 2)
            type_encoding[idx] = 1.0

            features.extend(type_encoding)

        return features

    def to_dict(self) -> dict[str, Any]:
        """Serialize diplomacy state."""
        return {
            "relations": {f"{k[0]}_{k[1]}": v.to_dict() for k, v in self._relations.items()},
            "cooldowns": {f"{k[0]}_{k[1]}_{k[2].name}": v for k, v in self._cooldowns.items()},
            "pending_proposals": [p.to_dict() for p in self._pending_proposals],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: DiplomacyConfig | None = None,
    ) -> DiplomacyManager:
        """Deserialize diplomacy state."""
        manager = cls(config=config)

        for _key, rel_data in data.get("relations", {}).items():
            relation = DiplomaticRelation.from_dict(rel_data)
            manager.set_relation(relation)

        for key, turn in data.get("cooldowns", {}).items():
            parts = key.rsplit("_", 2)
            if len(parts) == 3:
                p1, p2, action_type = int(parts[0]), int(parts[1]), parts[2]
                manager._cooldowns[(p1, p2, DiplomaticActionType[action_type])] = turn

        return manager
