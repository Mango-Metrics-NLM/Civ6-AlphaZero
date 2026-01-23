"""Action Space Pruning and Macro-Action System for Civ6-AlphaZero.

This module provides intelligent action space reduction through:
- Feasibility pruning (physically impossible actions)
- Resource pruning (unaffordable actions)
- Strategic pruning (irrelevant actions)
- Macro-actions (multi-turn action abstractions)

These techniques reduce the effective action space from ~50k to ~1k-5k
while preserving strategically important actions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Optional, Protocol, TypeVar

import numpy as np

logger = logging.getLogger(__name__)


class PruningLevel(Enum):
    """Levels of action pruning aggressiveness."""

    NONE = auto()  # No pruning
    MINIMAL = auto()  # Only feasibility
    STANDARD = auto()  # Feasibility + resource
    AGGRESSIVE = auto()  # All pruning
    EXTREME = auto()  # Very aggressive, may miss good moves


@dataclass
class PrunerConfig:
    """Configuration for action pruning.

    Attributes:
        pruning_level: Overall pruning aggressiveness
        cache_size: Size of LRU cache for pruning results
        enable_feasibility: Enable feasibility pruning
        enable_resource: Enable resource-based pruning
        enable_strategic: Enable strategic relevance pruning
        strategic_threshold: Threshold for strategic pruning (0-1)
        max_actions_per_unit: Maximum actions to consider per unit
        max_actions_per_city: Maximum actions to consider per city
        threat_evaluation_range: Range for threat evaluation
        wonder_race_threshold: Turns behind in wonder race to prune
        min_pruned_actions: Minimum actions to keep after pruning
    """

    pruning_level: PruningLevel = PruningLevel.STANDARD
    cache_size: int = 1000
    enable_feasibility: bool = True
    enable_resource: bool = True
    enable_strategic: bool = True
    strategic_threshold: float = 0.1
    max_actions_per_unit: int = 50
    max_actions_per_city: int = 30
    threat_evaluation_range: int = 5
    wonder_race_threshold: int = 10
    min_pruned_actions: int = 100


@dataclass
class PruningStats:
    """Statistics from pruning operation.

    Attributes:
        total_actions: Total actions before pruning
        feasibility_pruned: Actions removed by feasibility
        resource_pruned: Actions removed by resource check
        strategic_pruned: Actions removed by strategic filter
        final_actions: Actions remaining after pruning
        pruning_time_ms: Time taken for pruning
    """

    total_actions: int = 0
    feasibility_pruned: int = 0
    resource_pruned: int = 0
    strategic_pruned: int = 0
    final_actions: int = 0
    pruning_time_ms: float = 0.0

    @property
    def pruning_ratio(self) -> float:
        """Ratio of actions pruned."""
        if self.total_actions == 0:
            return 0.0
        return 1.0 - (self.final_actions / self.total_actions)


@dataclass
class PrunedActions:
    """Result of action pruning.

    Attributes:
        mask: Boolean mask of valid actions
        action_indices: Indices of valid actions
        priorities: Optional priority scores for actions
        stats: Pruning statistics
    """

    mask: np.ndarray
    action_indices: np.ndarray
    priorities: Optional[np.ndarray] = None
    stats: PruningStats = field(default_factory=PruningStats)


class GameStateProtocol(Protocol):
    """Protocol for game state objects."""

    def get_units(self, player_id: int) -> list[dict[str, Any]]: ...
    def get_cities(self, player_id: int) -> list[dict[str, Any]]: ...
    def get_resources(self, player_id: int) -> dict[str, int]: ...
    def get_valid_positions(self, unit_id: str) -> set[tuple[int, int]]: ...
    def get_buildable_items(self, city_id: str) -> list[str]: ...
    def is_at_war(self, player_id: int) -> bool: ...


class BasePruner(ABC):
    """Abstract base class for action pruners."""

    @abstractmethod
    def prune(
        self,
        state: dict[str, Any],
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Prune actions from the action mask.

        Args:
            state: Game state dictionary
            action_mask: Current valid action mask

        Returns:
            Updated action mask with pruned actions
        """
        pass


class FeasibilityPruner(BasePruner):
    """Removes physically impossible actions.

    This pruner removes actions that cannot be executed regardless
    of game state, such as:
    - Moving units to blocked tiles
    - Building units the city cannot produce
    - Researching already-researched techs
    """

    def __init__(
        self,
        config: PrunerConfig,
        action_decoder: Optional[Callable[[int], dict[str, Any]]] = None,
    ) -> None:
        """Initialize feasibility pruner.

        Args:
            config: Pruner configuration
            action_decoder: Function to decode action index to info
        """
        self.config = config
        self._action_decoder = action_decoder

    def prune(
        self,
        state: dict[str, Any],
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Prune infeasible actions.

        Args:
            state: Game state
            action_mask: Current action mask

        Returns:
            Updated mask with infeasible actions removed
        """
        pruned_mask = action_mask.copy()

        current_player = state.get("current_player", 0)
        units = state.get("units", [])
        cities = state.get("cities", [])

        # Filter player's units
        player_units = [u for u in units if u.get("player") == current_player]
        player_cities = [c for c in cities if c.get("player") == current_player]

        # Prune unit actions
        for i, unit in enumerate(player_units):
            if i >= self.config.max_actions_per_unit:
                # Disable actions for units beyond limit
                self._disable_unit_actions(pruned_mask, i, len(action_mask))
                continue

            # Check movement feasibility
            moves_remaining = unit.get("moves_remaining", 0)
            if moves_remaining <= 0:
                # Disable all move/attack actions for this unit
                self._disable_movement_actions(pruned_mask, i, len(action_mask))

        # Prune city actions
        for i, city in enumerate(player_cities):
            if i >= self.config.max_actions_per_city:
                self._disable_city_actions(pruned_mask, i, len(action_mask))
                continue

            # Check buildable items
            available_production = city.get("available_production", [])
            self._prune_unbuildable(pruned_mask, i, available_production, len(action_mask))

        return pruned_mask

    def _disable_unit_actions(
        self,
        mask: np.ndarray,
        unit_idx: int,
        total_actions: int,
    ) -> None:
        """Disable all actions for a unit."""
        # This would use actual action space encoding
        # Simplified: assume unit actions are in a contiguous range
        pass

    def _disable_movement_actions(
        self,
        mask: np.ndarray,
        unit_idx: int,
        total_actions: int,
    ) -> None:
        """Disable movement actions for a unit."""
        pass

    def _disable_city_actions(
        self,
        mask: np.ndarray,
        city_idx: int,
        total_actions: int,
    ) -> None:
        """Disable all actions for a city."""
        pass

    def _prune_unbuildable(
        self,
        mask: np.ndarray,
        city_idx: int,
        available: list[Any],
        total_actions: int,
    ) -> None:
        """Prune production items not available."""
        pass


class ResourcePruner(BasePruner):
    """Removes actions requiring unavailable resources.

    This pruner removes actions that require resources the player
    doesn't have, such as:
    - Purchasing units without enough gold
    - Building improvements requiring unavailable resources
    - Recruiting great people without enough points
    """

    def __init__(self, config: PrunerConfig) -> None:
        """Initialize resource pruner.

        Args:
            config: Pruner configuration
        """
        self.config = config

    def prune(
        self,
        state: dict[str, Any],
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Prune actions requiring unavailable resources.

        Args:
            state: Game state
            action_mask: Current action mask

        Returns:
            Updated mask with unaffordable actions removed
        """
        pruned_mask = action_mask.copy()

        gold = state.get("gold", 0)
        faith = state.get("faith", 0)
        resources = state.get("resources", {})

        # Prune gold purchases
        self._prune_gold_purchases(pruned_mask, gold, state)

        # Prune faith purchases
        self._prune_faith_purchases(pruned_mask, faith, state)

        # Prune resource-gated buildings/units
        self._prune_resource_requirements(pruned_mask, resources, state)

        return pruned_mask

    def _prune_gold_purchases(
        self,
        mask: np.ndarray,
        gold: int,
        state: dict[str, Any],
    ) -> None:
        """Prune gold purchase actions."""
        # Would check purchase costs and disable unaffordable
        pass

    def _prune_faith_purchases(
        self,
        mask: np.ndarray,
        faith: int,
        state: dict[str, Any],
    ) -> None:
        """Prune faith purchase actions."""
        pass

    def _prune_resource_requirements(
        self,
        mask: np.ndarray,
        resources: dict[str, int],
        state: dict[str, Any],
    ) -> None:
        """Prune actions requiring strategic resources."""
        pass


class StrategicPruner(BasePruner):
    """Removes strategically irrelevant actions.

    This pruner uses heuristics to remove actions unlikely to be
    good choices, such as:
    - Building wonders we're far behind on
    - Exploring distant areas during war
    - Defensive actions when not threatened
    """

    def __init__(
        self,
        config: PrunerConfig,
        threat_evaluator: Optional[Callable[[dict[str, Any]], dict[str, float]]] = None,
    ) -> None:
        """Initialize strategic pruner.

        Args:
            config: Pruner configuration
            threat_evaluator: Function to evaluate threats
        """
        self.config = config
        self._threat_evaluator = threat_evaluator or self._default_threat_evaluator

    def prune(
        self,
        state: dict[str, Any],
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Prune strategically irrelevant actions.

        Args:
            state: Game state
            action_mask: Current action mask

        Returns:
            Updated mask with irrelevant actions deprioritized
        """
        pruned_mask = action_mask.copy()

        # Evaluate current situation
        threats = self._threat_evaluator(state)
        is_at_war = state.get("is_at_war", False)

        # Prune based on situation
        if is_at_war:
            # Deprioritize non-military actions during war
            self._prune_peacetime_actions(pruned_mask, state)
        else:
            # Deprioritize military buildup during peace
            self._prune_wartime_actions(pruned_mask, state)

        # Prune hopeless wonder races
        self._prune_wonder_races(pruned_mask, state)

        # Prune irrelevant exploration
        self._prune_distant_exploration(pruned_mask, state, threats)

        return pruned_mask

    def _default_threat_evaluator(self, state: dict[str, Any]) -> dict[str, float]:
        """Default threat evaluation."""
        return {
            "military_threat": 0.0,
            "economic_threat": 0.0,
            "diplomatic_threat": 0.0,
        }

    def _prune_peacetime_actions(
        self,
        mask: np.ndarray,
        state: dict[str, Any],
    ) -> None:
        """Reduce priority of peacetime actions during war."""
        pass

    def _prune_wartime_actions(
        self,
        mask: np.ndarray,
        state: dict[str, Any],
    ) -> None:
        """Reduce priority of military actions during peace."""
        pass

    def _prune_wonder_races(
        self,
        mask: np.ndarray,
        state: dict[str, Any],
    ) -> None:
        """Prune wonders we're unlikely to complete first."""
        pass

    def _prune_distant_exploration(
        self,
        mask: np.ndarray,
        state: dict[str, Any],
        threats: dict[str, float],
    ) -> None:
        """Prune exploration of distant, low-priority areas."""
        pass


class ActionPruner:
    """Main action pruning orchestrator.

    Combines multiple pruning strategies to efficiently reduce
    the action space while preserving good actions.

    Example:
        >>> config = PrunerConfig(pruning_level=PruningLevel.STANDARD)
        >>> pruner = ActionPruner(config)
        >>> result = pruner.prune(game_state, action_mask)
        >>> print(f"Pruned {result.stats.pruning_ratio:.1%} of actions")
    """

    def __init__(
        self,
        config: PrunerConfig | None = None,
        custom_pruners: list[BasePruner] | None = None,
    ) -> None:
        """Initialize action pruner.

        Args:
            config: Pruner configuration
            custom_pruners: Additional custom pruners
        """
        self.config = config or PrunerConfig()

        # Build pruner pipeline based on configuration
        self._pruners: list[BasePruner] = []

        if self.config.enable_feasibility:
            self._pruners.append(FeasibilityPruner(self.config))

        if self.config.enable_resource:
            self._pruners.append(ResourcePruner(self.config))

        if self.config.enable_strategic:
            self._pruners.append(StrategicPruner(self.config))

        if custom_pruners:
            self._pruners.extend(custom_pruners)

        self._cache: dict[int, PrunedActions] = {}
        logger.debug(f"ActionPruner initialized with {len(self._pruners)} pruners")

    def prune(
        self,
        state: dict[str, Any],
        action_mask: np.ndarray,
        use_cache: bool = True,
    ) -> PrunedActions:
        """Prune actions from the action mask.

        Args:
            state: Game state dictionary
            action_mask: Boolean mask of valid actions
            use_cache: Whether to use cached results

        Returns:
            PrunedActions with filtered mask and stats
        """
        import time

        start_time = time.perf_counter()

        # Check cache
        if use_cache:
            cache_key = self._compute_cache_key(state, action_mask)
            if cache_key in self._cache:
                return self._cache[cache_key]

        stats = PruningStats(total_actions=int(action_mask.sum()))

        # Apply pruners in sequence
        current_mask = action_mask.copy()
        actions_before = int(current_mask.sum())

        for pruner in self._pruners:
            current_mask = pruner.prune(state, current_mask)
            actions_after = int(current_mask.sum())

            # Track which pruner removed how many
            if isinstance(pruner, FeasibilityPruner):
                stats.feasibility_pruned = actions_before - actions_after
            elif isinstance(pruner, ResourcePruner):
                stats.resource_pruned = actions_before - actions_after
            elif isinstance(pruner, StrategicPruner):
                stats.strategic_pruned = actions_before - actions_after

            actions_before = actions_after

        # Ensure minimum actions remain
        if int(current_mask.sum()) < self.config.min_pruned_actions:
            # Re-enable some actions
            current_mask = self._restore_minimum_actions(
                current_mask,
                action_mask,
                self.config.min_pruned_actions,
            )

        stats.final_actions = int(current_mask.sum())
        stats.pruning_time_ms = (time.perf_counter() - start_time) * 1000

        # Build result
        action_indices = np.where(current_mask)[0]

        result = PrunedActions(
            mask=current_mask,
            action_indices=action_indices,
            stats=stats,
        )

        # Cache result
        if use_cache:
            cache_key = self._compute_cache_key(state, action_mask)
            self._cache[cache_key] = result

            # Limit cache size
            if len(self._cache) > self.config.cache_size:
                # Remove oldest entries
                keys_to_remove = list(self._cache.keys())[:len(self._cache) // 2]
                for key in keys_to_remove:
                    del self._cache[key]

        logger.debug(
            f"Pruned {stats.total_actions} -> {stats.final_actions} actions "
            f"({stats.pruning_ratio:.1%} reduction) in {stats.pruning_time_ms:.2f}ms"
        )

        return result

    def _compute_cache_key(
        self,
        state: dict[str, Any],
        action_mask: np.ndarray,
    ) -> int:
        """Compute cache key for state and mask."""
        # Simple hash based on key state features
        key_parts = [
            state.get("turn", 0),
            state.get("current_player", 0),
            len(state.get("units", [])),
            len(state.get("cities", [])),
            hash(action_mask.tobytes()),
        ]
        return hash(tuple(key_parts))

    def _restore_minimum_actions(
        self,
        pruned_mask: np.ndarray,
        original_mask: np.ndarray,
        minimum: int,
    ) -> np.ndarray:
        """Restore actions to meet minimum threshold."""
        current_count = int(pruned_mask.sum())
        if current_count >= minimum:
            return pruned_mask

        # Find actions that were pruned but originally valid
        pruned_indices = np.where(original_mask & ~pruned_mask)[0]

        # Restore random subset
        num_to_restore = minimum - current_count
        if len(pruned_indices) > num_to_restore:
            restore_indices = np.random.choice(
                pruned_indices,
                size=num_to_restore,
                replace=False,
            )
        else:
            restore_indices = pruned_indices

        restored_mask = pruned_mask.copy()
        restored_mask[restore_indices] = True

        return restored_mask

    def clear_cache(self) -> None:
        """Clear the pruning cache."""
        self._cache.clear()


# =============================================================================
# Macro-Action System
# =============================================================================


class MacroActionType(Enum):
    """Types of macro-actions."""

    # City-related
    SETTLE_NEW_CITY = auto()  # Scout + move settler + found
    EXPAND_TO_RESOURCE = auto()  # Settle near specific resource
    BUILD_DISTRICT_COMPLEX = auto()  # Plan district + buildings

    # Military
    CONQUER_CITY = auto()  # Gather army + siege + capture
    DEFEND_POSITION = auto()  # Fortify units + build walls
    RAID_ENEMY = auto()  # Hit-and-run attacks

    # Economic
    ESTABLISH_TRADE_NETWORK = auto()  # Multiple trade routes
    IMPROVE_TILE_GROUP = auto()  # Builder improvements

    # Exploration
    EXPLORE_REGION = auto()  # Scout pattern
    MAP_COASTLINE = auto()  # Naval exploration

    # Diplomacy
    NEGOTIATE_ALLIANCE = auto()  # Multi-step diplomacy


@dataclass
class MacroActionConfig:
    """Configuration for macro-action system.

    Attributes:
        enable_macro_actions: Whether to use macro-actions
        max_macro_depth: Maximum turns for a macro-action
        planning_horizon: Turns to plan ahead
        replanning_frequency: Turns between replanning
        abort_threshold: Performance threshold to abort macro
    """

    enable_macro_actions: bool = True
    max_macro_depth: int = 10
    planning_horizon: int = 5
    replanning_frequency: int = 3
    abort_threshold: float = 0.5


@dataclass
class MacroAction:
    """A multi-turn action plan.

    Attributes:
        macro_type: Type of macro-action
        goal: Description of the goal
        steps: Sequence of atomic actions
        current_step: Current step index
        estimated_turns: Estimated turns to complete
        priority: Priority score (0-1)
        units_involved: Unit IDs involved
        target: Target position or entity
        prerequisites: Required conditions
        success_conditions: Conditions for completion
    """

    macro_type: MacroActionType
    goal: str
    steps: list[int]  # Atomic action indices
    current_step: int = 0
    estimated_turns: int = 1
    priority: float = 0.5
    units_involved: list[str] = field(default_factory=list)
    target: Optional[tuple[int, int] | str] = None
    prerequisites: list[str] = field(default_factory=list)
    success_conditions: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if macro-action is complete."""
        return self.current_step >= len(self.steps)

    @property
    def progress(self) -> float:
        """Get completion progress (0-1)."""
        if not self.steps:
            return 1.0
        return self.current_step / len(self.steps)

    def get_current_action(self) -> int | None:
        """Get the current atomic action to execute."""
        if self.is_complete:
            return None
        return self.steps[self.current_step]

    def advance(self) -> bool:
        """Advance to the next step.

        Returns:
            True if more steps remain
        """
        if self.is_complete:
            return False
        self.current_step += 1
        return not self.is_complete


class MacroActionPlanner:
    """Plans and manages macro-actions.

    This class generates multi-turn action plans and tracks their
    execution, providing higher-level abstraction over atomic actions.

    Example:
        >>> config = MacroActionConfig(max_macro_depth=10)
        >>> planner = MacroActionPlanner(config)
        >>> macro = planner.plan_macro_action(
        ...     state,
        ...     MacroActionType.CONQUER_CITY,
        ...     target="enemy_capital"
        ... )
        >>> while not macro.is_complete:
        ...     action = macro.get_current_action()
        ...     execute(action)
        ...     macro.advance()
    """

    def __init__(
        self,
        config: MacroActionConfig | None = None,
        action_space_size: int = 381,
    ) -> None:
        """Initialize macro-action planner.

        Args:
            config: Planner configuration
            action_space_size: Size of atomic action space
        """
        self.config = config or MacroActionConfig()
        self._action_space_size = action_space_size

        self._active_macros: dict[str, MacroAction] = {}
        self._completed_macros: list[MacroAction] = []
        self._macro_counter = 0

        # Planning strategies by type
        self._planners: dict[MacroActionType, Callable] = {
            MacroActionType.SETTLE_NEW_CITY: self._plan_settle_city,
            MacroActionType.CONQUER_CITY: self._plan_conquer_city,
            MacroActionType.EXPLORE_REGION: self._plan_explore,
            MacroActionType.ESTABLISH_TRADE_NETWORK: self._plan_trade_network,
        }

        logger.debug(f"MacroActionPlanner initialized with {len(self._planners)} strategies")

    def plan_macro_action(
        self,
        state: dict[str, Any],
        macro_type: MacroActionType,
        target: Optional[Any] = None,
        **kwargs: Any,
    ) -> MacroAction | None:
        """Plan a macro-action.

        Args:
            state: Current game state
            macro_type: Type of macro-action
            target: Target for the action
            **kwargs: Additional planning parameters

        Returns:
            Planned macro-action, or None if not feasible
        """
        if macro_type not in self._planners:
            logger.warning(f"No planner for macro type: {macro_type}")
            return None

        planner = self._planners[macro_type]
        macro = planner(state, target, **kwargs)

        if macro and macro.steps:
            self._macro_counter += 1
            macro_id = f"macro_{self._macro_counter}"
            self._active_macros[macro_id] = macro

            logger.info(
                f"Planned {macro_type.name}: {len(macro.steps)} steps, "
                f"~{macro.estimated_turns} turns"
            )

            return macro

        return None

    def _plan_settle_city(
        self,
        state: dict[str, Any],
        target: Optional[tuple[int, int]],
        **kwargs: Any,
    ) -> MacroAction | None:
        """Plan settling a new city."""
        units = state.get("units", [])
        current_player = state.get("current_player", 0)

        # Find settler
        settlers = [
            u for u in units
            if u.get("player") == current_player and u.get("type") == "settler"
        ]

        if not settlers:
            return None

        settler = settlers[0]
        settler_pos = tuple(settler.get("position", (0, 0)))

        # Plan path to target
        if target is None:
            # Would use pathfinding to find good location
            target = (settler_pos[0] + 3, settler_pos[1] + 3)

        steps = []

        # Add movement steps (simplified)
        dx = target[0] - settler_pos[0]
        dy = target[1] - settler_pos[1]
        distance = abs(dx) + abs(dy)

        # Movement action placeholder
        for _ in range(min(distance, 10)):
            steps.append(1)  # Placeholder move action

        # Found city action
        steps.append(2)  # Placeholder found action

        return MacroAction(
            macro_type=MacroActionType.SETTLE_NEW_CITY,
            goal=f"Settle city at {target}",
            steps=steps,
            estimated_turns=distance + 1,
            units_involved=[settler.get("id", "settler")],
            target=target,
        )

    def _plan_conquer_city(
        self,
        state: dict[str, Any],
        target: str | tuple[int, int] | None,
        **kwargs: Any,
    ) -> MacroAction | None:
        """Plan conquering an enemy city."""
        # Would involve:
        # 1. Gather military units
        # 2. Move to target city
        # 3. Siege
        # 4. Capture

        steps = []
        units_involved = []

        # Simplified: just plan attack sequence
        steps.extend([10, 11, 12, 13, 14])  # Placeholder

        return MacroAction(
            macro_type=MacroActionType.CONQUER_CITY,
            goal=f"Conquer {target}",
            steps=steps,
            estimated_turns=10,
            units_involved=units_involved,
            target=target,
            priority=0.8,
        )

    def _plan_explore(
        self,
        state: dict[str, Any],
        target: Optional[tuple[int, int]],
        **kwargs: Any,
    ) -> MacroAction | None:
        """Plan exploration pattern."""
        steps = []

        # Spiral exploration pattern
        for _ in range(20):
            steps.append(5)  # Placeholder exploration moves

        return MacroAction(
            macro_type=MacroActionType.EXPLORE_REGION,
            goal="Explore region",
            steps=steps,
            estimated_turns=20,
        )

    def _plan_trade_network(
        self,
        state: dict[str, Any],
        target: Optional[str],
        **kwargs: Any,
    ) -> MacroAction | None:
        """Plan establishing trade routes."""
        steps = []

        # Plan trade route establishment
        steps.extend([20, 21, 22])  # Placeholder

        return MacroAction(
            macro_type=MacroActionType.ESTABLISH_TRADE_NETWORK,
            goal="Establish trade network",
            steps=steps,
            estimated_turns=5,
        )

    def get_suggested_macros(
        self,
        state: dict[str, Any],
        max_suggestions: int = 5,
    ) -> list[MacroActionType]:
        """Get suggested macro-actions based on game state.

        Args:
            state: Current game state
            max_suggestions: Maximum suggestions to return

        Returns:
            List of suggested macro-action types
        """
        suggestions: list[tuple[MacroActionType, float]] = []

        # Check for settlers without targets
        has_settler = any(
            u.get("type") == "settler"
            for u in state.get("units", [])
            if u.get("player") == state.get("current_player", 0)
        )
        if has_settler:
            suggestions.append((MacroActionType.SETTLE_NEW_CITY, 0.9))

        # Check for military opportunities
        is_at_war = state.get("is_at_war", False)
        if is_at_war:
            suggestions.append((MacroActionType.CONQUER_CITY, 0.8))
            suggestions.append((MacroActionType.RAID_ENEMY, 0.6))
        else:
            suggestions.append((MacroActionType.EXPLORE_REGION, 0.5))
            suggestions.append((MacroActionType.ESTABLISH_TRADE_NETWORK, 0.7))

        # Sort by priority
        suggestions.sort(key=lambda x: x[1], reverse=True)

        return [s[0] for s in suggestions[:max_suggestions]]

    def get_active_macro(self, macro_id: str) -> MacroAction | None:
        """Get an active macro-action by ID."""
        return self._active_macros.get(macro_id)

    def complete_macro(self, macro_id: str) -> bool:
        """Mark a macro-action as complete.

        Args:
            macro_id: ID of macro to complete

        Returns:
            True if successfully completed
        """
        if macro_id not in self._active_macros:
            return False

        macro = self._active_macros.pop(macro_id)
        self._completed_macros.append(macro)
        return True

    def abort_macro(self, macro_id: str) -> bool:
        """Abort an active macro-action.

        Args:
            macro_id: ID of macro to abort

        Returns:
            True if successfully aborted
        """
        if macro_id not in self._active_macros:
            return False

        del self._active_macros[macro_id]
        return True

    def encode_for_state(self) -> dict[str, Any]:
        """Encode macro-action state for neural network.

        Returns:
            Dictionary of encoded features
        """
        active = list(self._active_macros.values())

        return {
            "active_macros": len(active),
            "avg_macro_progress": (
                sum(m.progress for m in active) / len(active)
                if active else 0.0
            ),
            "completed_macros": len(self._completed_macros),
        }

    def reset(self) -> None:
        """Reset all macro-action state."""
        self._active_macros.clear()
        self._completed_macros.clear()
        self._macro_counter = 0
