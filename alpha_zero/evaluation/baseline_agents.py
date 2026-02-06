"""Baseline agents for evaluation benchmarking.

This module provides baseline agents that can be used to measure
the performance of trained models. These agents provide benchmarks
ranging from random play to simple heuristic strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alpha_zero.envs.microciv import MicroCivEnv


class BaseAgent(ABC):
    """Abstract base class for agents."""

    @abstractmethod
    def select_action(self, env: MicroCivEnv, obs: dict[str, np.ndarray]) -> int:
        """Select an action given the current observation.

        Args:
            env: The game environment (for accessing game state).
            obs: Current observation dictionary with 'spatial', 'vector', 'action_mask'.

        Returns:
            The selected action index.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for a new game."""
        pass


class RandomAgent(BaseAgent):
    """Agent that selects uniformly from valid actions.

    This provides a baseline for measuring if the trained agent
    has learned anything beyond random play.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize random agent.

        Args:
            seed: Random seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)

    def select_action(self, env: MicroCivEnv, obs: dict[str, np.ndarray]) -> int:
        """Select a random valid action.

        Args:
            env: The game environment.
            obs: Current observation with action_mask.

        Returns:
            Randomly selected valid action index.
        """
        action_mask = obs["action_mask"]
        valid_actions = np.where(action_mask > 0)[0]

        if len(valid_actions) == 0:
            return 0  # END_TURN fallback

        return int(self._rng.choice(valid_actions))

    def reset(self) -> None:
        """Reset is a no-op for random agent."""
        pass


@dataclass
class ActionPriority:
    """Defines priority ordering for heuristic action selection."""

    attack: int = 100  # Highest priority: attack enemies
    found_city: int = 80  # High priority: expand
    produce_warrior: int = 60  # Medium-high: build army
    produce_archer: int = 55  # Slightly lower than warrior
    produce_settler: int = 50  # Medium: prepare for expansion
    move_toward_enemy: int = 40  # Move aggressively
    move_explore: int = 30  # Explore map
    skip: int = 10  # Low priority
    end_turn: int = 5  # Lowest: only when nothing else to do


class GreedyHeuristicAgent(BaseAgent):
    """Agent that uses simple heuristics to select actions.

    Priority order:
    1. Attack enemy units/cities if adjacent
    2. Found city with settlers (if good location)
    3. Produce units (warriors > archers > settlers)
    4. Move units toward enemies
    5. Move units to explore
    6. Skip/End turn

    This provides a stronger baseline than random and helps bootstrap
    imitation learning for value head initialization.
    """

    def __init__(
        self,
        seed: int | None = None,
        aggression: float = 0.8,
        expansion_priority: float = 0.6,
    ) -> None:
        """Initialize heuristic agent.

        Args:
            seed: Random seed for tie-breaking.
            aggression: How much to prioritize attacking (0-1).
            expansion_priority: How much to prioritize city founding (0-1).
        """
        self._rng = np.random.default_rng(seed)
        self.aggression = aggression
        self.expansion_priority = expansion_priority
        self.priority = ActionPriority()

    def select_action(self, env: MicroCivEnv, obs: dict[str, np.ndarray]) -> int:
        """Select action using heuristic priorities.

        Args:
            env: The game environment.
            obs: Current observation with action_mask.

        Returns:
            Selected action index based on heuristics.
        """
        action_mask = obs["action_mask"]
        valid_actions = np.where(action_mask > 0)[0]

        if len(valid_actions) == 0:
            return 0  # END_TURN fallback

        if len(valid_actions) == 1:
            return int(valid_actions[0])

        # Score each valid action
        action_scores = self._score_actions(env, valid_actions)

        # Select highest scoring action (with random tie-breaking)
        max_score = max(action_scores.values())
        best_actions = [a for a, s in action_scores.items() if s == max_score]

        return int(self._rng.choice(best_actions))

    def _score_actions(
        self,
        env: MicroCivEnv,
        valid_actions: np.ndarray,
    ) -> dict[int, float]:
        """Score each action based on heuristics.

        Args:
            env: The game environment for state access.
            valid_actions: Array of valid action indices.

        Returns:
            Dictionary mapping action index to score.
        """
        action_space = env.action_space_handler
        scores: dict[int, float] = {}

        # Get game state info
        current_player = env.current_player
        own_units = [u for u in env.units if u.player == current_player]
        enemy_units = [u for u in env.units if u.player != current_player]
        own_cities = [c for c in env.cities if c.player == current_player]
        enemy_cities = [c for c in env.cities if c.player != current_player]

        # Find nearest enemy for each unit
        enemy_positions = [(u.position[0], u.position[1]) for u in enemy_units]
        enemy_city_positions = [(c.position[0], c.position[1]) for c in enemy_cities]
        all_enemy_targets = enemy_positions + enemy_city_positions

        for action in valid_actions:
            score = self._score_single_action(
                action,
                action_space,
                own_units,
                own_cities,
                all_enemy_targets,
                env.board_size,
            )
            scores[action] = score

        return scores

    def _score_single_action(
        self,
        action: int,
        action_space,
        own_units: list,
        own_cities: list,
        enemy_targets: list[tuple[int, int]],
        board_size: int,
    ) -> float:
        """Score a single action.

        Args:
            action: Action index to score.
            action_space: MicroCivActionSpace for decoding.
            own_units: List of own units.
            own_cities: List of own cities.
            enemy_targets: List of enemy positions (units + cities).
            board_size: Size of the game board.

        Returns:
            Score for this action.
        """
        # END_TURN
        if action == action_space.end_turn_offset:
            return self.priority.end_turn

        # ATTACK actions (highest priority)
        if action_space.attack_offset <= action < action_space.skip_offset:
            return self.priority.attack * self.aggression + self._rng.random() * 5

        # FOUND_CITY actions
        if action_space.found_city_offset <= action < action_space.produce_offset:
            unit_idx = action - action_space.found_city_offset
            if unit_idx < len(own_units):
                unit = own_units[unit_idx]
                if unit.unit_type == "settler":
                    # Check if location is good (not too close to existing cities)
                    min_city_dist = float("inf")
                    for city in own_cities:
                        dist = abs(unit.position[0] - city.position[0]) + abs(
                            unit.position[1] - city.position[1]
                        )
                        min_city_dist = min(min_city_dist, dist)

                    if min_city_dist >= 4:  # Good distance from other cities
                        return self.priority.found_city * self.expansion_priority
            return 0  # Invalid found_city action

        # PRODUCE actions
        if action >= action_space.produce_offset:
            rel_action = action - action_space.produce_offset
            city_idx = rel_action // action_space.num_unit_types
            unit_type_idx = rel_action % action_space.num_unit_types

            if city_idx < len(own_cities):
                unit_types = ["warrior", "settler", "archer"]
                if unit_type_idx < len(unit_types):
                    unit_type = unit_types[unit_type_idx]
                    if unit_type == "warrior":
                        return self.priority.produce_warrior
                    elif unit_type == "archer":
                        return self.priority.produce_archer
                    elif unit_type == "settler":
                        # Only prioritize settlers if we have few cities
                        if len(own_cities) < 3:
                            return self.priority.produce_settler
                        return self.priority.produce_settler * 0.3
            return 0

        # SKIP actions
        if action_space.skip_offset <= action < action_space.found_city_offset:
            return self.priority.skip

        # MOVE actions
        if action_space.move_offset <= action < action_space.attack_offset:
            rel_action = action - action_space.move_offset
            unit_idx = rel_action // 8
            dir_idx = rel_action % 8

            if unit_idx < len(own_units):
                unit = own_units[unit_idx]
                directions = [
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ]
                dx, dy = directions[dir_idx]
                new_pos = (unit.position[0] + dx, unit.position[1] + dy)

                # Score based on whether this moves toward enemies
                if enemy_targets:
                    # Current distance to nearest enemy
                    current_dist = min(
                        abs(unit.position[0] - ex) + abs(unit.position[1] - ey)
                        for ex, ey in enemy_targets
                    )
                    # New distance
                    new_dist = min(
                        abs(new_pos[0] - ex) + abs(new_pos[1] - ey) for ex, ey in enemy_targets
                    )

                    if new_dist < current_dist:
                        return self.priority.move_toward_enemy * self.aggression
                    elif new_dist > current_dist:
                        return self.priority.move_explore * 0.5  # Moving away
                    else:
                        return self.priority.move_explore * 0.8  # Sideways

                # No enemies visible, explore
                return self.priority.move_explore

        return 0

    def reset(self) -> None:
        """Reset is a no-op for heuristic agent."""
        pass


class MCTSAgentWrapper(BaseAgent):
    """Wrapper to use MCTS-based model as a BaseAgent.

    This allows trained models to be evaluated against baseline agents
    using the same interface.
    """

    def __init__(
        self,
        mcts,
        temperature: float = 0.0,
    ) -> None:
        """Initialize MCTS agent wrapper.

        Args:
            mcts: Civ6MCTS or similar MCTS instance.
            temperature: Temperature for action selection (0 = greedy).
        """
        self.mcts = mcts
        self.temperature = temperature

    def select_action(self, env: MicroCivEnv, obs: dict[str, np.ndarray]) -> int:
        """Select action using MCTS search.

        Args:
            env: The game environment.
            obs: Current observation (used for consistency, MCTS uses env directly).

        Returns:
            Selected action index from MCTS.
        """
        action_probs, _ = self.mcts.search(env)
        return self.mcts.select_action(action_probs, self.temperature)

    def reset(self) -> None:
        """Reset MCTS tree for new game."""
        if hasattr(self.mcts, "reset"):
            self.mcts.reset()
