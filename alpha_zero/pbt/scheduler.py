"""Scheduler and selection strategies for Population-Based Training.

This module provides scheduling logic for when to perform exploit/explore
operations and various selection strategies for choosing agents.

Usage:
    from alpha_zero.pbt.scheduler import PBTScheduler

    scheduler = PBTScheduler(config)

    if scheduler.should_evaluate(current_step):
        # Run fitness evaluation
        scheduler.record_evaluation(current_step)

    if scheduler.should_exploit_explore(current_step):
        # Perform exploit/explore
        scheduler.record_exploit_explore(current_step)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from alpha_zero.pbt.config import AgentState, PBTConfig, SelectionMethod

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Types of scheduling strategies."""

    FIXED_INTERVAL = "fixed_interval"
    ADAPTIVE = "adaptive"
    EPISODE_BASED = "episode_based"
    FITNESS_BASED = "fitness_based"


@dataclass
class SchedulerState:
    """State tracking for the scheduler.

    Attributes:
        last_evaluation_step: Last step when evaluation was performed
        last_exploit_explore_step: Last step when exploit/explore was performed
        evaluation_count: Total number of evaluations
        exploit_explore_count: Total number of exploit/explore cycles
        fitness_plateau_count: Number of steps without fitness improvement
        best_fitness_seen: Best fitness observed
    """

    last_evaluation_step: int = 0
    last_exploit_explore_step: int = 0
    evaluation_count: int = 0
    exploit_explore_count: int = 0
    fitness_plateau_count: int = 0
    best_fitness_seen: float = float("-inf")


class SelectionStrategy(ABC):
    """Abstract base class for agent selection strategies."""

    @abstractmethod
    def select(
        self,
        agents: list[AgentState],
        num_to_select: int,
        rng: np.random.Generator,
    ) -> list[AgentState]:
        """Select agents from the population.

        Args:
            agents: List of agents to select from
            num_to_select: Number of agents to select
            rng: Random number generator

        Returns:
            List of selected agents.
        """
        pass


class TruncationSelection(SelectionStrategy):
    """Truncation selection strategy.

    Selects the top-performing agents based on fitness.
    """

    def select(
        self,
        agents: list[AgentState],
        num_to_select: int,
        rng: np.random.Generator,
    ) -> list[AgentState]:
        """Select top agents by fitness."""
        sorted_agents = sorted(agents, key=lambda a: a.get_smoothed_fitness(), reverse=True)
        return sorted_agents[:num_to_select]


class TournamentSelection(SelectionStrategy):
    """Tournament selection strategy.

    Runs tournaments among random subsets of agents.
    """

    def __init__(self, tournament_size: int = 3) -> None:
        """Initialize tournament selection.

        Args:
            tournament_size: Number of agents in each tournament
        """
        self.tournament_size = tournament_size

    def select(
        self,
        agents: list[AgentState],
        num_to_select: int,
        rng: np.random.Generator,
    ) -> list[AgentState]:
        """Select agents via tournament."""
        selected = []
        for _ in range(num_to_select):
            # Random tournament
            tournament = rng.choice(
                agents, size=min(self.tournament_size, len(agents)), replace=False
            )
            winner = max(tournament, key=lambda a: a.get_smoothed_fitness())
            selected.append(winner)
        return selected


class RouletteSelection(SelectionStrategy):
    """Roulette wheel (fitness proportionate) selection strategy."""

    def select(
        self,
        agents: list[AgentState],
        num_to_select: int,
        rng: np.random.Generator,
    ) -> list[AgentState]:
        """Select agents with probability proportional to fitness."""
        fitnesses = np.array([max(0.01, a.get_smoothed_fitness()) for a in agents])
        probabilities = fitnesses / fitnesses.sum()

        indices = rng.choice(len(agents), size=num_to_select, replace=False, p=probabilities)
        return [agents[i] for i in indices]


class RankSelection(SelectionStrategy):
    """Rank-based selection strategy.

    Selection probability based on rank rather than raw fitness.
    """

    def __init__(self, selection_pressure: float = 2.0) -> None:
        """Initialize rank selection.

        Args:
            selection_pressure: Higher values favor top ranks more
        """
        self.selection_pressure = selection_pressure

    def select(
        self,
        agents: list[AgentState],
        num_to_select: int,
        rng: np.random.Generator,
    ) -> list[AgentState]:
        """Select agents based on rank."""
        sorted_agents = sorted(agents, key=lambda a: a.get_smoothed_fitness(), reverse=True)

        n = len(sorted_agents)

        # Handle edge case: n=0 (empty population)
        if n == 0:
            return []

        # Handle edge case: n=1 (single agent)
        if n == 1:
            return sorted_agents[:num_to_select]

        # Cap num_to_select to population size
        actual_select = min(num_to_select, n)

        # Linear ranking: P(i) = (2 - sp + 2*(sp-1)*(n-i)/(n-1)) / n
        ranks = np.arange(n)
        sp = self.selection_pressure
        probabilities = (2 - sp + 2 * (sp - 1) * (n - 1 - ranks) / (n - 1)) / n

        # Ensure minimum probability to avoid zeros (needed for np.choice with replace=False)
        min_prob = 0.01 / n
        probabilities = np.maximum(probabilities, min_prob)
        probabilities = probabilities / probabilities.sum()

        indices = rng.choice(n, size=actual_select, replace=False, p=probabilities)
        return [sorted_agents[i] for i in indices]


class SteadyStateSelection(SelectionStrategy):
    """Steady-state selection for incremental population updates.

    Replaces only the worst individuals rather than entire generations.
    """

    def select(
        self,
        agents: list[AgentState],
        num_to_select: int,
        rng: np.random.Generator,
    ) -> list[AgentState]:
        """Select worst agents for replacement."""
        sorted_agents = sorted(agents, key=lambda a: a.get_smoothed_fitness())
        return sorted_agents[:num_to_select]


def create_selection_strategy(method: SelectionMethod, **kwargs: Any) -> SelectionStrategy:
    """Factory function to create a selection strategy.

    Args:
        method: Selection method enum
        **kwargs: Additional arguments for the strategy

    Returns:
        Configured SelectionStrategy.
    """
    strategies = {
        SelectionMethod.TRUNCATION: TruncationSelection,
        SelectionMethod.TOURNAMENT: lambda: TournamentSelection(kwargs.get("tournament_size", 3)),
        SelectionMethod.ROULETTE: RouletteSelection,
        SelectionMethod.RANK: lambda: RankSelection(kwargs.get("selection_pressure", 2.0)),
    }

    creator = strategies.get(method, TruncationSelection)
    if callable(creator) and not isinstance(creator, type):
        return creator()
    return creator()


@dataclass
class EvaluationResult:
    """Result of a fitness evaluation.

    Attributes:
        agent_id: ID of evaluated agent
        fitness: Computed fitness score
        games_played: Number of games played
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        details: Additional evaluation details
        timestamp: When evaluation was performed
    """

    agent_id: str
    fitness: float
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "fitness": self.fitness,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class FitnessEvaluator:
    """Evaluates agent fitness through game play.

    This class provides methods to evaluate agents by playing games
    against baseline opponents or other agents.
    """

    def __init__(
        self,
        games_per_evaluation: int = 20,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize fitness evaluator.

        Args:
            games_per_evaluation: Number of games per evaluation
            rng: Random number generator
        """
        self.games_per_evaluation = games_per_evaluation
        self.rng = rng or np.random.default_rng()

        # Evaluation callbacks
        self._play_game_fn: Callable | None = None
        self._get_agent_policy_fn: Callable | None = None

    def set_callbacks(
        self,
        play_game_fn: Callable | None = None,
        get_agent_policy_fn: Callable | None = None,
    ) -> None:
        """Set evaluation callbacks.

        Args:
            play_game_fn: Function to play a game (policy1, policy2) -> (result1, result2)
            get_agent_policy_fn: Function to get policy for agent (agent_id) -> policy
        """
        self._play_game_fn = play_game_fn
        self._get_agent_policy_fn = get_agent_policy_fn

    def evaluate_agent(
        self,
        agent: AgentState,
        opponent_agents: list[AgentState] | None = None,
        opponent_policy: Callable | None = None,
    ) -> EvaluationResult:
        """Evaluate a single agent.

        Args:
            agent: Agent to evaluate
            opponent_agents: Optional list of opponent agents
            opponent_policy: Optional baseline opponent policy

        Returns:
            EvaluationResult with fitness score.
        """
        if self._play_game_fn is None:
            # Return dummy result if no play function set
            logger.warning("No play_game_fn set, returning dummy evaluation")
            return EvaluationResult(
                agent_id=agent.agent_id,
                fitness=0.5,
                games_played=0,
            )

        wins = 0
        losses = 0
        draws = 0
        total_reward = 0.0

        for _i in range(self.games_per_evaluation):
            # Get agent policy
            if self._get_agent_policy_fn:
                agent_policy = self._get_agent_policy_fn(agent.agent_id)
            else:
                agent_policy = None

            # Get opponent policy
            if opponent_agents and self._get_agent_policy_fn:
                opponent = self.rng.choice(opponent_agents)
                opp_policy = self._get_agent_policy_fn(opponent.agent_id)
            elif opponent_policy:
                opp_policy = opponent_policy
            else:
                opp_policy = None

            # Play game
            try:
                result1, result2 = self._play_game_fn(agent_policy, opp_policy)
                total_reward += result1

                if result1 > result2:
                    wins += 1
                elif result1 < result2:
                    losses += 1
                else:
                    draws += 1
            except Exception as e:
                logger.warning(f"Game play failed: {e}")
                continue

        games_played = wins + losses + draws

        # Compute fitness as win rate + 0.5 * draw rate
        if games_played > 0:
            fitness = (wins + 0.5 * draws) / games_played
        else:
            fitness = 0.0

        return EvaluationResult(
            agent_id=agent.agent_id,
            fitness=fitness,
            games_played=games_played,
            wins=wins,
            losses=losses,
            draws=draws,
            details={"total_reward": total_reward},
        )

    def evaluate_population(
        self,
        agents: list[AgentState],
        opponent_policy: Callable | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate all agents in the population.

        Args:
            agents: List of agents to evaluate
            opponent_policy: Baseline opponent policy

        Returns:
            List of EvaluationResults.
        """
        results = []
        for agent in agents:
            result = self.evaluate_agent(
                agent,
                opponent_agents=[a for a in agents if a.agent_id != agent.agent_id],
                opponent_policy=opponent_policy,
            )
            results.append(result)

        return results


class PBTScheduler:
    """Scheduler for Population-Based Training operations.

    Manages timing of evaluations, exploit/explore cycles, and
    adaptive scheduling based on training progress.
    """

    def __init__(
        self,
        config: PBTConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize PBT scheduler.

        Args:
            config: PBT configuration
            rng: Random number generator
        """
        self.config = config
        self.rng = rng or np.random.default_rng()

        self._state = SchedulerState()
        self._selection_strategy = create_selection_strategy(
            config.get_selection_method_enum(),
            tournament_size=config.tournament_size,
        )
        self._evaluator = FitnessEvaluator(
            games_per_evaluation=config.evaluation_games,
            rng=self.rng,
        )

    def should_evaluate(self, current_step: int) -> bool:
        """Check if evaluation should be performed.

        Args:
            current_step: Current training step

        Returns:
            True if evaluation should be performed.
        """
        steps_since_eval = current_step - self._state.last_evaluation_step
        return steps_since_eval >= self.config.evaluation_interval

    def should_exploit_explore(self, current_step: int) -> bool:
        """Check if exploit/explore should be performed.

        Args:
            current_step: Current training step

        Returns:
            True if exploit/explore should be performed.
        """
        # Must have evaluated first
        if self._state.evaluation_count == 0:
            return False

        steps_since_ee = current_step - self._state.last_exploit_explore_step
        return steps_since_ee >= self.config.evaluation_interval

    def should_checkpoint(self, current_step: int, last_checkpoint_step: int) -> bool:
        """Check if checkpoint should be saved.

        Args:
            current_step: Current training step
            last_checkpoint_step: Step of last checkpoint

        Returns:
            True if checkpoint should be saved.
        """
        return current_step - last_checkpoint_step >= self.config.checkpoint_interval

    def record_evaluation(self, step: int, results: list[EvaluationResult]) -> None:
        """Record that an evaluation was performed.

        Args:
            step: Current training step
            results: Evaluation results
        """
        self._state.last_evaluation_step = step
        self._state.evaluation_count += 1

        # Track best fitness
        if results:
            max_fitness = max(r.fitness for r in results)
            if max_fitness > self._state.best_fitness_seen:
                self._state.best_fitness_seen = max_fitness
                self._state.fitness_plateau_count = 0
            else:
                self._state.fitness_plateau_count += 1

        logger.debug(f"Recorded evaluation at step {step} (count={self._state.evaluation_count})")

    def record_exploit_explore(self, step: int) -> None:
        """Record that exploit/explore was performed.

        Args:
            step: Current training step
        """
        self._state.last_exploit_explore_step = step
        self._state.exploit_explore_count += 1

        logger.debug(
            f"Recorded exploit/explore at step {step} (count={self._state.exploit_explore_count})"
        )

    def select_for_exploitation(
        self,
        agents: list[AgentState],
        num_to_select: int,
    ) -> list[AgentState]:
        """Select top agents for exploitation (to copy from).

        Args:
            agents: All agents in population
            num_to_select: Number to select

        Returns:
            List of selected agents.
        """
        return self._selection_strategy.select(agents, num_to_select, self.rng)

    def select_for_replacement(
        self,
        agents: list[AgentState],
        num_to_replace: int,
    ) -> list[AgentState]:
        """Select worst agents for replacement.

        Args:
            agents: All agents in population
            num_to_replace: Number to replace

        Returns:
            List of agents to replace.
        """
        # Use steady-state selection for replacement
        steady_state = SteadyStateSelection()
        return steady_state.select(agents, num_to_replace, self.rng)

    def get_evaluator(self) -> FitnessEvaluator:
        """Get the fitness evaluator.

        Returns:
            FitnessEvaluator instance.
        """
        return self._evaluator

    def get_state(self) -> SchedulerState:
        """Get current scheduler state.

        Returns:
            SchedulerState object.
        """
        return self._state

    def is_plateau_detected(self, threshold: int = 10) -> bool:
        """Check if fitness has plateaued.

        Args:
            threshold: Number of evaluations without improvement

        Returns:
            True if plateau detected.
        """
        return self._state.fitness_plateau_count >= threshold

    def reset_state(self) -> None:
        """Reset scheduler state."""
        self._state = SchedulerState()
