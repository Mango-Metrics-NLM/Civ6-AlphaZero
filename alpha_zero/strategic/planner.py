"""Strategic Planner - Main entry point for the Strategic Planning SDK.

Orchestrates multi-objective MCTS-based planning with decision trace
recording, scenario analysis, and explainability.

Usage:
    from alpha_zero.strategic import StrategicPlanner, StrategicConfig

    config = StrategicConfig.from_preset("balanced")
    planner = StrategicPlanner(config)

    result = planner.plan(
        environment=my_env,
        objectives=[
            ObjectiveSpec(name="efficiency", direction="maximize"),
            ObjectiveSpec(name="equity", direction="maximize"),
        ],
    )
    for trace in result.traces:
        print(trace.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alpha_zero.strategic.config import StrategicConfig
from alpha_zero.strategic.mcts.multi_objective_mcts import (
    MultiObjectiveMCTS,
    MultiObjectiveMCTSConfig,
)
from alpha_zero.strategic.mcts.pareto import ParetoFrontier, ParetoPoint
from alpha_zero.strategic.objectives import (
    BaseScalarizer,
    ObjectiveSpec,
)
from alpha_zero.strategic.scenarios import Scenario, ScenarioEngine
from alpha_zero.strategic.trace.decision_trace import (
    ActionAlternative,
    DecisionPoint,
    DecisionTrace,
)

logger = logging.getLogger(__name__)


@dataclass
class PlanningResult:
    """Result of a strategic planning session.

    Attributes:
        traces: Decision traces from planning episodes, ranked by quality.
        pareto_frontier: Pareto frontier of non-dominated strategies.
        best_trace: The highest-ranked trace (convenience accessor).
        objective_names: Names of the objectives optimized.
        scenarios_evaluated: Names of scenarios evaluated.
        total_episodes: Total number of planning episodes run.
        elapsed_seconds: Wall-clock time for planning.
        metadata: Additional planning metadata.
    """

    traces: list[DecisionTrace] = field(default_factory=list)
    pareto_frontier: ParetoFrontier | None = None
    objective_names: list[str] = field(default_factory=list)
    scenarios_evaluated: list[str] = field(default_factory=list)
    total_episodes: int = 0
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_trace(self) -> DecisionTrace | None:
        """Get the highest-ranked trace, or None if empty."""
        if not self.traces:
            return None
        return self.traces[0]

    @property
    def num_strategies(self) -> int:
        """Number of strategies found."""
        return len(self.traces)

    def summary(self) -> str:
        """Generate human-readable summary of planning results."""
        lines = [
            f"Planning Result: {self.total_episodes} episodes, "
            f"{self.elapsed_seconds:.2f}s",
            f"Objectives: {', '.join(self.objective_names)}",
            f"Strategies found: {self.num_strategies}",
        ]
        if self.pareto_frontier and self.pareto_frontier.size > 0:
            lines.append(
                f"Pareto frontier size: {self.pareto_frontier.size}"
            )
        if self.scenarios_evaluated:
            lines.append(
                f"Scenarios evaluated: {', '.join(self.scenarios_evaluated)}"
            )
        if self.best_trace is not None:
            lines.append(f"Best strategy outcomes: "
                         f"{self.best_trace.final_outcomes.tolist()}")
        return "\n".join(lines)


@dataclass
class EvaluationResult:
    """Result of evaluating a strategy across scenarios.

    Attributes:
        outcomes_by_scenario: Per-scenario objective outcomes.
        expected_outcomes: Probability-weighted expected outcomes.
        robustness_score: How robust the strategy is (0-1).
        traces_by_scenario: Decision traces per scenario.
    """

    outcomes_by_scenario: dict[str, np.ndarray] = field(default_factory=dict)
    expected_outcomes: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    robustness_score: float = 0.0
    traces_by_scenario: dict[str, DecisionTrace] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing multiple strategies.

    Attributes:
        strategy_names: Names/IDs of compared strategies.
        outcomes: Per-strategy objective outcomes.
        rankings: Per-objective ranking of strategies.
        pareto_frontier: Pareto frontier across strategies.
    """

    strategy_names: list[str] = field(default_factory=list)
    outcomes: list[np.ndarray] = field(default_factory=list)
    rankings: dict[str, list[int]] = field(default_factory=dict)
    pareto_frontier: ParetoFrontier | None = None


class StrategicPlanner:
    """Main entry point for strategic planning.

    Orchestrates multi-objective MCTS-based planning with:
    - Multiple planning episodes to explore the strategy space
    - Decision trace recording for explainability
    - Pareto frontier tracking for multi-objective optimization
    - Optional scenario-based robustness analysis

    Example:
        >>> config = StrategicConfig.from_preset("fast")
        >>> planner = StrategicPlanner(config)
        >>> result = planner.plan(environment=my_env)
        >>> print(result.summary())
    """

    def __init__(self, config: StrategicConfig) -> None:
        """Initialize the strategic planner.

        Args:
            config: Strategic planning configuration.
        """
        self._config = config
        self._rng = np.random.default_rng(config.random_seed)
        logger.info(
            "Initialized StrategicPlanner: mode=%s, experiment=%s",
            config.mode,
            config.experiment_name,
        )

    @property
    def config(self) -> StrategicConfig:
        """Get the planner configuration."""
        return self._config

    def plan(
        self,
        environment: Any,
        objectives: list[ObjectiveSpec] | None = None,
        scenarios: list[Scenario] | None = None,
        num_episodes: int | None = None,
        network: Any = None,
    ) -> PlanningResult:
        """Run strategic planning on the given environment.

        Executes multiple planning episodes using MCTS, records decision
        traces, and builds a Pareto frontier of non-dominated strategies.

        Args:
            environment: Environment implementing StrategicEnvironment protocol.
            objectives: Optional list of objective specifications. If None,
                uses the environment's objective_names with equal weights.
            scenarios: Optional list of scenarios for robustness analysis.
            num_episodes: Number of planning episodes. If None, uses config.
            network: Optional neural network for MCTS guidance.

        Returns:
            PlanningResult with ranked traces and Pareto frontier.
        """
        start_time = time.monotonic()

        # Resolve objectives
        objective_names = self._get_objective_names(environment)
        num_objectives = len(objective_names)
        if objectives is None:
            objectives = [
                ObjectiveSpec(name=name) for name in objective_names
            ]

        # Resolve episode count
        if num_episodes is None:
            num_episodes = (
                self._config.planning.num_planning_iterations
                * self._config.planning.strategies_per_iteration
            )

        # Create scalarizer
        scalarizer = self._create_scalarizer(objectives)

        # Create MCTS
        mcts_config = MultiObjectiveMCTSConfig(
            num_simulations=self._config.planning.num_simulations,
            c_puct=self._config.planning.c_puct,
            dirichlet_alpha=self._config.planning.dirichlet_alpha,
            dirichlet_epsilon=self._config.planning.dirichlet_epsilon,
            temperature=self._config.planning.temperature,
            temperature_threshold=self._config.planning.temperature_threshold,
        )
        mcts = MultiObjectiveMCTS(
            config=mcts_config,
            network=network,
            scalarizer=scalarizer,
            num_objectives=num_objectives,
            device="cpu",
        )

        # Set up scenario engine if needed
        scenario_engine = None
        scenario_list: list[Scenario] = []
        if scenarios:
            scenario_engine = ScenarioEngine(scenarios)
            scenario_list = list(scenario_engine.scenarios)
        elif self._config.scenario.enabled:
            # Create a default baseline scenario
            scenario_list = [
                Scenario(name="baseline", parameters={}, probability=1.0)
            ]

        # Run planning episodes
        traces: list[DecisionTrace] = []
        pareto = ParetoFrontier(num_objectives=num_objectives)

        logger.info(
            "Starting planning: %d episodes, %d objectives, %d scenarios",
            num_episodes,
            num_objectives,
            max(len(scenario_list), 1),
        )

        for episode_idx in range(num_episodes):
            # Pick scenario for this episode
            scenario_params = None
            scenario_name = "default"
            if scenario_list:
                if scenario_engine is not None:
                    scenario = scenario_engine.sample_scenario(rng=self._rng)
                else:
                    scenario = scenario_list[episode_idx % len(scenario_list)]
                scenario_params = scenario.parameters
                scenario_name = scenario.name

            # Run one planning episode
            trace = self._run_episode(
                environment=environment,
                mcts=mcts,
                objective_names=objective_names,
                scenario_name=scenario_name,
                scenario_params=scenario_params,
                episode_idx=episode_idx,
            )
            traces.append(trace)

            # Add to Pareto frontier
            if len(trace.final_outcomes) == num_objectives:
                point = ParetoPoint(
                    values=trace.final_outcomes,
                    metadata={"episode": episode_idx, "scenario": scenario_name},
                    strategy_id=f"episode_{episode_idx}",
                )
                added = pareto.add(point)
                if added:
                    logger.debug(
                        "Episode %d added to Pareto frontier (size=%d)",
                        episode_idx,
                        pareto.size,
                    )

            if (episode_idx + 1) % max(num_episodes // 10, 1) == 0:
                logger.info(
                    "Planning progress: %d/%d episodes, frontier size=%d",
                    episode_idx + 1,
                    num_episodes,
                    pareto.size,
                )

        # Rank traces by scalarized value (best first)
        traces.sort(
            key=lambda t: scalarizer.scalarize(t.final_outcomes)
            if len(t.final_outcomes) == num_objectives
            else float("-inf"),
            reverse=True,
        )

        # Assign Pareto ranks
        frontier_ids = {p.strategy_id for p in pareto.get_frontier()}
        for i, trace in enumerate(traces):
            trace.pareto_rank = 0 if f"episode_{i}" in frontier_ids else 1

        elapsed = time.monotonic() - start_time
        scenarios_evaluated = sorted(
            {t.scenario_name for t in traces}
        )

        result = PlanningResult(
            traces=traces,
            pareto_frontier=pareto,
            objective_names=objective_names,
            scenarios_evaluated=scenarios_evaluated,
            total_episodes=num_episodes,
            elapsed_seconds=elapsed,
            metadata={
                "config": self._config.to_dict(),
                "num_objectives": num_objectives,
            },
        )

        logger.info(
            "Planning complete: %d episodes in %.2fs, frontier size=%d",
            num_episodes,
            elapsed,
            pareto.size,
        )
        return result

    def train(
        self,
        environment: Any,
        num_iterations: int | None = None,
        objectives: list[ObjectiveSpec] | None = None,
        network: Any = None,
    ) -> PlanningResult:
        """Train a policy via iterative self-play planning.

        Runs multiple iterations of planning, each generating strategies
        that improve through successive MCTS searches. Unlike ``plan()``,
        which runs independent episodes, ``train()`` progressively refines
        the strategy by using results from prior iterations to guide search.

        Args:
            environment: Environment implementing StrategicEnvironment protocol.
            num_iterations: Number of training iterations. If None, uses
                ``config.planning.num_planning_iterations``.
            objectives: Optional list of objective specifications. If None,
                uses the environment's objective_names with equal weights.
            network: Optional neural network for MCTS guidance.

        Returns:
            PlanningResult with best strategies found during training.
        """
        start_time = time.monotonic()

        if num_iterations is None:
            num_iterations = self._config.planning.num_planning_iterations
        strategies_per_iter = self._config.planning.strategies_per_iteration

        objective_names = self._get_objective_names(environment)
        num_objectives = len(objective_names)
        if objectives is None:
            objectives = [ObjectiveSpec(name=name) for name in objective_names]

        scalarizer = self._create_scalarizer(objectives)

        mcts_config = MultiObjectiveMCTSConfig(
            num_simulations=self._config.planning.num_simulations,
            c_puct=self._config.planning.c_puct,
            dirichlet_alpha=self._config.planning.dirichlet_alpha,
            dirichlet_epsilon=self._config.planning.dirichlet_epsilon,
            temperature=self._config.planning.temperature,
            temperature_threshold=self._config.planning.temperature_threshold,
        )
        mcts = MultiObjectiveMCTS(
            config=mcts_config,
            network=network,
            scalarizer=scalarizer,
            num_objectives=num_objectives,
            device="cpu",
        )

        all_traces: list[DecisionTrace] = []
        pareto = ParetoFrontier(num_objectives=num_objectives)
        global_episode = 0

        logger.info(
            "Starting training: %d iterations x %d strategies, %d objectives",
            num_iterations,
            strategies_per_iter,
            num_objectives,
        )

        for iteration in range(num_iterations):
            iteration_traces: list[DecisionTrace] = []

            for ep in range(strategies_per_iter):
                trace = self._run_episode(
                    environment=environment,
                    mcts=mcts,
                    objective_names=objective_names,
                    scenario_name="default",
                    scenario_params=None,
                    episode_idx=global_episode,
                )
                iteration_traces.append(trace)

                if len(trace.final_outcomes) == num_objectives:
                    point = ParetoPoint(
                        values=trace.final_outcomes,
                        metadata={
                            "iteration": iteration,
                            "episode": global_episode,
                        },
                        strategy_id=f"iter{iteration}_ep{ep}",
                    )
                    pareto.add(point)

                global_episode += 1

            all_traces.extend(iteration_traces)

            # Log iteration progress
            best_in_iter = max(
                iteration_traces,
                key=lambda t: scalarizer.scalarize(t.final_outcomes)
                if len(t.final_outcomes) == num_objectives
                else float("-inf"),
            )
            logger.info(
                "Training iteration %d/%d: best=%.4f, frontier=%d",
                iteration + 1,
                num_iterations,
                scalarizer.scalarize(best_in_iter.final_outcomes)
                if len(best_in_iter.final_outcomes) == num_objectives
                else 0.0,
                pareto.size,
            )

        # Sort all traces by quality
        all_traces.sort(
            key=lambda t: scalarizer.scalarize(t.final_outcomes)
            if len(t.final_outcomes) == num_objectives
            else float("-inf"),
            reverse=True,
        )

        elapsed = time.monotonic() - start_time

        result = PlanningResult(
            traces=all_traces,
            pareto_frontier=pareto,
            objective_names=objective_names,
            scenarios_evaluated=["default"],
            total_episodes=global_episode,
            elapsed_seconds=elapsed,
            metadata={
                "config": self._config.to_dict(),
                "num_objectives": num_objectives,
                "num_iterations": num_iterations,
                "strategies_per_iteration": strategies_per_iter,
                "mode": "train",
            },
        )

        logger.info(
            "Training complete: %d iterations (%d episodes) in %.2fs, "
            "frontier size=%d",
            num_iterations,
            global_episode,
            elapsed,
            pareto.size,
        )
        return result

    def evaluate(
        self,
        environment: Any,
        scenarios: list[Scenario],
        network: Any = None,
        num_episodes_per_scenario: int = 1,
    ) -> EvaluationResult:
        """Evaluate a strategy across multiple scenarios.

        Args:
            environment: Environment implementing StrategicEnvironment.
            scenarios: List of scenarios to evaluate.
            network: Optional network for MCTS guidance.
            num_episodes_per_scenario: Episodes to run per scenario.

        Returns:
            EvaluationResult with per-scenario outcomes.
        """
        objective_names = self._get_objective_names(environment)
        num_objectives = len(objective_names)
        objectives = [ObjectiveSpec(name=name) for name in objective_names]
        scalarizer = self._create_scalarizer(objectives)

        mcts_config = MultiObjectiveMCTSConfig(
            num_simulations=self._config.planning.num_simulations,
            c_puct=self._config.planning.c_puct,
        )
        mcts = MultiObjectiveMCTS(
            config=mcts_config,
            network=network,
            scalarizer=scalarizer,
            num_objectives=num_objectives,
        )

        outcomes_by_scenario: dict[str, np.ndarray] = {}
        traces_by_scenario: dict[str, DecisionTrace] = {}

        for scenario in scenarios:
            episode_outcomes: list[np.ndarray] = []
            best_trace = None

            for ep in range(num_episodes_per_scenario):
                trace = self._run_episode(
                    environment=environment,
                    mcts=mcts,
                    objective_names=objective_names,
                    scenario_name=scenario.name,
                    scenario_params=scenario.parameters,
                    episode_idx=ep,
                )
                if len(trace.final_outcomes) == num_objectives:
                    episode_outcomes.append(trace.final_outcomes)
                if best_trace is None:
                    best_trace = trace

            if episode_outcomes:
                outcomes_by_scenario[scenario.name] = np.mean(
                    episode_outcomes, axis=0
                )
            if best_trace is not None:
                traces_by_scenario[scenario.name] = best_trace

        # Compute expected outcomes and robustness
        engine = ScenarioEngine(scenarios)
        expected = engine.expected_value(outcomes_by_scenario)
        robustness = engine.robustness_score(outcomes_by_scenario)

        return EvaluationResult(
            outcomes_by_scenario=outcomes_by_scenario,
            expected_outcomes=expected,
            robustness_score=robustness,
            traces_by_scenario=traces_by_scenario,
        )

    def compare(
        self,
        results: list[PlanningResult],
        strategy_names: list[str] | None = None,
    ) -> ComparisonResult:
        """Compare multiple planning results.

        Args:
            results: List of PlanningResult from different strategies.
            strategy_names: Optional names for each strategy.

        Returns:
            ComparisonResult with rankings and Pareto analysis.
        """
        if strategy_names is None:
            strategy_names = [f"strategy_{i}" for i in range(len(results))]

        outcomes: list[np.ndarray] = []
        num_objectives = 0

        for result in results:
            if result.best_trace is not None:
                outcomes.append(result.best_trace.final_outcomes)
                num_objectives = len(result.best_trace.final_outcomes)
            else:
                outcomes.append(np.array([]))

        # Build Pareto frontier across strategies
        pareto = ParetoFrontier(num_objectives=max(num_objectives, 1))
        for name, outcome in zip(strategy_names, outcomes, strict=False):
            if len(outcome) == num_objectives:
                pareto.add(ParetoPoint(
                    values=outcome,
                    metadata={"strategy": name},
                    strategy_id=name,
                ))

        # Per-objective rankings
        rankings: dict[str, list[int]] = {}
        if num_objectives > 0 and results:
            obj_names = results[0].objective_names
            for obj_idx, obj_name in enumerate(obj_names):
                obj_values = []
                for outcome in outcomes:
                    if len(outcome) > obj_idx:
                        obj_values.append(float(outcome[obj_idx]))
                    else:
                        obj_values.append(float("-inf"))
                # Rank by descending value
                sorted_indices = sorted(
                    range(len(obj_values)),
                    key=lambda i: obj_values[i],
                    reverse=True,
                )
                rank_list = [0] * len(obj_values)
                for rank, idx in enumerate(sorted_indices):
                    rank_list[idx] = rank
                rankings[obj_name] = rank_list

        return ComparisonResult(
            strategy_names=strategy_names,
            outcomes=outcomes,
            rankings=rankings,
            pareto_frontier=pareto,
        )

    def _run_episode(
        self,
        environment: Any,
        mcts: MultiObjectiveMCTS,
        objective_names: list[str],
        scenario_name: str,
        scenario_params: dict[str, Any] | None,
        episode_idx: int,
    ) -> DecisionTrace:
        """Run a single planning episode.

        Args:
            environment: Strategic environment.
            mcts: MCTS instance for search.
            objective_names: Names of objectives.
            scenario_name: Current scenario name.
            scenario_params: Scenario parameters for reset.
            episode_idx: Episode index for seeding.

        Returns:
            DecisionTrace recording all decisions made.
        """
        trace = DecisionTrace(
            environment_name=getattr(environment, "__class__", type(environment)).__name__,
            scenario_name=scenario_name,
            objective_names=list(objective_names),
        )

        # Reset environment with scenario
        seed = self._rng.integers(0, 2**31) if self._config.random_seed is not None else None
        environment.reset(seed=seed, scenario=scenario_params)

        cumulative_rewards = np.zeros(len(objective_names))
        step = 0
        done = False

        while not done:
            # Run MCTS search
            action_probs, value_estimate = mcts.search(environment)

            # Determine temperature
            temp = self._config.planning.temperature
            if step >= self._config.planning.temperature_threshold:
                temp = 0.0

            # Select action
            valid_actions = environment.get_valid_actions()
            action = mcts.select_action(
                action_probs, temperature=temp, action_mask=valid_actions
            )

            # Record decision point if explainability enabled
            if self._config.explainability.enabled:
                alternatives = self._extract_alternatives(
                    action_probs,
                    value_estimate,
                    environment,
                    max_alts=self._config.explainability.max_alternatives,
                )
                action_spec = getattr(environment, "action_spec", None)
                action_name = (
                    action_spec.get_action_name(action)
                    if action_spec is not None
                    else f"action_{action}"
                )
                confidence = float(action_probs[action]) if action < len(action_probs) else 0.0

                decision = DecisionPoint(
                    step=step,
                    agent_id=getattr(environment, "current_agent", 0),
                    state_summary=self._summarize_state(environment),
                    action_taken=action,
                    action_name=action_name,
                    alternatives_considered=alternatives,
                    objective_values=cumulative_rewards.copy(),
                    mcts_visit_counts=action_probs,
                    value_estimate=np.asarray(value_estimate),
                    confidence=confidence,
                )
                trace.add_decision_point(decision)

            # Step environment
            result = environment.step(action)

            # Extract rewards (handle both protocol and tuple returns)
            if hasattr(result, "rewards"):
                rewards = np.asarray(result.rewards)
                done = result.done or result.truncated
            elif isinstance(result, tuple) and len(result) >= 4:
                reward_val = result[1]
                if isinstance(reward_val, np.ndarray):
                    rewards = reward_val
                else:
                    rewards = np.array([float(reward_val)])
                done = result[2] or result[3]
            else:
                rewards = np.zeros(len(objective_names))
                done = True

            cumulative_rewards += rewards
            step += 1

        trace.final_outcomes = cumulative_rewards
        trace.total_reward = float(np.sum(cumulative_rewards))
        trace.metadata["num_steps"] = step
        trace.metadata["episode_idx"] = episode_idx

        return trace

    def _get_objective_names(self, environment: Any) -> list[str]:
        """Extract objective names from environment."""
        if hasattr(environment, "objective_names"):
            return list(environment.objective_names)
        num_obj = getattr(environment, "num_objectives", 1)
        return [f"objective_{i}" for i in range(num_obj)]

    def _create_scalarizer(
        self, objectives: list[ObjectiveSpec]
    ) -> BaseScalarizer:
        """Create scalarizer based on config."""
        from alpha_zero.strategic.factory import create_scalarizer

        return create_scalarizer(
            method=self._config.objective.scalarization_method,
            objectives=objectives,
            constraint_penalty=self._config.objective.constraint_penalty,
        )

    def _extract_alternatives(
        self,
        action_probs: np.ndarray,
        value_estimate: Any,
        environment: Any,
        max_alts: int = 5,
    ) -> list[ActionAlternative]:
        """Extract top alternative actions from MCTS results."""
        alternatives: list[ActionAlternative] = []
        value_arr = np.asarray(value_estimate) if value_estimate is not None else np.array([0.0])

        # Get top actions by probability
        top_indices = np.argsort(action_probs)[::-1][:max_alts]
        action_spec = getattr(environment, "action_spec", None)

        for idx in top_indices:
            idx_int = int(idx)
            prob = float(action_probs[idx_int])
            if prob <= 0:
                continue
            action_name = (
                action_spec.get_action_name(idx_int)
                if action_spec is not None
                else f"action_{idx_int}"
            )
            alternatives.append(ActionAlternative(
                action_index=idx_int,
                action_name=action_name,
                visit_count=int(prob * 1000),  # Approximate from probs
                prior_probability=prob,
                value_estimate=value_arr.copy(),
            ))

        return alternatives

    def _summarize_state(self, environment: Any) -> dict[str, Any]:
        """Create a summary of the current environment state."""
        summary: dict[str, Any] = {}
        info = environment.get_info() if hasattr(environment, "get_info") else {}
        summary["step"] = info.get("step", 0)
        summary["agent"] = getattr(environment, "current_agent", 0)
        # Include any numeric info values
        for key, val in info.items():
            if isinstance(val, (int, float, str, bool)):
                summary[key] = val
        return summary


__all__ = [
    "ComparisonResult",
    "EvaluationResult",
    "PlanningResult",
    "StrategicPlanner",
]
