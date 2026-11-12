"""Batch MCTS inference engine for efficient parallel search.

This module implements BatchGumbelMCTS, which accumulates leaf node evaluations
from multiple parallel MCTS trees and submits them as single batched GPU forward
passes. This significantly reduces GPU idle time and improves throughput compared
to sequential MCTS search.

Key features:
- Parallel tree traversal across N environments
- Batched neural network evaluation (single forward pass for multiple leaves)
- Compatible with GumbelMCTS algorithm (sequential halving, Gumbel sampling)
- W&B metrics logging for batch efficiency tracking

Usage:
    from alpha_zero.core.batch_mcts import BatchGumbelMCTS, BatchMCTSConfig

    config = BatchMCTSConfig(
        num_simulations=16,
        num_parallel_envs=8,
        max_batch_size=64,
    )
    mcts = BatchGumbelMCTS(network, config, device="cuda")

    # Batch search across multiple environments
    results = mcts.batch_search([env1, env2, env3])
    for action_probs, root_value in results:
        action = mcts.select_action(action_probs)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from alpha_zero.core.civ6_mcts import (
    GumbelMCTSNode,
    compute_completed_q,
    gumbel_sample_actions,
)
from alpha_zero.core.mcts.backup import (
    backpropagate_path,
    env_current_player,
    log_backup_diagnostics,
    should_track_to_play,
)
from alpha_zero.core.mcts.gumbel import (
    POLICY_TARGET_COMPLETED_Q,
)
from alpha_zero.core.mcts.gumbel_ops import (
    HalvingRound,
    SearchStats,
    choose_among_most_visited,
    halving_schedule,
)
from alpha_zero.core.mcts_constants import (
    DEFAULT_C_VISIT,
    DEFAULT_C_VISIT_DIVISOR,
    EPSILON,
    LOG_SEARCH_STATS_INTERVAL,
)
from alpha_zero.core.mcts_constants import (
    GREEDY_THRESHOLD as _GREEDY_THRESHOLD,
)
from alpha_zero.utils.device import place_network
from alpha_zero.utils.experiment_tracker import log_metrics

if TYPE_CHECKING:
    from alpha_zero.core.civ6_mcts import MCTSEnvironment, MCTSNetwork

logger = logging.getLogger(__name__)

# Threshold below which action probability sums are treated as zero
_PROBABILITY_SUM_EPSILON = 1e-8


@dataclass(slots=True)
class BatchMCTSConfig:
    """Configuration for batch MCTS inference.

    Attributes:
        num_simulations: Total simulation budget per environment.
        num_sampled_actions: Number of actions to sample at root (Gumbel-Top-k).
        max_batch_size: Maximum batch size for neural network evaluation.
        num_parallel_envs: Number of environments to search in parallel.
        max_considered_actions: Maximum actions to consider before Gumbel sampling.
        c_visit: Controls value vs visit count trade-off in completed Q.
        c_scale: Scale factor for value estimates.
        use_sequential_halving: Use sequential halving for simulation allocation.
        min_actions_after_halving: Lower bound on survivors per halving round.
            Values below 2 act as 2 (see
            :func:`~alpha_zero.core.mcts.gumbel_ops.halving_schedule`); it
            reshapes the rounds but never changes the total budget.
        temperature: Temperature for final action selection.
        temperature_threshold: Turn after which to use temperature=0.
    """

    num_simulations: int = 16
    num_sampled_actions: int = 16
    max_batch_size: int = 64
    num_parallel_envs: int = 8
    max_considered_actions: int = 32
    c_visit: float = DEFAULT_C_VISIT
    c_scale: float = 1.0
    use_sequential_halving: bool = True
    min_actions_after_halving: int = 1
    temperature: float = 1.0
    temperature_threshold: int = 30
    # Seed for this search's own RNG. ``None`` means unseeded, which is what
    # every existing caller gets, so behaviour is unchanged unless a seed is
    # passed. `create_mcts` filters kwargs against this dataclass's fields, so
    # adding it here makes `create_mcts(..., seed=42)` work with no factory
    # change.
    seed: int | None = None
    policy_target: str = POLICY_TARGET_COMPLETED_Q
    greedy_threshold: float = _GREEDY_THRESHOLD

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_parallel_envs < 1:
            raise ValueError(f"num_parallel_envs must be >= 1, got {self.num_parallel_envs}")
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")


class _PendingEvaluation(NamedTuple):
    """Represents a pending neural network evaluation for a leaf node.

    Attributes:
        env_idx: Index of the environment this evaluation belongs to.
        search_path: Path from root to leaf node (for backpropagation).
        node: The leaf node to expand after evaluation.
        spatial: Spatial features for network input.
        vector: Vector features for network input.
        action_mask: Valid action mask.
    """

    env_idx: int
    search_path: list[GumbelMCTSNode]
    node: GumbelMCTSNode
    spatial: torch.Tensor
    vector: torch.Tensor
    action_mask: np.ndarray


class BatchGumbelMCTS:
    """Batch Gumbel MCTS - efficient search with batched neural network evaluation.

    This class implements Gumbel MuZero MCTS with batched inference across multiple
    parallel environments. Instead of evaluating each leaf node individually, it
    accumulates leaf nodes from all environments and evaluates them in a single
    batched forward pass.

    The algorithm follows these steps:
    1. Initialize root nodes for all environments (batched evaluation)
    2. Expand roots and sample candidate actions using Gumbel sampling
    3. For each simulation round:
       - Traverse trees to find leaf nodes (parallel across envs)
       - Batch all leaf evaluations into single forward pass
       - Expand leaves and backpropagate values
    4. Select best actions using sequential halving or visit counts
    """

    def __init__(
        self,
        network: MCTSNetwork,
        config: BatchMCTSConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize Batch Gumbel MCTS.

        Args:
            network: Neural network implementing MCTSNetwork protocol.
            config: Batch MCTS configuration.
            device: Device for neural network inference.
        """
        self.network = network
        self.config = config or BatchMCTSConfig()
        # Resolve the request and place the network in one step: storing an
        # unresolved "cuda" here while the network sat on cpu was a real
        # source of mixed-device RuntimeErrors.
        self.device = place_network(network, device)
        # This search owns its randomness. Drawing from the global `np.random`
        # state meant a caller could seed its environments and still get a
        # different search on every run -- the defect that made a "seeded"
        # self-play run only half determined.
        self._rng = np.random.default_rng(self.config.seed)

        # What the most recent `batch_search` actually spent, per environment and
        # summed. Measured, never copied from the plan.
        self.last_batch_search_stats: list[SearchStats] = []
        self.last_search_stats = SearchStats(root_evals=0, sim_evals=0)
        self._env_root_evals: list[int] = []
        self._env_sim_evals: list[int] = []

        # Metrics tracking for W&B logging
        self._search_count = 0
        self._track_to_play = True
        self._log_interval = LOG_SEARCH_STATS_INTERVAL

    def search(
        self,
        env: MCTSEnvironment,
        add_noise: bool = False,  # Not used - Gumbel provides stochasticity
    ) -> tuple[np.ndarray, float]:
        """Run Gumbel MCTS search from current state (single environment).

        This is a convenience wrapper for MCTSProtocol compliance. For batched
        search, use batch_search() directly.

        Args:
            env: Game environment.
            add_noise: Ignored - Gumbel sampling provides exploration.

        Returns:
            Tuple of (action_probs, root_value).
        """
        results = self.batch_search([env], add_noise=add_noise)
        return results[0]

    def batch_search(
        self,
        envs: list[MCTSEnvironment],
        add_noise: bool = False,  # Not used - Gumbel provides stochasticity
    ) -> list[tuple[np.ndarray, float]]:
        """Run Gumbel MCTS search across multiple environments in parallel.

        Args:
            envs: List of game environments.
            add_noise: Ignored - Gumbel sampling provides exploration.

        Returns:
            List of (action_probs, root_value) tuples, one per environment.

        Raises:
            ValueError: If ``envs`` is empty.
        """
        if not envs:
            # Previously an accidental `max()` of an empty sequence inside the
            # halving loop; now an explicit contract.
            raise ValueError("batch_search requires at least one environment")
        start_time = time.perf_counter()
        num_envs = len(envs)
        total_evals = 0
        self._env_root_evals = [0] * num_envs
        self._env_sim_evals = [0] * num_envs

        # Step 1: Initialize root nodes and get initial evaluations (batched)
        # Above two players a scalar sign flip cannot express the game; fall back
        # to alternation there, matching Civ6MCTS and GumbelMCTS.
        self._track_to_play = all(should_track_to_play(env) for env in envs)
        roots: list[GumbelMCTSNode] = [
            GumbelMCTSNode(to_play=env_current_player(env) if self._track_to_play else None)
            for env in envs
        ]
        observations = [env.get_observation() for env in envs]

        # Batch evaluate initial states
        spatial_list = [
            torch.from_numpy(obs["spatial"]).float().to(self.device) for obs in observations
        ]
        vector_list = [
            torch.from_numpy(obs["vector"]).float().to(self.device) for obs in observations
        ]
        mask_list = [
            torch.from_numpy(obs["action_mask"]).float().to(self.device) for obs in observations
        ]

        eval_start = time.perf_counter()
        initial_results = self._batch_evaluate(spatial_list, vector_list, mask_list)
        eval_time = (time.perf_counter() - eval_start) * 1000
        for i in range(len(initial_results)):
            self._env_root_evals[i] += 1
        total_evals += len(initial_results)

        # Step 2: Expand roots and sample candidate actions
        candidate_actions_list: list[np.ndarray] = []
        gumbel_scores_list: list[np.ndarray] = []
        root_values: list[float] = []

        for i, (_env, root, (policy, value)) in enumerate(
            zip(envs, roots, initial_results, strict=True)
        ):
            action_mask = observations[i]["action_mask"]
            root.expand(policy, action_mask)

            if not root.is_expanded():
                # No valid actions - return empty policy
                candidate_actions_list.append(np.array([], dtype=np.int64))
                gumbel_scores_list.append(np.array([], dtype=np.float32))
                root_values.append(0.0)
                continue

            # Gumbel sampling at root
            sampled_actions, gumbel_scores = gumbel_sample_actions(
                policy,
                action_mask,
                self.config.num_sampled_actions,
                temperature=self.config.temperature,
                rng=self._rng,
            )

            # Store gumbel scores in children
            for action, score in zip(sampled_actions, gumbel_scores, strict=False):
                if action in root.children:
                    root.children[action].gumbel_score = score

            candidate_actions_list.append(sampled_actions)
            gumbel_scores_list.append(gumbel_scores)
            root_values.append(value)

        # Step 3: Run simulations with batched evaluation
        if self.config.use_sequential_halving:
            best_actions = self._batch_sequential_halving(
                envs, roots, candidate_actions_list, gumbel_scores_list, root_values
            )
        else:
            best_actions = self._batch_simple_search(envs, roots, candidate_actions_list)

        # Count total evaluations as measured, not as configured.
        self.last_batch_search_stats = [
            SearchStats(root_evals=roots, sim_evals=sims)
            for roots, sims in zip(self._env_root_evals, self._env_sim_evals, strict=True)
        ]
        self.last_search_stats = SearchStats(
            root_evals=sum(self._env_root_evals), sim_evals=sum(self._env_sim_evals)
        )
        total_evals += self.last_search_stats.sim_evals

        # Step 4: Construct one-hot policies
        results: list[tuple[np.ndarray, float]] = []
        for env, root, best_action in zip(envs, roots, best_actions, strict=True):
            action_probs = np.zeros(env.action_space.n, dtype=np.float32)
            if best_action >= 0 and best_action < env.action_space.n:
                action_probs[best_action] = 1.0
            results.append((action_probs, root.value))

        # Log metrics periodically
        total_time = (time.perf_counter() - start_time) * 1000
        self._search_count += 1
        if self._search_count % self._log_interval == 0:
            avg_batch_size = total_evals / max(self.config.num_simulations, 1)
            log_metrics(
                {
                    "batch_mcts/actual_batch_size": avg_batch_size,
                    "batch_mcts/network_eval_ms": eval_time / max(num_envs, 1),
                    "batch_mcts/total_search_ms": total_time / max(num_envs, 1),
                    "batch_mcts/evals_per_search": total_evals / max(num_envs, 1),
                }
            )

        return results

    def _batch_sequential_halving(
        self,
        envs: list[MCTSEnvironment],
        roots: list[GumbelMCTSNode],
        candidate_actions_list: list[np.ndarray],
        gumbel_scores_list: list[np.ndarray],
        root_values: list[float],
    ) -> list[int]:
        """Run sequential halving across all environments with batched evaluation.

        Args:
            envs: List of game environments.
            roots: List of root nodes.
            candidate_actions_list: Candidate actions per environment.
            gumbel_scores_list: Gumbel scores per environment.
            root_values: Initial value estimates per environment.

        Returns:
            List of best action indices, one per environment.
        """
        num_envs = len(envs)

        # Per-env state. Each environment follows its OWN schedule from
        # gumbel_ops: the round count used to be derived from the largest
        # candidate set in the batch, so the same position spent a different
        # budget depending on what it was searched beside.
        active_actions_list: list[list[int]] = []
        scores_list: list[dict[int, float]] = []
        schedules: list[tuple[HalvingRound, ...]] = []

        for i in range(num_envs):
            actions = [int(a) for a in candidate_actions_list[i]]
            scores = {
                int(a): float(g)
                for a, g in zip(candidate_actions_list[i], gumbel_scores_list[i], strict=False)
            }
            active_actions_list.append(actions)
            scores_list.append(scores)
            schedules.append(
                halving_schedule(
                    self.config.num_sampled_actions,
                    self.config.num_simulations,
                    len(roots[i].children),
                    min_survivors=self.config.min_actions_after_halving,
                )
                if actions
                else ()
            )

        if not any(active_actions_list):
            return [0] * num_envs

        num_rounds = max(len(schedule) for schedule in schedules)

        # Run halving rounds. Round r of every env is evaluated in two batches:
        # the full passes, then -- after re-ranking on their evidence -- the odd
        # visits (mctx re-scores before every simulation).
        for round_idx in range(num_rounds):
            in_round = [
                env_idx for env_idx in range(num_envs) if round_idx < len(schedules[env_idx])
            ]

            pending_evals: list[_PendingEvaluation] = []
            for env_idx in in_round:
                halving_round = schedules[env_idx][round_idx]
                survivors = active_actions_list[env_idx][: halving_round.survivors]
                active_actions_list[env_idx] = survivors
                for _ in range(halving_round.full_passes):
                    for action in survivors:
                        pending = self._run_simulation_for_env(
                            env_idx, envs[env_idx], roots[env_idx], np.array([action])
                        )
                        if pending is not None:
                            pending_evals.append(pending)
            self._evaluate_pending(pending_evals)

            pending_evals = []
            for env_idx in in_round:
                halving_round = schedules[env_idx][round_idx]
                if not halving_round.remainder:
                    continue
                survivors = self._rank_by_completed_q(
                    roots[env_idx],
                    active_actions_list[env_idx],
                    scores_list[env_idx],
                    root_values[env_idx],
                )
                active_actions_list[env_idx] = survivors
                for action in survivors[: halving_round.remainder]:
                    pending = self._run_simulation_for_env(
                        env_idx, envs[env_idx], roots[env_idx], np.array([action])
                    )
                    if pending is not None:
                        pending_evals.append(pending)
            self._evaluate_pending(pending_evals)

            # After the round, rank each participating env's survivors by
            # completed Q plus Gumbel score; the next round takes the top ones.
            for env_idx in in_round:
                active_actions_list[env_idx] = self._rank_by_completed_q(
                    roots[env_idx],
                    active_actions_list[env_idx],
                    scores_list[env_idx],
                    root_values[env_idx],
                )

        # Return best action for each env
        best_actions = []
        for env_idx in range(num_envs):
            actions = active_actions_list[env_idx]
            if len(actions) == 0:
                best_actions.append(0)
            else:
                root = roots[env_idx]
                best_actions.append(
                    choose_among_most_visited(
                        actions,
                        [
                            root.children[a].visit_count if a in root.children else 0
                            for a in actions
                        ],
                    )
                )

        return best_actions

    def _evaluate_pending(self, pending_evals: list[_PendingEvaluation]) -> None:
        """Evaluate queued leaves in one batched forward pass, expand and back up."""
        if not pending_evals:
            return
        spatial_list = [p.spatial for p in pending_evals]
        vector_list = [p.vector for p in pending_evals]
        mask_list = [torch.from_numpy(p.action_mask).float().to(self.device) for p in pending_evals]
        results = self._batch_evaluate(spatial_list, vector_list, mask_list)
        for pending, (policy, value) in zip(pending_evals, results, strict=True):
            self._env_sim_evals[pending.env_idx] += 1
            pending.node.expand(policy, pending.action_mask)
            self._backpropagate(pending.search_path, value)

    @staticmethod
    def _rank_by_completed_q(
        root: GumbelMCTSNode,
        actions: list[int],
        scores: dict[int, float],
        root_value: float,
    ) -> list[int]:
        """Order ``actions`` best-first by completed Q plus Gumbel score.

        Keeps this path's historical unconditional ``-child.value`` (CLAUDE.md
        §5.3; the sign rule is task T18).
        """
        max_visits = max(
            (root.children[a].visit_count for a in actions if a in root.children), default=0
        )
        combined: dict[int, float] = {}
        for action in actions:
            completed = 0.0
            child = root.children.get(action)
            if child is not None:
                q = -child.value if child.visit_count > 0 else 0.0
                completed = compute_completed_q(q, child.visit_count, max_visits, root_value)
            combined[action] = completed + scores.get(action, 0.0)
        return sorted(actions, key=lambda a: -combined[a])

    def _batch_simple_search(
        self,
        envs: list[MCTSEnvironment],
        roots: list[GumbelMCTSNode],
        candidate_actions_list: list[np.ndarray],
    ) -> list[int]:
        """Simple search without sequential halving (batched evaluation).

        Args:
            envs: List of game environments.
            roots: List of root nodes.
            candidate_actions_list: Candidate actions per environment.

        Returns:
            List of best action indices, one per environment.
        """
        num_envs = len(envs)

        # Run simulations
        for _ in range(self.config.num_simulations):
            pending_evals: list[_PendingEvaluation] = []

            # Collect pending evaluations from all envs
            for env_idx in range(num_envs):
                pending = self._run_simulation_for_env(
                    env_idx, envs[env_idx], roots[env_idx], candidate_actions_list[env_idx]
                )
                if pending is not None:
                    pending_evals.append(pending)

            # Batch evaluate all pending leaves
            if pending_evals:
                spatial_list = [p.spatial for p in pending_evals]
                vector_list = [p.vector for p in pending_evals]
                mask_list = [
                    torch.from_numpy(p.action_mask).float().to(self.device) for p in pending_evals
                ]

                results = self._batch_evaluate(spatial_list, vector_list, mask_list)

                # Expand and backpropagate
                for pending, (policy, value) in zip(pending_evals, results, strict=True):
                    self._env_sim_evals[pending.env_idx] += 1
                    pending.node.expand(policy, pending.action_mask)
                    self._backpropagate(pending.search_path, value)

        # Select action with highest visit count
        best_actions = []
        for env_idx in range(num_envs):
            candidates = candidate_actions_list[env_idx]
            if len(candidates) == 0:
                best_actions.append(0)
            else:
                root = roots[env_idx]
                best_action = max(
                    candidates,
                    key=lambda a: root.children[a].visit_count if a in root.children else 0,
                )
                best_actions.append(int(best_action))

        return best_actions

    def _run_simulation_for_env(
        self,
        env_idx: int,
        env: MCTSEnvironment,
        root: GumbelMCTSNode,
        candidate_actions: np.ndarray,
    ) -> _PendingEvaluation | None:
        """Run tree traversal until a leaf is found (or terminal state).

        Args:
            env_idx: Environment index.
            env: Game environment.
            root: Root node.
            candidate_actions: Candidate actions to consider.

        Returns:
            Pending evaluation info if leaf found, None if terminal state reached.
        """
        if len(candidate_actions) == 0:
            return None

        # Select action to simulate (prefer least visited)
        min_visits = float("inf")
        best_action = int(candidate_actions[0])

        for action in candidate_actions:
            if int(action) in root.children:
                visits = root.children[int(action)].visit_count
                if visits < min_visits:
                    min_visits = visits
                    best_action = int(action)

        # Clone environment and take initial action
        sim_env = env.clone()
        _, _reward, terminated, truncated, _ = sim_env.step(best_action)

        node = root.children.get(best_action)
        if node is None:
            return None

        # A node describes the state after its action.
        node.to_play = env_current_player(sim_env) if self._track_to_play else None
        search_path = [root, node]

        # Traverse tree until leaf or terminal
        while not (terminated or truncated):
            if node.is_expanded():
                # Selection: use PUCT
                action, next_node = node.select_child(self.config.c_visit / DEFAULT_C_VISIT_DIVISOR)
                if next_node is None:
                    break
                node = next_node
                search_path.append(node)
                _, _reward, terminated, truncated, _ = sim_env.step(action)
                node.to_play = env_current_player(sim_env) if self._track_to_play else None
            else:
                # Found a leaf - return pending evaluation
                obs = sim_env.get_observation()
                spatial = torch.from_numpy(obs["spatial"]).float().to(self.device)
                vector = torch.from_numpy(obs["vector"]).float().to(self.device)
                action_mask = obs["action_mask"]

                return _PendingEvaluation(
                    env_idx=env_idx,
                    search_path=search_path,
                    node=node,
                    spatial=spatial,
                    vector=vector,
                    action_mask=action_mask,
                )

        # Terminal state - the game's outcome is this simulation's leaf evaluation.
        info = sim_env.get_info()
        winner = info.get("winner")
        if winner is None:
            value = 0.0
        elif winner == sim_env.current_player:
            value = 1.0
        else:
            value = -1.0

        self._env_sim_evals[env_idx] += 1
        self._backpropagate(search_path, value)
        return None

    def _batch_evaluate(
        self,
        spatial_list: list[torch.Tensor],
        vector_list: list[torch.Tensor],
        mask_list: list[torch.Tensor],
    ) -> list[tuple[np.ndarray, float]]:
        """Evaluate multiple states in a single batched forward pass.

        Args:
            spatial_list: List of spatial feature tensors.
            vector_list: List of vector feature tensors.
            mask_list: List of action mask tensors.

        Returns:
            List of (policy, value) tuples, one per input state.
        """
        if len(spatial_list) == 0:
            return []

        self.network.eval()
        with torch.no_grad():
            # Stack inputs into batch
            spatial_batch = torch.stack(spatial_list)
            vector_batch = torch.stack(vector_list)
            mask_batch = torch.stack(mask_list)

            # Single forward pass
            policy_probs, value_tensor = self.network.predict(
                spatial_batch,
                vector_batch,
                mask_batch,
            )

            # Split outputs
            results = []
            for i in range(len(spatial_list)):
                policy = policy_probs[i].cpu().numpy()
                value = float(value_tensor[i].item())
                results.append((policy, value))

        return results

    def _backpropagate(
        self,
        search_path: list[GumbelMCTSNode],
        value: float,
    ) -> None:
        """Backpropagate value through search path.

        Args:
            search_path: List of nodes from root to leaf.
            value: Value to propagate (from leaf player's perspective).
        """
        log_backup_diagnostics(search_path, search_name="BatchGumbelMCTS")
        backpropagate_path(search_path, value)

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float = 0.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Select action from probabilities.

        For Gumbel MCTS, action_probs is typically one-hot from search(),
        so this usually just returns the argmax.

        Args:
            action_probs: Action probability distribution.
            temperature: Temperature for selection (0 = greedy).
            action_mask: Optional valid action mask (1=valid, 0=invalid).

        Returns:
            Selected action index.

        Raises:
            ValueError: If no valid actions available when mask is provided.
        """
        # Apply mask if provided
        if action_mask is not None:
            action_probs = action_probs * action_mask
            if action_probs.sum() < _PROBABILITY_SUM_EPSILON:
                # Fallback: select first valid action
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) == 0:
                    raise ValueError("No valid actions available")
                return int(valid_actions[0])
            # Renormalize after masking
            action_probs = action_probs / action_probs.sum()

        if temperature == 0 or action_probs.max() > _GREEDY_THRESHOLD:
            # Greedy selection (usual case for Gumbel MCTS)
            return int(np.argmax(action_probs))

        # Apply temperature for exploration
        probs = np.power(action_probs + EPSILON, 1.0 / temperature)
        probs = probs / probs.sum()

        return int(self._rng.choice(len(probs), p=probs))
