"""Gumbel MuZero MCTS -- strong play with 4-32 simulations.

Deliberately independent of :mod:`alpha_zero.core.mcts.base`: this variant
carries its own node and configuration types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from alpha_zero.core.mcts.gumbel_ops import compute_completed_q, gumbel_sample_actions
from alpha_zero.core.mcts.mcts_protocols import MCTSEnvironment, MCTSNetwork
from alpha_zero.core.mcts_constants import (
    DEFAULT_C_VISIT,
    DEFAULT_C_VISIT_DIVISOR,
    EPSILON,
    GREEDY_THRESHOLD,
)
from alpha_zero.utils.constants import MCTS
from alpha_zero.utils.device import place_network


@dataclass(slots=True)
class GumbelMCTSConfig:
    """Configuration for Gumbel MuZero MCTS.

    Gumbel MuZero achieves strong performance with fewer simulations than
    classic MCTS by using Gumbel sampling and sequential halving for efficient
    exploration. Default 64 sims balances quality and speed.

    Attributes:
        num_simulations: Total simulation budget (default 64, can be as low as 4).
        num_sampled_actions: Number of actions to sample at root (top-k).
        max_considered_actions: Maximum actions to consider before Gumbel sampling.
        c_visit: Controls value vs visit count trade-off in completed Q.
        c_scale: Scale factor for value estimates.
        use_sequential_halving: Use sequential halving for simulation allocation.
        min_actions_after_halving: Stop halving when this many actions remain.
        temperature: Temperature for final action selection (usually 0 for greedy).
        temperature_threshold: Turn after which to use temperature=0.
    """

    num_simulations: int = MCTS.GUMBEL_NUM_SIMULATIONS
    num_sampled_actions: int = MCTS.GUMBEL_NUM_SAMPLED_ACTIONS
    max_considered_actions: int = MCTS.GUMBEL_MAX_CONSIDERED_ACTIONS
    c_visit: float = DEFAULT_C_VISIT
    c_scale: float = MCTS.GUMBEL_C_SCALE
    use_sequential_halving: bool = True
    min_actions_after_halving: int = MCTS.GUMBEL_MIN_ACTIONS_AFTER_HALVING
    temperature: float = MCTS.DEFAULT_TEMPERATURE
    temperature_threshold: int = MCTS.DEFAULT_TEMPERATURE_THRESHOLD

    def __post_init__(self) -> None:
        if not MCTS.MIN_SIMULATIONS <= self.num_simulations <= MCTS.MAX_SIMULATIONS:
            raise ValueError(
                f"num_simulations must be in [{MCTS.MIN_SIMULATIONS}, "
                f"{MCTS.MAX_SIMULATIONS}]; got {self.num_simulations}"
            )
        if self.num_sampled_actions <= 0:
            raise ValueError(f"num_sampled_actions must be > 0; got {self.num_sampled_actions}")
        if self.max_considered_actions < self.num_sampled_actions:
            raise ValueError(
                "max_considered_actions must be >= num_sampled_actions; "
                f"got {self.max_considered_actions} < {self.num_sampled_actions}"
            )
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0; got {self.temperature}")
        if self.temperature_threshold < 0:
            raise ValueError(
                f"temperature_threshold must be >= 0; got {self.temperature_threshold}"
            )


@dataclass(slots=True)
class GumbelMCTSNode:
    """Node in the Gumbel MCTS tree.

    Similar to MCTSNode but includes additional fields for Gumbel MuZero:
    - gumbel_score: The Gumbel score (log(pi) + g) for root actions.
    - completed_q: The completed Q-value with sigma penalty.

    Attributes:
        prior: Prior probability from neural network.
        visit_count: Number of times node was visited.
        value_sum: Sum of values from all visits.
        children: Child nodes keyed by action.
        parent: Parent node.
        action: Action that led to this node.
        gumbel_score: Gumbel score for this action (root only).
    """

    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, GumbelMCTSNode] = field(default_factory=dict)
    parent: GumbelMCTSNode | None = None
    action: int | None = None
    gumbel_score: float = 0.0

    @property
    def value(self) -> float:
        """Mean value estimate (Q-value)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0

    def select_child(self, c_puct: float) -> tuple[int, GumbelMCTSNode | None]:
        """Select child with highest UCB score (for internal tree traversal).

        Args:
            c_puct: Exploration constant.

        Returns:
            Tuple of (action, child_node).
        """
        best_score = -float("inf")
        best_action = -1
        best_child = None

        sqrt_parent_visits = np.sqrt(max(self.visit_count, 1))

        for action, child in self.children.items():
            # PUCT formula (negamax: child's value is opponent's)
            q_value = -child.value if child.visit_count > 0 else 0.0
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None, "select_child called on node with no children"
        return best_action, best_child

    def expand(self, policy: np.ndarray, action_mask: np.ndarray) -> None:
        """Expand node with children for valid actions.

        Args:
            policy: Policy probabilities for all actions.
            action_mask: Mask of valid actions.
        """
        valid_actions = np.where(action_mask > 0)[0]

        # Normalize policy over valid actions
        valid_policy = policy[valid_actions]
        policy_sum = valid_policy.sum()
        if policy_sum > 0:
            valid_policy = valid_policy / policy_sum
        else:
            valid_policy = np.ones_like(valid_policy) / len(valid_policy)

        # Create child nodes
        for i, action in enumerate(valid_actions):
            self.children[int(action)] = GumbelMCTSNode(
                prior=float(valid_policy[i]),
                parent=self,
                action=int(action),
            )

    def backpropagate(self, value: float) -> None:
        """Backpropagate value through tree.

        Args:
            value: Value to propagate (from current player's perspective).
        """
        node: GumbelMCTSNode | None = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            node = node.parent


class GumbelMCTS:
    """Gumbel MuZero MCTS - efficient search with very few simulations.

    Gumbel MuZero achieves strong performance with only 4-32 simulations
    (compared to 800+ in standard AlphaZero) by:
    1. Using Gumbel sampling to select promising actions at the root.
    2. Using sequential halving to allocate simulations efficiently.
    3. Computing "completed" Q-values to fairly compare actions with
       different visit counts.

    Reference: "Policy Improvement by Planning with Gumbel" (Danihelka et al., 2022)
    """

    def __init__(
        self,
        network: MCTSNetwork,
        config: GumbelMCTSConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize Gumbel MCTS.

        Args:
            network: Neural network implementing MCTSNetwork protocol.
            config: Gumbel MCTS configuration.
            device: Device for neural network inference.
        """
        self.network = network
        self.config = config or GumbelMCTSConfig()
        # Resolve the request and place the network in one step: storing an
        # unresolved "cuda" here while the network sat on cpu was a real
        # source of mixed-device RuntimeErrors.
        self.device = place_network(network, device)

    def search(
        self,
        env: MCTSEnvironment,
        add_noise: bool = False,  # Not used - Gumbel provides stochasticity
    ) -> tuple[np.ndarray, float]:
        """Run Gumbel MCTS search from current state.

        Unlike standard MCTS which returns visit-count-based probabilities,
        Gumbel MCTS returns a one-hot policy for the selected action (when
        using sequential halving) or softmax over completed Q-values.

        Args:
            env: Game environment (must support clone()).
            add_noise: Ignored - Gumbel sampling provides exploration.

        Returns:
            Tuple of (action_probs, root_value).
        """
        # Create root node
        root = GumbelMCTSNode()

        # Get initial policy and value from network
        obs = env.get_observation()
        spatial = torch.from_numpy(obs["spatial"]).float().to(self.device)
        vector = torch.from_numpy(obs["vector"]).float().to(self.device)
        action_mask = obs["action_mask"]

        policy, value = self._evaluate(spatial, vector, action_mask)

        # Expand root
        root.expand(policy, action_mask)

        if not root.is_expanded():
            # No valid actions - return empty policy
            return np.zeros(env.action_space.n, dtype=np.float32), 0.0

        # Phase 1: Gumbel sampling at root
        sampled_actions, gumbel_scores = gumbel_sample_actions(
            policy,
            action_mask,
            self.config.num_sampled_actions,
            temperature=self.config.temperature,
        )

        if len(sampled_actions) == 0:
            return np.zeros(env.action_space.n, dtype=np.float32), 0.0

        # Store gumbel scores in children
        for action, score in zip(sampled_actions, gumbel_scores, strict=False):
            if action in root.children:
                root.children[action].gumbel_score = score

        # Phase 2: Run simulations
        if self.config.use_sequential_halving:
            best_action = self._sequential_halving(env, root, sampled_actions, gumbel_scores, value)
        else:
            best_action = self._simple_search(env, root, sampled_actions)

        # Return one-hot policy for selected action
        action_probs = np.zeros(env.action_space.n, dtype=np.float32)
        action_probs[best_action] = 1.0

        return action_probs, root.value

    def _sequential_halving(
        self,
        env: MCTSEnvironment,
        root: GumbelMCTSNode,
        candidate_actions: np.ndarray,
        gumbel_scores: np.ndarray,
        root_value: float,
    ) -> int:
        """Allocate simulations using sequential halving.

        Sequential halving progressively eliminates less promising actions,
        concentrating simulations on the best candidates.

        Args:
            env: Game environment.
            root: Root node of the search tree.
            candidate_actions: Actions selected by Gumbel sampling.
            gumbel_scores: Gumbel scores for candidate actions.
            root_value: Initial value estimate at root.

        Returns:
            Best action index.
        """
        actions = list(candidate_actions)
        scores = dict(zip(candidate_actions, gumbel_scores, strict=False))

        if len(actions) == 0:
            return 0

        if len(actions) == 1:
            return int(actions[0])

        # Calculate number of halving rounds
        num_rounds = max(1, int(np.ceil(np.log2(len(actions)))))
        sims_per_round = max(1, self.config.num_simulations // num_rounds)

        for _round_idx in range(num_rounds):
            if len(actions) <= self.config.min_actions_after_halving:
                break

            # Allocate simulations evenly among remaining actions
            sims_per_action = max(1, sims_per_round // len(actions))

            # Run simulations for each remaining action
            for action in actions:
                for _ in range(sims_per_action):
                    self._run_simulation_from_action(env, root, action)

            # Compute completed Q-values
            max_visits = max(root.children[a].visit_count for a in actions if a in root.children)

            completed_qs = {}
            for action in actions:
                if action not in root.children:
                    continue
                child = root.children[action]
                q = -child.value if child.visit_count > 0 else 0.0
                completed_qs[action] = compute_completed_q(
                    q, child.visit_count, max_visits, root_value
                )

            # Combine with Gumbel scores and halve
            combined = {a: completed_qs.get(a, 0.0) + scores.get(a, 0.0) for a in actions}
            actions = sorted(actions, key=lambda a: -combined[a])
            actions = actions[: max(len(actions) // 2, self.config.min_actions_after_halving)]

        return int(actions[0])

    def _simple_search(
        self,
        env: MCTSEnvironment,
        root: GumbelMCTSNode,
        candidate_actions: np.ndarray,
    ) -> int:
        """Simple search without sequential halving.

        Distributes simulations evenly, then selects by visit count.

        Args:
            env: Game environment.
            root: Root node.
            candidate_actions: Actions to consider.

        Returns:
            Best action index.
        """
        if len(candidate_actions) == 0:
            return 0

        # Run simulations
        for _ in range(self.config.num_simulations):
            # Select action to simulate (round-robin or by visit count)
            action = self._select_simulation_action(root, candidate_actions)
            self._run_simulation_from_action(env, root, action)

        # Select action with highest visit count
        best_action = max(
            candidate_actions,
            key=lambda a: root.children[a].visit_count if a in root.children else 0,
        )
        return int(best_action)

    def _select_simulation_action(
        self,
        root: GumbelMCTSNode,
        candidate_actions: np.ndarray,
    ) -> int:
        """Select which action to simulate next.

        Uses a simple heuristic: prefer actions with fewer visits,
        weighted by their Gumbel score.

        Args:
            root: Root node.
            candidate_actions: Actions to choose from.

        Returns:
            Action to simulate.
        """
        # Find action with lowest visit count among candidates
        min_visits = float("inf")
        best_action = candidate_actions[0]

        for action in candidate_actions:
            if action in root.children:
                visits = root.children[action].visit_count
                if visits < min_visits:
                    min_visits = visits
                    best_action = action

        return int(best_action)

    def _run_simulation_from_action(
        self,
        env: MCTSEnvironment,
        root: GumbelMCTSNode,
        action: int,
    ) -> None:
        """Run a single MCTS simulation starting with the given action.

        Args:
            env: Game environment.
            root: Root node.
            action: Action to take from root.
        """
        sim_env = env.clone()

        # Take the initial action
        _, _reward, terminated, truncated, _ = sim_env.step(action)

        node = root.children.get(action)
        if node is None:
            return

        search_path = [root, node]

        # Continue traversing/expanding tree
        while not (terminated or truncated):
            if node.is_expanded():
                # Selection: use PUCT
                action, next_node = node.select_child(self.config.c_visit / DEFAULT_C_VISIT_DIVISOR)
                if next_node is None:
                    break  # No valid children to select
                node = next_node
                search_path.append(node)
                _, _reward, terminated, truncated, _ = sim_env.step(action)
            else:
                # Expansion and evaluation
                obs = sim_env.get_observation()
                spatial = torch.from_numpy(obs["spatial"]).float().to(self.device)
                vector = torch.from_numpy(obs["vector"]).float().to(self.device)
                action_mask = obs["action_mask"]

                policy, value = self._evaluate(spatial, vector, action_mask)
                node.expand(policy, action_mask)

                # Backpropagate
                self._backpropagate(search_path, value)
                return

        # Terminal state
        info = sim_env.get_info()
        winner = info.get("winner")
        if winner is None:
            value = 0.0
        elif winner == sim_env.current_player:
            value = 1.0
        else:
            value = -1.0

        self._backpropagate(search_path, value)

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
        # Propagate from leaf to root, flipping value at each level
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _evaluate(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Evaluate state using neural network.

        Args:
            spatial: Spatial features.
            vector: Vector features.
            action_mask: Valid action mask.

        Returns:
            Tuple of (policy, value).
        """
        self.network.eval()
        with torch.no_grad():
            mask_tensor = torch.from_numpy(action_mask).float().to(self.device)
            policy_probs, value_tensor = self.network.predict(
                spatial.unsqueeze(0),
                vector.unsqueeze(0),
                mask_tensor.unsqueeze(0),
            )
            policy = policy_probs.squeeze(0).cpu().numpy()
            value = float(value_tensor.item())

        return policy, value

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
                        If provided, ensures only valid actions are selected.

        Returns:
            Selected action index.

        Raises:
            ValueError: If no valid actions available when mask is provided.
        """
        # Apply mask if provided to ensure only valid actions are selected
        if action_mask is not None:
            action_probs = action_probs * action_mask
            if action_probs.sum() < 1e-8:
                # Fallback: select first valid action
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) == 0:
                    raise ValueError("No valid actions available")
                return int(valid_actions[0])
            # Renormalize after masking
            action_probs = action_probs / action_probs.sum()

        if temperature == 0 or action_probs.max() > GREEDY_THRESHOLD:
            # Greedy selection (usual case for Gumbel MCTS)
            return int(np.argmax(action_probs))

        # Apply temperature for exploration
        probs = np.power(action_probs + EPSILON, 1.0 / temperature)
        probs = probs / probs.sum()

        return int(np.random.choice(len(probs), p=probs))
