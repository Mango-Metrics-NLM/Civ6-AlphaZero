"""Monte Carlo Tree Search for Civ6 AlphaZero.

Implements PUCT-based MCTS with neural network guidance, supporting:
- Action masking for valid moves
- Dirichlet noise for exploration
- Temperature-based action selection

Also implements Gumbel MuZero MCTS for efficient search with few simulations:
- Gumbel-Top-k sampling for root action selection
- Sequential halving for simulation allocation
- Works effectively with 4-32 simulations instead of 200+
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import torch


class MCTSNetwork(Protocol):
    """Protocol for neural networks used in MCTS."""

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy and value from state.

        Args:
            spatial: Spatial features (B, C, H, W).
            vector: Vector features (B, D).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_probs, value).
        """
        ...


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""

    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30  # Turn after which to use temperature=0


@dataclass
class GumbelMCTSConfig:
    """Configuration for Gumbel MuZero MCTS.

    Gumbel MuZero achieves strong performance with very few simulations (4-32)
    by using Gumbel sampling and sequential halving for efficient exploration.

    Attributes:
        num_simulations: Total simulation budget (default 16, can be as low as 4).
        num_sampled_actions: Number of actions to sample at root (top-k).
        max_considered_actions: Maximum actions to consider before Gumbel sampling.
        c_visit: Controls value vs visit count trade-off in completed Q.
        c_scale: Scale factor for value estimates.
        use_sequential_halving: Use sequential halving for simulation allocation.
        min_actions_after_halving: Stop halving when this many actions remain.
        temperature: Temperature for final action selection (usually 0 for greedy).
        temperature_threshold: Turn after which to use temperature=0.
    """

    num_simulations: int = 16
    num_sampled_actions: int = 16
    max_considered_actions: int = 32
    c_visit: float = 50.0
    c_scale: float = 1.0
    use_sequential_halving: bool = True
    min_actions_after_halving: int = 1
    temperature: float = 1.0
    temperature_threshold: int = 30


@dataclass
class MCTSNode:
    """Node in the MCTS tree.

    Attributes:
        prior: Prior probability from neural network.
        visit_count: Number of times node was visited.
        value_sum: Sum of values from all visits.
        children: Child nodes keyed by action.
        parent: Parent node.
        action: Action that led to this node.
    """

    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, MCTSNode] = field(default_factory=dict)
    parent: MCTSNode | None = None
    action: int | None = None

    @property
    def value(self) -> float:
        """Mean value estimate."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0

    def select_child(self, c_puct: float) -> tuple[int, MCTSNode]:
        """Select child with highest UCB score.

        Args:
            c_puct: Exploration constant.

        Returns:
            Tuple of (action, child_node).
        """
        best_score = -float("inf")
        best_action = -1
        best_child = None

        sqrt_parent_visits = np.sqrt(self.visit_count)

        for action, child in self.children.items():
            # PUCT formula
            q_value = -child.value  # Negamax: child's value is opponent's
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, policy: np.ndarray, action_mask: np.ndarray) -> None:
        """Expand node with children for valid actions.

        Args:
            policy: Policy probabilities for all actions.
            action_mask: Mask of valid actions.
        """
        # Get valid actions
        valid_actions = np.where(action_mask > 0)[0]

        # Normalize policy over valid actions
        valid_policy = policy[valid_actions]
        valid_policy = valid_policy / (valid_policy.sum() + 1e-8)

        # Create child nodes
        for i, action in enumerate(valid_actions):
            self.children[int(action)] = MCTSNode(
                prior=float(valid_policy[i]),
                parent=self,
                action=int(action),
            )

    def backpropagate(self, value: float) -> None:
        """Backpropagate value through tree.

        Args:
            value: Value to propagate.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            node = node.parent


class Civ6MCTS:
    """Monte Carlo Tree Search for Civ6.

    Uses neural network for policy prior and value estimation.
    Supports action masking for valid moves.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        config: MCTSConfig | None = None,
        device: str = "cpu",
    ):
        """Initialize MCTS.

        Args:
            network: Neural network for evaluation.
            config: MCTS configuration.
            device: Device for neural network inference.
        """
        self.network = network
        self.config = config or MCTSConfig()
        self.device = device

    def search(
        self,
        env,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Run MCTS search from current state.

        Args:
            env: Game environment (must support clone()).
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            Tuple of (action_probs, root_value).
        """
        # Create root node
        root = MCTSNode()

        # Get initial policy and value from network
        obs = env._get_observation()
        spatial = torch.from_numpy(obs["spatial"]).float().to(self.device)
        vector = torch.from_numpy(obs["vector"]).float().to(self.device)
        action_mask = obs["action_mask"]

        policy, value = self._evaluate(spatial, vector, action_mask)

        # Expand root
        root.expand(policy, action_mask)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.is_expanded():
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            sim_env = env.clone()

            # Selection: traverse tree to leaf
            search_path = [node]
            while node.is_expanded():
                action, node = node.select_child(self.config.c_puct)
                search_path.append(node)

                # Take action in environment
                _, _, terminated, truncated, _ = sim_env.step(action)
                if terminated or truncated:
                    break

            # Evaluation
            if terminated or truncated:
                # Game over: use actual result
                info = sim_env._get_info()
                winner = info.get("winner")
                if winner is None:
                    value = 0.0  # Draw
                elif winner == sim_env.current_player:
                    value = 1.0
                else:
                    value = -1.0
            else:
                # Use neural network
                obs = sim_env._get_observation()
                spatial = torch.from_numpy(obs["spatial"]).float().to(self.device)
                vector = torch.from_numpy(obs["vector"]).float().to(self.device)
                action_mask = obs["action_mask"]

                policy, value = self._evaluate(spatial, vector, action_mask)

                # Expand leaf node
                node.expand(policy, action_mask)

            # Backpropagation
            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                value = -value  # Flip for opponent

        # Get action probabilities from visit counts
        action_probs = self._get_action_probs(root, env.action_space.n)

        return action_probs, root.value

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
            policy_probs, value = self.network.predict(
                spatial.unsqueeze(0),
                vector.unsqueeze(0),
                mask_tensor.unsqueeze(0),
            )
            policy = policy_probs.squeeze(0).cpu().numpy()
            value = value.item()

        return policy, value

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """Add Dirichlet noise to root node priors.

        Args:
            node: Root node.
        """
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))

        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _get_action_probs(
        self,
        root: MCTSNode,
        action_space_size: int,
    ) -> np.ndarray:
        """Get action probabilities from root visit counts.

        Args:
            root: Root node.
            action_space_size: Total number of actions.

        Returns:
            Action probability distribution.
        """
        action_probs = np.zeros(action_space_size, dtype=np.float32)

        # Collect visit counts
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        # Normalize
        total_visits = action_probs.sum()
        if total_visits > 0:
            action_probs /= total_visits

        return action_probs

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float = 1.0,
    ) -> int:
        """Select action based on probabilities and temperature.

        Args:
            action_probs: Action probability distribution.
            temperature: Temperature for action selection.

        Returns:
            Selected action index.
        """
        if temperature == 0:
            # Greedy selection
            return int(np.argmax(action_probs))

        # Apply temperature
        if temperature != 1.0:
            probs = np.power(action_probs, 1.0 / temperature)
            probs = probs / probs.sum()
        else:
            probs = action_probs

        # Sample action
        return int(np.random.choice(len(probs), p=probs))


class SimpleMCTS:
    """Simplified MCTS for testing without neural network.

    Uses random policy and rollout for evaluation.
    """

    def __init__(
        self,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        rollout_depth: int = 20,
    ):
        """Initialize simple MCTS.

        Args:
            num_simulations: Number of MCTS simulations.
            c_puct: Exploration constant.
            rollout_depth: Maximum rollout depth.
        """
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self._rng = np.random.default_rng()

    def search(self, env) -> tuple[np.ndarray, float]:
        """Run MCTS search.

        Args:
            env: Game environment.

        Returns:
            Tuple of (action_probs, root_value).
        """
        root = MCTSNode()
        action_mask = env.get_valid_actions()

        # Initialize root with uniform policy
        num_valid = int(action_mask.sum())
        uniform_policy = action_mask / max(num_valid, 1)
        root.expand(uniform_policy, action_mask)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_env = env.clone()

            # Selection
            search_path = [node]
            while node.is_expanded():
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
                _, _, terminated, truncated, _ = sim_env.step(action)
                if terminated or truncated:
                    break

            # Evaluation
            if terminated or truncated:
                info = sim_env._get_info()
                winner = info.get("winner")
                if winner is None:
                    value = 0.0
                elif winner == sim_env.current_player:
                    value = 1.0
                else:
                    value = -1.0
            else:
                # Expand with uniform policy
                action_mask = sim_env.get_valid_actions()
                num_valid = int(action_mask.sum())
                uniform_policy = action_mask / max(num_valid, 1)
                node.expand(uniform_policy, action_mask)

                # Random rollout
                value = self._rollout(sim_env)

            # Backpropagation
            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                value = -value

        # Get action probabilities
        action_probs = np.zeros(env.action_space.n, dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        if action_probs.sum() > 0:
            action_probs /= action_probs.sum()

        return action_probs, root.value

    def _rollout(self, env) -> float:
        """Perform random rollout.

        Args:
            env: Game environment.

        Returns:
            Value estimate from rollout.
        """
        original_player = env.current_player

        for _ in range(self.rollout_depth):
            action_mask = env.get_valid_actions()
            valid_actions = np.where(action_mask > 0)[0]

            if len(valid_actions) == 0:
                break

            action = self._rng.choice(valid_actions)
            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                info = env._get_info()
                winner = info.get("winner")
                if winner is None:
                    return 0.0
                elif winner == original_player:
                    return 1.0
                else:
                    return -1.0

        # Game not finished: return 0
        return 0.0

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float = 1.0,
    ) -> int:
        """Select action based on probabilities.

        Args:
            action_probs: Action probabilities.
            temperature: Selection temperature.

        Returns:
            Selected action.
        """
        if temperature == 0:
            return int(np.argmax(action_probs))

        if temperature != 1.0:
            probs = np.power(action_probs + 1e-8, 1.0 / temperature)
            probs = probs / probs.sum()
        else:
            probs = action_probs

        return int(np.random.choice(len(probs), p=probs))


# =============================================================================
# Gumbel MuZero MCTS Implementation
# =============================================================================


def gumbel_sample_actions(
    policy: np.ndarray,
    action_mask: np.ndarray,
    num_actions: int,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample actions using Gumbel-Top-k trick.

    The Gumbel-Top-k trick provides a way to sample k elements from a categorical
    distribution without replacement, while maintaining the policy improvement
    guarantee of Gumbel MuZero.

    Args:
        policy: Policy probabilities for all actions (A,).
        action_mask: Mask of valid actions (A,).
        num_actions: Number of actions to sample (k).
        temperature: Temperature for sampling (higher = more exploration).

    Returns:
        Tuple of (selected_actions, gumbel_scores).
        - selected_actions: Indices of selected actions (k,).
        - gumbel_scores: Gumbel scores for selected actions (k,).
    """
    valid_actions = np.where(action_mask > 0)[0]

    if len(valid_actions) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # Get log probabilities for valid actions
    valid_policy = policy[valid_actions]
    # Ensure numerical stability
    valid_policy = np.clip(valid_policy, 1e-8, 1.0)
    log_probs = np.log(valid_policy)

    # Sample Gumbel noise: g ~ Gumbel(0, 1)
    # Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1)
    uniform = np.random.uniform(size=len(valid_actions))
    gumbel_noise = -np.log(-np.log(np.clip(uniform, 1e-8, 1 - 1e-8)))

    # Gumbel-Top-k: score = log(pi) / temperature + g
    scores = log_probs / temperature + gumbel_noise

    # Select top-k by score
    k = min(num_actions, len(valid_actions))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    selected_actions = valid_actions[top_indices]
    selected_scores = scores[top_indices]

    return selected_actions, selected_scores


def compute_completed_q(
    q_value: float,
    visit_count: int,
    max_visit_count: int,
    root_value: float,
) -> float:
    """Compute completed Q-value with sigma penalty for low visits.

    In Gumbel MuZero, we "complete" Q-values by hallucinating that missing
    visits would have returned the root value. This provides a principled
    way to compare actions with different visit counts.

    Args:
        q_value: Current Q-value estimate for the action.
        visit_count: Number of visits to this action.
        max_visit_count: Maximum visits among all candidate actions.
        root_value: Value estimate at the root node.

    Returns:
        Completed Q-value.
    """
    if max_visit_count == 0:
        return q_value

    # sigma = fraction of "missing" visits
    sigma = (max_visit_count - visit_count) / max_visit_count

    # Completed Q assumes missing visits would return root value
    completed_q = q_value + sigma * root_value

    return completed_q


@dataclass
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

    def select_child(self, c_puct: float) -> tuple[int, GumbelMCTSNode]:
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
        node = self
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
        network: torch.nn.Module,
        config: GumbelMCTSConfig | None = None,
        device: str = "cpu",
    ):
        """Initialize Gumbel MCTS.

        Args:
            network: Neural network for evaluation.
            config: Gumbel MCTS configuration.
            device: Device for neural network inference.
        """
        self.network = network
        self.config = config or GumbelMCTSConfig()
        self.device = device

    def search(
        self,
        env,
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
        obs = env._get_observation()
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
        for action, score in zip(sampled_actions, gumbel_scores):
            if action in root.children:
                root.children[action].gumbel_score = score

        # Phase 2: Run simulations
        if self.config.use_sequential_halving:
            best_action = self._sequential_halving(
                env, root, sampled_actions, gumbel_scores, value
            )
        else:
            best_action = self._simple_search(env, root, sampled_actions)

        # Return one-hot policy for selected action
        action_probs = np.zeros(env.action_space.n, dtype=np.float32)
        action_probs[best_action] = 1.0

        return action_probs, root.value

    def _sequential_halving(
        self,
        env,
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
        scores = dict(zip(candidate_actions, gumbel_scores))

        if len(actions) == 0:
            return 0

        if len(actions) == 1:
            return int(actions[0])

        # Calculate number of halving rounds
        num_rounds = max(1, int(np.ceil(np.log2(len(actions)))))
        sims_per_round = max(1, self.config.num_simulations // num_rounds)

        for round_idx in range(num_rounds):
            if len(actions) <= self.config.min_actions_after_halving:
                break

            # Allocate simulations evenly among remaining actions
            sims_per_action = max(1, sims_per_round // len(actions))

            # Run simulations for each remaining action
            for action in actions:
                for _ in range(sims_per_action):
                    self._run_simulation_from_action(env, root, action)

            # Compute completed Q-values
            max_visits = max(
                root.children[a].visit_count for a in actions if a in root.children
            )

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
        env,
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
        env,
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
        _, reward, terminated, truncated, _ = sim_env.step(action)

        node = root.children.get(action)
        if node is None:
            return

        search_path = [root, node]

        # Continue traversing/expanding tree
        while not (terminated or truncated):
            if node.is_expanded():
                # Selection: use PUCT
                action, node = node.select_child(self.config.c_visit / 50.0)
                search_path.append(node)
                _, reward, terminated, truncated, _ = sim_env.step(action)
            else:
                # Expansion and evaluation
                obs = sim_env._get_observation()
                spatial = torch.from_numpy(obs["spatial"]).float().to(self.device)
                vector = torch.from_numpy(obs["vector"]).float().to(self.device)
                action_mask = obs["action_mask"]

                policy, value = self._evaluate(spatial, vector, action_mask)
                node.expand(policy, action_mask)

                # Backpropagate
                self._backpropagate(search_path, value)
                return

        # Terminal state
        info = sim_env._get_info()
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
            policy_probs, value = self.network.predict(
                spatial.unsqueeze(0),
                vector.unsqueeze(0),
                mask_tensor.unsqueeze(0),
            )
            policy = policy_probs.squeeze(0).cpu().numpy()
            value = value.item()

        return policy, value

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float = 0.0,
    ) -> int:
        """Select action from probabilities.

        For Gumbel MCTS, action_probs is typically one-hot from search(),
        so this usually just returns the argmax.

        Args:
            action_probs: Action probability distribution.
            temperature: Temperature for selection (0 = greedy).

        Returns:
            Selected action index.
        """
        if temperature == 0 or action_probs.max() > 0.99:
            # Greedy selection (usual case for Gumbel MCTS)
            return int(np.argmax(action_probs))

        # Apply temperature for exploration
        probs = np.power(action_probs + 1e-8, 1.0 / temperature)
        probs = probs / probs.sum()

        return int(np.random.choice(len(probs), p=probs))
