"""Simplified MCTS implementation for AlphaZero-style training.

This module provides a lightweight Monte Carlo Tree Search implementation
that uses a neural network for policy priors and value estimation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""

    num_simulations: int = 50  # Number of simulations per move
    c_puct: float = 1.5  # Exploration constant for UCB
    temperature: float = 1.0  # Temperature for action selection
    dirichlet_alpha: float = 0.3  # Dirichlet noise alpha for root
    dirichlet_epsilon: float = 0.25  # Weight of Dirichlet noise at root
    max_depth: int = 50  # Maximum tree depth


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(
        self,
        prior: float = 0.0,
        parent: MCTSNode | None = None,
        action: int | None = None,
    ) -> None:
        """Initialize MCTS node.

        Args:
            prior: Prior probability from policy network.
            parent: Parent node.
            action: Action that led to this node.
        """
        self.prior = prior
        self.parent = parent
        self.action = action

        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, MCTSNode] = {}

        # For leaf evaluation
        self.is_terminal = False
        self.terminal_value: float | None = None

    @property
    def value(self) -> float:
        """Mean value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded."""
        return len(self.children) > 0

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """Calculate UCB score for this node.

        Args:
            parent_visit_count: Visit count of parent node.
            c_puct: Exploration constant.

        Returns:
            UCB score combining exploitation and exploration.
        """
        # Exploitation term: Q(s, a)
        q_value = self.value

        # Exploration term: c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)

        return q_value + exploration

    def select_child(self, c_puct: float) -> MCTSNode | None:
        """Select child with highest UCB score.

        Args:
            c_puct: Exploration constant.

        Returns:
            Child node with highest UCB score, or None if no children.
        """
        best_score = float("-inf")
        best_child: MCTSNode | None = None

        for child in self.children.values():
            score = child.ucb_score(self.visit_count, c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, policy: np.ndarray, action_mask: np.ndarray) -> None:
        """Expand node with children for valid actions.

        Args:
            policy: Policy probabilities from network.
            action_mask: Binary mask of valid actions.
        """
        valid_actions = np.where(action_mask > 0)[0]

        if len(valid_actions) == 0:
            self.is_terminal = True
            return

        # Mask and renormalize policy
        masked_policy = policy * action_mask
        policy_sum = masked_policy.sum()

        if policy_sum > 0:
            masked_policy = masked_policy / policy_sum
        else:
            # Uniform over valid actions if policy is zero
            masked_policy = action_mask / action_mask.sum()

        for action in valid_actions:
            self.children[action] = MCTSNode(
                prior=masked_policy[action],
                parent=self,
                action=action,
            )

    def backpropagate(self, value: float) -> None:
        """Backpropagate value up the tree.

        Args:
            value: Value to backpropagate (from current player's perspective).
        """
        self.visit_count += 1
        self.value_sum += value

        if self.parent is not None:
            # Negate value for opponent's perspective
            self.parent.backpropagate(-value)


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""

    def __init__(
        self,
        config: MCTSConfig | None = None,
        network_fn: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, float]] | None = None,
    ) -> None:
        """Initialize MCTS.

        Args:
            config: MCTS configuration.
            network_fn: Function that takes observation and returns (policy, value).
        """
        self.config = config or MCTSConfig()
        self.network_fn = network_fn
        self.root: MCTSNode | None = None

    def set_network(
        self, network_fn: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, float]]
    ) -> None:
        """Set the network function for evaluation.

        Args:
            network_fn: Function that takes observation and returns (policy, value).
        """
        self.network_fn = network_fn

    def search(
        self,
        obs: dict[str, np.ndarray],
        step_fn: Callable[[dict[str, np.ndarray], int], tuple[dict[str, np.ndarray], float, bool]],
        add_noise: bool = True,
    ) -> np.ndarray:
        """Run MCTS search from current observation.

        Args:
            obs: Current observation dictionary.
            step_fn: Function that simulates stepping environment.
                     Takes (obs, action) and returns (next_obs, reward, done).
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            Action probabilities based on visit counts.
        """
        if self.network_fn is None:
            raise ValueError("Network function not set. Call set_network() first.")

        action_mask = obs["action_mask"]
        num_actions = len(action_mask)

        # Get initial policy and value for root
        policy, _value = self.network_fn(obs)

        # Create root node
        self.root = MCTSNode()
        self.root.expand(policy, action_mask)

        # Add Dirichlet noise at root for exploration
        if add_noise and self.root.is_expanded:
            self._add_dirichlet_noise(self.root, action_mask)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = self.root
            current_obs = obs.copy()
            depth = 0

            # Selection: traverse tree using UCB until we reach a leaf
            while node.is_expanded and depth < self.config.max_depth:
                child = node.select_child(self.config.c_puct)
                if child is None:
                    break

                # Simulate step
                action = child.action
                if action is None:
                    break
                next_obs, reward, done = step_fn(current_obs, action)

                node = child
                current_obs = next_obs
                depth += 1

                if done:
                    node.is_terminal = True
                    node.terminal_value = reward
                    break

            # Evaluation
            if node.is_terminal:
                leaf_value = node.terminal_value or 0.0
            else:
                # Expand if not terminal
                if not node.is_expanded:
                    new_policy, leaf_value = self.network_fn(current_obs)
                    node.expand(new_policy, current_obs["action_mask"])
                else:
                    _, leaf_value = self.network_fn(current_obs)

            # Backpropagation
            node.backpropagate(leaf_value)

        # Extract action probabilities from visit counts
        return self._get_action_probs(num_actions)

    def _add_dirichlet_noise(self, node: MCTSNode, action_mask: np.ndarray) -> None:
        """Add Dirichlet noise to root node priors.

        Args:
            node: Root node to add noise to.
            action_mask: Binary mask of valid actions.
        """
        valid_actions = list(node.children.keys())
        if not valid_actions:
            return

        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_actions))

        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(valid_actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _get_action_probs(self, num_actions: int) -> np.ndarray:
        """Get action probabilities from root visit counts.

        Args:
            num_actions: Total number of actions.

        Returns:
            Probability distribution over actions.
        """
        action_probs = np.zeros(num_actions, dtype=np.float32)

        if self.root is None or not self.root.children:
            return action_probs

        # Extract visit counts
        visit_counts = np.zeros(num_actions, dtype=np.float32)
        for action, child in self.root.children.items():
            visit_counts[action] = child.visit_count

        if visit_counts.sum() == 0:
            # Uniform over children if no visits
            for action in self.root.children.keys():
                action_probs[action] = 1.0 / len(self.root.children)
            return action_probs

        # Apply temperature
        if self.config.temperature == 0:
            # Greedy: select most visited
            best_action = np.argmax(visit_counts)
            action_probs[best_action] = 1.0
        else:
            # Softmax with temperature
            visit_counts_temp = np.power(visit_counts, 1.0 / self.config.temperature)
            action_probs = visit_counts_temp / visit_counts_temp.sum()

        return action_probs

    def select_action(self, action_probs: np.ndarray) -> int:
        """Select an action from the probability distribution.

        Args:
            action_probs: Probability distribution over actions.

        Returns:
            Selected action index.
        """
        if action_probs.sum() == 0:
            valid_actions = np.where(action_probs > 0)[0]
            if len(valid_actions) == 0:
                # Fallback: return 0 if no valid actions
                return 0
            return int(np.random.choice(valid_actions))

        return int(np.random.choice(len(action_probs), p=action_probs))


def create_network_fn(
    network: torch.nn.Module,
    device: str = "cpu",
) -> Callable[[dict[str, np.ndarray]], tuple[np.ndarray, float]]:
    """Create a network evaluation function for MCTS.

    Args:
        network: Neural network with forward(spatial, vector) method.
        device: Device to run network on.

    Returns:
        Function that takes observation and returns (policy, value).
    """

    def network_fn(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        spatial = torch.tensor(obs["spatial"]).unsqueeze(0).float().to(device)
        vector = torch.tensor(obs["vector"]).unsqueeze(0).float().to(device)

        with torch.no_grad():
            policy_logits, value = network(spatial, vector)
            policy = torch.softmax(policy_logits, dim=-1)

        policy_np = policy.squeeze().cpu().numpy()
        value_np = float(value.cpu().numpy().squeeze())

        return policy_np, value_np

    return network_fn


class SimpleMCTS:
    """Simplified MCTS that doesn't require environment stepping.

    This version uses the neural network for both policy guidance and
    value estimation without actually simulating game states. Useful
    for cases where environment simulation is expensive or complex.
    """

    def __init__(
        self,
        config: MCTSConfig | None = None,
    ) -> None:
        """Initialize simplified MCTS.

        Args:
            config: MCTS configuration.
        """
        self.config = config or MCTSConfig()

    def get_action_probs(
        self,
        policy: np.ndarray,
        action_mask: np.ndarray,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Get improved action probabilities using policy and noise.

        This is a simplified version that uses Dirichlet noise for exploration
        without full tree search. Useful when environment simulation is not
        available or too expensive.

        Args:
            policy: Policy probabilities from network.
            action_mask: Binary mask of valid actions.
            add_noise: Whether to add Dirichlet noise.

        Returns:
            Improved action probabilities.
        """
        # Apply mask
        masked_policy = policy * action_mask

        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return np.zeros_like(policy)

        # Normalize
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy = masked_policy / policy_sum
        else:
            masked_policy = action_mask / action_mask.sum()

        # Add Dirichlet noise for exploration
        if add_noise:
            noise = np.zeros_like(policy)
            noise[valid_actions] = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(valid_actions),
            )

            eps = self.config.dirichlet_epsilon
            masked_policy = (1 - eps) * masked_policy + eps * noise

        # Apply temperature
        if self.config.temperature != 1.0 and self.config.temperature > 0:
            valid_probs = masked_policy[valid_actions]
            valid_probs = np.power(valid_probs, 1.0 / self.config.temperature)
            valid_probs = valid_probs / valid_probs.sum()

            result = np.zeros_like(policy)
            result[valid_actions] = valid_probs
            return result

        return masked_policy  # type: ignore[no-any-return]

    def select_action(
        self,
        policy: np.ndarray,
        action_mask: np.ndarray,
        add_noise: bool = True,
    ) -> tuple[int, np.ndarray]:
        """Select action using improved policy.

        Args:
            policy: Policy probabilities from network.
            action_mask: Binary mask of valid actions.
            add_noise: Whether to add Dirichlet noise.

        Returns:
            Tuple of (selected action, action probabilities).
        """
        action_probs = self.get_action_probs(policy, action_mask, add_noise)

        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return 0, action_probs

        # Sample action
        if action_probs.sum() > 0:
            action = int(np.random.choice(len(action_probs), p=action_probs))
        else:
            action = int(np.random.choice(valid_actions))

        return action, action_probs


def create_mcts(
    num_simulations: int = 50,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    use_full_mcts: bool = False,
    network: torch.nn.Module | None = None,
) -> MCTS | SimpleMCTS:
    """Factory function to create MCTS instance.

    Args:
        num_simulations: Number of simulations per move.
        c_puct: Exploration constant for UCB.
        temperature: Temperature for action selection.
        dirichlet_alpha: Dirichlet noise alpha.
        dirichlet_epsilon: Weight of Dirichlet noise.
        use_full_mcts: Whether to use full MCTS with tree search.
        network: Neural network for full MCTS (required if use_full_mcts=True).

    Returns:
        MCTS or SimpleMCTS instance.
    """
    config = MCTSConfig(
        num_simulations=num_simulations,
        c_puct=c_puct,
        temperature=temperature,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )

    if use_full_mcts:
        mcts = MCTS(config)
        if network is not None:
            mcts.set_network(create_network_fn(network))
        return mcts
    else:
        return SimpleMCTS(config)
