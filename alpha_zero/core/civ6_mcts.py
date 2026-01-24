"""Monte Carlo Tree Search for Civ6 AlphaZero.

Implements PUCT-based MCTS with neural network guidance, supporting:
- Action masking for valid moves
- Dirichlet noise for exploration
- Temperature-based action selection
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


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

        assert best_child is not None, "select_child called on node with no children"
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
        node: MCTSNode | None = self
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
