"""Multi-objective Monte Carlo Tree Search engine for strategic planning.

Implements PUCT-based MCTS with multi-objective value backpropagation
and scalarized node selection. Supports both neural network guidance
and random rollout fallback for standalone operation.

Key features:
- Multi-objective value vectors at each node
- Scalarized UCB selection (pluggable scalarization strategies)
- Dirichlet noise at root for exploration
- Temperature-based action selection
- Compatible with StrategicEnvironment protocol

Example:
    >>> from alpha_zero.strategic.mcts import MultiObjectiveMCTS, MultiObjectiveMCTSConfig
    >>> from alpha_zero.strategic.objectives import WeightedSumScalarizer, ObjectiveSpec
    >>>
    >>> objectives = [
    ...     ObjectiveSpec(name="efficiency", weight=0.6),
    ...     ObjectiveSpec(name="equity", weight=0.4),
    ... ]
    >>> scalarizer = WeightedSumScalarizer(objectives)
    >>> config = MultiObjectiveMCTSConfig(num_simulations=100)
    >>> mcts = MultiObjectiveMCTS(config, scalarizer=scalarizer, num_objectives=2)
    >>> action_probs, values = mcts.search(env)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from alpha_zero.strategic.objectives import (
    BaseScalarizer,
    ObjectiveSpec,
    WeightedSumScalarizer,
)

logger = logging.getLogger(__name__)

# Small constant for numerical stability
_EPSILON = 1e-8


@dataclass
class MultiObjectiveMCTSConfig:
    """Configuration for multi-objective MCTS.

    Attributes:
        num_simulations: Number of MCTS simulations per search call.
        c_puct: Exploration constant for PUCT formula.
        dirichlet_alpha: Alpha parameter for Dirichlet noise at root.
        dirichlet_epsilon: Weight of Dirichlet noise vs prior policy.
        temperature: Temperature for action selection from visit counts.
        temperature_threshold: Move number after which temperature drops to near-zero.
    """

    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiObjectiveMCTSConfig:
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration values. Unknown keys are ignored.

        Returns:
            New configuration instance.
        """
        known_fields = {
            "num_simulations",
            "c_puct",
            "dirichlet_alpha",
            "dirichlet_epsilon",
            "temperature",
            "temperature_threshold",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class MultiObjectiveNode:
    """Node in a multi-objective MCTS tree.

    Each node maintains a vector of value sums (one per objective) and
    a visit count. The scalar value used for UCB selection is computed
    on-the-fly via a scalarizer.

    Attributes:
        prior: Prior probability from the policy network.
        visit_count: Number of times this node has been visited.
        value_sum: Sum of backpropagated values per objective.
        children: Child nodes keyed by action index.
        parent: Parent node (None for root).
        action: Action that led to this node (None for root).
    """

    __slots__ = (
        "_num_objectives",
        "action",
        "children",
        "parent",
        "prior",
        "value_sum",
        "visit_count",
    )

    def __init__(self, prior: float = 0.0, num_objectives: int = 1) -> None:
        """Initialize a tree node.

        Args:
            prior: Prior probability from policy network.
            num_objectives: Number of optimization objectives.
        """
        self.prior = prior
        self.visit_count: int = 0
        self._num_objectives = num_objectives
        self.value_sum: np.ndarray = np.zeros(num_objectives, dtype=np.float64)
        self.children: dict[int, MultiObjectiveNode] = {}
        self.parent: MultiObjectiveNode | None = None
        self.action: int | None = None

    @property
    def value(self) -> np.ndarray:
        """Mean value vector across all visits.

        Returns:
            Per-objective mean values, shape (num_objectives,).
            Returns zeros if unvisited.
        """
        if self.visit_count == 0:
            return np.zeros(self._num_objectives, dtype=np.float64)
        return self.value_sum / self.visit_count

    def scalar_value(self, scalarizer: BaseScalarizer) -> float:
        """Compute scalarized value for UCB selection.

        Args:
            scalarizer: Scalarization strategy to combine objectives.

        Returns:
            Single scalar value.
        """
        return scalarizer.scalarize(self.value)

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded (has children)."""
        return len(self.children) > 0

    def expand(self, policy: np.ndarray, action_mask: np.ndarray) -> None:
        """Expand this node by creating child nodes for valid actions.

        Args:
            policy: Prior policy distribution over all actions, shape (A,).
            action_mask: Binary mask of valid actions, shape (A,).
                         1 = valid, 0 = invalid.
        """
        # Mask and renormalize policy
        masked_policy = policy * action_mask
        policy_sum = masked_policy.sum()
        if policy_sum > _EPSILON:
            masked_policy = masked_policy / policy_sum
        else:
            # Fallback: uniform over valid actions
            valid_count = action_mask.sum()
            if valid_count > 0:
                masked_policy = action_mask / valid_count
            else:
                logger.warning("No valid actions during node expansion")
                return

        # Create children for valid actions
        valid_actions = np.where(action_mask > 0)[0]
        for action_idx in valid_actions:
            child = MultiObjectiveNode(
                prior=float(masked_policy[action_idx]),
                num_objectives=self._num_objectives,
            )
            child.parent = self
            child.action = int(action_idx)
            self.children[int(action_idx)] = child

    def select_child(
        self,
        c_puct: float,
        scalarizer: BaseScalarizer,
    ) -> tuple[int, MultiObjectiveNode]:
        """Select the child with the highest UCB score.

        Uses the PUCT formula:
            UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        where Q is the scalarized value of the child.

        Args:
            c_puct: Exploration constant.
            scalarizer: Scalarization strategy for multi-objective values.

        Returns:
            Tuple of (action, child_node) with highest UCB.

        Raises:
            ValueError: If node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select child from unexpanded node")

        best_score = -float("inf")
        best_action = -1
        best_child: MultiObjectiveNode | None = None

        sqrt_parent = np.sqrt(self.visit_count + _EPSILON)

        for action, child in self.children.items():
            # Q-value: scalarized mean value of child
            if child.visit_count > 0:
                q_value = child.scalar_value(scalarizer)
            else:
                q_value = 0.0

            # Exploration bonus
            u_value = c_puct * child.prior * sqrt_parent / (1.0 + child.visit_count)

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None, "Should have found at least one child"
        return best_action, best_child

    def backpropagate(self, value: np.ndarray) -> None:
        """Backpropagate a value vector up to the root.

        Updates visit counts and value sums for this node and
        all ancestor nodes.

        Args:
            value: Per-objective value vector, shape (num_objectives,).
        """
        node: MultiObjectiveNode | None = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent


class MultiObjectiveMCTS:
    """Multi-objective Monte Carlo Tree Search engine.

    Performs PUCT-based tree search with multi-objective value vectors.
    Supports neural network guidance or random rollout fallback.

    The search produces an action probability distribution based on
    visit counts and a value vector estimate for the root state.

    Args:
        config: MCTS configuration.
        network: Optional neural network for policy/value prediction.
            Must implement predict(spatial, vector, action_mask) ->
            (policy_probs, value).
        scalarizer: Strategy for combining objectives into scalar.
            Defaults to WeightedSumScalarizer with equal weights.
        num_objectives: Number of optimization objectives.
        device: PyTorch device for network inference.
    """

    def __init__(
        self,
        config: MultiObjectiveMCTSConfig,
        network: Any = None,
        scalarizer: BaseScalarizer | None = None,
        num_objectives: int = 1,
        device: str = "cpu",
    ) -> None:
        self._config = config
        self._network = network
        self._num_objectives = num_objectives
        self._device = device

        # Default scalarizer: equal-weight sum
        if scalarizer is not None:
            self._scalarizer = scalarizer
        else:
            default_objectives = [
                ObjectiveSpec(name=f"objective_{i}", weight=1.0)
                for i in range(num_objectives)
            ]
            self._scalarizer = WeightedSumScalarizer(default_objectives)

    @property
    def config(self) -> MultiObjectiveMCTSConfig:
        """Get MCTS configuration."""
        return self._config

    @property
    def scalarizer(self) -> BaseScalarizer:
        """Get the scalarization strategy."""
        return self._scalarizer

    def search(
        self,
        env: Any,
        num_simulations: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run MCTS search from the current environment state.

        Performs the specified number of simulations, each consisting of:
        1. Select: traverse tree using UCB to find a leaf
        2. Expand: expand the leaf node with policy priors
        3. Evaluate: get value estimate (network or rollout)
        4. Backpropagate: update value sums along the path

        Args:
            env: Environment implementing clone(), step(),
                get_valid_actions(), and observation access.
                Compatible with StrategicEnvironment protocol or
                MCTSEnvironment protocol.
            num_simulations: Override config num_simulations.

        Returns:
            Tuple of:
            - action_probs: Action probability distribution from visit
              counts, shape (action_space_size,).
            - value_vector: Multi-objective value estimate for root
              state, shape (num_objectives,).
        """
        sims = (
            num_simulations
            if num_simulations is not None
            else self._config.num_simulations
        )

        # Get action space size and initial valid actions
        action_mask = self._get_action_mask(env)
        action_space_size = len(action_mask)

        # Create root node
        root = MultiObjectiveNode(
            prior=1.0, num_objectives=self._num_objectives
        )

        # Get initial policy and value for root expansion
        policy, value = self._evaluate(env)

        # Expand root
        root.expand(policy, action_mask)

        # Add Dirichlet noise to root for exploration
        self._add_noise(root)

        # Backpropagate initial value through root
        root.backpropagate(value)

        # Run simulations
        for _sim_idx in range(sims):
            # Clone environment for this simulation
            sim_env = env.clone()

            # Select leaf node
            leaf, _actions_taken = self._select_leaf(root, sim_env)

            # Check if game is over at this leaf
            if self._is_terminal(sim_env):
                terminal_value = self._get_terminal_value(sim_env)
                leaf.backpropagate(terminal_value)
                continue

            # Evaluate leaf position
            leaf_policy, leaf_value = self._evaluate(sim_env)

            # Get valid actions at leaf
            leaf_mask = self._get_action_mask(sim_env)

            # Expand leaf if not already expanded
            if not leaf.is_expanded:
                leaf.expand(leaf_policy, leaf_mask)

            # Backpropagate value
            leaf.backpropagate(leaf_value)

        # Extract action probabilities from visit counts
        action_probs = np.zeros(action_space_size, dtype=np.float64)
        for action, child in root.children.items():
            if action < action_space_size:
                action_probs[action] = child.visit_count

        # Normalize to probabilities
        total_visits = action_probs.sum()
        if total_visits > 0:
            action_probs = action_probs / total_visits
        else:
            # Fallback: uniform over valid actions
            valid_count = action_mask.sum()
            if valid_count > 0:
                action_probs = action_mask.astype(np.float64) / valid_count

        # Root value estimate
        value_vector = root.value.copy()

        logger.debug(
            "MCTS search complete: %d simulations, root visits=%d, "
            "value=%s, top action=%d (%.3f)",
            sims,
            root.visit_count,
            value_vector,
            int(np.argmax(action_probs)),
            float(np.max(action_probs)),
        )

        return action_probs, value_vector

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float = 1.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Select an action from the probability distribution.

        With temperature=0 (or very small), selects greedily.
        With temperature=1, samples proportionally to visit counts.
        Higher temperatures increase randomness.

        Args:
            action_probs: Action probability distribution, shape (A,).
            temperature: Temperature for sampling. 0 = greedy.
            action_mask: Optional mask of valid actions.

        Returns:
            Selected action index.
        """
        if action_mask is not None:
            action_probs = action_probs * action_mask
            prob_sum = action_probs.sum()
            if prob_sum > _EPSILON:
                action_probs = action_probs / prob_sum
            else:
                # No valid actions with probability; uniform over valid
                valid_count = action_mask.sum()
                if valid_count > 0:
                    action_probs = action_mask.astype(np.float64) / valid_count
                else:
                    # Truly no valid actions; return 0
                    return 0

        if temperature < _EPSILON:
            # Greedy selection
            return int(np.argmax(action_probs))

        if abs(temperature - 1.0) < _EPSILON:
            # Direct sampling
            probs = action_probs.copy()
            prob_sum = probs.sum()
            if prob_sum > _EPSILON:
                probs = probs / prob_sum
            else:
                probs = np.ones_like(probs) / len(probs)
            return int(np.random.choice(len(probs), p=probs))

        # Temperature-scaled sampling
        # Raise probabilities to 1/temperature power
        log_probs = np.log(action_probs + _EPSILON)
        scaled = log_probs / temperature
        # Numerical stability: subtract max
        scaled = scaled - scaled.max()
        exp_scaled = np.exp(scaled)
        probs = exp_scaled / exp_scaled.sum()

        return int(np.random.choice(len(probs), p=probs))

    # ----------------------------------------------------------------
    # Private helper methods
    # ----------------------------------------------------------------

    def _select_leaf(
        self,
        root: MultiObjectiveNode,
        sim_env: Any,
    ) -> tuple[MultiObjectiveNode, list[int]]:
        """Traverse the tree from root to a leaf node.

        Follows the UCB-best path, stepping the simulation environment
        along the way.

        Args:
            root: Root node of the search tree.
            sim_env: Cloned environment for simulation.

        Returns:
            Tuple of (leaf_node, actions_taken).
        """
        node = root
        actions_taken: list[int] = []

        while node.is_expanded:
            action, child = node.select_child(
                self._config.c_puct,
                self._scalarizer,
            )
            # Step the simulation environment
            self._step_env(sim_env, action)
            actions_taken.append(action)
            node = child

            # Stop if terminal
            if self._is_terminal(sim_env):
                break

        return node, actions_taken

    def _evaluate(self, env: Any) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a position, returning policy and value.

        If a neural network is available, uses it for evaluation.
        Otherwise, falls back to uniform policy and random rollout.

        Args:
            env: Environment at the position to evaluate.

        Returns:
            Tuple of:
            - policy: Action probability distribution, shape (A,).
            - value: Multi-objective value vector, shape (num_objectives,).
        """
        if self._network is not None:
            return self._evaluate_with_network(env)
        return self._evaluate_with_rollout(env)

    def _evaluate_with_network(
        self, env: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate position using neural network.

        Args:
            env: Environment at the position to evaluate.

        Returns:
            Tuple of (policy, value_vector).
        """
        obs = self._get_observation(env)
        action_mask = self._get_action_mask(env)

        spatial = obs.get("spatial")
        vector = obs.get("vector")

        # Convert to tensors
        if spatial is not None:
            spatial_t = torch.tensor(
                spatial, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
        else:
            spatial_t = None

        if vector is not None:
            vector_t = torch.tensor(
                vector, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
        else:
            # Create dummy vector
            vector_t = torch.zeros(
                1, 64, dtype=torch.float32, device=self._device
            )

        mask_t = torch.tensor(
            action_mask, dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        # Network inference
        with torch.no_grad():
            policy_t, value_t = self._network.predict(
                spatial_t, vector_t, mask_t
            )

        # Convert to numpy
        policy = policy_t.squeeze(0).cpu().numpy().astype(np.float64)
        value_raw = value_t.squeeze(0).cpu().numpy().astype(np.float64)

        # Ensure value has correct shape
        if value_raw.ndim == 0:
            # Scalar value: replicate for all objectives
            value = np.full(
                self._num_objectives, float(value_raw), dtype=np.float64
            )
        elif len(value_raw) == self._num_objectives:
            value = value_raw
        else:
            # Mismatch: pad or truncate
            value = np.zeros(self._num_objectives, dtype=np.float64)
            n = min(len(value_raw), self._num_objectives)
            value[:n] = value_raw[:n]

        return policy, value

    def _evaluate_with_rollout(
        self, env: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate position using uniform policy and random rollout.

        Performs a random rollout from the current position until
        terminal or a maximum depth, collecting rewards.

        Args:
            env: Environment at the position to evaluate.

        Returns:
            Tuple of (uniform_policy, rollout_value).
        """
        action_mask = self._get_action_mask(env)
        action_space_size = len(action_mask)

        # Uniform policy over valid actions
        valid_count = action_mask.sum()
        if valid_count > 0:
            policy = action_mask.astype(np.float64) / valid_count
        else:
            policy = (
                np.ones(action_space_size, dtype=np.float64) / action_space_size
            )

        # Random rollout for value estimate
        rollout_env = env.clone()
        cumulative_reward = np.zeros(self._num_objectives, dtype=np.float64)
        max_rollout_depth = 50
        discount = 0.99

        for depth in range(max_rollout_depth):
            if self._is_terminal(rollout_env):
                break

            rollout_mask = self._get_action_mask(rollout_env)
            valid_actions = np.where(rollout_mask > 0)[0]

            if len(valid_actions) == 0:
                break

            # Random action
            action = int(np.random.choice(valid_actions))
            reward = self._step_env(rollout_env, action)

            # Accumulate discounted reward
            cumulative_reward += (discount**depth) * reward

        # Also get terminal value if applicable
        if self._is_terminal(rollout_env):
            terminal_val = self._get_terminal_value(rollout_env)
            cumulative_reward += (discount**max_rollout_depth) * terminal_val

        return policy, cumulative_reward

    def _expand(
        self,
        node: MultiObjectiveNode,
        policy: np.ndarray,
        action_mask: np.ndarray,
    ) -> None:
        """Expand a leaf node with policy priors.

        Args:
            node: Leaf node to expand.
            policy: Prior policy distribution, shape (A,).
            action_mask: Valid action mask, shape (A,).
        """
        node.expand(policy, action_mask)

    def _add_noise(self, root: MultiObjectiveNode) -> None:
        """Add Dirichlet noise to root node priors for exploration.

        Only applied to valid child actions. The noise weight is
        controlled by dirichlet_epsilon.

        Args:
            root: Root node whose children receive noise.
        """
        if not root.children:
            return

        epsilon = self._config.dirichlet_epsilon
        alpha = self._config.dirichlet_alpha

        num_children = len(root.children)
        if num_children == 0:
            return

        noise = np.random.dirichlet([alpha] * num_children)

        for i, child in enumerate(root.children.values()):
            child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i]

    # ----------------------------------------------------------------
    # Environment interface helpers
    # ----------------------------------------------------------------

    def _get_observation(self, env: Any) -> dict[str, Any]:
        """Get observation from environment, handling both protocols.

        Supports StrategicEnvironment (observation attribute) and
        MCTSEnvironment (get_observation method).

        Args:
            env: Environment instance.

        Returns:
            Dictionary with 'spatial' and 'vector' keys.
        """
        # Try StrategicEnvironment / MCTSEnvironment protocol
        if hasattr(env, "get_observation"):
            obs = env.get_observation()
            if isinstance(obs, dict):
                return obs
            # StrategicObservation dataclass
            if hasattr(obs, "spatial") and hasattr(obs, "vector"):
                return {"spatial": obs.spatial, "vector": obs.vector}

        # Try direct observation access
        if hasattr(env, "observation"):
            obs = env.observation
            if isinstance(obs, dict):
                return obs
            if hasattr(obs, "spatial") and hasattr(obs, "vector"):
                return {"spatial": obs.spatial, "vector": obs.vector}

        # Fallback: empty observation
        logger.warning("Could not extract observation from environment")
        return {
            "spatial": None,
            "vector": np.zeros(64, dtype=np.float32),
        }

    def _get_action_mask(self, env: Any) -> np.ndarray:
        """Get valid action mask from environment.

        Args:
            env: Environment instance.

        Returns:
            Binary mask array, shape (A,).
        """
        if hasattr(env, "get_valid_actions"):
            mask = env.get_valid_actions()
            return np.asarray(mask, dtype=np.float64)

        # Fallback: try action_spec
        if hasattr(env, "action_spec"):
            size = env.action_spec.total_size
            return np.ones(size, dtype=np.float64)

        # Fallback: try action_space
        if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
            return np.ones(env.action_space.n, dtype=np.float64)

        logger.warning("Could not get action mask; defaulting to size 100")
        return np.ones(100, dtype=np.float64)

    def _step_env(self, env: Any, action: int) -> np.ndarray:
        """Step the environment and extract multi-objective reward.

        Handles both StrategicEnvironment (returns StrategicStepResult)
        and standard gym-like environments (returns tuple).

        Args:
            env: Environment instance.
            action: Action to take.

        Returns:
            Reward vector, shape (num_objectives,).
        """
        result = env.step(action)

        # StrategicStepResult
        if hasattr(result, "rewards"):
            rewards = np.asarray(result.rewards, dtype=np.float64)
            if len(rewards) >= self._num_objectives:
                return rewards[: self._num_objectives]
            padded = np.zeros(self._num_objectives, dtype=np.float64)
            padded[: len(rewards)] = rewards
            return padded

        # Gym-like tuple: (obs, reward, terminated, truncated, info)
        if isinstance(result, tuple):
            if len(result) >= 2:
                reward = result[1]
                if isinstance(reward, np.ndarray):
                    if len(reward) >= self._num_objectives:
                        return reward[: self._num_objectives].astype(
                            np.float64
                        )
                    padded = np.zeros(self._num_objectives, dtype=np.float64)
                    padded[: len(reward)] = reward
                    return padded
                # Scalar reward
                rewards = np.zeros(self._num_objectives, dtype=np.float64)
                rewards[0] = float(reward)
                return rewards

        return np.zeros(self._num_objectives, dtype=np.float64)

    def _is_terminal(self, env: Any) -> bool:
        """Check if the environment is in a terminal state.

        Args:
            env: Environment instance.

        Returns:
            True if the episode is over.
        """
        # StrategicEnvironment: check get_info or done attribute
        if hasattr(env, "get_info"):
            info = env.get_info()
            if isinstance(info, dict):
                if info.get("done", False) or info.get("terminated", False):
                    return True

        # Check done attribute directly
        if hasattr(env, "done") and env.done:
            return True

        # Check terminated attribute
        if hasattr(env, "terminated") and env.terminated:
            return True

        return False

    def _get_terminal_value(self, env: Any) -> np.ndarray:
        """Get terminal value for a finished episode.

        Args:
            env: Environment in terminal state.

        Returns:
            Terminal value vector, shape (num_objectives,).
        """
        value = np.zeros(self._num_objectives, dtype=np.float64)

        if hasattr(env, "get_info"):
            info = env.get_info()
            if isinstance(info, dict):
                # Check for explicit terminal values
                if "terminal_values" in info:
                    tv = np.asarray(
                        info["terminal_values"], dtype=np.float64
                    )
                    n = min(len(tv), self._num_objectives)
                    value[:n] = tv[:n]
                    return value

                # Check for winner (game-style)
                if "winner" in info:
                    winner = info["winner"]
                    if winner is not None:
                        # Positive value for win
                        value[0] = 1.0
                    else:
                        # Draw
                        value[0] = 0.0
                    return value

                # Check for final reward
                if "final_reward" in info:
                    reward = info["final_reward"]
                    if isinstance(reward, np.ndarray):
                        n = min(len(reward), self._num_objectives)
                        value[:n] = reward[:n]
                    else:
                        value[0] = float(reward)
                    return value

        return value


__all__ = [
    "MultiObjectiveMCTS",
    "MultiObjectiveMCTSConfig",
    "MultiObjectiveNode",
]
