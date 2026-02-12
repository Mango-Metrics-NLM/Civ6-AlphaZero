"""Domain-agnostic MCTS search engine."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.config import MCTSConfig
from src.core.constants import EPSILON, EPSILON_SOFTMAX
from src.core.mcts.node import MCTSNode
from src.core.mcts.policies import PUCTPolicy

logger = logging.getLogger(__name__)

# Type for the evaluation function: (spatial, vector, action_mask) -> (policy, value)
EvalFn = Callable[
    [NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]],
    tuple[NDArray[np.float32], float],
]


class MCTSEngine:
    """Generic MCTS search engine that works with any StrategicEnvironment.

    The engine is decoupled from the neural network -- it accepts an evaluation
    function that takes (spatial, vector, action_mask) and returns (policy, value).

    Args:
        config: MCTS configuration.
        eval_fn: Evaluation function (neural network inference).
        selection_policy: Optional child selection policy. Defaults to PUCT.
    """

    def __init__(
        self,
        config: MCTSConfig,
        eval_fn: EvalFn,
        selection_policy: PUCTPolicy | None = None,
    ) -> None:
        self._config = config
        self._eval_fn = eval_fn
        self._selection_policy = selection_policy or PUCTPolicy(c_puct=config.c_puct)

    @property
    def config(self) -> MCTSConfig:
        """Return the MCTS configuration."""
        return self._config

    def search(
        self,
        env: Any,  # StrategicEnvironment protocol
        num_simulations: int | None = None,
        add_noise: bool = True,
    ) -> tuple[NDArray[np.float32], float]:
        """Run MCTS search from the current environment state.

        Args:
            env: Environment at the root state. Must support clone().
            num_simulations: Override for number of simulations.
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            Tuple of (action_probabilities, root_value) where action_probs
            sums to approximately 1.0.
        """
        n_sims = num_simulations or self._config.num_simulations
        logger.debug(
            "MCTS search: %d simulations, noise=%s", n_sims, add_noise,
        )

        # Initialize root
        root = MCTSNode()
        obs = env.get_observation()
        action_mask = env.get_valid_actions()

        policy, _value = self._eval_fn(obs["spatial"], obs["vector"], action_mask)
        root.expand(policy, action_mask)

        # Add Dirichlet noise for exploration
        if add_noise and root.is_expanded:
            self._add_dirichlet_noise(root)

        # Simulation loop
        for _ in range(n_sims):
            node = root
            sim_env = env.clone()
            search_path: list[MCTSNode] = [node]

            # Track terminal state
            terminated = False
            truncated = False

            # SELECTION: traverse tree using selection policy
            while node.is_expanded:
                action, node = self._select_child(node)
                search_path.append(node)
                _, _, terminated, truncated, _ = sim_env.step(action)
                if terminated or truncated:
                    break

            # EVALUATION
            if terminated or truncated:
                # Terminal node - use actual game result
                # Convention: value from perspective of player who just moved
                leaf_value = 0.0  # draw by default
            else:
                # Leaf node - evaluate with neural network
                leaf_obs = sim_env.get_observation()
                leaf_mask = sim_env.get_valid_actions()
                leaf_policy, leaf_value = self._eval_fn(
                    leaf_obs["spatial"], leaf_obs["vector"], leaf_mask
                )
                if np.any(leaf_mask):
                    node.expand(leaf_policy, leaf_mask)

            # BACKPROPAGATION
            self._backpropagate(search_path, leaf_value)

        # Extract action probabilities from visit counts
        action_probs = get_action_probabilities(root, env.action_space_size)
        logger.debug(
            "MCTS search complete: root_value=%.3f, root_visits=%d, best_action=%d",
            root.value,
            root.visit_count,
            int(np.argmax(action_probs)),
        )
        return action_probs, root.value

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select best child using the configured selection policy."""
        if not node.children:
            msg = "Cannot select child from unexpanded node"
            raise ValueError(msg)

        actions = sorted(node.children.keys())
        priors = np.array([node.children[a].prior for a in actions], dtype=np.float32)
        values = np.array([node.children[a].value for a in actions], dtype=np.float32)
        visits = np.array([node.children[a].visit_count for a in actions], dtype=np.int64)
        valid = np.ones(len(actions), dtype=np.bool_)

        idx = self._selection_policy.select_child(
            parent_visit_count=node.visit_count,
            children_priors=priors,
            children_values=values,
            children_visit_counts=visits,
            valid_mask=valid,
        )

        action = actions[idx]
        return action, node.children[action]

    def _backpropagate(self, search_path: list[MCTSNode], value: float) -> None:
        """Backpropagate value through the search path with negamax flipping."""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Add Dirichlet noise to root priors for exploration."""
        actions = list(root.children.keys())
        if not actions:
            return

        alpha = self._config.dirichlet_alpha
        epsilon = self._config.dirichlet_epsilon
        noise = np.random.dirichlet([alpha] * len(actions))

        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


def get_action_probabilities(
    root: MCTSNode,
    action_space_size: int,
    temperature: float = 1.0,
) -> NDArray[np.float32]:
    """Convert root visit counts to action probability distribution.

    Args:
        root: Root node of the search tree.
        action_space_size: Total action space size.
        temperature: Sampling temperature (0=greedy, 1=proportional).

    Returns:
        Normalized probability array of shape (action_space_size,).
    """
    counts = root.child_visit_counts(action_space_size)

    if temperature == 0:
        # Greedy: all mass on most-visited action
        probs = np.zeros_like(counts)
        best = int(np.argmax(counts))
        probs[best] = 1.0
        return probs

    # Temperature scaling
    if counts.sum() == 0:
        return counts  # No visits, return zeros

    # Exponentiate with temperature
    scaled = np.power(counts, 1.0 / temperature)
    total = scaled.sum()
    if total > 0:
        result: NDArray[np.float32] = (scaled / total).astype(np.float32)
        return result
    return counts


def select_action(
    action_probs: NDArray[np.float32],
    temperature: float = 1.0,
    action_mask: NDArray[np.bool_] | None = None,
) -> int:
    """Select a single action from a probability distribution.

    Args:
        action_probs: Probability distribution over actions.
        temperature: 0=greedy (argmax), >0=proportional sampling.
        action_mask: Optional mask filtering valid actions.

    Returns:
        Selected action index.
    """
    probs = action_probs.copy().astype(np.float64)

    # Apply mask
    if action_mask is not None:
        probs = probs * action_mask
        if probs.sum() < EPSILON:
            valid = np.where(action_mask)[0]
            if len(valid) == 0:
                return 0
            return int(valid[0])

    # Normalize
    total = probs.sum()
    if total < EPSILON:
        return int(np.argmax(action_probs))
    probs = probs / total

    # Temperature selection
    if temperature == 0:
        return int(np.argmax(probs))

    # Apply temperature
    if temperature != 1.0:
        probs = np.power(probs + EPSILON_SOFTMAX, 1.0 / temperature)
        probs = probs / probs.sum()

    return int(np.random.choice(len(probs), p=probs))
