"""
Hierarchical Monte Carlo Tree Search with Options.

This module implements MCTS that operates at multiple hierarchy levels,
using options (temporally extended actions) for higher-level planning
and primitive actions for operational decisions.

Example:
    >>> from alpha_zero.hierarchical.mcts import HierarchicalMCTS, MCTSConfig
    >>> config = MCTSConfig()
    >>> mcts = HierarchicalMCTS(network, encoder, config)
    >>> action, stats = mcts.search(state, active_goals)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from alpha_zero.hierarchical.config import HierarchyLevel, MCTSLevelConfig
from alpha_zero.hierarchical.goals.goal_types import Goal

logger = logging.getLogger(__name__)


class OptionTermination(Enum):
    """Reasons for option termination."""

    GOAL_ACHIEVED = auto()
    TIMEOUT = auto()
    FAILURE = auto()
    INTERRUPTED = auto()
    STILL_ACTIVE = auto()


@dataclass
class Option:
    """Temporally extended action (option) for hierarchical planning.

    An option consists of:
    - Initiation set: states where option can start
    - Policy: how to select actions while option is active
    - Termination condition: when option ends

    Attributes:
        id: Unique option identifier.
        goal: Goal this option aims to achieve.
        level: Hierarchy level of this option.
        policy: Function mapping state to action distribution.
        termination: Function mapping state to termination probability.
        initiation: Function checking if option can start in state.
        duration: Expected duration in turns.
        reward_modifier: Reward shaping for this option.
    """

    id: str
    goal: Goal
    level: HierarchyLevel
    policy: Callable[[dict[str, Any]], np.ndarray] | None = None
    termination: Callable[[dict[str, Any]], float] | None = None
    initiation: Callable[[dict[str, Any]], bool] | None = None
    duration: int = 10
    reward_modifier: float = 1.0

    # Runtime state
    steps_active: int = 0
    cumulative_reward: float = 0.0
    status: OptionTermination = OptionTermination.STILL_ACTIVE

    def can_initiate(self, state: dict[str, Any]) -> bool:
        """Check if option can be initiated in given state."""
        if self.initiation is None:
            return True
        return self.initiation(state)

    def should_terminate(self, state: dict[str, Any]) -> bool:
        """Check if option should terminate."""
        if self.steps_active >= self.duration:
            self.status = OptionTermination.TIMEOUT
            return True

        if self.goal.progress >= 1.0:
            self.status = OptionTermination.GOAL_ACHIEVED
            return True

        if self.termination is not None:
            prob = self.termination(state)
            if np.random.random() < prob:
                self.status = OptionTermination.INTERRUPTED
                return True

        return False

    def get_action(self, state: dict[str, Any], valid_actions: np.ndarray) -> int:
        """Get action from option policy."""
        if self.policy is None:
            # Default: random valid action
            valid_indices = np.where(valid_actions > 0)[0]
            return int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0

        probs = self.policy(state)
        probs = probs * valid_actions  # Mask invalid
        probs = probs / (probs.sum() + 1e-8)  # Renormalize
        return int(np.random.choice(len(probs), p=probs))

    def step(self, reward: float) -> None:
        """Update option state after one step."""
        self.steps_active += 1
        self.cumulative_reward += reward * self.reward_modifier


@dataclass
class MCTSConfig:
    """Configuration for hierarchical MCTS.

    Attributes:
        num_simulations: Number of MCTS simulations.
        c_puct: Exploration constant for UCB.
        c_visit: Visit count bonus factor.
        temperature: Temperature for action selection.
        dirichlet_alpha: Dirichlet noise alpha for exploration.
        dirichlet_epsilon: Weight of Dirichlet noise at root.
        discount: Discount factor for future rewards.
        horizon: Maximum planning horizon.
        use_options: Whether to use options (temporal abstraction).
        option_penalty: Penalty for switching options.
        virtual_loss: Virtual loss for parallel MCTS.
    """

    num_simulations: int = 100
    c_puct: float = 1.5
    c_visit: float = 1.0
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    discount: float = 0.99
    horizon: int = 50
    use_options: bool = True
    option_penalty: float = 0.1
    virtual_loss: float = 3.0

    @classmethod
    def from_level_config(cls, level_config: MCTSLevelConfig) -> MCTSConfig:
        """Create from MCTSLevelConfig.

        Args:
            level_config: Level-specific MCTS configuration.

        Returns:
            MCTSConfig instance.
        """
        return cls(
            num_simulations=level_config.num_simulations,
            c_puct=level_config.c_puct,
            temperature=level_config.temperature,
            dirichlet_alpha=level_config.dirichlet_alpha,
            dirichlet_epsilon=level_config.dirichlet_epsilon,
            horizon=level_config.horizon,
        )

    @classmethod
    def for_level(cls, level: HierarchyLevel) -> MCTSConfig:
        """Create configuration for specific hierarchy level.

        Args:
            level: Hierarchy level.

        Returns:
            Level-appropriate configuration.
        """
        configs = {
            HierarchyLevel.STRATEGIC: cls(
                num_simulations=50,
                c_puct=2.0,
                horizon=100,
                temperature=1.2,
                use_options=True,
                discount=0.995,
            ),
            HierarchyLevel.TACTICAL: cls(
                num_simulations=100,
                c_puct=1.5,
                horizon=25,
                temperature=1.0,
                use_options=True,
                discount=0.99,
            ),
            HierarchyLevel.OPERATIONAL: cls(
                num_simulations=200,
                c_puct=1.5,
                horizon=5,
                temperature=1.0,
                use_options=False,
                discount=0.99,
            ),
        }
        return configs.get(level, cls())


@dataclass
class SearchStats:
    """Statistics from MCTS search.

    Attributes:
        root_value: Value estimate at root node.
        visit_counts: Visit counts for each action.
        prior_probs: Prior probabilities from network.
        q_values: Q-values for each action.
        depth_reached: Maximum depth reached in search.
        simulations: Number of simulations performed.
        options_considered: Number of options considered (if applicable).
    """

    root_value: float = 0.0
    visit_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    prior_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    q_values: np.ndarray = field(default_factory=lambda: np.array([]))
    depth_reached: int = 0
    simulations: int = 0
    options_considered: int = 0

    def get_policy(self, temperature: float = 1.0) -> np.ndarray:
        """Get policy from visit counts with temperature.

        Args:
            temperature: Temperature for softmax.

        Returns:
            Policy distribution.
        """
        if temperature == 0:
            # Deterministic: pick most visited
            policy = np.zeros_like(self.visit_counts, dtype=np.float32)
            policy[np.argmax(self.visit_counts)] = 1.0
            return policy

        counts = self.visit_counts ** (1.0 / temperature)
        return counts / (counts.sum() + 1e-8)


class MCTSNode:
    """Node in the MCTS search tree.

    Attributes:
        state: Game state at this node.
        parent: Parent node.
        action: Action taken to reach this node.
        prior: Prior probability of this action.
        children: Child nodes.
        visit_count: Number of visits to this node.
        value_sum: Sum of values from simulations through this node.
        virtual_loss: Virtual loss for parallel MCTS.
    """

    def __init__(
        self,
        state: dict[str, Any],
        parent: MCTSNode | None = None,
        action: int | None = None,
        prior: float = 1.0,
    ) -> None:
        """Initialize MCTS node.

        Args:
            state: Game state at this node.
            parent: Parent node.
            action: Action taken to reach this node.
            prior: Prior probability of this action.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children: dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.virtual_loss: float = 0.0

        # Option-related
        self.option: Option | None = None
        self.option_children: dict[str, MCTSNode] = {}

    @property
    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0 or len(self.option_children) > 0

    @property
    def q_value(self) -> float:
        """Get Q-value (average value) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """Calculate UCB score for node selection.

        Args:
            c_puct: Exploration constant.
            parent_visits: Visit count of parent node.

        Returns:
            UCB score.
        """
        # Q-value component (with virtual loss)
        q = self.q_value - self.virtual_loss

        # Exploration component
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

        return q + u

    def select_child(self, c_puct: float) -> tuple[int, MCTSNode]:
        """Select child node with highest UCB score.

        Args:
            c_puct: Exploration constant.

        Returns:
            Tuple of (action, child_node).
        """
        best_score = float("-inf")
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def select_option_child(self, c_puct: float) -> tuple[str, MCTSNode]:
        """Select option child with highest UCB score.

        Args:
            c_puct: Exploration constant.

        Returns:
            Tuple of (option_id, child_node).
        """
        best_score = float("-inf")
        best_id = ""
        best_child = None

        for opt_id, child in self.option_children.items():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_id = opt_id
                best_child = child

        return best_id, best_child

    def expand(
        self,
        priors: np.ndarray,
        valid_actions: np.ndarray,
        state_transition: Callable[[dict[str, Any], int], dict[str, Any]],
    ) -> None:
        """Expand node with child nodes for each valid action.

        Args:
            priors: Prior probabilities from network.
            valid_actions: Mask of valid actions.
            state_transition: Function to get next state.
        """
        for action in np.where(valid_actions > 0)[0]:
            if action not in self.children:
                next_state = state_transition(self.state, int(action))
                self.children[action] = MCTSNode(
                    state=next_state,
                    parent=self,
                    action=action,
                    prior=priors[action],
                )

    def expand_with_options(
        self,
        options: list[Option],
        priors: dict[str, float],
        state_predictor: Callable[[dict[str, Any], Option], dict[str, Any]],
    ) -> None:
        """Expand node with option children.

        Args:
            options: Available options.
            priors: Prior probabilities for each option.
            state_predictor: Function to predict state after option.
        """
        for option in options:
            if option.can_initiate(self.state):
                predicted_state = state_predictor(self.state, option)
                self.option_children[option.id] = MCTSNode(
                    state=predicted_state,
                    parent=self,
                    action=None,
                    prior=priors.get(option.id, 1.0 / len(options)),
                )
                self.option_children[option.id].option = option

    def backpropagate(self, value: float, discount: float = 1.0) -> None:
        """Backpropagate value up the tree.

        Args:
            value: Value to backpropagate.
            discount: Discount factor per level.
        """
        node = self
        current_value = value

        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            node.virtual_loss = max(0, node.virtual_loss - 1)

            current_value *= discount
            node = node.parent


class HierarchicalMCTS:
    """Hierarchical Monte Carlo Tree Search.

    This class implements MCTS that operates at multiple abstraction levels,
    using options for temporal abstraction at higher levels and primitive
    actions at the operational level.
    """

    def __init__(
        self,
        network: Any,  # HierarchicalNetwork
        state_encoder: Any,  # HierarchicalStateEncoder
        goal_encoder: Any,  # HierarchicalGoalEncoder
        config: MCTSConfig | None = None,
        level: HierarchyLevel = HierarchyLevel.OPERATIONAL,
        spatial_shape: tuple[int, int, int] = (13, 16, 16),
        vector_dim: int = 64,
        goal_dim: int = 64,
    ) -> None:
        """Initialize hierarchical MCTS.

        Args:
            network: Neural network for policy/value estimation.
            state_encoder: State encoder.
            goal_encoder: Goal encoder.
            config: MCTS configuration.
            level: Hierarchy level for this MCTS instance.
            spatial_shape: Shape of spatial input (channels, height, width).
            vector_dim: Dimension of vector input.
            goal_dim: Dimension of goal embedding.
        """
        self.network = network
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.config = config or MCTSConfig.for_level(level)
        self.level = level

        # Store shape parameters for tensor creation
        self.spatial_shape = spatial_shape
        self.vector_dim = vector_dim
        self.goal_dim = goal_dim

        # Available options (for higher levels)
        self.options: list[Option] = []

        logger.debug(
            "HierarchicalMCTS initialized for %s with %d simulations",
            level.value,
            self.config.num_simulations,
        )

    def register_option(self, option: Option) -> None:
        """Register an option for use in search.

        Args:
            option: Option to register.
        """
        self.options.append(option)

    def search(
        self,
        state: dict[str, Any],
        goals: list[Goal],
        valid_actions: np.ndarray | None = None,
        state_transition: Callable[[dict[str, Any], int], dict[str, Any]] | None = None,
    ) -> tuple[int, SearchStats]:
        """Perform MCTS search from given state.

        Args:
            state: Current game state.
            goals: Active goals for conditioning.
            valid_actions: Mask of valid actions.
            state_transition: Function to transition to next state.

        Returns:
            Tuple of (selected_action, search_stats).
        """
        if valid_actions is None:
            # Default to all actions valid
            action_space = self.config.num_simulations  # Placeholder
            valid_actions = np.ones(action_space)

        if state_transition is None:
            state_transition = self._default_state_transition

        # Create root node
        root = MCTSNode(state)

        # Get initial policy and value from network
        priors, value = self._evaluate(state, goals)

        # Add Dirichlet noise at root for exploration
        priors = self._add_dirichlet_noise(priors, valid_actions)

        # Expand root
        root.expand(priors, valid_actions, state_transition)

        # Run simulations
        max_depth = 0
        for _sim in range(self.config.num_simulations):
            node = root
            depth = 0

            # Selection: traverse tree to leaf
            while node.is_expanded:
                if self.config.use_options and node.option_children:
                    # Option selection at higher levels
                    opt_id, node = node.select_option_child(self.config.c_puct)
                else:
                    # Action selection
                    _, child = node.select_child(self.config.c_puct)
                    if child is None:
                        break
                    node = child

                # Add virtual loss
                node.virtual_loss += self.config.virtual_loss
                depth += 1

                if depth >= self.config.horizon:
                    break

            max_depth = max(max_depth, depth)

            # Expansion and evaluation
            if depth < self.config.horizon:
                child_priors, leaf_value = self._evaluate(node.state, goals)
                child_valid = self._get_valid_actions(node.state)
                node.expand(child_priors, child_valid, state_transition)
            else:
                _, leaf_value = self._evaluate(node.state, goals)

            # Backpropagation
            node.backpropagate(leaf_value, self.config.discount)

        # Collect statistics
        stats = self._collect_stats(root, max_depth, self.config.num_simulations)

        # Select action
        action = self._select_action(root, self.config.temperature)

        return action, stats

    def search_with_options(
        self,
        state: dict[str, Any],
        goals: list[Goal],
        state_predictor: Callable[[dict[str, Any], Option], dict[str, Any]],
    ) -> tuple[Option | int, SearchStats]:
        """Search at higher levels using options.

        Args:
            state: Current game state.
            goals: Active goals.
            state_predictor: Function to predict state after option execution.

        Returns:
            Tuple of (selected_option_or_action, search_stats).
        """
        if not self.config.use_options or not self.options:
            # Fall back to primitive action search
            return self.search(state, goals)

        root = MCTSNode(state)

        # Filter applicable options
        applicable = [opt for opt in self.options if opt.can_initiate(state)]

        if not applicable:
            # No applicable options, use primitive actions
            return self.search(state, goals)

        # Get option priors from network (placeholder - could use separate head)
        option_priors = {opt.id: 1.0 / len(applicable) for opt in applicable}

        # Expand root with options
        root.expand_with_options(applicable, option_priors, state_predictor)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            depth = 0

            # Selection
            while node.is_expanded and node.option_children:
                opt_id, node = node.select_option_child(self.config.c_puct)
                node.virtual_loss += self.config.virtual_loss
                depth += 1

                if depth >= self.config.horizon:
                    break

            # Evaluation
            _, leaf_value = self._evaluate(node.state, goals)

            # Account for option execution cost
            if node.option:
                # Penalty for longer options
                leaf_value -= self.config.option_penalty * node.option.duration

            # Backpropagation
            node.backpropagate(leaf_value, self.config.discount)

        # Select best option
        best_option = self._select_option(root)
        stats = self._collect_option_stats(root)

        return best_option, stats

    def _evaluate(
        self,
        state: dict[str, Any],
        goals: list[Goal],
    ) -> tuple[np.ndarray, float]:
        """Evaluate state with neural network.

        Args:
            state: Game state.
            goals: Active goals.

        Returns:
            Tuple of (policy_priors, value).
        """
        if not TORCH_AVAILABLE or self.network is None:
            # Fallback: uniform policy, zero value
            action_space = 356  # Default action space
            return np.ones(action_space) / action_space, 0.0

        # Encode state
        self.state_encoder.encode(state, self.level)

        # Encode goals
        level_goals = [g for g in goals if g.level == self.level]
        goal_encoding = self.goal_encoder.encode_goals_for_level(level_goals, self.level, state)

        # Convert to tensors using configured shapes
        with torch.no_grad():
            # Get spatial and vector from state, use instance shape params for defaults
            spatial = (
                torch.from_numpy(
                    state.get("spatial", np.zeros(self.spatial_shape))
                ).float().unsqueeze(0)
            )
            vector = torch.from_numpy(
                state.get("vector", np.zeros(self.vector_dim))
            ).float().unsqueeze(0)
            goal_emb = (
                torch.from_numpy(
                    goal_encoding[0] if len(goal_encoding) > 0 else np.zeros(self.goal_dim)
                )
                .float()
                .unsqueeze(0)
            )

            # Forward pass
            policy_logits, value = self.network(spatial, vector, goal_emb, self.level)

            # Convert to numpy
            policy = torch.softmax(policy_logits, dim=-1).squeeze(0).numpy()
            value = value.item()

        return policy, value

    def _add_dirichlet_noise(
        self,
        priors: np.ndarray,
        valid_actions: np.ndarray,
    ) -> np.ndarray:
        """Add Dirichlet noise to priors at root.

        Args:
            priors: Prior probabilities.
            valid_actions: Valid action mask.

        Returns:
            Priors with added noise.
        """
        if self.config.dirichlet_epsilon == 0:
            return priors

        # Generate Dirichlet noise
        num_valid = int(valid_actions.sum())
        if num_valid == 0:
            return priors

        noise = np.zeros_like(priors)
        valid_indices = np.where(valid_actions > 0)[0]
        noise[valid_indices] = np.random.dirichlet([self.config.dirichlet_alpha] * num_valid)

        # Mix priors with noise
        noisy_priors = (
            1 - self.config.dirichlet_epsilon
        ) * priors + self.config.dirichlet_epsilon * noise

        # Renormalize
        noisy_priors = noisy_priors * valid_actions
        return noisy_priors / (noisy_priors.sum() + 1e-8)

    def _get_valid_actions(self, state: dict[str, Any]) -> np.ndarray:
        """Get valid action mask for state."""
        return state.get("action_mask", np.ones(356))

    def _default_state_transition(
        self,
        state: dict[str, Any],
        action: int,
    ) -> dict[str, Any]:
        """Default state transition (returns copy of state)."""
        # In practice, this should simulate the game
        return dict(state)

    def _select_action(self, root: MCTSNode, temperature: float) -> int:
        """Select action from root based on visit counts.

        Args:
            root: Root node.
            temperature: Selection temperature.

        Returns:
            Selected action index.
        """
        # Handle edge case: root has no children (never expanded)
        if not root.children:
            return 0  # Return fallback action

        max_action = max(root.children.keys())
        visits = np.array(
            [
                root.children[a].visit_count if a in root.children else 0
                for a in range(max_action + 1)
            ]
        )

        if temperature == 0:
            return int(np.argmax(visits))

        # Softmax with temperature
        visits_temp = visits ** (1.0 / temperature)
        probs = visits_temp / (visits_temp.sum() + 1e-8)

        return int(np.random.choice(len(probs), p=probs))

    def _select_option(self, root: MCTSNode) -> Option | None:
        """Select best option from root.

        Args:
            root: Root node.

        Returns:
            Selected option or None.
        """
        if not root.option_children:
            return None

        best_visits = -1
        best_option = None

        for _opt_id, child in root.option_children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_option = child.option

        return best_option

    def _collect_stats(
        self,
        root: MCTSNode,
        max_depth: int,
        simulations: int,
    ) -> SearchStats:
        """Collect search statistics from root.

        Args:
            root: Root node.
            max_depth: Maximum depth reached.
            simulations: Number of simulations.

        Returns:
            SearchStats instance.
        """
        if not root.children:
            return SearchStats()

        max_action = max(root.children.keys()) + 1
        visit_counts = np.zeros(max_action)
        q_values = np.zeros(max_action)
        prior_probs = np.zeros(max_action)

        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
            q_values[action] = child.q_value
            prior_probs[action] = child.prior

        return SearchStats(
            root_value=root.q_value,
            visit_counts=visit_counts,
            prior_probs=prior_probs,
            q_values=q_values,
            depth_reached=max_depth,
            simulations=simulations,
            options_considered=len(root.option_children),
        )

    def _collect_option_stats(self, root: MCTSNode) -> SearchStats:
        """Collect statistics for option search.

        Args:
            root: Root node.

        Returns:
            SearchStats for options.
        """
        return SearchStats(
            root_value=root.q_value,
            simulations=self.config.num_simulations,
            options_considered=len(root.option_children),
            depth_reached=0,
        )
