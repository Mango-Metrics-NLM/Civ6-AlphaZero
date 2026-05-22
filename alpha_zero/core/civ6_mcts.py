"""Monte Carlo Tree Search for Civ6 AlphaZero.

Implements PUCT-based MCTS with neural network guidance, supporting:
- Action masking for valid moves
- Dirichlet noise for exploration
- Temperature-based action selection

Also implements Gumbel MuZero MCTS for efficient search with few simulations:
- Gumbel-Top-k sampling for root action selection
- Sequential halving for simulation allocation
- Works effectively with 4-32 simulations instead of 200+

MuZero MCTS variant that uses learned dynamics (no environment cloning):
- Uses representation function to create initial hidden state
- Uses dynamics function for tree traversal (world model)
- No environment cloning during search
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, SupportsFloat

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from alpha_zero.core.mcts_constants import (
    DEFAULT_C_VISIT,
    DEFAULT_C_VISIT_DIVISOR,
    DEFAULT_ROLLOUT_DEPTH,
    EPSILON,
    GREEDY_THRESHOLD,
)
from alpha_zero.utils.constants import MCTS

# Module-level logger
logger = logging.getLogger(__name__)


class MCTSNetwork(Protocol):
    """Protocol for neural networks used in MCTS."""

    def __call__(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Call the network directly."""
        ...

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

    def eval(self) -> Any:
        """Set network to evaluation mode."""
        ...


class MuZeroNetworkProtocol(Protocol):
    """Protocol for MuZero-compatible networks with learned dynamics."""

    def initial_inference(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> Any:
        """Initial inference from observation.

        Args:
            spatial: Spatial features (B, C, H, W).
            vector: Vector features (B, D).
            action_mask: Valid action mask (B, A).

        Returns:
            Output with hidden_state, policy_logits, value attributes.
        """
        ...

    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recurrent inference during MCTS search.

        Args:
            hidden_state: Current hidden state (B, C, H, W).
            action: Action indices (B,).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (next_hidden_state, reward, policy_logits, value).
        """
        ...

    def prediction_function(
        self,
        hidden_state: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get policy and value from hidden state.

        Args:
            hidden_state: Hidden state (B, C, H, W).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_logits, value).
        """
        ...

    def eval(self) -> Any:
        """Set network to evaluation mode."""
        ...


class ActionSpace(Protocol):
    """Protocol for action space with size attribute."""

    n: int


class MCTSEnvironment(Protocol):
    """Protocol for environments compatible with MCTS.

    MCTS requires environments that can be cloned for simulation,
    stepped through, and queried for observations and valid actions.

    Attributes:
        action_space: Action space with `n` attribute for size.
        current_player: Index of the current player (for two-player games).

    Note:
        Implementations must provide the public `get_observation()` and
        `get_info()` methods. The private `_get_observation()` and
        `_get_info()` methods are deprecated and should not be used
        in new code.
    """

    action_space: ActionSpace
    current_player: int

    def clone(self) -> MCTSEnvironment:
        """Create a deep copy of the environment for simulation."""
        ...

    def step(self, action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        ...

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation dict with 'spatial', 'vector', 'action_mask'."""
        ...

    def get_info(self) -> dict[str, Any]:
        """Get environment info dict (may contain 'winner' key)."""
        ...

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions (1 = valid, 0 = invalid)."""
        ...


@dataclass(slots=True)
class MCTSConfig:
    """Configuration for MCTS.

    Defaults sourced from alpha_zero.utils.constants.MCTS to keep hyperparameter
    values consistent across all MCTS variants and the orchestrator config.
    """

    num_simulations: int = MCTS.CLASSIC_NUM_SIMULATIONS
    c_puct: float = MCTS.DEFAULT_C_PUCT
    dirichlet_alpha: float = MCTS.DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = MCTS.DEFAULT_DIRICHLET_EPSILON
    temperature: float = MCTS.DEFAULT_TEMPERATURE
    temperature_threshold: int = MCTS.DEFAULT_TEMPERATURE_THRESHOLD

    def __post_init__(self) -> None:
        if not MCTS.MIN_SIMULATIONS <= self.num_simulations <= MCTS.MAX_SIMULATIONS:
            raise ValueError(
                f"num_simulations must be in [{MCTS.MIN_SIMULATIONS}, "
                f"{MCTS.MAX_SIMULATIONS}]; got {self.num_simulations}"
            )
        if self.c_puct <= 0:
            raise ValueError(f"c_puct must be > 0; got {self.c_puct}")
        if self.dirichlet_alpha <= 0:
            raise ValueError(f"dirichlet_alpha must be > 0; got {self.dirichlet_alpha}")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError(f"dirichlet_epsilon must be in [0, 1]; got {self.dirichlet_epsilon}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0; got {self.temperature}")
        if self.temperature_threshold < 0:
            raise ValueError(
                f"temperature_threshold must be >= 0; got {self.temperature_threshold}"
            )


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
class MCTSNode:
    """Node in the MCTS tree.

    Attributes:
        prior: Prior probability from neural network.
        visit_count: Number of times node was visited.
        value_sum: Sum of values from all visits.
        children: Child nodes keyed by action.
        parent: Parent node.
        action: Action that led to this node.
        virtual_loss: Virtual loss for parallel MCTS (threads in flight).
    """

    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, MCTSNode] = field(default_factory=dict)
    parent: MCTSNode | None = None
    action: int | None = None
    virtual_loss: int = 0

    @property
    def value(self) -> float:
        """Mean value estimate."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def effective_value(self) -> float:
        """Value adjusted for virtual loss (for parallel MCTS)."""
        total_visits = self.visit_count + self.virtual_loss
        if total_visits == 0:
            return 0.0
        # Virtual loss assumes worst case (-1)
        adjusted_sum = self.value_sum - self.virtual_loss
        return adjusted_sum / total_visits

    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0

    def add_virtual_loss(self, loss: int = 1) -> None:
        """Add virtual loss for parallel MCTS.

        Virtual loss discourages other threads from exploring the same path.

        Args:
            loss: Amount of virtual loss to add.
        """
        self.virtual_loss += loss

    def remove_virtual_loss(self, loss: int = 1) -> None:
        """Remove virtual loss after backpropagation.

        Args:
            loss: Amount of virtual loss to remove.
        """
        self.virtual_loss = max(0, self.virtual_loss - loss)

    def select_child(self, c_puct: float, use_virtual_loss: bool = False) -> tuple[int, MCTSNode]:
        """Select child with highest UCB score.

        Args:
            c_puct: Exploration constant.
            use_virtual_loss: Whether to use virtual loss in selection.

        Returns:
            Tuple of (action, child_node).
        """
        best_score = -float("inf")
        best_action = -1
        best_child = None

        sqrt_parent_visits = np.sqrt(
            self.visit_count + self.virtual_loss if use_virtual_loss else self.visit_count
        )

        for action, child in self.children.items():
            # PUCT formula with optional virtual loss
            if use_virtual_loss:
                q_value = -child.effective_value  # Use virtual-loss-adjusted value
                effective_visits = child.visit_count + child.virtual_loss
            else:
                q_value = -child.value  # Negamax: child's value is opponent's
                effective_visits = child.visit_count
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + effective_visits)
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
        valid_policy = valid_policy / (valid_policy.sum() + EPSILON)

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
    Optionally integrates with ReflectionMemory for LATS-style learning.
    """

    def __init__(
        self,
        network: MCTSNetwork,
        config: MCTSConfig | None = None,
        device: str = "cpu",
        reflection_memory: Any | None = None,
        llm_prior_fn: Callable[[Any, np.ndarray], np.ndarray] | None = None,
        llm_prior_weight: float = 0.5,
    ) -> None:
        """Initialize MCTS.

        Args:
            network: Neural network implementing MCTSNetwork protocol.
            config: MCTS configuration.
            device: Device for neural network inference.
            reflection_memory: Optional ReflectionMemory for value adjustments
                based on past failures. When provided, actions that led to
                failures in similar states will be penalized.
            llm_prior_fn: Optional callable ``(env, action_mask) -> prior``
                returning a 1-D distribution over actions (same length as
                ``action_mask``). When set, the network policy at expansion
                time is blended as
                ``policy = (1 - w) * network_policy + w * llm_prior``,
                renormalised over legal actions. ``None`` (default) leaves
                MCTS behaviour byte-identical to the pre-extension code path.
            llm_prior_weight: Blend weight ``w`` in ``[0, 1]``. Only used
                when ``llm_prior_fn`` is provided.
        """
        self.network = network
        self.config = config or MCTSConfig()
        self.device = device
        self._reflection_memory = reflection_memory
        if not 0.0 <= llm_prior_weight <= 1.0:
            raise ValueError(f"llm_prior_weight must be in [0, 1]; got {llm_prior_weight}")
        self._llm_prior_fn = llm_prior_fn
        self._llm_prior_weight = llm_prior_weight
        self._llm_warned = False
        self._search_count = 0
        logger.debug(
            "Civ6MCTS initialized",
            extra={
                "num_simulations": self.config.num_simulations,
                "c_puct": self.config.c_puct,
                "device": device,
                "has_reflection_memory": reflection_memory is not None,
                "has_llm_prior": llm_prior_fn is not None,
                "llm_prior_weight": llm_prior_weight,
            },
        )

    def search(
        self,
        env: MCTSEnvironment,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Run MCTS search from current state.

        Args:
            env: Game environment (must support clone()).
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            Tuple of (action_probs, root_value).
        """
        self._search_count += 1
        logger.debug(
            "Starting MCTS search",
            extra={
                "search_id": self._search_count,
                "num_simulations": self.config.num_simulations,
                "add_noise": add_noise,
            },
        )

        # Create root node
        root = MCTSNode()

        # Get initial policy and value from network
        obs = env.get_observation()
        spatial = torch.from_numpy(obs["spatial"]).float().to(self.device)
        vector = torch.from_numpy(obs["vector"]).float().to(self.device)
        action_mask = obs["action_mask"]

        policy, value = self._evaluate(spatial, vector, action_mask)

        # Optional LLM-prior blend (no-op when llm_prior_fn is None).
        policy = self._blend_llm_prior(env, policy, action_mask)

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
                info = sim_env.get_info()
                winner = info.get("winner")
                if winner is None:
                    value = 0.0  # Draw
                elif winner == sim_env.current_player:
                    value = 1.0
                else:
                    value = -1.0
            else:
                # Use neural network
                obs = sim_env.get_observation()
                spatial = torch.from_numpy(obs["spatial"]).float().to(self.device)
                vector = torch.from_numpy(obs["vector"]).float().to(self.device)
                action_mask = obs["action_mask"]

                policy, value = self._evaluate(spatial, vector, action_mask)

                # Optional LLM-prior blend (no-op when llm_prior_fn is None).
                policy = self._blend_llm_prior(sim_env, policy, action_mask)

                # Expand leaf node
                node.expand(policy, action_mask)

            # Backpropagation
            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                value = -value  # Flip for opponent

            # Explicit cleanup to prevent memory accumulation
            del sim_env

        # Get action probabilities from visit counts
        action_probs = self._get_action_probs(root, env.action_space.n)

        return action_probs, root.value

    def _evaluate(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: np.ndarray,
        action: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Evaluate state using neural network.

        Optionally applies value adjustments from reflection memory
        to penalize actions that led to failures in similar states.

        Args:
            spatial: Spatial features.
            vector: Vector features.
            action_mask: Valid action mask.
            action: Optional action to get reflection-based adjustment for.

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
            # Ensure value is a scalar - squeeze all dimensions then convert
            value_scalar = float(value.squeeze().cpu().item())

        # Apply reflection-based value adjustment if available
        if self._reflection_memory is not None and action is not None:
            try:
                # Check if reflection memory is initialized
                if (
                    hasattr(self._reflection_memory, "initialized")
                    and self._reflection_memory.initialized
                ):
                    state = {
                        "spatial": spatial.cpu().numpy(),
                        "vector": vector.cpu().numpy(),
                    }
                    adjustment = self._reflection_memory.get_value_adjustment_sync(state, action)
                    value_scalar = value_scalar + adjustment
            except Exception as e:
                logger.debug(f"Reflection memory adjustment failed: {e}")

        return policy, value_scalar

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

    def _blend_llm_prior(
        self,
        env: Any,
        network_policy: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Blend the network policy with an optional LLM-supplied prior.

        Returns the input policy unchanged when ``llm_prior_fn`` is ``None``,
        so behaviour is byte-identical to the pre-extension code path.

        On any failure inside the user-supplied callable (exception or
        shape mismatch), we fall back to the network policy and log a
        warning **once per Civ6MCTS instance** (gated by ``_llm_warned``)
        so a misbehaving LLM cannot spam the MCTS hot loop. MCTS never
        crashes due to advisor failure.
        """
        if self._llm_prior_fn is None:
            return network_policy
        try:
            llm_prior = self._llm_prior_fn(env, action_mask)
        except Exception as exc:  # pragma: no cover - defensive logging
            if not self._llm_warned:
                logger.warning("llm_prior_fn raised %s; using network policy", exc)
                self._llm_warned = True
            return network_policy
        llm_prior = np.asarray(llm_prior, dtype=np.float64)
        if llm_prior.shape != network_policy.shape:
            if not self._llm_warned:
                logger.warning(
                    "llm_prior shape %s != policy shape %s; using network policy",
                    llm_prior.shape,
                    network_policy.shape,
                )
                self._llm_warned = True
            return network_policy
        # Mask + blend + renormalise.
        masked_llm = llm_prior * (action_mask > 0).astype(np.float64)
        total = masked_llm.sum()
        if total <= 0:
            return network_policy
        masked_llm = masked_llm / total
        w = float(self._llm_prior_weight)
        blended = (1.0 - w) * np.asarray(network_policy, dtype=np.float64) + w * masked_llm
        blended = blended * (action_mask > 0).astype(np.float64)
        blended_sum = blended.sum()
        if blended_sum <= 0:
            return network_policy
        return (blended / blended_sum).astype(network_policy.dtype)

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
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Select action based on probabilities and temperature.

        Args:
            action_probs: Action probability distribution.
            temperature: Temperature for action selection.
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

        if temperature == 0:
            # Greedy selection
            return int(np.argmax(action_probs))

        # Apply temperature
        if temperature != 1.0:
            probs = np.power(action_probs + 1e-8, 1.0 / temperature)
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
        rollout_depth: int = DEFAULT_ROLLOUT_DEPTH,
    ) -> None:
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
        self._search_count = 0
        logger.debug(
            "SimpleMCTS initialized",
            extra={
                "num_simulations": num_simulations,
                "c_puct": c_puct,
                "rollout_depth": rollout_depth,
            },
        )

    def search(self, env: MCTSEnvironment) -> tuple[np.ndarray, float]:
        """Run MCTS search.

        Args:
            env: Game environment (must support clone(), step(), get_valid_actions()).

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
                info = sim_env.get_info()
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

    def _rollout(self, env: MCTSEnvironment) -> float:
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
                info = env.get_info()
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
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Select action based on probabilities.

        Args:
            action_probs: Action probabilities.
            temperature: Selection temperature.
            action_mask: Optional valid action mask (1=valid, 0=invalid).
                        If provided, ensures only valid actions are selected.

        Returns:
            Selected action.

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

        if temperature == 0:
            return int(np.argmax(action_probs))

        if temperature != 1.0:
            probs = np.power(action_probs + EPSILON, 1.0 / temperature)
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
    valid_policy = np.clip(valid_policy, EPSILON, 1.0)
    log_probs = np.log(valid_policy)

    # Sample Gumbel noise: g ~ Gumbel(0, 1)
    # Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1)
    uniform = np.random.uniform(size=len(valid_actions))
    gumbel_noise = -np.log(-np.log(np.clip(uniform, EPSILON, 1 - EPSILON)))

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
        self.device = device

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


# =============================================================================
# MuZero MCTS Implementation (Hidden State Based)
# =============================================================================


@dataclass
class MuZeroMCTSConfig:
    """Configuration for MuZero MCTS with learned dynamics.

    This MCTS variant uses the network's dynamics function for tree traversal
    instead of cloning environments. This enables planning in games where
    environment cloning is expensive or impossible.

    Attributes:
        num_simulations: Number of MCTS simulations.
        c_puct: Exploration constant for PUCT formula.
        dirichlet_alpha: Alpha parameter for Dirichlet noise at root.
        dirichlet_epsilon: Fraction of noise to mix with prior.
        use_gumbel: Use Gumbel-Top-k sampling at root.
        num_sampled_actions: Number of actions to sample with Gumbel.
        use_sequential_halving: Use sequential halving for simulation allocation.
        temperature: Temperature for action selection.
        discount: Discount factor for value computation.
        use_categorical_reward: If True, network outputs categorical reward distribution.
            If False, network outputs scalar reward. Must match network configuration.
        reward_support_size: Size of categorical support (only used if use_categorical_reward=True).
    """

    num_simulations: int = MCTS.MUZERO_NUM_SIMULATIONS
    c_puct: float = MCTS.DEFAULT_C_PUCT
    dirichlet_alpha: float = MCTS.DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = MCTS.DEFAULT_DIRICHLET_EPSILON
    use_gumbel: bool = True
    num_sampled_actions: int = MCTS.GUMBEL_NUM_SAMPLED_ACTIONS
    use_sequential_halving: bool = True
    temperature: float = MCTS.DEFAULT_TEMPERATURE
    temperature_threshold: int = MCTS.DEFAULT_TEMPERATURE_THRESHOLD
    discount: float = MCTS.MUZERO_DISCOUNT
    use_categorical_reward: bool = True
    reward_support_size: int = MCTS.MUZERO_REWARD_SUPPORT_SIZE

    def __post_init__(self) -> None:
        if not MCTS.MIN_SIMULATIONS <= self.num_simulations <= MCTS.MAX_SIMULATIONS:
            raise ValueError(
                f"num_simulations must be in [{MCTS.MIN_SIMULATIONS}, "
                f"{MCTS.MAX_SIMULATIONS}]; got {self.num_simulations}"
            )
        if self.c_puct <= 0:
            raise ValueError(f"c_puct must be > 0; got {self.c_puct}")
        if self.dirichlet_alpha <= 0:
            raise ValueError(f"dirichlet_alpha must be > 0; got {self.dirichlet_alpha}")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError(f"dirichlet_epsilon must be in [0, 1]; got {self.dirichlet_epsilon}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0; got {self.temperature}")
        if self.temperature_threshold < 0:
            raise ValueError(
                f"temperature_threshold must be >= 0; got {self.temperature_threshold}"
            )
        # num_sampled_actions is used by _gumbel_search via min(num_sampled_actions,
        # len(valid_actions)); a non-positive value would propagate into
        # gumbel_sample_actions and crash at search time.
        if self.use_gumbel and self.num_sampled_actions <= 0:
            raise ValueError(
                f"num_sampled_actions must be > 0 when use_gumbel=True; "
                f"got {self.num_sampled_actions}"
            )
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1]; got {self.discount}")
        if self.use_categorical_reward and self.reward_support_size <= 0:
            raise ValueError(
                f"reward_support_size must be > 0 when use_categorical_reward=True; "
                f"got {self.reward_support_size}"
            )


@dataclass
class MuZeroNode:
    """Node in MuZero MCTS tree that stores hidden state.

    Unlike standard MCTS nodes, MuZeroNode stores the learned hidden state
    representation rather than relying on environment state.

    Attributes:
        hidden_state: Hidden state tensor from dynamics/representation function.
        reward: Predicted reward for transition to this state.
        prior: Prior probability from policy network.
        visit_count: Number of times node was visited.
        value_sum: Sum of values from all visits.
        children: Child nodes keyed by action.
        parent: Parent node.
        action: Action that led to this node.
    """

    hidden_state: torch.Tensor | None = None
    reward: float = 0.0
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, MuZeroNode] = field(default_factory=dict)
    parent: MuZeroNode | None = None
    action: int | None = None

    @property
    def value(self) -> float:
        """Mean value estimate (Q-value)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0

    def select_child(
        self, c_puct: float, min_q: float, max_q: float
    ) -> tuple[int, MuZeroNode | None]:
        """Select child with highest UCB score using normalized Q-values.

        Args:
            c_puct: Exploration constant.
            min_q: Minimum Q-value in tree for normalization.
            max_q: Maximum Q-value in tree for normalization.

        Returns:
            Tuple of (action, child_node). child_node may be None if no children.
        """
        best_score = -float("inf")
        best_action = -1
        best_child: MuZeroNode | None = None

        sqrt_parent_visits = np.sqrt(self.visit_count)

        for action, child in self.children.items():
            # Normalized Q-value
            if child.visit_count > 0:
                q_value = child.value
                # Normalize to [0, 1] range
                if max_q > min_q:
                    q_value = (q_value - min_q) / (max_q - min_q)
                else:
                    q_value = 0.5
            else:
                q_value = 0.0

            # PUCT formula
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None, "select_child called on node with no children"
        return best_action, best_child

    def expand(
        self,
        actions: list[int],
        priors: np.ndarray,
    ) -> None:
        """Expand node with children for given actions.

        Args:
            actions: List of valid action indices.
            priors: Prior probabilities for each action.
        """
        for i, action in enumerate(actions):
            self.children[action] = MuZeroNode(
                prior=float(priors[i]),
                parent=self,
                action=action,
            )


class MuZeroMCTS:
    """MuZero MCTS that uses learned dynamics instead of environment cloning.

    This implementation follows the MuZero algorithm where:
    1. The representation function converts observations to hidden states.
    2. The dynamics function predicts next hidden states and rewards.
    3. The prediction function outputs policy and value from hidden states.

    No environment cloning is needed during search - all tree traversal
    happens in the learned hidden state space.

    Reference: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
    """

    def __init__(
        self,
        network: MuZeroNetworkProtocol,
        config: MuZeroMCTSConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize MuZero MCTS.

        Args:
            network: MuZeroNetwork with initial_inference and recurrent_inference.
            config: MuZero MCTS configuration.
            device: Device for neural network inference.
        """
        self.network = network
        self.config = config or MuZeroMCTSConfig()
        self.device = device

    @classmethod
    def from_network(
        cls,
        network: torch.nn.Module,
        config: MuZeroMCTSConfig | None = None,
        device: str = "cpu",
    ) -> MuZeroMCTS:
        """Create MuZeroMCTS with config derived from network.

        This factory method ensures the MCTS config matches the network
        configuration for categorical reward settings.

        Args:
            network: MuZeroNetwork instance.
            config: Base config (reward settings will be overridden from network).
            device: Device for inference.

        Returns:
            MuZeroMCTS instance with consistent configuration.
        """
        base_config = config or MuZeroMCTSConfig()

        # Extract network config if available
        if hasattr(network, "config"):
            net_config = network.config
            synced_config = MuZeroMCTSConfig(
                num_simulations=base_config.num_simulations,
                c_puct=base_config.c_puct,
                dirichlet_alpha=base_config.dirichlet_alpha,
                dirichlet_epsilon=base_config.dirichlet_epsilon,
                use_gumbel=base_config.use_gumbel,
                num_sampled_actions=base_config.num_sampled_actions,
                use_sequential_halving=base_config.use_sequential_halving,
                temperature=base_config.temperature,
                temperature_threshold=base_config.temperature_threshold,
                discount=base_config.discount,
                # Sync with network config
                use_categorical_reward=getattr(net_config, "use_categorical_reward", True),
                reward_support_size=getattr(
                    net_config, "reward_support_size", base_config.reward_support_size
                ),
            )
            return cls(network, synced_config, device)  # type: ignore[arg-type]

        return cls(network, base_config, device)  # type: ignore[arg-type]

    def _convert_reward_to_scalar(self, reward: torch.Tensor) -> float:
        """Convert network reward output to scalar value.

        Handles both categorical and scalar reward formats based on config.

        Args:
            reward: Reward tensor from network. Expected shapes:
                - Categorical: (B, support_size) or (support_size,)
                - Scalar: (B,) or (B, 1) or scalar

        Returns:
            Scalar reward value.

        Raises:
            ValueError: If reward shape doesn't match expected format.
        """
        if self.config.use_categorical_reward:
            # Expect categorical distribution (B, support_size) or (support_size,)
            from alpha_zero.core.muzero_network import categorical_to_scalar

            if reward.dim() == 0:
                raise ValueError(
                    f"use_categorical_reward=True but got scalar reward with shape {reward.shape}"
                )
            elif reward.dim() == 1:
                # Shape (support_size,) - single sample without batch dim
                # Or shape (B,) where B == support_size (ambiguous - assume categorical)
                if reward.shape[0] == self.config.reward_support_size:
                    scalar_reward = categorical_to_scalar(
                        reward.unsqueeze(0),
                        apply_softmax=True,
                    )
                    return scalar_reward[0].item()
                else:
                    raise ValueError(
                        f"Expected categorical reward with support_size={self.config.reward_support_size}, "
                        f"got shape {reward.shape}"
                    )
            elif reward.dim() == 2:
                # Shape (B, support_size)
                if reward.shape[1] != self.config.reward_support_size:
                    raise ValueError(
                        f"Expected categorical reward with support_size={self.config.reward_support_size}, "
                        f"got shape {reward.shape}"
                    )
                scalar_reward = categorical_to_scalar(
                    reward,
                    apply_softmax=True,
                )
                return scalar_reward[0].item()
            else:
                raise ValueError(f"Unexpected categorical reward shape: {reward.shape}")
        else:
            # Expect scalar reward: shape (), (1,), (B,), or (B, 1)
            if reward.dim() == 0:
                return reward.item()
            elif reward.dim() == 1:
                return reward[0].item()
            elif reward.dim() == 2 and reward.shape[1] == 1:
                return reward[0, 0].item()
            else:
                raise ValueError(
                    f"use_categorical_reward=False but got unexpected reward shape {reward.shape}. "
                    f"Expected scalar with shape (), (1,), (B,), or (B, 1)"
                )

    def search(
        self,
        observation: dict,
        action_mask: np.ndarray,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Run MuZero MCTS search from observation.

        Unlike standard MCTS, this does not require an environment object -
        only the observation and valid action mask.

        Args:
            observation: Dictionary with 'spatial' and 'vector' keys.
            action_mask: Valid action mask (A,).
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            Tuple of (action_probs, root_value).
        """
        # Convert observation to tensors
        spatial = torch.from_numpy(observation["spatial"]).float().to(self.device)
        vector = torch.from_numpy(observation["vector"]).float().to(self.device)
        mask_tensor = torch.from_numpy(action_mask).float().to(self.device)

        # Initial inference: observation -> hidden_state, policy, value
        self.network.eval()
        with torch.no_grad():
            output = self.network.initial_inference(
                spatial.unsqueeze(0),
                vector.unsqueeze(0),
                mask_tensor.unsqueeze(0),
            )

        # Create root node
        root = MuZeroNode(
            hidden_state=output.hidden_state,  # (1, C, H, W)
        )

        # Get valid actions and their priors
        policy = F.softmax(output.policy_logits, dim=-1).squeeze(0).cpu().numpy()
        valid_actions = np.where(action_mask > 0)[0].tolist()

        if len(valid_actions) == 0:
            return np.zeros(len(action_mask), dtype=np.float32), 0.0

        # Normalize policy over valid actions
        valid_priors = policy[valid_actions]
        valid_priors = valid_priors / (valid_priors.sum() + EPSILON)

        # Expand root
        root.expand(valid_actions, valid_priors)
        root_value = output.value.item()

        # Add Dirichlet noise at root for exploration
        if add_noise and not self.config.use_gumbel:
            self._add_dirichlet_noise(root)

        # Track min/max Q-values for normalization
        min_q = float("inf")
        max_q = float("-inf")

        # Run simulations
        if self.config.use_gumbel:
            # Gumbel-based simulation allocation
            best_action = self._gumbel_search(root, valid_actions, policy, action_mask, root_value)
            action_probs = np.zeros(len(action_mask), dtype=np.float32)
            action_probs[best_action] = 1.0
        else:
            # Standard PUCT simulations
            for _ in range(self.config.num_simulations):
                node = root
                search_path = [node]

                # Selection: traverse tree using PUCT until we find unexpanded node
                while node.is_expanded():
                    action, child = node.select_child(self.config.c_puct, min_q, max_q)
                    if child is None:
                        break
                    search_path.append(child)

                    # If child has no hidden state, compute it but don't expand yet
                    if child.hidden_state is None:
                        if node.hidden_state is not None:
                            self._compute_hidden_state(
                                child, node.hidden_state, action, action_mask
                            )
                        # Stop traversal - this is our leaf
                        node = child
                        break

                    node = child

                # Expansion and evaluation of leaf
                if node.hidden_state is not None:
                    value = self._get_value(node, action_mask)

                    # Expand leaf node for future simulations
                    if not node.is_expanded():
                        self._expand_leaf(node, action_mask)

                    # Update min/max Q
                    min_q = min(min_q, value)
                    max_q = max(max_q, value)

                    # Backpropagation
                    self._backpropagate(search_path, value)

            # Get action probabilities from visit counts
            action_probs = self._get_action_probs(root, len(action_mask))

        return action_probs, root.value

    def _gumbel_search(
        self,
        root: MuZeroNode,
        valid_actions: list[int],
        policy: np.ndarray,
        action_mask: np.ndarray,
        root_value: float,
    ) -> int:
        """Run Gumbel-based search with sequential halving.

        Args:
            root: Root node with hidden state.
            valid_actions: List of valid action indices.
            policy: Full policy distribution.
            action_mask: Valid action mask.
            root_value: Initial value estimate.

        Returns:
            Best action index.
        """
        # Gumbel sampling at root
        sampled_actions, gumbel_scores = gumbel_sample_actions(
            policy,
            action_mask,
            min(self.config.num_sampled_actions, len(valid_actions)),
        )

        if len(sampled_actions) == 0:
            return valid_actions[0] if valid_actions else 0

        if len(sampled_actions) == 1:
            return int(sampled_actions[0])

        # Store gumbel scores
        scores = dict(zip(sampled_actions, gumbel_scores, strict=False))

        if self.config.use_sequential_halving:
            return self._sequential_halving_muzero(
                root, sampled_actions, scores, action_mask, root_value
            )
        else:
            return self._simple_muzero_search(root, sampled_actions, action_mask)

    def _sequential_halving_muzero(
        self,
        root: MuZeroNode,
        candidate_actions: np.ndarray,
        gumbel_scores: dict[int, float],
        action_mask: np.ndarray,
        root_value: float,
    ) -> int:
        """Sequential halving with MuZero dynamics.

        Args:
            root: Root node.
            candidate_actions: Actions selected by Gumbel sampling.
            gumbel_scores: Gumbel scores for candidates.
            action_mask: Valid action mask.
            root_value: Initial value estimate.

        Returns:
            Best action index.
        """
        actions = list(candidate_actions)

        # Calculate number of halving rounds
        num_rounds = max(1, int(np.ceil(np.log2(len(actions)))))
        sims_per_round = max(1, self.config.num_simulations // num_rounds)

        for _round_idx in range(num_rounds):
            if len(actions) <= 1:
                break

            # Allocate simulations evenly
            sims_per_action = max(1, sims_per_round // len(actions))

            # Run simulations for each action
            for action in actions:
                for _ in range(sims_per_action):
                    self._run_muzero_simulation(root, action, action_mask)

            # Compute completed Q-values
            max_visits = max(root.children[a].visit_count for a in actions if a in root.children)

            completed_qs = {}
            for action in actions:
                if action not in root.children:
                    continue
                child = root.children[action]
                q = child.value if child.visit_count > 0 else 0.0
                completed_qs[action] = compute_completed_q(
                    q, child.visit_count, max_visits, root_value
                )

            # Combine with Gumbel scores and halve
            combined = {a: completed_qs.get(a, 0.0) + gumbel_scores.get(a, 0.0) for a in actions}
            actions = sorted(actions, key=lambda a: -combined[a])
            actions = actions[: max(len(actions) // 2, 1)]

        return int(actions[0])

    def _simple_muzero_search(
        self,
        root: MuZeroNode,
        candidate_actions: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """Simple search without sequential halving.

        Args:
            root: Root node.
            candidate_actions: Actions to consider.
            action_mask: Valid action mask.

        Returns:
            Best action index.
        """
        # Distribute simulations
        for _ in range(self.config.num_simulations):
            # Pick action with lowest visit count
            min_visits = float("inf")
            best_action = candidate_actions[0]
            for action in candidate_actions:
                if action in root.children:
                    visits = root.children[action].visit_count
                    if visits < min_visits:
                        min_visits = visits
                        best_action = action

            self._run_muzero_simulation(root, best_action, action_mask)

        # Select by visit count
        return int(
            max(
                candidate_actions,
                key=lambda a: root.children[a].visit_count if a in root.children else 0,
            )
        )

    def _run_muzero_simulation(
        self,
        root: MuZeroNode,
        action: int,
        action_mask: np.ndarray,
    ) -> None:
        """Run a single MCTS simulation using dynamics.

        Args:
            root: Root node.
            action: Action to take from root.
            action_mask: Valid action mask for expansion.
        """
        if action not in root.children:
            return

        node = root.children[action]
        search_path = [root, node]

        # Compute hidden state for first child if needed
        if node.hidden_state is None and root.hidden_state is not None:
            self._expand_node(node, root.hidden_state, action, action_mask)

        # Track min/max for normalization
        min_q = 0.0
        max_q = 1.0

        # Traverse tree
        depth = 0
        max_depth = DEFAULT_ROLLOUT_DEPTH  # Prevent infinite loops

        while node.is_expanded() and depth < max_depth:
            action, child = node.select_child(self.config.c_puct, min_q, max_q)
            if child is None:
                break

            search_path.append(child)

            # Compute hidden state via dynamics if needed
            if child.hidden_state is None and node.hidden_state is not None:
                self._expand_node(child, node.hidden_state, action, action_mask)

            node = child
            depth += 1

        # Get value from leaf
        if node.hidden_state is not None:
            value = self._get_value(node, action_mask)
        else:
            value = 0.0

        # Backpropagate with discounting
        self._backpropagate_with_rewards(search_path, value)

    def _compute_hidden_state(
        self,
        node: MuZeroNode,
        parent_hidden: torch.Tensor,
        action: int,
        action_mask: np.ndarray,
    ) -> None:
        """Compute hidden state for node using dynamics (without expanding).

        Args:
            node: Node to compute hidden state for.
            parent_hidden: Parent's hidden state.
            action: Action taken to reach this node.
            action_mask: Valid action mask.
        """
        self.network.eval()
        with torch.no_grad():
            action_tensor = torch.tensor([action], device=self.device)
            mask_tensor = torch.from_numpy(action_mask).float().to(self.device)

            next_hidden, reward, _policy_logits, _value = self.network.recurrent_inference(
                parent_hidden,
                action_tensor,
                mask_tensor.unsqueeze(0),
            )

        node.hidden_state = next_hidden
        node.reward = self._convert_reward_to_scalar(reward)

    def _expand_leaf(
        self,
        node: MuZeroNode,
        action_mask: np.ndarray,
    ) -> None:
        """Expand a leaf node by creating children based on policy.

        Args:
            node: Leaf node to expand.
            action_mask: Valid action mask.
        """
        if node.hidden_state is None or node.is_expanded():
            return

        self.network.eval()
        with torch.no_grad():
            mask_tensor = torch.from_numpy(action_mask).float().to(self.device)
            policy_logits, _ = self.network.prediction_function(
                node.hidden_state,
                mask_tensor.unsqueeze(0),
            )
            policy = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        valid_actions = np.where(action_mask > 0)[0].tolist()
        if len(valid_actions) > 0:
            valid_priors = policy[valid_actions]
            valid_priors = valid_priors / (valid_priors.sum() + EPSILON)
            node.expand(valid_actions, valid_priors)

    def _expand_node(
        self,
        node: MuZeroNode,
        parent_hidden: torch.Tensor,
        action: int,
        action_mask: np.ndarray,
    ) -> None:
        """Expand node using dynamics function (compute hidden state + expand).

        Args:
            node: Node to expand.
            parent_hidden: Parent's hidden state.
            action: Action taken to reach this node.
            action_mask: Valid action mask.
        """
        self.network.eval()
        with torch.no_grad():
            action_tensor = torch.tensor([action], device=self.device)
            mask_tensor = torch.from_numpy(action_mask).float().to(self.device)

            next_hidden, reward, policy_logits, _value = self.network.recurrent_inference(
                parent_hidden,
                action_tensor,
                mask_tensor.unsqueeze(0),
            )

        node.hidden_state = next_hidden
        node.reward = self._convert_reward_to_scalar(reward)

        # Get valid actions and expand
        valid_actions = np.where(action_mask > 0)[0].tolist()
        if len(valid_actions) > 0:
            policy = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            valid_priors = policy[valid_actions]
            valid_priors = valid_priors / (valid_priors.sum() + EPSILON)
            node.expand(valid_actions, valid_priors)

    def _get_value(self, node: MuZeroNode, action_mask: np.ndarray) -> float:
        """Get value estimate from node's hidden state.

        Args:
            node: Node with hidden state.
            action_mask: Valid action mask.

        Returns:
            Value estimate.
        """
        if node.hidden_state is None:
            return 0.0

        self.network.eval()
        with torch.no_grad():
            mask_tensor = torch.from_numpy(action_mask).float().to(self.device)
            _, value = self.network.prediction_function(
                node.hidden_state,
                mask_tensor.unsqueeze(0),
            )
        return value.item()

    def _backpropagate(self, search_path: list[MuZeroNode], value: float) -> None:
        """Backpropagate value through search path.

        Args:
            search_path: List of nodes from root to leaf.
            value: Value to propagate.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent (if applicable)

    def _backpropagate_with_rewards(
        self,
        search_path: list[MuZeroNode],
        value: float,
    ) -> None:
        """Backpropagate value with discounted rewards.

        For MuZero, we accumulate discounted rewards along the path.

        Args:
            search_path: List of nodes from root to leaf.
            value: Terminal value estimate.
        """
        # Start from leaf, accumulate discounted value
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            # Add discounted reward from this transition
            value = node.reward + self.config.discount * value

    def _add_dirichlet_noise(self, node: MuZeroNode) -> None:
        """Add Dirichlet noise to root node priors.

        Args:
            node: Root node.
        """
        actions = list(node.children.keys())
        if len(actions) == 0:
            return

        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))

        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _get_action_probs(
        self,
        root: MuZeroNode,
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

        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

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
        if temperature == 0 or action_probs.max() > GREEDY_THRESHOLD:
            return int(np.argmax(action_probs))

        # Apply temperature
        probs = np.power(action_probs + EPSILON, 1.0 / temperature)
        probs = probs / probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    def search_from_env(
        self,
        env: Any,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Convenience method to search from an environment.

        Extracts observation and action mask from environment.

        Args:
            env: Game environment with get_observation() method.
            add_noise: Whether to add exploration noise.

        Returns:
            Tuple of (action_probs, root_value).
        """
        obs = env.get_observation()
        return self.search(obs, obs["action_mask"], add_noise)
