"""MuZero MCTS -- searches a learned world model, never cloning the environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from alpha_zero.core.mcts.backup import backpropagate_path, log_backup_diagnostics
from alpha_zero.core.mcts.gumbel_ops import (
    POLICY_TARGET_COMPLETED_Q,
    VALID_POLICY_TARGETS,
    SearchStats,
    choose_among_most_visited,
    compute_completed_q,
    gumbel_sample_actions,
    halving_schedule,
    policy_target_from_root_stats,
)
from alpha_zero.core.mcts.mcts_protocols import MuZeroNetworkProtocol
from alpha_zero.core.mcts_constants import (
    DEFAULT_C_VISIT,
    DEFAULT_ROLLOUT_DEPTH,
    EPSILON,
    GREEDY_THRESHOLD,
)
from alpha_zero.utils.constants import MCTS
from alpha_zero.utils.device import place_network


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
        policy_target: How the Gumbel path (``use_gumbel=True``) reports its
            improved policy; the modes and builder of ``GumbelMCTSConfig``.
            ``one_hot`` reproduces runs from before T18. The PUCT path always
            returns normalised visit counts.
        c_visit: Visit constant of the completed-Q policy target.
        c_scale: Scale constant of the completed-Q policy target.
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
    # Seed for this search's own RNG. ``None`` means unseeded, which is what
    # every existing caller gets, so behaviour is unchanged unless a seed is
    # passed. `create_mcts` filters kwargs against this dataclass's fields, so
    # adding it here makes `create_mcts(..., seed=42)` work with no factory
    # change.
    seed: int | None = None
    policy_target: str = POLICY_TARGET_COMPLETED_Q
    c_visit: float = DEFAULT_C_VISIT
    c_scale: float = MCTS.GUMBEL_C_SCALE

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
        if self.policy_target not in VALID_POLICY_TARGETS:
            raise ValueError(
                f"policy_target must be one of {sorted(VALID_POLICY_TARGETS)}; "
                f"got {self.policy_target!r}"
            )
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
    # Always None today: the latent dynamics model cannot report the
    # acting player. Present so MuZeroNode satisfies the BackupNode
    # protocol and so a future variant can populate it.
    to_play: int | None = None
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
        # Resolve the request and place the network in one step: storing an
        # unresolved "cuda" here while the network sat on cpu was a real
        # source of mixed-device RuntimeErrors.
        self.device = place_network(network, device)
        # This search owns its randomness. Drawing from the global `np.random`
        # state meant a caller could seed its environments and still get a
        # different search on every run -- the defect that made a "seeded"
        # self-play run only half determined.
        self._rng = np.random.default_rng(self.config.seed)
        # What the most recent `search` actually spent. Measured, never copied
        # from the plan, so a test can hold it against `planned_halving_sims`.
        self.last_search_stats = SearchStats(root_evals=0, sim_evals=0)
        self._root_evals = 0
        self._sim_evals = 0

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
                seed=base_config.seed,
                policy_target=base_config.policy_target,
                c_visit=base_config.c_visit,
                c_scale=base_config.c_scale,
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
        observation: dict[str, Any],
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
        self._root_evals = 0
        self._sim_evals = 0
        self.network.eval()
        with torch.no_grad():
            output = self.network.initial_inference(
                spatial.unsqueeze(0),
                vector.unsqueeze(0),
                mask_tensor.unsqueeze(0),
            )
        self._root_evals += 1
        self.last_search_stats = SearchStats(root_evals=self._root_evals, sim_evals=0)

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
            action_probs = self._gumbel_policy_target(root, best_action, len(action_mask))
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
                    self._sim_evals += 1

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

        self.last_search_stats = SearchStats(root_evals=self._root_evals, sim_evals=self._sim_evals)
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
            rng=self._rng,
        )

        if len(sampled_actions) == 0:
            return valid_actions[0] if valid_actions else 0

        # No early return for a single candidate: with k = 1 and several legal
        # moves the schedule still spends the budget on it (gumbel_ops).

        # Store gumbel scores
        scores = dict(zip(sampled_actions, gumbel_scores, strict=False))

        if self.config.use_sequential_halving:
            return self._sequential_halving_muzero(
                root, sampled_actions, scores, action_mask, root_value, len(valid_actions)
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
        num_legal: int,
    ) -> int:
        """Sequential halving with MuZero dynamics.

        Args:
            root: Root node.
            candidate_actions: Actions selected by Gumbel sampling.
            gumbel_scores: Gumbel scores for candidates.
            action_mask: Valid action mask.
            root_value: Initial value estimate.
            num_legal: Legal actions at the root.

        Returns:
            Best action index.
        """
        actions = [int(a) for a in candidate_actions]  # rank order: best Gumbel score first

        # The schedule is defined once, in gumbel_ops, for every Gumbel search.
        # MuZero has no min_actions_after_halving field, so the default floor
        # of two survivors applies.
        for halving_round in halving_schedule(
            self.config.num_sampled_actions, self.config.num_simulations, num_legal
        ):
            survivors = actions[: halving_round.survivors]
            for _ in range(halving_round.full_passes):
                for action in survivors:
                    self._run_muzero_simulation(root, action, action_mask)
            if halving_round.remainder:
                # The odd visits follow the evidence of the passes just run,
                # not the ranking the round started with (mctx re-scores
                # before every simulation).
                survivors = self._rank_by_completed_q(root, survivors, gumbel_scores, root_value)
                for action in survivors[: halving_round.remainder]:
                    self._run_muzero_simulation(root, action, action_mask)

            # The next round takes the top survivors of this ranking.
            actions = self._rank_by_completed_q(root, survivors, gumbel_scores, root_value)

        return choose_among_most_visited(
            actions,
            [root.children[a].visit_count if a in root.children else 0 for a in actions],
        )

    def _gumbel_policy_target(
        self, root: MuZeroNode, best_action: int, action_space_size: int
    ) -> np.ndarray:
        """The Gumbel path's policy target, from the builder ``GumbelMCTS`` uses.

        This path returned a one-hot on the halving winner. Survivors are ranked
        with a coefficient of 1 on completed Q (LF-A12), so a large enough
        Gumbel draw outranks a forced win and the one-hot reported the losing
        move (T18 forced-win oracle: 4 of 12 positions). The completed-Q target
        scales Q by ``c_visit + max N``. ``policy_target="one_hot"`` restores it.
        """
        priors = np.zeros(action_space_size, dtype=np.float32)
        visits = np.zeros(action_space_size, dtype=np.float32)
        qs = np.zeros(action_space_size, dtype=np.float32)
        for action, child in root.children.items():
            if 0 <= action < action_space_size:
                priors[action] = child.prior
                visits[action] = child.visit_count
                qs[action] = self._root_q(child)
        return policy_target_from_root_stats(
            policy_target=self.config.policy_target,
            best_action=best_action,
            priors=priors,
            visit_counts=visits,
            q_values=qs,
            root_value=root.value,
            c_visit=self.config.c_visit,
            c_scale=self.config.c_scale,
        )

    def _root_q(self, child: MuZeroNode) -> float:
        """Q of a root action: the transition's reward plus the discounted value after it.

        ``_backpropagate_with_rewards`` stores in each node the return *from*
        that node onward, so a child's ``value`` excludes the reward on the
        edge into it. Ranking by ``child.value`` alone therefore dropped the
        immediate reward -- and when a game ends on that edge, the reward is the
        whole outcome: the T18 forced-win oracle at depth 1 saw every action
        valued 0 and chose by Gumbel noise. MuZero's reference ``ucb_score``
        uses ``reward + discount * value`` for the same reason.
        """
        if child.visit_count == 0:
            return 0.0
        return child.reward + self.config.discount * child.value

    def _rank_by_completed_q(
        self,
        root: MuZeroNode,
        actions: list[int],
        gumbel_scores: dict[int, float],
        root_value: float,
    ) -> list[int]:
        """Order ``actions`` best-first by completed Q plus Gumbel score."""
        max_visits = max(
            (root.children[a].visit_count for a in actions if a in root.children), default=0
        )
        combined: dict[int, float] = {}
        for action in actions:
            completed = 0.0
            child = root.children.get(action)
            if child is not None:
                q = self._root_q(child)
                completed = compute_completed_q(q, child.visit_count, max_visits, root_value)
            combined[action] = completed + gumbel_scores.get(action, 0.0)
        return sorted(actions, key=lambda a: -combined[a])

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
            self._sim_evals += 1
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
        # Delegates to the shared primitive so this rule has one implementation
        # across every variant. MuZero searches a *learned latent* model: past
        # the root there is no environment to ask who is to move, so its nodes
        # carry no ``to_play`` and the shared function applies its documented
        # per-ply alternation fallback -- byte-identical to the loop this
        # replaced. If a future MuZero variant can predict the acting player,
        # populating ``to_play`` is all that is needed to make this correct for
        # multi-action turns, exactly as it now is for the Gumbel variants.
        log_backup_diagnostics(search_path, search_name="MuZeroMCTS")
        backpropagate_path(search_path, value)

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

        noise = self._rng.dirichlet([self.config.dirichlet_alpha] * len(actions))

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

        return int(self._rng.choice(len(probs), p=probs))

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
