"""Simultaneous action MCTS using Decoupled UCT (DUCT).

In simultaneous-move games, all players choose actions at the same time
without knowledge of each other's choices. Standard MCTS cannot handle
this because it assumes sequential moves. Decoupled UCT solves this by
maintaining independent action statistics per player at each node, then
combining independently-selected actions into joint actions.

Key design:
- SimultaneousNode stores per-player action stats independently (decoupled).
- joint_children uses tuple[int, ...] as key (one action per active player).
- Only visited joint actions create child nodes (lazy expansion).

Reference:
    Lisy et al. "Online Monte Carlo Counterfactual Regret Minimization
    for Search in Imperfect Information Games" (2015)

Example:
    >>> config = SimultaneousMCTSConfig(num_simulations=100)
    >>> duct = DecoupledUCT(config, num_players=2, action_space_size=10)
    >>> action_probs = duct.search(env, network, active_players=[0, 1])
    >>> action_probs[0]  # Player 0's action distribution
    array([0.1, 0.3, ...])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from alpha_zero.core.mcts_constants import EPSILON
from alpha_zero.multiplayer.action_resolution import (
    ActionResolutionConfig,
    ActionResolutionStrategy,
    ActionResolver,
)
from alpha_zero.multiplayer.protocols import MultiplayerEnvironment, MultiplayerNetwork

logger = logging.getLogger(__name__)

# Sentinel action index for inactive players in joint action keys.
# Using -1 avoids collisions with valid action indices (which are >= 0).
_INACTIVE_PLAYER_ACTION = -1


@dataclass
class ChildStats:
    """Statistics for a single action of a single player.

    Tracks visit counts, accumulated value, and prior probability
    for one action in one player's decoupled action space.

    Attributes:
        visit_count: Number of times this action was selected.
        value_sum: Accumulated value from simulations selecting this action.
        prior: Prior probability from the neural network policy.
    """

    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0


@dataclass
class SimultaneousNode:
    """MCTS node for simultaneous-move games with decoupled statistics.

    Unlike sequential MCTS nodes, this node stores independent action
    statistics per player. Each player's actions are evaluated and
    selected independently using their own visit counts and values.

    Joint children are created lazily: only when a specific combination
    of player actions is visited does the corresponding child node exist.

    Attributes:
        num_players: Number of players in the game.
        visit_count: Total visit count for this node.
        value_sums: Per-player accumulated values at this node.
        per_player_children: Independent action stats per player.
            Maps player_id -> {action_idx -> ChildStats}.
        joint_children: Lazily expanded joint action children.
            Maps tuple(action_per_player...) -> SimultaneousNode.
    """

    num_players: int = 2
    visit_count: int = 0
    value_sums: dict[int, float] = field(default_factory=dict)
    per_player_children: dict[int, dict[int, ChildStats]] = field(default_factory=dict)
    joint_children: dict[tuple[int, ...], SimultaneousNode] = field(
        default_factory=dict
    )

    def is_expanded(self) -> bool:
        """Check if node has been expanded with per-player statistics."""
        return len(self.per_player_children) > 0

    def select_action_for_player(
        self,
        player_id: int,
        c_puct: float,
    ) -> int:
        """Select action for a player using UCB formula.

        Uses PUCT: Q_i(a_i) + c * P_i(a_i) * sqrt(N) / (1 + N_i(a_i))
        where Q_i is the mean value for player i, P_i is the prior,
        N is the node visit count, and N_i is the action visit count.

        Args:
            player_id: Player whose action to select.
            c_puct: Exploration constant.

        Returns:
            Selected action index.

        Raises:
            AssertionError: If the player has no action stats.
        """
        player_stats = self.per_player_children.get(player_id, {})
        assert player_stats, f"No action stats for player {player_id}"

        best_score = -float("inf")
        best_action = -1
        sqrt_total = np.sqrt(self.visit_count) if self.visit_count > 0 else 0.0

        for action, stats in player_stats.items():
            # Q-value: mean value for this player from this action
            q_value = (
                stats.value_sum / stats.visit_count
                if stats.visit_count > 0
                else 0.0
            )

            # Exploration bonus
            u_value = c_puct * stats.prior * sqrt_total / (1 + stats.visit_count)

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def expand_simultaneous(
        self,
        policies: dict[int, np.ndarray],
        action_masks: dict[int, np.ndarray],
    ) -> None:
        """Expand node with per-player action statistics.

        Creates independent ChildStats for each valid action of each player.
        Policies are normalized over valid actions.

        Args:
            policies: Per-player policy distributions. Maps player_id -> array.
            action_masks: Per-player valid action masks. Maps player_id -> array.
        """
        for player_id, policy in policies.items():
            mask = action_masks.get(player_id, np.ones_like(policy))
            valid_actions = np.where(mask > 0)[0]

            if len(valid_actions) == 0:
                logger.warning("No valid actions for player %d during expansion", player_id)
                continue

            # Normalize policy over valid actions
            valid_policy = policy[valid_actions].copy()
            policy_sum = valid_policy.sum()
            if policy_sum > EPSILON:
                valid_policy = valid_policy / policy_sum
            else:
                valid_policy = np.ones_like(valid_policy) / len(valid_policy)

            self.per_player_children[player_id] = {}
            for i, action in enumerate(valid_actions):
                self.per_player_children[player_id][int(action)] = ChildStats(
                    prior=float(valid_policy[i]),
                )

    def update_stats(
        self,
        joint_action: dict[int, int],
        values: dict[int, float],
    ) -> None:
        """Update per-player statistics after backpropagation.

        Each player's action stats are updated independently with
        the value obtained from that player's perspective.

        Args:
            joint_action: Maps player_id -> action_idx.
            values: Maps player_id -> value for this simulation.
        """
        self.visit_count += 1

        for player_id, value in values.items():
            self.value_sums[player_id] = self.value_sums.get(player_id, 0.0) + value

            # Update per-action stats for this player
            action = joint_action.get(player_id)
            if action is not None and player_id in self.per_player_children:
                stats = self.per_player_children[player_id].get(action)
                if stats is not None:
                    stats.visit_count += 1
                    stats.value_sum += value

    def get_action_probabilities_for_player(
        self,
        player_id: int,
        action_space_size: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Compute action probabilities from visit counts for a player.

        Args:
            player_id: Player whose probabilities to compute.
            action_space_size: Total size of the action space.
            temperature: Sampling temperature (0 = argmax, 1 = proportional).

        Returns:
            Probability distribution over all actions.
        """
        action_probs = np.zeros(action_space_size)

        player_stats = self.per_player_children.get(player_id, {})
        if not player_stats:
            return action_probs

        if temperature < EPSILON:
            # Greedy: pick most visited
            best_action = max(
                player_stats, key=lambda a: player_stats[a].visit_count
            )
            action_probs[best_action] = 1.0
        else:
            actions = sorted(player_stats.keys())
            visits = np.array(
                [player_stats[a].visit_count for a in actions], dtype=np.float64
            )

            if temperature == 1.0:
                total = visits.sum()
                if total > 0:
                    probs = visits / total
                else:
                    probs = np.ones(len(visits)) / len(visits)
            else:
                # Apply temperature
                log_visits = np.log(visits + EPSILON)
                log_visits = log_visits / temperature
                log_visits -= log_visits.max()  # Numerical stability
                probs = np.exp(log_visits)
                probs = probs / (probs.sum() + EPSILON)

            for i, action in enumerate(actions):
                action_probs[action] = probs[i]

        return action_probs


@dataclass(frozen=True)
class SimultaneousMCTSConfig:
    """Configuration for Decoupled UCT (simultaneous-move MCTS).

    Attributes:
        num_simulations: Total simulation budget per search.
        c_puct: Exploration constant for PUCT formula.
        dirichlet_alpha: Dirichlet noise parameter for root exploration.
        dirichlet_epsilon: Weight of Dirichlet noise vs prior.
        temperature: Sampling temperature for action selection.
        temperature_threshold: Turn after which temperature becomes greedy.
        action_resolution: Strategy for resolving action conflicts.
    """

    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30
    action_resolution: str = ActionResolutionStrategy.INDEPENDENT.value

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.num_simulations < 1:
            msg = f"num_simulations must be >= 1, got {self.num_simulations}"
            raise ValueError(msg)
        if self.c_puct <= 0:
            msg = f"c_puct must be > 0, got {self.c_puct}"
            raise ValueError(msg)
        if not 0 <= self.dirichlet_epsilon <= 1:
            msg = f"dirichlet_epsilon must be in [0, 1], got {self.dirichlet_epsilon}"
            raise ValueError(msg)
        valid_strategies = {s.value for s in ActionResolutionStrategy}
        if self.action_resolution not in valid_strategies:
            msg = (
                f"action_resolution must be one of {valid_strategies}, "
                f"got {self.action_resolution!r}"
            )
            raise ValueError(msg)


class DecoupledUCT:
    """Decoupled UCT (DUCT) for simultaneous-move multiplayer games.

    In each simulation:
    1. Selection: For each active player independently, select action via UCB.
    2. Joint action: Combine independently-selected actions.
    3. Environment step: Apply the joint action.
    4. Expansion: At unexpanded node, evaluate with network and expand.
    5. Backpropagation: Update per-player statistics independently.

    Attributes:
        config: DUCT configuration.
        num_players: Number of players in the game.
        action_space_size: Size of the action space.
    """

    def __init__(
        self,
        config: SimultaneousMCTSConfig,
        num_players: int,
        action_space_size: int,
    ) -> None:
        """Initialize Decoupled UCT.

        Args:
            config: DUCT configuration.
            num_players: Number of players in the game.
            action_space_size: Size of the action space.
        """
        self.config = config
        self.num_players = num_players
        self.action_space_size = action_space_size
        self._search_count = 0
        self._action_resolver = ActionResolver(
            ActionResolutionConfig(
                strategy=config.action_resolution,
                allow_same_target=False,
            )
        )

        logger.debug(
            "DecoupledUCT initialized",
            extra={
                "num_players": num_players,
                "action_space_size": action_space_size,
                "num_simulations": config.num_simulations,
                "c_puct": config.c_puct,
                "action_resolution": config.action_resolution,
            },
        )

    def search(
        self,
        env: MultiplayerEnvironment,
        network: MultiplayerNetwork,
        active_players: list[int],
        turn_number: int = 0,
    ) -> dict[int, np.ndarray]:
        """Run DUCT search from the current state.

        Args:
            env: Game environment (must support clone()).
            network: Neural network for state evaluation.
            active_players: List of player IDs acting simultaneously.
            turn_number: Current game turn for temperature threshold.
                When turn_number >= config.temperature_threshold,
                temperature is set to near-zero (greedy).

        Returns:
            Dict mapping player_id -> action probability distribution.
        """
        self._search_count += 1

        # Apply temperature threshold: switch to greedy after threshold turn
        temperature = self.config.temperature
        if turn_number >= self.config.temperature_threshold:
            temperature = 0.0
            logger.debug(
                "Temperature threshold reached, using greedy selection",
                extra={"turn_number": turn_number, "threshold": self.config.temperature_threshold},
            )

        # Create root node
        root = SimultaneousNode(num_players=self.num_players)

        # Evaluate and expand root
        self._expand_node(root, env, network, active_players)

        # Add Dirichlet noise at root for exploration
        self._add_dirichlet_noise(root, active_players)

        # Run simulations
        for sim_idx in range(self.config.num_simulations):
            sim_env = env.clone()
            self._run_simulation(sim_env, root, network, active_players)

            if logger.isEnabledFor(logging.DEBUG) and (sim_idx + 1) % 50 == 0:
                logger.debug(
                    "DUCT simulation progress",
                    extra={"sim": sim_idx + 1, "total": self.config.num_simulations,
                           "root_visits": root.visit_count},
                )

        # Extract per-player action probabilities
        result: dict[int, np.ndarray] = {}
        for player_id in active_players:
            result[player_id] = root.get_action_probabilities_for_player(
                player_id,
                self.action_space_size,
                temperature=temperature,
            )

        logger.debug(
            "DUCT search complete",
            extra={
                "search_id": self._search_count,
                "root_visits": root.visit_count,
                "active_players": active_players,
                "temperature": temperature,
            },
        )

        return result

    def _expand_node(
        self,
        node: SimultaneousNode,
        env: MultiplayerEnvironment,
        network: MultiplayerNetwork,
        active_players: list[int],
    ) -> dict[int, float]:
        """Evaluate and expand a node with per-player action statistics.

        Calls the neural network, builds per-player action masks,
        and populates the node's per_player_children.

        Args:
            node: Node to expand.
            env: Environment at the node's state.
            network: Neural network for evaluation.
            active_players: Players acting simultaneously.

        Returns:
            Per-player values from network evaluation.
        """
        policies, values = self._evaluate(env, network)

        action_masks: dict[int, np.ndarray] = {}
        for player_id in active_players:
            action_masks[player_id] = env.get_valid_actions()

        active_policies = {
            pid: policies[pid] for pid in active_players if pid in policies
        }
        active_masks = {
            pid: action_masks[pid] for pid in active_players if pid in action_masks
        }

        node.expand_simultaneous(active_policies, active_masks)

        logger.debug(
            "Node expanded",
            extra={"active_players": active_players, "values": values},
        )
        return values

    def _step_env_with_resolution(
        self,
        env: MultiplayerEnvironment,
        joint_action: dict[int, int],
        active_players: list[int],
    ) -> tuple[bool, dict[str, Any]]:
        """Step the environment using the action resolver.

        Resolves the joint action through the configured strategy,
        then applies the resolved actions to the environment.

        Args:
            env: Environment to step.
            joint_action: Maps player_id -> action_idx.
            active_players: Active player IDs.

        Returns:
            Tuple of (is_terminal, info_dict).
        """
        resolved = self._action_resolver.resolve(joint_action)

        # Apply the first resolved action (environment handles turn semantics)
        if resolved:
            _primary_player, primary_action = resolved[0]
        else:
            # All actions cancelled by conflict resolution — use first player's action
            primary_action = joint_action.get(active_players[0], 0)
            primary_player = active_players[0]
            logger.debug(
                "All actions cancelled by resolver, falling back",
                extra={"player": primary_player, "action": primary_action},
            )

        _obs, _reward, terminated, truncated, info = env.step(primary_action)
        return (terminated or truncated), info

    def _run_simulation(
        self,
        env: MultiplayerEnvironment,
        root: SimultaneousNode,
        network: MultiplayerNetwork,
        active_players: list[int],
    ) -> None:
        """Run a single DUCT simulation.

        Steps:
        1. Selection: Independently select action per player via UCB.
        2. Form joint action tuple.
        3. Check if joint child exists; if so, descend.
        4. If leaf: expand with network, backpropagate.
        5. If terminal: backpropagate terminal values.

        Args:
            env: Cloned environment for simulation.
            root: Root node of the search tree.
            network: Neural network for evaluation.
            active_players: Players acting simultaneously.
        """
        node = root
        search_path: list[SimultaneousNode] = [node]
        joint_actions_path: list[dict[int, int]] = []
        depth = 0

        # === Selection: traverse existing tree ===
        while node.is_expanded():
            # Select action for each active player independently
            joint_action: dict[int, int] = {}
            for player_id in active_players:
                if player_id in node.per_player_children:
                    joint_action[player_id] = node.select_action_for_player(
                        player_id, self.config.c_puct
                    )

            joint_actions_path.append(joint_action)

            # Form joint action key using sentinel for inactive players
            joint_key = tuple(
                joint_action.get(pid, _INACTIVE_PLAYER_ACTION)
                for pid in range(self.num_players)
            )

            # Check if joint child exists (lazy expansion)
            if joint_key in node.joint_children:
                node = node.joint_children[joint_key]
                search_path.append(node)
                depth += 1

                is_terminal, info = self._step_env_with_resolution(
                    env, joint_action, active_players
                )

                if is_terminal:
                    terminal_values = self._extract_terminal_values(
                        info, active_players
                    )
                    self._backpropagate(
                        search_path, joint_actions_path, terminal_values
                    )
                    logger.debug(
                        "Simulation hit terminal at depth %d", depth,
                        extra={"terminal_values": terminal_values},
                    )
                    return
            else:
                # Joint action not yet visited: create child and expand
                child = SimultaneousNode(num_players=self.num_players)
                node.joint_children[joint_key] = child
                search_path.append(child)
                depth += 1

                is_terminal, info = self._step_env_with_resolution(
                    env, joint_action, active_players
                )

                if is_terminal:
                    terminal_values = self._extract_terminal_values(
                        info, active_players
                    )
                    self._backpropagate(
                        search_path, joint_actions_path, terminal_values
                    )
                    logger.debug(
                        "Simulation hit terminal at new node depth %d", depth,
                        extra={"terminal_values": terminal_values},
                    )
                    return

                # Expand the new child node
                values = self._expand_node(child, env, network, active_players)

                # Backpropagate network values
                self._backpropagate(search_path, joint_actions_path, values)
                return

        # Node is not expanded (should only happen at root before expansion)
        logger.debug("Simulation reached unexpanded non-root node at depth %d", depth)
        _policies, values = self._evaluate(env, network)
        self._backpropagate(search_path, joint_actions_path, values)

    def _evaluate(
        self,
        env: MultiplayerEnvironment,
        network: MultiplayerNetwork,
    ) -> tuple[dict[int, np.ndarray], dict[int, float]]:
        """Evaluate current state using neural network.

        Args:
            env: Game environment at current state.
            network: Neural network for inference.

        Returns:
            Tuple of (per_player_policies, per_player_values).
        """
        obs = env.get_observation()

        spatial = torch.from_numpy(obs["spatial"]).float().unsqueeze(0)
        vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0)
        action_mask_np = obs.get("action_mask", np.ones(1))
        action_mask = torch.from_numpy(action_mask_np).float().unsqueeze(0)

        with torch.no_grad():
            policy_probs, value_tensor = network.predict(spatial, vector, action_mask)

        policy = policy_probs.squeeze(0).cpu().numpy()
        values_np = value_tensor.squeeze(0).cpu().numpy()

        # Build per-player policy dicts
        # In simultaneous games, all players share the same policy space
        # but may have different valid actions
        policies: dict[int, np.ndarray] = {}
        for pid in range(self.num_players):
            policies[pid] = policy.copy()

        # Convert values to per-player dict
        player_values: dict[int, float] = {}
        if values_np.ndim == 0:
            # Scalar value (legacy 2-player zero-sum)
            player_values[0] = float(values_np)
            player_values[1] = float(-values_np)
        else:
            num_values = len(values_np)
            if num_values < self.num_players:
                logger.warning(
                    "Network returned %d values but expected %d players",
                    num_values, self.num_players,
                )
            for i in range(min(num_values, self.num_players)):
                player_values[i] = float(values_np[i])

        return policies, player_values

    def _add_dirichlet_noise(
        self,
        node: SimultaneousNode,
        active_players: list[int],
    ) -> None:
        """Add Dirichlet noise to root node priors for exploration.

        Noise is added independently to each player's action priors.

        Args:
            node: Root node to add noise to.
            active_players: Players whose priors should get noise.
        """
        eps = self.config.dirichlet_epsilon
        alpha = self.config.dirichlet_alpha

        for player_id in active_players:
            player_stats = node.per_player_children.get(player_id, {})
            if not player_stats:
                continue

            actions = sorted(player_stats.keys())
            noise = np.random.dirichlet([alpha] * len(actions))

            for i, action in enumerate(actions):
                stats = player_stats[action]
                stats.prior = (1 - eps) * stats.prior + eps * noise[i]

    def _backpropagate(
        self,
        search_path: list[SimultaneousNode],
        joint_actions_path: list[dict[int, int]],
        values: dict[int, float],
    ) -> None:
        """Backpropagate values through the search path.

        Updates per-player action statistics at each node independently.

        Args:
            search_path: Nodes from root to leaf.
            joint_actions_path: Joint actions taken at each step.
            values: Per-player values from evaluation/terminal.
        """
        for i, node in enumerate(search_path):
            joint_action = joint_actions_path[i] if i < len(joint_actions_path) else {}
            node.update_stats(joint_action, values)

    def _extract_terminal_values(
        self,
        info: dict[str, Any],
        active_players: list[int],
    ) -> dict[int, float]:
        """Extract per-player values from terminal game info.

        Args:
            info: Environment info dict from terminal state.
            active_players: Active player IDs.

        Returns:
            Dict mapping player_id -> terminal value.
        """
        values: dict[int, float] = {}

        winner = info.get("winner")
        player_scores = info.get("player_scores", {})

        if player_scores:
            for pid in active_players:
                values[pid] = float(player_scores.get(pid, 0.0))
        elif winner is not None:
            for pid in active_players:
                values[pid] = 1.0 if pid == winner else -1.0
        else:
            # Draw — no winner or player_scores in terminal info
            logger.debug(
                "Terminal state with no winner/scores, treating as draw",
                extra={"info_keys": list(info.keys())},
            )
            for pid in active_players:
                values[pid] = 0.0

        return values
