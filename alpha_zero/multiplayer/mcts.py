"""N-player Monte Carlo Tree Search.

Implements MCTS variants for games with 2 or more players:
- Max-N: Each player maximizes their own value component.
- Paranoid: All opponents assumed to collude against root player.
- Best-Reply: Only the strongest opponent reply is considered.

Backward compatible with 2-player games: for num_players=2 with
Paranoid strategy, behavior is equivalent to standard negamax MCTS.

Example:
    >>> from alpha_zero.multiplayer import NPlayerMCTS, NPlayerMCTSConfig
    >>> config = NPlayerMCTSConfig(num_simulations=100)
    >>> mcts = NPlayerMCTS(network, config, num_players=4)
    >>> action_probs, values = mcts.search(env)
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from alpha_zero.core.mcts_constants import EPSILON
from alpha_zero.multiplayer.config import NPlayerMCTSConfig, SearchStrategy
from alpha_zero.multiplayer.mcts_node import NPlayerMCTSNode
from alpha_zero.multiplayer.protocols import MultiplayerEnvironment, MultiplayerNetwork
from alpha_zero.multiplayer.scoring import extract_game_result

logger = logging.getLogger(__name__)


class NPlayerMCTS:
    """Monte Carlo Tree Search for N-player games.

    Supports Max-N, Paranoid, and Best-Reply search strategies.
    Uses a neural network for policy guidance and value estimation.

    Attributes:
        network: Neural network for policy/value prediction.
        config: MCTS configuration.
        num_players: Number of players in the game.
        device: PyTorch device for inference.
    """

    def __init__(
        self,
        network: MultiplayerNetwork,
        config: NPlayerMCTSConfig | None = None,
        num_players: int = 2,
        device: str = "cpu",
    ) -> None:
        """Initialize N-player MCTS.

        Args:
            network: Neural network implementing MultiplayerNetwork protocol.
            config: MCTS configuration. Defaults to NPlayerMCTSConfig().
            num_players: Number of players in the game.
            device: Device for neural network inference.
        """
        self.network = network
        self.config = config or NPlayerMCTSConfig()
        self.num_players = num_players
        self.device = device
        self._search_count = 0

        logger.debug(
            "NPlayerMCTS initialized",
            extra={
                "num_players": num_players,
                "num_simulations": self.config.num_simulations,
                "strategy": self.config.search_strategy,
                "device": device,
            },
        )

    def search(
        self,
        env: MultiplayerEnvironment,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, dict[int, float]]:
        """Run MCTS search from current state.

        Args:
            env: Game environment (must support clone()).
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            Tuple of (action_probs, player_values).
            action_probs: Array of shape (action_space_size,).
            player_values: Dict mapping player_id -> estimated value.
        """
        self._search_count += 1
        root_player = env.current_player

        # Create root node
        root = NPlayerMCTSNode(
            num_players=self.num_players,
            player_to_move=root_player,
        )

        # Get initial evaluation
        obs = env.get_observation()
        policy, _player_values = self._evaluate(obs)
        action_mask = obs.get("action_mask", env.get_valid_actions())

        # Determine next player after root acts
        next_player = self._get_next_player(root_player)

        # Expand root
        root.expand(policy, action_mask, next_player=next_player)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.is_expanded():
            self._add_dirichlet_noise(root)

        # Run simulations
        strategy = self.config.strategy
        for _sim_idx in range(self.config.num_simulations):
            sim_env = env.clone()
            self._run_simulation(sim_env, root, root_player, strategy)

        # Extract action probabilities from visit counts
        action_probs = root.get_action_probabilities(
            action_space_size=env.action_space.n,
            temperature=self.config.temperature,
        )

        # Get root value estimates
        root_values = root.get_all_values()

        logger.debug(
            "MCTS search complete",
            extra={
                "search_id": self._search_count,
                "root_visits": root.visit_count,
                "root_values": {k: f"{v:.3f}" for k, v in root_values.items()},
                "num_actions_explored": len(root.children),
            },
        )

        return action_probs, root_values

    def _run_simulation(
        self,
        env: MultiplayerEnvironment,
        root: NPlayerMCTSNode,
        root_player: int,
        strategy: SearchStrategy,
    ) -> None:
        """Run a single MCTS simulation.

        Phases: Selection -> Expansion -> Evaluation -> Backpropagation.

        Args:
            env: Cloned environment for simulation.
            root: Root node of the search tree.
            root_player: Player ID at the root.
            strategy: Search strategy to use.
        """
        node = root
        search_path: list[NPlayerMCTSNode] = [node]

        # === Selection: traverse tree to a leaf ===
        while node.is_expanded():
            action, node = self._select(node, root_player, strategy)
            search_path.append(node)

            # Step environment
            obs, _reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                # Terminal state: extract game values
                values = extract_game_result(info, self.num_players)
                self._backpropagate(search_path, values, root_player, strategy)
                return

        # === Expansion: expand leaf with network policy ===
        obs = env.get_observation()
        policy, player_values = self._evaluate(obs)
        action_mask = obs.get("action_mask", env.get_valid_actions())

        next_player = self._get_next_player(env.current_player)
        node.expand(policy, action_mask, next_player=next_player)

        # === Backpropagation: propagate values up the tree ===
        self._backpropagate(search_path, player_values, root_player, strategy)

    def _select(
        self,
        node: NPlayerMCTSNode,
        root_player: int,
        strategy: SearchStrategy,
    ) -> tuple[int, NPlayerMCTSNode]:
        """Select child node based on search strategy.

        Args:
            node: Current node.
            root_player: Root player for paranoid search.
            strategy: Search strategy.

        Returns:
            Tuple of (action, child_node).
        """
        if strategy == SearchStrategy.PARANOID:
            return node.select_child_paranoid(
                c_puct=self.config.c_puct,
                root_player=root_player,
            )
        elif strategy == SearchStrategy.BEST_REPLY:
            return self._select_best_reply(node, root_player)
        else:
            # Max-N: current player maximizes their own value
            return node.select_child(
                c_puct=self.config.c_puct,
                selecting_player=node.player_to_move,
            )

    def _select_best_reply(
        self,
        node: NPlayerMCTSNode,
        root_player: int,
    ) -> tuple[int, NPlayerMCTSNode]:
        """Best-Reply search: only strongest opponent responds.

        If it's the root player's turn, maximize root player's value.
        If it's any opponent's turn, find the opponent response that
        minimizes root player's value (strongest counter-move).

        Args:
            node: Current node.
            root_player: Root player ID.

        Returns:
            Tuple of (action, child_node).
        """
        if node.player_to_move == root_player:
            return node.select_child(
                c_puct=self.config.c_puct,
                selecting_player=root_player,
            )
        else:
            # Find opponent move that most hurts root player
            return node.select_child_paranoid(
                c_puct=self.config.c_puct,
                root_player=root_player,
            )

    def _backpropagate(
        self,
        search_path: list[NPlayerMCTSNode],
        values: dict[int, float],
        root_player: int,
        strategy: SearchStrategy,
    ) -> None:
        """Backpropagate values through the search path.

        Args:
            search_path: List of nodes from root to leaf.
            values: Per-player values from evaluation.
            root_player: Root player ID.
            strategy: Search strategy (affects paranoid backprop).
        """
        leaf = search_path[-1]

        if strategy == SearchStrategy.PARANOID:
            root_value = values.get(root_player, 0.0)
            leaf.backpropagate_paranoid(root_value, root_player)
        else:
            # Max-N and Best-Reply use per-player values
            leaf.backpropagate(values)

    def _evaluate(
        self,
        obs: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[int, float]]:
        """Evaluate state using neural network.

        Args:
            obs: Observation dict with 'spatial', 'vector', 'action_mask'.

        Returns:
            Tuple of (policy_probs, player_values).
        """
        spatial = torch.from_numpy(obs["spatial"]).float().unsqueeze(0).to(self.device)
        vector = torch.from_numpy(obs["vector"]).float().unsqueeze(0).to(self.device)
        action_mask_np = obs.get("action_mask", np.ones(1))
        action_mask = torch.from_numpy(action_mask_np).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_probs, value_tensor = self.network.predict(spatial, vector, action_mask)

        policy = policy_probs.squeeze(0).cpu().numpy()
        values_np = value_tensor.squeeze(0).cpu().numpy()

        # Convert to per-player dict
        if values_np.ndim == 0:
            # Scalar value (legacy 2-player network)
            player_values = {
                0: float(values_np),
                1: float(-values_np),
            }
        else:
            # Vector of per-player values
            player_values = {
                i: float(values_np[i]) for i in range(min(len(values_np), self.num_players))
            }

        logger.debug(
            "Network evaluation completed",
            extra={
                "policy_shape": policy.shape,
                "player_values": player_values,
            },
        )

        return policy, player_values

    def _add_dirichlet_noise(self, node: NPlayerMCTSNode) -> None:
        """Add Dirichlet noise to root node priors for exploration.

        Args:
            node: Root node to add noise to.
        """
        actions = list(node.children.keys())
        if not actions:
            return

        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _get_next_player(self, current_player: int) -> int:
        """Get the next player in turn order.

        Args:
            current_player: Currently active player.

        Returns:
            Next player ID (wraps around).
        """
        return (current_player + 1) % self.num_players

    @staticmethod
    def select_action(
        action_probs: np.ndarray,
        temperature: float = 1.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Select action from probability distribution.

        Args:
            action_probs: Action probabilities from search.
            temperature: Sampling temperature (0 = argmax, 1 = sample).
            action_mask: Optional mask of valid actions.

        Returns:
            Selected action index.
        """
        if action_mask is not None:
            action_probs = action_probs * action_mask
            prob_sum = action_probs.sum()
            if prob_sum < EPSILON:
                # Fallback: uniform over valid actions
                valid = np.where(action_mask > 0)[0]
                if len(valid) == 0:
                    return 0
                return int(np.random.choice(valid))
            action_probs = action_probs / prob_sum

        if temperature < EPSILON:
            # Greedy
            return int(np.argmax(action_probs))

        # Ensure valid probability distribution
        action_probs = np.maximum(action_probs, 0)
        prob_sum = action_probs.sum()
        if prob_sum < EPSILON:
            return int(np.random.randint(len(action_probs)))
        action_probs = action_probs / prob_sum

        return int(np.random.choice(len(action_probs), p=action_probs))
