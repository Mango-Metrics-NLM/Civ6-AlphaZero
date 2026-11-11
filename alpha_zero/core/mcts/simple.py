"""Random-rollout MCTS used for testing and as a no-network baseline."""

from __future__ import annotations

import logging

import numpy as np

from alpha_zero.core.mcts.backup import (
    backpropagate_path,
    env_current_player,
    log_backup_diagnostics,
    should_track_to_play,
)
from alpha_zero.core.mcts.base import MCTSNode
from alpha_zero.core.mcts.mcts_protocols import MCTSEnvironment
from alpha_zero.core.mcts_constants import DEFAULT_ROLLOUT_DEPTH, EPSILON

logger = logging.getLogger(__name__)


class SimpleMCTS:
    """Simplified MCTS for testing without neural network.

    Uses random policy and rollout for evaluation.
    """

    def __init__(
        self,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        rollout_depth: int = DEFAULT_ROLLOUT_DEPTH,
        seed: int | None = None,
    ) -> None:
        """Initialize simple MCTS.

        Args:
            num_simulations: Number of MCTS simulations.
            c_puct: Exploration constant.
            rollout_depth: Maximum rollout depth.
            seed: Seed for this search's RNG. ``None`` means unseeded, matching
                the previous behaviour. A parameter rather than a config field
                because ``SimpleMCTS`` is the one variant that takes no config
                object.
        """
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        # One generator for the whole class. It previously had this attribute
        # *and* reached for the global `np.random` in `select_action`, so the
        # same object drew from two different sources and seeding it determined
        # only half its behaviour.
        self._rng = np.random.default_rng(seed)
        self._search_count = 0
        logger.debug(
            "SimpleMCTS initialized",
            extra={
                "num_simulations": num_simulations,
                "c_puct": c_puct,
                "rollout_depth": rollout_depth,
            },
        )

    def search(self, env: MCTSEnvironment, add_noise: bool = False) -> tuple[np.ndarray, float]:
        """Run MCTS search.

        Args:
            env: Game environment (must support clone(), step(), get_valid_actions()).
            add_noise: Accepted for call-site parity with the network-guided
                variants and ignored -- random rollouts need no Dirichlet
                noise. Without it ``run_arena`` (which passes
                ``add_noise=False`` to every non-MuZero search) could not use
                this variant as an opponent.

        Returns:
            Tuple of (action_probs, root_value).
        """
        track_to_play = should_track_to_play(env)
        root = MCTSNode(to_play=env_current_player(env) if track_to_play else None)
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
                # A node describes the state after its action.
                node.to_play = env_current_player(sim_env) if track_to_play else None
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

            # Backpropagation. `_rollout` and the terminal branch above both
            # express `value` from the leaf player's perspective, which is the
            # reference `backpropagate_path` defaults to.
            log_backup_diagnostics(search_path, search_name="SimpleMCTS")
            backpropagate_path(search_path, value)

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

        return int(self._rng.choice(len(probs), p=probs))
