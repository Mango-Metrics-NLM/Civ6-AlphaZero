"""Baseline player implementations for model evaluation.

This module provides various baseline opponents for measuring model performance:
- RandomPlayer: Uniform random action selection
- HeuristicPlayer: Rule-based player with configurable weights
- NeuralNetworkPlayer: Trained model with optional MCTS

Usage:
    from alpha_zero.evaluation.baselines import RandomPlayer, HeuristicPlayer
    from alpha_zero.evaluation.config import BaselineConfig, BaselineType

    # Create random player
    player = RandomPlayer(seed=42)

    # Create via factory
    config = BaselineConfig(type=BaselineType.HEURISTIC)
    player = BaselineFactory.create(config)

    # Use in evaluation
    action = player.select_action(env, observation)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from alpha_zero.evaluation.config import (
    BaselineConfig,
    BaselineType,
    HeuristicWeights,
)

if TYPE_CHECKING:
    from alpha_zero.core.civ6_mcts import Civ6MCTS
    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.envs.microciv import MicroCivEnv

logger = logging.getLogger(__name__)


@runtime_checkable
class BaselinePlayer(Protocol):
    """Protocol for baseline player implementations.

    All baseline players must implement this interface for compatibility
    with the ModelEvaluator.
    """

    @property
    def name(self) -> str:
        """Player name for identification in reports."""
        ...

    def select_action(
        self,
        env: MicroCivEnv,
        observation: dict[str, np.ndarray],
    ) -> int:
        """Select an action given the current game state.

        Args:
            env: Current environment instance (for action mask access)
            observation: Current observation dictionary containing:
                - spatial: (C, H, W) spatial features
                - vector: (D,) vector features
                - action_mask: (A,) valid action mask

        Returns:
            Selected action index
        """
        ...

    def reset(self) -> None:
        """Reset player state for a new game."""
        ...


class RandomPlayer:
    """Uniformly random action selection from valid actions.

    Selects actions uniformly at random from the set of valid actions
    according to the action mask.

    Attributes:
        name: Player name for identification
    """

    def __init__(self, seed: int | None = None, name: str = "random") -> None:
        """Initialize random player.

        Args:
            seed: Random seed for reproducibility
            name: Display name for this player
        """
        self._name = name
        self._rng = np.random.default_rng(seed)
        self._games_played = 0

    @property
    def name(self) -> str:
        """Player name for identification."""
        return self._name

    def select_action(
        self,
        env: MicroCivEnv,
        observation: dict[str, np.ndarray],
    ) -> int:
        """Select random valid action.

        Args:
            env: Current environment
            observation: Current observation with action_mask

        Returns:
            Randomly selected valid action index
        """
        action_mask = observation.get("action_mask")
        if action_mask is None:
            # Fallback: get mask from env
            action_mask = env.get_valid_actions()

        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            logger.warning("No valid actions available, returning action 0")
            return 0

        return int(self._rng.choice(valid_actions))

    def reset(self) -> None:
        """Reset player state for new game."""
        self._games_played += 1

    def get_stats(self) -> dict[str, Any]:
        """Get player statistics.

        Returns:
            Dictionary of player stats
        """
        return {
            "name": self._name,
            "type": "random",
            "games_played": self._games_played,
        }


class HeuristicPlayer:
    """Rule-based heuristic player with configurable weights.

    Uses hand-crafted heuristics to evaluate and select actions:
    - Prefer attacking when advantageous
    - Found cities in strategic locations
    - Produce units when cities are idle
    - Move units toward objectives

    Attributes:
        name: Player name for identification
        weights: HeuristicWeights controlling behavior
        aggression: Overall aggression level (0-1)
    """

    def __init__(
        self,
        weights: HeuristicWeights | None = None,
        aggression: float = 0.5,
        name: str = "heuristic",
        seed: int | None = None,
    ) -> None:
        """Initialize heuristic player.

        Args:
            weights: Configurable weights for action evaluation
            aggression: Overall aggression level (0=defensive, 1=aggressive)
            name: Display name for this player
            seed: Random seed for tie-breaking
        """
        self._name = name
        self.weights = weights or HeuristicWeights()
        self.aggression = max(0.0, min(1.0, aggression))
        self._rng = np.random.default_rng(seed)
        self._games_played = 0

    @property
    def name(self) -> str:
        """Player name for identification."""
        return self._name

    def select_action(
        self,
        env: MicroCivEnv,
        observation: dict[str, np.ndarray],
    ) -> int:
        """Select action using heuristic evaluation.

        Evaluates each valid action and selects the highest-scoring one.

        Args:
            env: Current environment
            observation: Current observation with action_mask

        Returns:
            Selected action index
        """
        action_mask = observation.get("action_mask")
        if action_mask is None:
            action_mask = env.get_valid_actions()

        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            logger.warning("No valid actions available, returning action 0")
            return 0

        if len(valid_actions) == 1:
            return int(valid_actions[0])

        # Score each valid action
        scores = np.zeros(len(valid_actions), dtype=np.float32)
        for i, action in enumerate(valid_actions):
            scores[i] = self._evaluate_action(env, action, observation)

        # Add small random noise for tie-breaking
        scores += self._rng.random(len(scores), dtype=np.float32) * 0.001

        best_idx = np.argmax(scores)
        return int(valid_actions[best_idx])

    def _evaluate_action(
        self,
        env: MicroCivEnv,
        action: int,
        observation: dict[str, np.ndarray],
    ) -> float:
        """Evaluate action using heuristic rules.

        Args:
            env: Current environment
            action: Action to evaluate
            observation: Current observation

        Returns:
            Heuristic score for the action
        """
        score = 0.0

        # Decode action to understand its type
        try:
            action_info = env.action_space_handler.decode_action(action)
        except (AttributeError, ValueError):
            # If we can't decode, return neutral score
            return score

        action_type = action_info.get("type", "unknown")

        # Score based on action type
        if action_type == "attack":
            score += self._score_attack(action_info, env)
        elif action_type == "move":
            score += self._score_move(action_info, env, observation)
        elif action_type == "found_city":
            score += self._score_found_city(action_info, env)
        elif action_type == "produce":
            score += self._score_production(action_info, env)
        elif action_type == "end_turn":
            # Slight penalty for ending turn early
            score -= 0.1

        return score

    def _score_attack(self, action_info: dict, env: MicroCivEnv) -> float:
        """Score an attack action."""
        base_score = 1.0 * self.weights.attack_weight

        # Bonus for attacking weaker targets
        if "target_strength" in action_info and "attacker_strength" in action_info:
            strength_ratio = action_info["attacker_strength"] / max(
                1, action_info["target_strength"]
            )
            if strength_ratio > 1.5:
                base_score *= 1.5
            elif strength_ratio < 0.7:
                base_score *= 0.5

        return base_score * (0.5 + self.aggression)

    def _score_move(
        self,
        action_info: dict,
        env: MicroCivEnv,
        observation: dict[str, np.ndarray],
    ) -> float:
        """Score a move action."""
        score = 0.2

        # Prefer exploration if there's fog of war
        if "to_position" in action_info:
            # to_position available for future exploration heuristics

            # Check if moving toward unexplored area
            spatial = observation.get("spatial")
            if spatial is not None:
                # Assuming channel 0 or 1 contains visibility info
                if len(spatial.shape) >= 2:
                    # Basic heuristic: prefer edges of explored area
                    score += 0.1 * self.weights.exploration_weight

        return score

    def _score_found_city(self, action_info: dict, env: MicroCivEnv) -> float:
        """Score founding a city."""
        base_score = 1.5 * self.weights.expansion_weight

        # Check proximity to other cities
        if "position" in action_info:
            pos = action_info["position"]
            min_dist = float("inf")

            for city in getattr(env, "cities", {}).values():
                dist = abs(city.position[0] - pos[0]) + abs(city.position[1] - pos[1])
                min_dist = min(min_dist, dist)

            # Prefer some distance from other cities
            if min_dist < 3:
                base_score *= 0.5
            elif min_dist > 5:
                base_score *= 1.2

        return base_score

    def _score_production(self, action_info: dict, env: MicroCivEnv) -> float:
        """Score a production action."""
        base_score = 0.8 * self.weights.production_weight

        unit_type = action_info.get("unit_type", "warrior")

        # Adjust based on unit type and current game state
        if unit_type == "settler":
            base_score *= self.weights.expansion_weight
        elif unit_type == "warrior":
            base_score *= 0.5 + 0.5 * self.aggression
        elif unit_type == "archer":
            base_score *= self.weights.defense_weight

        return base_score

    def reset(self) -> None:
        """Reset player state for new game."""
        self._games_played += 1

    def get_stats(self) -> dict[str, Any]:
        """Get player statistics."""
        return {
            "name": self._name,
            "type": "heuristic",
            "aggression": self.aggression,
            "games_played": self._games_played,
            "weights": self.weights.to_dict(),
        }


class NeuralNetworkPlayer:
    """Neural network player with optional MCTS.

    Uses a trained neural network for action selection, with optional
    Monte Carlo Tree Search for improved play strength.

    Attributes:
        name: Player name for identification
        use_mcts: Whether to use MCTS for action selection
    """

    def __init__(
        self,
        network: MicroCivNetwork,
        device: str = "cpu",
        use_mcts: bool = True,
        mcts_simulations: int = 50,
        temperature: float = 0.0,
        name: str = "nn_player",
    ) -> None:
        """Initialize neural network player.

        Args:
            network: Trained neural network
            device: Device for inference (cpu, cuda, etc.)
            use_mcts: Whether to use MCTS
            mcts_simulations: Number of MCTS simulations if using MCTS
            temperature: Temperature for action sampling (0 = greedy)
            name: Display name for this player
        """
        import torch

        self._name = name
        self.device = device
        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self._games_played = 0

        # Move network to device
        self.network = network.to(torch.device(device))
        self.network.eval()

        # Initialize MCTS if needed
        self.mcts: Civ6MCTS | None = None
        if use_mcts:
            self._init_mcts()

    def _init_mcts(self) -> None:
        """Initialize MCTS search."""
        try:
            from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig

            config = MCTSConfig(
                num_simulations=self.mcts_simulations,
                c_puct=1.5,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.0,  # No noise during evaluation
            )
            self.mcts = Civ6MCTS(
                network=self.network,
                config=config,
                device=self.device,
            )
        except ImportError:
            logger.warning("Could not import MCTS, falling back to direct policy")
            self.use_mcts = False
            self.mcts = None

    @property
    def name(self) -> str:
        """Player name for identification."""
        return self._name

    def select_action(
        self,
        env: MicroCivEnv,
        observation: dict[str, np.ndarray],
    ) -> int:
        """Select action using neural network.

        Args:
            env: Current environment
            observation: Current observation

        Returns:
            Selected action index
        """

        action_mask = observation.get("action_mask")
        if action_mask is None:
            action_mask = env.get_valid_actions()

        if self.use_mcts and self.mcts is not None:
            return self._select_with_mcts(env, observation, action_mask)
        else:
            return self._select_with_policy(observation, action_mask)

    def _select_with_mcts(
        self,
        env: MicroCivEnv,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
    ) -> int:
        """Select action using MCTS."""
        assert self.mcts is not None
        action_probs, _ = self.mcts.search(env, add_noise=False)  # type: ignore[arg-type]

        # Apply temperature
        if self.temperature > 0:
            action_probs = np.power(action_probs, 1.0 / self.temperature)
            action_probs = action_probs / action_probs.sum()
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            action = int(np.argmax(action_probs))

        return action

    def _select_with_policy(
        self,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
    ) -> int:
        """Select action directly from policy network."""
        import torch

        with torch.no_grad():
            # Prepare inputs
            spatial = torch.from_numpy(observation["spatial"]).float()
            spatial = spatial.unsqueeze(0).to(self.device)

            vector = None
            if "vector" in observation:
                vector = torch.from_numpy(observation["vector"]).float()
                vector = vector.unsqueeze(0).to(self.device)

            # Get policy output
            policy_logits, _ = self.network(spatial, vector)
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]

        # Apply action mask
        policy = policy * action_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Fall back to uniform over valid actions
            valid_count = action_mask.sum()
            if valid_count > 0:
                policy = action_mask / valid_count
            else:
                policy = np.zeros_like(action_mask)
                policy[0] = 1.0

        # Select action
        if self.temperature > 0:
            action = np.random.choice(len(policy), p=policy)
        else:
            action = int(np.argmax(policy))

        return action

    def reset(self) -> None:
        """Reset player state for new game."""
        self._games_played += 1
        if self.mcts is not None:
            if hasattr(self.mcts, "reset"):
                self.mcts.reset()

    def get_stats(self) -> dict[str, Any]:
        """Get player statistics."""
        return {
            "name": self._name,
            "type": "neural_network",
            "use_mcts": self.use_mcts,
            "mcts_simulations": self.mcts_simulations,
            "temperature": self.temperature,
            "games_played": self._games_played,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        use_mcts: bool = True,
        mcts_simulations: int = 50,
        name: str | None = None,
    ) -> NeuralNetworkPlayer:
        """Create player from checkpoint file.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference
            use_mcts: Whether to use MCTS
            mcts_simulations: Number of MCTS simulations
            name: Player name (defaults to checkpoint filename)

        Returns:
            NeuralNetworkPlayer instance
        """
        import torch

        from alpha_zero.core.civ6_network import MicroCivNetwork

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device)

        # Extract network config from checkpoint
        network_config = checkpoint.get("network_config", {})
        board_size = network_config.get("board_size", 16)
        action_size = network_config.get("action_size", 8000)

        # Create network
        network = MicroCivNetwork(
            board_size=board_size,
            action_space_size=action_size,
            **{k: v for k, v in network_config.items() if k not in ["board_size", "action_size"]},
        )

        # Load weights
        if "model_state_dict" in checkpoint:
            network.load_state_dict(checkpoint["model_state_dict"])
        elif "network" in checkpoint:
            network.load_state_dict(checkpoint["network"])
        else:
            network.load_state_dict(checkpoint)

        player_name = name or path.stem
        return cls(
            network=network,
            device=device,
            use_mcts=use_mcts,
            mcts_simulations=mcts_simulations,
            name=player_name,
        )


class BaselineFactory:
    """Factory for creating baseline players from configuration."""

    @staticmethod
    def create(
        config: BaselineConfig,
        device: str = "cpu",
        seed: int | None = None,
    ) -> BaselinePlayer:
        """Create baseline player from configuration.

        Args:
            config: Baseline configuration
            device: Device for neural network inference
            seed: Random seed for reproducibility

        Returns:
            BaselinePlayer instance

        Raises:
            ValueError: If baseline type is unknown
            FileNotFoundError: If checkpoint is required but not found
        """
        if config.type == BaselineType.RANDOM:
            return RandomPlayer(seed=seed, name=config.name)

        elif config.type == BaselineType.HEURISTIC:
            return HeuristicPlayer(
                weights=config.heuristic_weights,
                name=config.name,
                seed=seed,
            )

        elif config.type in (BaselineType.PREVIOUS_MODEL, BaselineType.SPECIFIC_CHECKPOINT):
            if config.checkpoint_path is None:
                raise ValueError(f"checkpoint_path required for baseline type {config.type}")
            return NeuralNetworkPlayer.from_checkpoint(
                checkpoint_path=config.checkpoint_path,
                device=device,
                use_mcts=config.use_mcts,
                mcts_simulations=config.mcts_simulations,
                name=config.name,
            )

        else:
            raise ValueError(f"Unknown baseline type: {config.type}")

    @staticmethod
    def create_all(
        configs: list[BaselineConfig],
        device: str = "cpu",
        seed: int | None = None,
    ) -> dict[str, BaselinePlayer]:
        """Create all baseline players from configurations.

        Args:
            configs: List of baseline configurations
            device: Device for neural network inference
            seed: Random seed (each player gets seed + index)

        Returns:
            Dictionary mapping player names to instances
        """
        players = {}
        for i, config in enumerate(configs):
            if not config.enabled:
                continue
            player_seed = seed + i if seed is not None else None
            player = BaselineFactory.create(config, device=device, seed=player_seed)
            players[config.name] = player
        return players
