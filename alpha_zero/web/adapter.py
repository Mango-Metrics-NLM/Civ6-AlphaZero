"""Adapters bridging MicroCivEnv and AI agents to GameController protocol.

GameController expects:
- game_interface with: reset(seed) -> obs, step(action) -> (obs, r, d, t, info),
  get_valid_actions() -> list[int], action_space_size property
- ai_agent with: select_action(obs, valid_actions=) -> {"action", "value", "policy"}

MicroCivEnv provides: reset() -> (obs, info), get_valid_actions() -> np.ndarray mask
Baseline players provide: select_action(env, obs) -> int

These adapters bridge the gap.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from alpha_zero.envs.microciv import MicroCivEnv

logger = logging.getLogger(__name__)


class MicroCivGameAdapter:
    """Adapts MicroCivEnv to GameController's game_interface protocol.

    The GameController calls:
    - reset(seed) expecting just the observation dict
    - step(action) expecting (obs, reward, done, truncated, info)
    - get_valid_actions() expecting list[int]
    - action_space_size property

    MicroCivEnv.reset() returns (obs, info) tuple and
    get_valid_actions() returns a float mask array.
    This adapter bridges both differences.
    """

    def __init__(self, env: MicroCivEnv) -> None:
        """Initialize adapter wrapping a MicroCivEnv instance.

        Args:
            env: The MicroCiv environment to adapt.
        """
        self._env = env
        self._last_info: dict[str, Any] = {}

    @property
    def env(self) -> MicroCivEnv:
        """Access the underlying environment (for MCTS cloning)."""
        return self._env

    @property
    def action_space_size(self) -> int:
        """Total number of actions in the action space."""
        return int(self._env.action_space.n)  # type: ignore[arg-type]

    @property
    def board_size(self) -> int:
        """Board size of the environment."""
        return int(self._env.board_size)

    @property
    def current_player(self) -> int:
        """Current player index."""
        return int(self._env.current_player)

    @property
    def turn(self) -> int:
        """Current turn number."""
        return int(self._env.turn)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset environment and return observation.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Observation dictionary with spatial, vector, and action_mask arrays.
        """
        obs, info = self._env.reset(seed=seed)
        self._last_info = info
        logger.debug("Environment reset (seed=%s)", seed)
        return obs

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute action and return step result.

        Args:
            action: Action index to execute.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        result = self._env.step(action)
        obs, reward, terminated, truncated, info = result
        self._last_info = info
        return obs, float(reward), terminated, truncated, info

    def get_valid_actions(self) -> list[int]:
        """Get list of valid action indices.

        Returns:
            List of integer action indices that are currently valid.
        """
        mask = self._env.get_valid_actions()
        indices: list[int] = np.where(mask > 0)[0].tolist()
        return indices

    def get_board_data(self) -> dict[str, Any]:
        """Get structured board data for web rendering.

        Returns a dict with terrain grid, units, cities, and game metadata
        suitable for HTML template rendering.

        Returns:
            Structured board data dictionary.
        """
        game_state = self._env._get_game_state()

        # Build terrain grid (2D list for template iteration)
        board_size = self.board_size
        terrain_grid: list[list[dict[str, Any]]] = []
        for y in range(board_size):
            row: list[dict[str, Any]] = []
            for x in range(board_size):
                tile = game_state["terrain"].get((x, y), {})
                row.append(
                    {
                        "x": x,
                        "y": y,
                        "terrain": tile.get("terrain_type", "plains"),
                    }
                )
            terrain_grid.append(row)

        # Units with positions
        units = []
        for u in game_state["units"]:
            units.append(
                {
                    "id": u["id"],
                    "player": u["player"],
                    "type": u["type"],
                    "x": u["position"][0],
                    "y": u["position"][1],
                    "health": u["health"],
                    "moves": u.get("moves_remaining", 0),
                }
            )

        # Cities with positions
        cities = []
        for c in game_state["cities"]:
            cities.append(
                {
                    "id": c["id"],
                    "player": c["player"],
                    "name": c["name"],
                    "x": c["position"][0],
                    "y": c["position"][1],
                    "population": c.get("population", 1),
                    "producing": c.get("producing"),
                }
            )

        return {
            "terrain_grid": terrain_grid,
            "units": units,
            "cities": cities,
            "board_size": board_size,
            "current_player": self.current_player,
            "turn": self.turn,
        }

    def get_last_info(self) -> dict[str, Any]:
        """Get info dict from the last step/reset call."""
        return self._last_info


class MCTSAIAgent:
    """AI agent wrapping neural network + MCTS for GameController compatibility.

    GameController.play_ai_move() calls select_action(obs, valid_actions=)
    and expects a dict with keys: "action", "value", "policy".

    This agent loads a trained network and optionally uses MCTS search.
    """

    def __init__(
        self,
        network: Any,
        device: str = "cpu",
        use_mcts: bool = True,
        mcts_simulations: int = 50,
        temperature: float = 0.0,
        board_size: int = 16,
    ) -> None:
        """Initialize MCTS AI agent.

        Args:
            network: Trained neural network with predict_single method.
            device: PyTorch device for inference.
            use_mcts: Whether to use MCTS search (vs policy-only).
            mcts_simulations: Number of MCTS simulations per move.
            temperature: Temperature for action selection (0=greedy).
            board_size: Board size for MCTS configuration.
        """
        self._network = network
        self._device = device
        self._use_mcts = use_mcts
        self._mcts_simulations = mcts_simulations
        self._temperature = temperature
        self._board_size = board_size
        self._env: MicroCivEnv | None = None
        self._mcts: Any | None = None

        if use_mcts:
            self._init_mcts()

        logger.info(
            "MCTSAIAgent initialized (mcts=%s, sims=%d, temp=%.2f)",
            use_mcts,
            mcts_simulations,
            temperature,
        )

    def _init_mcts(self) -> None:
        """Initialize MCTS searcher."""
        try:
            from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig

            mcts_config = MCTSConfig(
                num_simulations=self._mcts_simulations,
                temperature=self._temperature,
            )
            self._mcts = Civ6MCTS(
                network=self._network,
                config=mcts_config,
                device=self._device,
            )
            logger.debug("MCTS searcher initialized")
        except ImportError:
            logger.warning("Could not import Civ6MCTS, falling back to policy-only")
            self._use_mcts = False

    def set_env(self, env: MicroCivEnv) -> None:
        """Set the environment reference for MCTS cloning.

        Must be called before select_action when using MCTS.

        Args:
            env: The current game environment.
        """
        self._env = env

    def select_action(
        self,
        observation: dict[str, np.ndarray] | Any,
        valid_actions: list[int] | None = None,
    ) -> dict[str, Any]:
        """Select an action given the current observation.

        Args:
            observation: Current observation dict with spatial, vector, action_mask.
            valid_actions: List of valid action indices (optional).

        Returns:
            Dict with keys: action (int), value (float|None), policy (ndarray|None).
        """
        start_time = time.time()

        # Handle observation formats
        if isinstance(observation, dict):
            spatial = observation.get("spatial")
            vector = observation.get("vector")
            action_mask = observation.get("action_mask")
        else:
            # Numpy array observation - shouldn't happen with MicroCiv
            logger.warning("Unexpected observation format: %s", type(observation))
            spatial = observation
            vector = None
            action_mask = None

        if self._use_mcts and self._mcts is not None and self._env is not None:
            return self._select_with_mcts(action_mask, start_time)

        return self._select_with_policy(spatial, vector, action_mask, valid_actions, start_time)

    def _select_with_mcts(
        self,
        action_mask: np.ndarray | None,
        start_time: float,
    ) -> dict[str, Any]:
        """Select action using MCTS search."""
        assert self._mcts is not None
        assert self._env is not None

        action_probs, root_value = self._mcts.search(self._env, add_noise=False)

        # Apply action mask
        if action_mask is not None:
            action_probs = action_probs * action_mask
            prob_sum = action_probs.sum()
            if prob_sum > 0:
                action_probs = action_probs / prob_sum

        # Select action
        if self._temperature == 0:
            action = int(np.argmax(action_probs))
        else:
            action = int(np.random.choice(len(action_probs), p=action_probs))

        elapsed = time.time() - start_time
        logger.debug("MCTS search completed in %.3fs, action=%d", elapsed, action)

        return {
            "action": action,
            "value": float(root_value) if root_value is not None else None,
            "policy": action_probs,
        }

    def _select_with_policy(
        self,
        spatial: np.ndarray | None,
        vector: np.ndarray | None,
        action_mask: np.ndarray | None,
        valid_actions: list[int] | None,
        start_time: float,
    ) -> dict[str, Any]:
        """Select action using network policy only (no MCTS)."""
        import torch

        with torch.no_grad():
            if spatial is not None:
                spatial_t = torch.from_numpy(spatial).unsqueeze(0).float().to(self._device)
            else:
                msg = "No spatial observation available"
                raise ValueError(msg)

            vector_t = None
            if vector is not None:
                vector_t = torch.from_numpy(vector).unsqueeze(0).float().to(self._device)

            mask_t = None
            if action_mask is not None:
                mask_t = torch.from_numpy(action_mask).unsqueeze(0).float().to(self._device)

            if hasattr(self._network, "predict_single"):
                policy, value = self._network.predict_single(
                    spatial_t.squeeze(0),
                    vector_t.squeeze(0) if vector_t is not None else None,
                    mask_t.squeeze(0) if mask_t is not None else None,
                )
                if isinstance(policy, torch.Tensor):
                    policy = policy.cpu().numpy()
                if isinstance(value, torch.Tensor):
                    value = float(value.cpu().item())
            else:
                # Generic forward pass
                outputs = self._network(spatial_t, vector_t, mask_t)
                if isinstance(outputs, tuple):
                    policy_logits, value_t = outputs
                else:
                    policy_logits = outputs
                    value_t = None

                # Softmax on logits
                policy = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
                value = float(value_t.squeeze().cpu().item()) if value_t is not None else None

        # Apply mask and select
        if action_mask is not None:
            policy = policy * action_mask
            prob_sum = policy.sum()
            if prob_sum > 0:
                policy = policy / prob_sum
            elif valid_actions:
                # Uniform over valid actions
                policy = np.zeros_like(policy)
                for a in valid_actions:
                    policy[a] = 1.0 / len(valid_actions)

        if self._temperature == 0:
            action = int(np.argmax(policy))
        else:
            action = int(np.random.choice(len(policy), p=policy))

        elapsed = time.time() - start_time
        logger.debug("Policy inference completed in %.3fs, action=%d", elapsed, action)

        return {
            "action": action,
            "value": value,
            "policy": policy,
        }


class BaselineAgentAdapter:
    """Adapts evaluation baseline players to GameController's ai_agent protocol.

    Baseline players (RandomPlayer, HeuristicPlayer) have:
        select_action(env, observation) -> int

    GameController expects:
        select_action(observation, valid_actions=) -> {"action", "value", "policy"}

    This adapter bridges the interface gap.
    """

    def __init__(self, baseline: Any, env: MicroCivEnv | None = None) -> None:
        """Initialize adapter for a baseline player.

        Args:
            baseline: A baseline player instance (RandomPlayer, HeuristicPlayer, etc.)
            env: Environment reference for the baseline's select_action call.
        """
        self._baseline = baseline
        self._env: MicroCivEnv | None = env
        self._name: str = getattr(baseline, "name", "baseline")
        logger.debug("BaselineAgentAdapter wrapping: %s", self._name)

    @property
    def name(self) -> str:
        """Name of the wrapped baseline player."""
        return self._name

    def set_env(self, env: MicroCivEnv) -> None:
        """Set the environment reference.

        Args:
            env: The current game environment.
        """
        self._env = env

    def select_action(
        self,
        observation: dict[str, np.ndarray] | Any,
        valid_actions: list[int] | None = None,
    ) -> dict[str, Any]:
        """Select an action using the wrapped baseline player.

        Args:
            observation: Current observation.
            valid_actions: List of valid action indices (unused by baselines).

        Returns:
            Dict with keys: action (int), value (None), policy (None).
        """
        if self._env is None:
            msg = "Environment not set. Call set_env() before select_action()."
            raise RuntimeError(msg)

        start_time = time.time()

        # Baseline expects (env, observation) -> int
        action = self._baseline.select_action(self._env, observation)

        elapsed = time.time() - start_time
        logger.debug(
            "Baseline %s selected action %d in %.3fs",
            self._name,
            action,
            elapsed,
        )

        return {
            "action": int(action),
            "value": None,
            "policy": None,
        }

    def reset(self) -> None:
        """Reset the baseline player state for a new game."""
        if hasattr(self._baseline, "reset"):
            self._baseline.reset()
