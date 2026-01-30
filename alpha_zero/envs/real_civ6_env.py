"""Real Civ6 Gymnasium environment.

This module provides a Gymnasium-compliant environment that communicates
with an actual Civilization VI game instance through the bridge system.

The environment wraps the file-based IPC bridge to provide the standard
Gymnasium interface (reset, step, observation_space, action_space).

Usage:
    from alpha_zero.envs.real_civ6_env import RealCiv6Env

    env = RealCiv6Env()
    obs, info = env.reset()

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from alpha_zero.bridge.action_converter import ActionConverter, InvalidActionError
from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.file_bridge import (
    ConnectionLostError,
    FileBridge,
    MockFileBridge,
    StateTimeoutError,
)
from alpha_zero.bridge.protocol import (
    CommandMessage,
    GameStateData,
)
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_actions import ActionSpaceConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward scaling in Civ6 environment.

    All reward scales are multiplied by the respective state change value
    (e.g., score_scale * delta_score). This allows fine-tuning the relative
    importance of different game metrics for reinforcement learning.

    Attributes:
        score_scale: Reward per point of score change (default: 0.001).
        city_scale: Reward per city gained/lost (default: 0.1).
        unit_scale: Reward per unit gained/lost (default: 0.01).
        gold_scale: Reward per gold change (default: 0.0001).
        win_bonus: Bonus reward for winning the game (default: 1.0).
        loss_penalty: Penalty for losing the game (default: -1.0).

    Example:
        >>> config = RewardConfig(score_scale=0.002, win_bonus=2.0)
        >>> env = RealCiv6Env(reward_config=config)
    """

    score_scale: float = 0.001
    city_scale: float = 0.1
    unit_scale: float = 0.01
    gold_scale: float = 0.0001
    win_bonus: float = 1.0
    loss_penalty: float = -1.0

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> RewardConfig:
        """Create RewardConfig from dictionary.

        Only recognized keys are used; unknown keys are ignored.
        This enables loading from partial configs or YAML files.

        Args:
            data: Dictionary with reward configuration values.

        Returns:
            RewardConfig instance with values from dict.
        """
        valid_keys = {
            "score_scale", "city_scale", "unit_scale",
            "gold_scale", "win_bonus", "loss_penalty",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_env(cls, prefix: str = "CIV6AZ_REWARD_") -> RewardConfig:
        """Create RewardConfig from environment variables.

        Environment variables:
            {prefix}SCORE_SCALE: Score change reward scale (default: 0.001)
            {prefix}CITY_SCALE: City change reward scale (default: 0.1)
            {prefix}UNIT_SCALE: Unit change reward scale (default: 0.01)
            {prefix}GOLD_SCALE: Gold change reward scale (default: 0.0001)
            {prefix}WIN_BONUS: Win bonus reward (default: 1.0)
            {prefix}LOSS_PENALTY: Loss penalty (default: -1.0)

        Args:
            prefix: Environment variable prefix (default: "CIV6AZ_REWARD_").

        Returns:
            RewardConfig instance with values from environment or defaults.
        """
        return cls(
            score_scale=float(os.environ.get(f"{prefix}SCORE_SCALE", "0.001")),
            city_scale=float(os.environ.get(f"{prefix}CITY_SCALE", "0.1")),
            unit_scale=float(os.environ.get(f"{prefix}UNIT_SCALE", "0.01")),
            gold_scale=float(os.environ.get(f"{prefix}GOLD_SCALE", "0.0001")),
            win_bonus=float(os.environ.get(f"{prefix}WIN_BONUS", "1.0")),
            loss_penalty=float(os.environ.get(f"{prefix}LOSS_PENALTY", "-1.0")),
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary.

        Returns:
            Dictionary with all reward configuration values.
        """
        return asdict(self)


class RealCiv6Env(gym.Env):
    """Gymnasium environment for real Civilization VI gameplay.

    This environment communicates with an actual Civ6 game instance
    through a file-based IPC bridge. The Lua mod in the game exports
    state and executes commands based on this communication.

    Attributes:
        observation_space: Dict space with spatial, vector, and action_mask.
        action_space: Discrete action space.
        metadata: Environment metadata including render modes.

    Example:
        >>> env = RealCiv6Env(bridge_config=BridgeConfig())
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> env.close()
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 1,
    }

    # Default reward configuration (use RewardConfig for customization)
    DEFAULT_REWARD_CONFIG: RewardConfig = RewardConfig()

    def __init__(
        self,
        bridge_config: BridgeConfig | None = None,
        board_size: int = 16,
        max_units: int = 100,
        max_cities: int = 20,
        render_mode: str | None = None,
        state_timeout: float = 60.0,
        action_timeout: float = 30.0,
        invalid_action_penalty: float = -0.1,
        reward_config: RewardConfig | dict[str, float] | None = None,
    ):
        """Initialize the Civ6 environment.

        Args:
            bridge_config: Configuration for the file bridge.
            board_size: Internal board size for neural network.
            max_units: Maximum units per player.
            max_cities: Maximum cities per player.
            render_mode: Render mode (only "human" supported).
            state_timeout: Timeout for initial state (game load).
            action_timeout: Timeout for action acknowledgment.
            invalid_action_penalty: Penalty for invalid actions.
            reward_config: Reward configuration. Can be:
                - RewardConfig instance
                - dict with keys: score_scale, city_scale, unit_scale,
                  gold_scale, win_bonus, loss_penalty
                - None (uses defaults)
        """
        super().__init__()

        self.bridge_config = bridge_config or BridgeConfig()
        self.board_size = board_size
        self.render_mode = render_mode
        self.state_timeout = state_timeout
        self.action_timeout = action_timeout
        self.invalid_action_penalty = invalid_action_penalty

        # Initialize reward config with backwards compatibility
        if reward_config is None:
            self._reward_config = RewardConfig()
        elif isinstance(reward_config, RewardConfig):
            self._reward_config = reward_config
        elif isinstance(reward_config, dict):
            self._reward_config = RewardConfig.from_dict(reward_config)
        else:
            raise TypeError(
                f"reward_config must be RewardConfig, dict, or None, "
                f"got {type(reward_config).__name__}"
            )

        # Create components
        self._state_converter = StateConverter(
            board_size=board_size,
            create_encoder=True,
        )
        self._action_converter = ActionConverter(
            board_size=board_size,
            action_space_config=ActionSpaceConfig(
                board_size=board_size,
                max_units=max_units,
                max_cities=max_cities,
            ),
        )

        # Bridge (created on reset)
        self._bridge: FileBridge | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # State
        self._current_state: dict[str, Any] | None = None
        self._raw_state: GameStateData | None = None
        self._msg_counter: int = 0
        self._turn: int = 0
        self._done: bool = False

        # Define spaces
        self._action_space_size = self._action_converter.action_space_size
        self.action_space = spaces.Discrete(self._action_space_size)

        # Observation space: spatial + vector + action_mask
        self.observation_space = spaces.Dict({
            "spatial": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(84, board_size, board_size),
                dtype=np.float32,
            ),
            "vector": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(512,),
                dtype=np.float32,
            ),
            "action_mask": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._action_space_size,),
                dtype=np.float32,
            ),
        })

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment and wait for game state.

        Note: The actual game must be started/loaded externally in Civ6.
        This method waits for the first state update from the game.

        Args:
            seed: Random seed (not used for real game).
            options: Additional options (not used).

        Returns:
            Tuple of (observation, info).

        Raises:
            StateTimeoutError: If no state received within timeout.
            ConnectionLostError: If connection to game is lost.
        """
        super().reset(seed=seed)

        # Initialize bridge if needed
        if self._bridge is None:
            self._bridge = FileBridge(self.bridge_config)
            self._run_async(self._bridge.start())

        # Clear any old state
        self._bridge.clear_files()
        self._current_state = None
        self._raw_state = None
        self._done = False
        self._msg_counter = 0

        # Wait for initial state from game
        logger.info("Waiting for game state...")
        try:
            self._raw_state = self._run_async(
                self._bridge.wait_for_state(timeout=self.state_timeout)
            )
        except StateTimeoutError:
            logger.error("Timeout waiting for initial game state")
            raise
        except ConnectionLostError:
            logger.error("Lost connection to game")
            raise

        # Convert state
        self._current_state = self._state_converter.convert(self._raw_state)
        self._turn = self._raw_state.turn_id

        logger.info(f"Received initial state: turn={self._turn}")

        return self.get_observation(), self.get_info()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Take an action in the game.

        Args:
            action: Action index from policy network.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self._done:
            logger.warning("Step called on terminated environment")
            return self.get_observation(), 0.0, True, False, self.get_info()

        if self._current_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Get legal actions from current state
        legal_actions = self._current_state.get("_legal_actions", [])

        # Convert action to command
        try:
            result = self._action_converter.convert_with_result(
                action, self._current_state, legal_actions
            )
            command = result.command

            if not result.is_valid:
                logger.warning(f"Invalid action {action}: {result.validation_error}")
                # Return penalty but continue
                return (
                    self.get_observation(),
                    self.invalid_action_penalty,
                    False,
                    False,
                    {"error": "invalid_action", "message": result.validation_error},
                )

        except InvalidActionError as e:
            logger.warning(f"Action conversion failed: {e}")
            return (
                self.get_observation(),
                self.invalid_action_penalty,
                False,
                False,
                {"error": "conversion_error", "message": str(e)},
            )

        # Send command to game
        self._msg_counter += 1
        cmd_msg = CommandMessage(
            msg_id=self._msg_counter,
            commands=[command],
        )

        try:
            ack = self._run_async(
                self._bridge.send_and_wait(cmd_msg, timeout=self.action_timeout)
            )

            if not ack.success:
                logger.warning(f"Action failed: {ack.error_message}")
                return (
                    self.get_observation(),
                    self.invalid_action_penalty,
                    False,
                    False,
                    {"error": "action_failed", "message": ack.error_message},
                )

        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return (
                self.get_observation(),
                0.0,
                False,
                True,  # Truncated due to error
                {"error": "send_failed", "message": str(e)},
            )

        # Wait for next state
        prev_state = self._current_state
        try:
            self._raw_state = self._run_async(
                self._bridge.wait_for_state(
                    timeout=self.bridge_config.command_timeout_ms / 1000.0
                )
            )
            self._current_state = self._state_converter.convert(self._raw_state)

        except StateTimeoutError:
            logger.warning("Timeout waiting for next state")
            return (
                self.get_observation(),
                0.0,
                False,
                True,
                {"error": "state_timeout"},
            )

        except ConnectionLostError:
            logger.error("Connection to game lost")
            self._done = True
            return (
                self.get_observation(),
                0.0,
                True,
                False,
                {"error": "connection_lost"},
            )

        # Calculate reward
        reward = self._calculate_reward(prev_state, self._current_state)

        # Check for game end
        terminated = self._raw_state.is_game_over
        if terminated:
            self._done = True
            # Add win/loss reward from config
            if self._raw_state.winner_id == self._raw_state.player_id:
                reward += self._reward_config.win_bonus
            elif self._raw_state.winner_id is not None:
                reward += self._reward_config.loss_penalty
            # Draw or unknown outcome: no additional reward

        # Check for turn change
        new_turn = self._current_state.get("turn", 0)
        turn_changed = new_turn > self._turn
        self._turn = new_turn

        info = self.get_info()
        info["turn_changed"] = turn_changed
        info["action_executed"] = command.action_type.value

        return self.get_observation(), reward, terminated, False, info

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation.

        This is the public API for retrieving the current observation.
        MCTS and other components should use this method.

        Returns:
            Observation dictionary with spatial, vector, and action_mask.
        """
        if self._current_state is None or self._raw_state is None:
            # Return zeros if no state
            return {
                "spatial": np.zeros(
                    (84, self.board_size, self.board_size), dtype=np.float32
                ),
                "vector": np.zeros(512, dtype=np.float32),
                "action_mask": np.zeros(self._action_space_size, dtype=np.float32),
            }

        # Encode raw state to tensors (use _raw_state for encoding, not _current_state)
        spatial, vector = self._state_converter.convert_and_encode(self._raw_state)

        # Generate action mask
        legal_actions = self._current_state.get("_legal_actions", [])
        action_mask = self._action_converter.generate_action_mask(
            legal_actions, self._current_state
        )

        return {
            "spatial": spatial.numpy(),
            "vector": vector.numpy(),
            "action_mask": action_mask,
        }

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation (deprecated).

        .. deprecated::
            Use :meth:`get_observation` instead. This method will be removed
            in a future version.

        Returns:
            Observation dictionary with spatial, vector, and action_mask.
        """
        import warnings
        warnings.warn(
            "_get_observation() is deprecated, use get_observation() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_observation()

    def get_info(self) -> dict[str, Any]:
        """Get current info dictionary.

        This is the public API for retrieving the current info.
        MCTS and other components should use this method.

        Returns:
            Info dictionary with game state details including winner_id
            for proper game end detection.
        """
        if self._current_state is None:
            return {"turn": 0, "winner_id": None}

        # Extract winner_id from raw state if available
        winner_id = None
        if self._raw_state is not None:
            winner_id = self._raw_state.winner_id

        return {
            "turn": self._turn,
            "units": len(self._current_state.get("units", [])),
            "cities": len(self._current_state.get("cities", [])),
            "gold": self._current_state.get("gold", 0),
            "is_connected": self._bridge.is_connected if self._bridge else False,
            "winner_id": winner_id,
            "is_game_over": self._raw_state.is_game_over if self._raw_state else False,
        }

    def _get_info(self) -> dict[str, Any]:
        """Get current info dictionary (deprecated).

        .. deprecated::
            Use :meth:`get_info` instead. This method will be removed
            in a future version.

        Returns:
            Info dictionary with game state details.
        """
        import warnings
        warnings.warn(
            "_get_info() is deprecated, use get_info() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_info()

    def _calculate_reward(
        self,
        prev_state: dict[str, Any],
        curr_state: dict[str, Any],
    ) -> float:
        """Calculate reward based on state transition.

        Uses configurable reward scaling from self._reward_config.

        Args:
            prev_state: Previous game state.
            curr_state: Current game state.

        Returns:
            Reward value (unbounded, caller should clip if needed).
        """
        reward = 0.0
        config = self._reward_config

        # Score change (filtered for current player)
        prev_score = prev_state.get("score", 0)
        curr_score = curr_state.get("score", 0)
        reward += (curr_score - prev_score) * config.score_scale

        # City count (for current player only)
        prev_cities = len([c for c in prev_state.get("cities", [])
                          if c.get("player") == 0])
        curr_cities = len([c for c in curr_state.get("cities", [])
                          if c.get("player") == 0])
        reward += (curr_cities - prev_cities) * config.city_scale

        # Unit count (for current player only)
        prev_units = len([u for u in prev_state.get("units", [])
                         if u.get("player") == 0])
        curr_units = len([u for u in curr_state.get("units", [])
                         if u.get("player") == 0])
        reward += (curr_units - prev_units) * config.unit_scale

        # Gold change (top-level, already player-specific from StateConverter)
        prev_gold = prev_state.get("gold", 0)
        curr_gold = curr_state.get("gold", 0)
        reward += (curr_gold - prev_gold) * config.gold_scale

        return reward

    def _run_async(self, coro):
        """Run async coroutine from sync context.

        Manages the event loop lifecycle safely, handling closed loops
        and cancellation errors.

        Args:
            coro: Coroutine to run.

        Returns:
            Result of the coroutine.

        Raises:
            RuntimeError: If the event loop cannot be created or is closed.
        """
        # Create new loop if none exists or if existing loop is closed
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        try:
            return self._loop.run_until_complete(coro)
        except asyncio.CancelledError:
            logger.warning("Async operation was cancelled")
            raise
        except Exception as e:
            logger.debug(f"Async operation error: {e}")
            raise

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Currently only supports "human" mode which prints state info.

        Returns:
            None (rendering is to console).
        """
        if self.render_mode == "human":
            info = self.get_info()
            print(f"Turn: {info.get('turn', 0)}")
            print(f"Units: {info.get('units', 0)}")
            print(f"Cities: {info.get('cities', 0)}")
            print(f"Gold: {info.get('gold', 0)}")
        return None

    def close(self) -> None:
        """Close the environment and clean up resources.

        Properly cleans up the bridge connection and event loop,
        cancelling any pending async tasks before closing.
        """
        # Stop bridge first
        if self._bridge is not None:
            try:
                self._run_async(self._bridge.stop())
            except Exception as e:
                logger.warning(f"Error stopping bridge: {e}")
            self._bridge = None

        # Clean up event loop
        if self._loop is not None:
            try:
                # Cancel all pending tasks
                if not self._loop.is_closed():
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    # Allow cancelled tasks to complete
                    if pending:
                        self._loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    self._loop.close()
            except Exception as e:
                logger.warning(f"Error closing event loop: {e}")
            self._loop = None

        logger.info("Environment closed")

    @property
    def unwrapped(self) -> RealCiv6Env:
        """Return the unwrapped environment."""
        return self


class MockCiv6Env(RealCiv6Env):
    """Mock Civ6 environment for testing.

    Uses MockFileBridge instead of real file I/O.
    """

    def __init__(self, **kwargs):
        """Initialize mock environment."""
        super().__init__(**kwargs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset with mock bridge."""
        # Call gym.Env.reset directly (not RealCiv6Env.reset which uses real bridge)
        gym.Env.reset(self, seed=seed)

        # Stop old bridge if it exists to prevent resource leaks
        if self._bridge is not None:
            try:
                self._run_async(self._bridge.stop())
            except Exception as e:
                logger.debug(f"Error stopping old mock bridge: {e}")
            self._bridge = None

        # Use mock bridge
        self._bridge = MockFileBridge(self.bridge_config)
        self._run_async(self._bridge.start())

        # Set mock initial state
        self._set_mock_state(turn_id=1)

        self._done = False
        self._msg_counter = 0

        return self.get_observation(), self.get_info()

    def _set_mock_state(self, turn_id: int = 1) -> None:
        """Set a mock game state.

        Args:
            turn_id: Turn number.
        """
        mock_state = {
            "turn_id": turn_id,
            "player_id": 0,
            "map": {
                "width": 44,
                "height": 26,
                "tiles": [
                    {"x": x, "y": y, "terrain": "TERRAIN_PLAINS"}
                    for x in range(44) for y in range(26)
                ],
            },
            "units": [
                {
                    "id": 1,
                    "type_name": "UNIT_WARRIOR",
                    "owner": 0,
                    "x": 5,
                    "y": 5,
                    "hp": 100,
                    "movement": 2,
                    "combat_strength": 20,
                },
                {
                    "id": 2,
                    "type_name": "UNIT_SETTLER",
                    "owner": 0,
                    "x": 6,
                    "y": 5,
                    "hp": 100,
                    "movement": 2,
                    "is_civilian": True,
                },
            ],
            "cities": [
                {
                    "id": 1,
                    "name": "Capital",
                    "owner": 0,
                    "x": 10,
                    "y": 10,
                    "population": 3,
                    "is_capital": True,
                },
            ],
            "player": {
                "id": 0,
                "gold": 100,
                "science_per_turn": 10,
                "culture_per_turn": 5,
                "faith": 0,
                "score": turn_id * 10,  # Score scales with turn for testing
            },
            "legal_actions": [
                {"action": "end_turn"},
                {"action": "move", "unit_id": 1, "targets": [[4, 5], [5, 4], [6, 5], [5, 6]]},
            ],
        }

        self._bridge.set_mock_state(mock_state)
        self._raw_state = self._run_async(self._bridge.poll_state())
        self._current_state = self._state_converter.convert(self._raw_state)
        self._turn = turn_id

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Take an action in the mock environment.

        Auto-acknowledges commands and provides mock state updates.

        Args:
            action: Action index from policy network.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self._done:
            logger.warning("Step called on terminated environment")
            return self.get_observation(), 0.0, True, False, self.get_info()

        if self._current_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Get legal actions from current state
        legal_actions = self._current_state.get("_legal_actions", [])

        # Convert action to command
        try:
            result = self._action_converter.convert_with_result(
                action, self._current_state, legal_actions
            )
            command = result.command

            if not result.is_valid:
                logger.debug(f"Invalid action {action}: {result.validation_error}")
                return (
                    self.get_observation(),
                    self.invalid_action_penalty,
                    False,
                    False,
                    {"error": "invalid_action", "message": result.validation_error},
                )

        except InvalidActionError as e:
            logger.debug(f"Action conversion failed: {e}")
            return (
                self.get_observation(),
                self.invalid_action_penalty,
                False,
                False,
                {"error": "conversion_error", "message": str(e)},
            )

        # In mock mode, just record the command and advance state
        self._msg_counter += 1

        # Store previous state for reward calculation
        prev_state = self._current_state

        # Advance turn for mock (simulate game progress)
        new_turn = self._turn + 1

        # Generate new mock state
        self._set_mock_state(turn_id=new_turn)

        # Calculate reward
        reward = self._calculate_reward(prev_state, self._current_state)

        # Mock never terminates unless we explicitly set it
        terminated = False
        truncated = False

        info = self.get_info()
        info["turn_changed"] = True
        info["action_executed"] = command.action_type.value

        return self.get_observation(), reward, terminated, truncated, info
