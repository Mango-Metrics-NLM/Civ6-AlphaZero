"""Vectorized MicroCiv environment for batch self-play.

Manages N MicroCivEnv instances with shared pre-allocated observation buffers
to minimize memory allocation overhead during parallel self-play.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from alpha_zero.envs.civ6_encoder import MicroCivEncoder
from alpha_zero.envs.env_factory import create_game_env
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.utils.wandb_logger import log_metrics

logger = logging.getLogger(__name__)


@dataclass
class VectorizedEnvConfig:
    """Configuration for vectorized environment.

    Args:
        num_envs: Number of parallel environments (>= 1).
        board_size: Size of the square game board (>= 1).
        max_turns: Maximum turns before draw (>= 1).
        seed_offset: Offset for environment seeds. If 0, envs are unseeded.
    """

    num_envs: int = 8
    board_size: int = 8
    max_turns: int = 30
    seed_offset: int = 0
    gameplay_preset: str = "default"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
        if self.board_size < 1:
            raise ValueError(f"board_size must be >= 1, got {self.board_size}")
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")
        from alpha_zero.envs.env_factory import GAMEPLAY_PRESETS

        if self.gameplay_preset not in GAMEPLAY_PRESETS:
            raise ValueError(
                f"Unknown gameplay_preset {self.gameplay_preset!r}. "
                f"Valid presets: {GAMEPLAY_PRESETS}"
            )


class VectorizedMicroCivEnv:
    """Vectorized MicroCiv environment for batch self-play.

    Manages multiple MicroCivEnv instances with pre-allocated numpy buffers
    to reduce memory allocation overhead during parallel training.

    Example:
        >>> config = VectorizedEnvConfig(num_envs=16, board_size=8, max_turns=30)
        >>> vec_env = VectorizedMicroCivEnv(config)
        >>> vec_env.reset_all()
        >>> spatial, vector, action_mask = vec_env.get_observations()
        >>> actions = np.zeros(16, dtype=np.int32)  # Select actions
        >>> rewards, terminated, truncated, infos = vec_env.step_all(actions)
    """

    def __init__(self, config: VectorizedEnvConfig | None = None) -> None:
        """Initialize vectorized environment.

        Args:
            config: Configuration for vectorized environment. If None, uses defaults.
        """
        self.config = config or VectorizedEnvConfig()

        # Create individual environments via centralized factory
        self._envs: list[MicroCivEnv] = []
        for i in range(self.config.num_envs):
            seed = None if self.config.seed_offset == 0 else self.config.seed_offset + i
            env = create_game_env(
                board_size=self.config.board_size,
                max_turns=self.config.max_turns,
                gameplay_preset=self.config.gameplay_preset,
                seed=seed,
            )
            self._envs.append(env)

        # Get dimensions from first environment
        sample_obs = self._envs[0].get_observation()
        spatial_shape = sample_obs["spatial"].shape
        vector_shape = sample_obs["vector"].shape
        action_mask_shape = sample_obs["action_mask"].shape

        # Validate shapes match expected values
        expected_spatial_channels = MicroCivEncoder.TOTAL_CHANNELS
        if spatial_shape[0] != expected_spatial_channels:
            raise ValueError(
                f"Expected {expected_spatial_channels} spatial channels, got {spatial_shape[0]}"
            )

        # Pre-allocate observation buffers
        self._spatial_buffer = np.zeros(
            (self.config.num_envs, *spatial_shape),
            dtype=np.float32,
        )
        self._vector_buffer = np.zeros(
            (self.config.num_envs, *vector_shape),
            dtype=np.float32,
        )
        self._action_mask_buffer = np.zeros(
            (self.config.num_envs, *action_mask_shape),
            dtype=np.float32,
        )

        # State tracking
        self._done = np.zeros(self.config.num_envs, dtype=bool)
        self._rewards = np.zeros(self.config.num_envs, dtype=np.float32)

        # Store dimensions
        self._action_space_size = action_mask_shape[0]

        logger.info(
            f"Initialized VectorizedMicroCivEnv: {self.config.num_envs} envs, "
            f"board_size={self.config.board_size}, max_turns={self.config.max_turns}, "
            f"spatial={spatial_shape}, vector={vector_shape}, "
            f"action_space={self._action_space_size}"
        )

    def reset_all(self) -> None:
        """Reset all environments and update buffers.

        Sets all done flags to False and fills all observation buffers
        with fresh environment states.
        """
        start_time = time.perf_counter()

        for i, env in enumerate(self._envs):
            obs, _ = env.reset()
            self._spatial_buffer[i] = obs["spatial"]
            self._vector_buffer[i] = obs["vector"]
            self._action_mask_buffer[i] = obs["action_mask"]
            self._done[i] = False

        self._rewards.fill(0.0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"reset_all() completed in {elapsed_ms:.2f}ms")

    def reset_done(self) -> int:
        """Reset only environments where done=True.

        Returns:
            Number of environments that were reset.
        """
        done_indices = np.where(self._done)[0]
        if len(done_indices) == 0:
            return 0

        start_time = time.perf_counter()

        for i in done_indices:
            obs, _ = self._envs[i].reset()
            self._spatial_buffer[i] = obs["spatial"]
            self._vector_buffer[i] = obs["vector"]
            self._action_mask_buffer[i] = obs["action_mask"]
            self._done[i] = False
            self._rewards[i] = 0.0

        num_resets = len(done_indices)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(f"reset_done() reset {num_resets} envs in {elapsed_ms:.2f}ms")

        # Log metrics
        log_metrics({"vectorized_env/resets_triggered": num_resets})

        return num_resets

    def step_all(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step all non-done environments with given actions.

        Only environments where done=False are stepped. Done environments
        retain their previous state until explicitly reset.

        Args:
            actions: Array of actions shape (num_envs,) with dtype int.

        Returns:
            Tuple of (rewards, terminated, truncated, infos):
                - rewards: Array of shape (num_envs,) with latest rewards
                - terminated: Array of shape (num_envs,) indicating game over
                - truncated: Array of shape (num_envs,) indicating time limit
                - infos: List of info dicts for each environment
        """
        if len(actions) != self.config.num_envs:
            raise ValueError(f"Expected {self.config.num_envs} actions, got {len(actions)}")

        start_time = time.perf_counter()

        # Prepare output arrays
        rewards = np.zeros(self.config.num_envs, dtype=np.float32)
        terminated = np.zeros(self.config.num_envs, dtype=bool)
        truncated = np.zeros(self.config.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        # Step each environment
        for i in range(self.config.num_envs):
            if self._done[i]:
                # Environment already done, skip stepping
                infos.append({})
                continue

            # Step environment
            obs, reward, term, trunc, info = self._envs[i].step(int(actions[i]))

            # Update buffers
            self._spatial_buffer[i] = obs["spatial"]
            self._vector_buffer[i] = obs["vector"]
            self._action_mask_buffer[i] = obs["action_mask"]

            # Store outputs
            rewards[i] = float(reward)
            terminated[i] = term
            truncated[i] = trunc
            infos.append(info)

            # Update done status
            if term or trunc:
                self._done[i] = True

        # Store rewards
        self._rewards = rewards

        # Compute metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        active_envs = (~self._done).sum()

        logger.debug(
            f"step_all() completed in {elapsed_ms:.2f}ms, "
            f"active={active_envs}/{self.config.num_envs}"
        )

        # Log metrics
        log_metrics(
            {
                "vectorized_env/active_envs": int(active_envs),
                "vectorized_env/step_time_ms": elapsed_ms,
            }
        )

        return rewards, terminated, truncated, infos

    def get_observations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current observations for all environments.

        Returns zero-copy views into the internal buffers.

        Returns:
            Tuple of (spatial, vector, action_mask):
                - spatial: shape (num_envs, spatial_channels, H, W)
                - vector: shape (num_envs, vector_dim)
                - action_mask: shape (num_envs, action_space_size)
        """
        return (
            self._spatial_buffer,
            self._vector_buffer,
            self._action_mask_buffer,
        )

    def get_active_envs(self) -> list[int]:
        """Get indices of non-done environments.

        Returns:
            List of environment indices where done=False.
        """
        return list(np.where(~self._done)[0].tolist())

    def get_env(self, idx: int) -> MicroCivEnv:
        """Get individual environment by index.

        Useful for MCTS which needs access to underlying environment.

        Args:
            idx: Environment index in [0, num_envs).

        Returns:
            The MicroCivEnv instance at the given index.

        Raises:
            IndexError: If idx is out of bounds.
        """
        if idx < 0 or idx >= self.config.num_envs:
            raise IndexError(f"Environment index {idx} out of bounds [0, {self.config.num_envs})")
        return self._envs[idx]

    def clone_env(self, idx: int) -> MicroCivEnv:
        """Clone an environment by index.

        Creates a deep copy suitable for MCTS tree search.

        Args:
            idx: Environment index in [0, num_envs).

        Returns:
            Cloned MicroCivEnv instance.

        Raises:
            IndexError: If idx is out of bounds.
        """
        return self.get_env(idx).clone()

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self.config.num_envs

    @property
    def board_size(self) -> int:
        """Board size of each environment."""
        return self.config.board_size

    @property
    def action_space_size(self) -> int:
        """Size of the discrete action space."""
        return int(self._action_space_size)

    @property
    def active_count(self) -> int:
        """Number of non-done environments."""
        return int((~self._done).sum())
