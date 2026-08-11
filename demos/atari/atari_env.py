"""Atari environment wrapper implementing the MCTSEnvironment protocol.

Wraps a Gymnasium Atari environment with standard preprocessing (frame
stacking, grayscale, resizing, reward clipping) and adapts it to the
``MCTSEnvironment`` protocol used by the core MCTS implementations.

Example:
    >>> from demos.atari.atari_env import AtariEnv, AtariEnvConfig
    >>> env = AtariEnv(AtariEnvConfig(game_name="Pong"))
    >>> obs = env.get_observation()
    >>> print(obs["spatial"].shape)  # (4, 84, 84) by default
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from demos.atari.config import AtariEnvConfig

logger = logging.getLogger(__name__)


def _preprocess_frame(
    frame: np.ndarray,
    height: int,
    width: int,
    grayscale: bool,
    grayscale_weights: tuple[float, float, float] = (0.2989, 0.5870, 0.1140),
) -> np.ndarray:
    """Preprocess a raw Atari frame.

    Pipeline: grayscale conversion -> resize -> float32 normalisation [0, 1].

    Args:
        frame: Raw RGB frame from the emulator (H, W, 3).
        height: Target height after resize.
        width: Target width after resize.
        grayscale: Whether to convert to grayscale.
        grayscale_weights: (R, G, B) luminance weights for grayscale conversion.

    Returns:
        Preprocessed frame as float32 in [0, 1], shape (H, W) or (H, W, 3).
    """
    # Grayscale conversion (configurable luminance weights)
    if grayscale and frame.ndim == 3 and frame.shape[2] == 3:
        frame = np.dot(frame[..., :3], list(grayscale_weights)).astype(np.uint8)

    # Resize via simple nearest-neighbour interpolation (avoids cv2 dependency)
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = _resize_nearest(frame, height, width)

    return frame.astype(np.float32) / 255.0


def _resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize using nearest-neighbour interpolation (no opencv needed).

    Args:
        image: Source image, 2-D or 3-D.
        target_h: Target height.
        target_w: Target width.

    Returns:
        Resized image.
    """
    src_h, src_w = image.shape[:2]
    row_idx = (np.arange(target_h) * src_h / target_h).astype(int)
    col_idx = (np.arange(target_w) * src_w / target_w).astype(int)
    row_idx = np.clip(row_idx, 0, src_h - 1)
    col_idx = np.clip(col_idx, 0, src_w - 1)
    return image[np.ix_(row_idx, col_idx)]


class AtariEnv:
    """Atari environment adapter for the ``MCTSEnvironment`` protocol.

    This class wraps a Gymnasium Atari environment and provides:
    - Frame stacking with configurable depth
    - Grayscale conversion and resizing
    - Reward clipping
    - Life-loss-as-terminal signal
    - The ``clone()`` / ``get_observation()`` / ``get_valid_actions()`` interface
      required by the core MCTS implementations.
    """

    def __init__(self, config: AtariEnvConfig | None = None) -> None:
        """Initialise the wrapper.

        Args:
            config: Environment configuration.  Defaults are used when *None*.
        """
        self.config = config or AtariEnvConfig()
        self._build_env()

        # Frame stack buffer
        self._frame_stack: deque[np.ndarray] = deque(maxlen=self.config.frame_stack)

        # Episode bookkeeping
        self._lives: int = 0
        self._episode_reward: float = 0.0
        self._step_count: int = 0
        self._done: bool = False
        self._last_raw_frame: np.ndarray | None = None

        # Protocol attributes
        self.current_player: int = 0  # single-player

        logger.debug(
            "AtariEnv created: game=%s, spatial=%s, action_space=%d",
            self.config.game_name,
            self.config.spatial_size,
            self.action_space.n,  # type: ignore[attr-defined,attr-defined]
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_env(self) -> None:
        """Construct the underlying Gymnasium environment."""
        kwargs: dict[str, Any] = {"render_mode": None}
        if self.config.frame_skip > 1:
            kwargs["frameskip"] = self.config.frame_skip
        if self.config.sticky_actions > 0:
            kwargs["repeat_action_probability"] = self.config.sticky_actions

        env_id = self.config.gymnasium_id

        try:
            self._env = gym.make(env_id, **kwargs)
        except gym.error.NameNotFound:
            # Fallback: try without ALE/ prefix for older Gymnasium versions
            fallback_id = f"{self.config.game_name}NoFrameskip-v4"
            logger.warning("Env %s not found, trying %s", env_id, fallback_id)
            self._env = gym.make(fallback_id)

        self.action_space = self._env.action_space

        if self.config.max_episode_steps > 0:
            self._env = gym.wrappers.TimeLimit(
                self._env,
                max_episode_steps=self.config.max_episode_steps,
            )

    # ------------------------------------------------------------------
    # Gymnasium-like interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment and return initial observation.

        Args:
            seed: Optional random seed.

        Returns:
            Tuple of (observation_dict, info_dict).
        """
        obs, info = self._env.reset(seed=seed)
        self._last_raw_frame = np.asarray(obs)

        # Initialise lives tracking
        self._lives = info.get("lives", 0)
        self._episode_reward = 0.0
        self._step_count = 0
        self._done = False

        # Fill frame stack with the initial frame
        processed = self._process_frame(self._last_raw_frame)
        for _ in range(self.config.frame_stack):
            self._frame_stack.append(processed)

        return self.get_observation(), self.get_info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute an action.

        Args:
            action: Action index.

        Returns:
            (obs, reward, terminated, truncated, info) tuple.
        """
        raw_obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_raw_frame = np.asarray(raw_obs)
        self._step_count += 1

        # Life-loss-as-terminal
        real_terminated = terminated
        current_lives = info.get("lives", 0)
        if self.config.terminal_on_life_loss and current_lives < self._lives and not terminated:
            terminated = True
        self._lives = current_lives

        # Reward clipping
        clipped_reward = float(reward)
        if self.config.clip_rewards:
            clipped_reward = float(np.sign(reward))  # type: ignore[call-overload,call-overload]

        self._episode_reward += float(reward)

        # Update frame stack
        processed = self._process_frame(self._last_raw_frame)
        self._frame_stack.append(processed)

        self._done = real_terminated or truncated

        return self.get_observation(), clipped_reward, terminated, truncated, self.get_info()

    # ------------------------------------------------------------------
    # MCTSEnvironment protocol
    # ------------------------------------------------------------------

    def clone(self) -> AtariEnv:
        """Create a deep copy for MCTS simulation.

        Returns:
            Independent copy of this environment.
        """
        cloned = AtariEnv.__new__(AtariEnv)
        cloned.config = self.config
        cloned._env = copy.deepcopy(self._env)
        cloned._frame_stack = deque(
            [f.copy() for f in self._frame_stack],
            maxlen=self.config.frame_stack,
        )
        cloned._lives = self._lives
        cloned._episode_reward = self._episode_reward
        cloned._step_count = self._step_count
        cloned._done = self._done
        cloned._last_raw_frame = (
            self._last_raw_frame.copy() if self._last_raw_frame is not None else None
        )
        cloned.current_player = self.current_player
        cloned.action_space = cloned._env.action_space
        return cloned

    def get_observation(self) -> dict[str, np.ndarray]:
        """Build the observation dict expected by the network / MCTS.

        Returns:
            Dict with keys ``spatial``, ``vector``, ``action_mask``.
        """
        # Stack frames → (C, H, W)
        frames = list(self._frame_stack)
        if self.config.grayscale:
            spatial = np.stack(frames, axis=0)  # (stack, H, W)
        else:
            spatial = np.concatenate(
                [f.transpose(2, 0, 1) for f in frames], axis=0
            )  # (stack*3, H, W)

        # Auxiliary vector: normalised episode stats
        vector = np.zeros(self.config.vector_dim, dtype=np.float32)
        if self.config.vector_dim >= 4:
            max_steps = max(self.config.max_episode_steps, 10_000)
            vector[0] = self._step_count / max_steps  # progress
            vector[1] = self._episode_reward / max(abs(self._episode_reward), 1.0)  # norm reward
            vector[2] = self._lives / max(self._lives, 1)  # norm lives
            vector[3] = float(self._done)

        # All actions always valid in Atari
        action_mask = np.ones(self.action_space.n, dtype=np.float32)  # type: ignore[attr-defined,attr-defined]

        return {
            "spatial": spatial.astype(np.float32),
            "vector": vector,
            "action_mask": action_mask,
        }

    def get_info(self) -> dict[str, Any]:
        """Return environment info dict.

        Returns:
            Info dict (compatible with MCTS terminal-value logic).
        """
        info: dict[str, Any] = {
            "episode_reward": self._episode_reward,
            "step_count": self._step_count,
            "lives": self._lives,
        }
        # For MCTS: single-player → no "winner" concept; use reward signal
        if self._done:
            info["terminal"] = True
        return info

    def get_valid_actions(self) -> np.ndarray:
        """Return mask of valid actions (all ones for Atari).

        Returns:
            Binary mask array of shape ``(num_actions,)``.
        """
        return np.ones(self.action_space.n, dtype=np.float32)  # type: ignore[attr-defined,attr-defined]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to a single frame."""
        return _preprocess_frame(
            frame,
            height=self.config.frame_height,
            width=self.config.frame_width,
            grayscale=self.config.grayscale,
            grayscale_weights=self.config.grayscale_weights,
        )

    def close(self) -> None:
        """Release resources."""
        self._env.close()

    def __repr__(self) -> str:
        return (
            f"AtariEnv(game={self.config.game_name!r}, "
            f"spatial={self.config.spatial_channels}x{self.config.frame_height}x"
            f"{self.config.frame_width}, actions={self.action_space.n})"  # type: ignore[attr-defined,attr-defined]
        )
