"""Adapter bridging the existing Civ6 MicroCivEnv to SDK's StrategicEnvironment protocol.

This module demonstrates how the SDK's protocol-based design enables
wrapping existing game environments without modifying their source code.

Usage::
    from src.adapters.civ6 import MicroCivAdapter

    env = MicroCivAdapter(board_size=16)
    # Now satisfies StrategicEnvironment protocol
"""
from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.constants import (
    DEFAULT_BOARD_SIZE,
    DEFAULT_MAX_STEPS,
    MICROCIV_ACTION_SPACE,
    MICROCIV_SPATIAL_CHANNELS,
    MICROCIV_VECTOR_DIM,
    STUB_ACTION_DIRECTIONS,
    STUB_SPATIAL_CHANNELS,
    STUB_VECTOR_DIM,
)
from src.core.types import ObservationSpec

logger = logging.getLogger(__name__)


class MicroCivAdapter:
    """Wraps MicroCivEnv to satisfy the StrategicEnvironment protocol.

    This adapter translates between the existing MicroCivEnv interface
    and the SDK's StrategicEnvironment protocol, demonstrating the SDK's
    backwards-compatible, non-invasive integration approach.

    When alpha_zero is not available (e.g. in SDK-only installations),
    this adapter can be used in "stub mode" with a simple fallback environment.

    Args:
        board_size: Grid size for MicroCiv.
        max_steps: Maximum steps per game.
        use_stub: If True, use a simple stub instead of actual MicroCivEnv.
    """

    def __init__(
        self,
        board_size: int = DEFAULT_BOARD_SIZE,
        max_steps: int = DEFAULT_MAX_STEPS,
        use_stub: bool = False,
    ) -> None:
        self._board_size = board_size
        self._max_steps = max_steps
        self._use_stub = use_stub
        self._env: Any = None
        self._step_count = 0
        self._terminated = False
        self._truncated = False

        # Try to import actual MicroCivEnv
        if not use_stub:
            try:
                from alpha_zero.envs.microciv import MicroCivEnv
                self._env = MicroCivEnv(board_size=board_size)
                self._action_space_size_val = (
                    self._env.action_space.n
                    if hasattr(self._env, "action_space")
                    else MICROCIV_ACTION_SPACE
                )
                logger.info("MicroCivAdapter: using real MicroCivEnv")
            except ImportError:
                logger.warning(
                    "alpha_zero not available, falling back to stub mode"
                )
                self._use_stub = True

        if self._use_stub:
            self._action_space_size_val = (
                STUB_ACTION_DIRECTIONS * board_size * board_size + 1
            )
            logger.info("MicroCivAdapter: using stub mode (board=%d)", board_size)

        # Observation spec: MicroCiv uses 13 channels
        channels = MICROCIV_SPATIAL_CHANNELS if not self._use_stub else STUB_SPATIAL_CHANNELS
        vector_dim = MICROCIV_VECTOR_DIM if not self._use_stub else STUB_VECTOR_DIM
        self._observation_spec = ObservationSpec(
            spatial_shape=(channels, board_size, board_size),
            vector_dim=vector_dim,
        )

    @property
    def observation_spec(self) -> ObservationSpec:
        """Specification describing observation tensor shapes."""
        return self._observation_spec

    @property
    def action_space_size(self) -> int:
        """Total number of discrete actions in the flat action space."""
        return self._action_space_size_val

    def reset(self, *, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            The initial observation dictionary with 'spatial' and 'vector' keys.
        """
        self._step_count = 0
        self._terminated = False
        self._truncated = False

        if not self._use_stub and self._env is not None:
            obs, _ = self._env.reset(seed=seed)
            return self._translate_obs(obs)

        # Stub mode
        return self._stub_observation()

    def step(
        self, action: int,
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        """Execute an action and advance the environment by one step.

        Args:
            action: A flat action index.

        Returns:
            A 5-tuple (obs, reward, terminated, truncated, info).
        """
        self._step_count += 1

        if not self._use_stub and self._env is not None:
            obs, reward, terminated, truncated, info = self._env.step(action)
            self._terminated = terminated
            self._truncated = truncated
            reward = max(-1.0, min(1.0, float(reward)))
            return self._translate_obs(obs), reward, terminated, truncated, info

        # Stub mode
        if self._step_count >= self._max_steps:
            self._truncated = True
        obs = self._stub_observation()
        reward = 0.0
        return obs, reward, self._terminated, self._truncated, {"step": self._step_count}

    def get_valid_actions(self) -> NDArray[np.bool_]:
        """Return a boolean mask over the flat action space.

        Returns:
            A boolean array of shape (action_space_size,).
        """
        if not self._use_stub and self._env is not None:
            mask = self._env.get_valid_actions()
            # Ensure bool dtype
            return np.asarray(mask, dtype=np.bool_)
        return np.ones(self._action_space_size_val, dtype=np.bool_)

    def get_observation(self) -> dict[str, NDArray[np.float32]]:
        """Return the current observation without advancing the state.

        Returns:
            Observation dictionary with 'spatial' and 'vector' keys.
        """
        if not self._use_stub and self._env is not None:
            obs = self._env.get_observation()
            return self._translate_obs(obs)
        return self._stub_observation()

    def is_terminal(self) -> bool:
        """Check whether the current episode has ended.

        Returns:
            True if the episode is over, False otherwise.
        """
        if not self._use_stub and self._env is not None:
            return bool(
                getattr(self._env, "done", False)
                or getattr(self._env, "_done", False)
                or self._terminated
                or self._truncated
            )
        return self._terminated or self._truncated

    def clone(self) -> MicroCivAdapter:
        """Create a deep copy of the environment and its internal state.

        Returns:
            A new MicroCivAdapter instance whose state is identical
            to this one but fully independent.
        """
        return copy.deepcopy(self)

    def get_symmetries(
        self,
        observation: NDArray[np.float32],
        policy: NDArray[np.float32],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Return symmetric equivalents of an observation-policy pair.

        For MicroCiv, no spatial symmetries are exploited, so only
        the identity pair is returned.

        Args:
            observation: A spatial observation tensor.
            policy: The MCTS-improved policy distribution over actions.

        Returns:
            A list containing the original (observation, policy) pair.
        """
        return [(observation, policy)]

    def _translate_obs(self, obs: Any) -> dict[str, NDArray[np.float32]]:
        """Translate MicroCivEnv observation to SDK format."""
        if isinstance(obs, dict):
            spatial = np.asarray(
                obs.get(
                    "spatial",
                    obs.get("board", np.zeros(self._observation_spec.spatial_shape)),
                ),
                dtype=np.float32,
            )
            vector = np.asarray(
                obs.get(
                    "vector",
                    obs.get("global", np.zeros(self._observation_spec.vector_dim)),
                ),
                dtype=np.float32,
            )
        elif isinstance(obs, tuple):
            spatial = np.asarray(obs[0], dtype=np.float32)
            vector = (
                np.asarray(obs[1], dtype=np.float32)
                if len(obs) > 1
                else np.zeros(self._observation_spec.vector_dim, dtype=np.float32)
            )
        else:
            spatial = np.zeros(self._observation_spec.spatial_shape, dtype=np.float32)
            vector = np.zeros(self._observation_spec.vector_dim, dtype=np.float32)
        return {"spatial": spatial, "vector": vector}

    def _stub_observation(self) -> dict[str, NDArray[np.float32]]:
        """Generate a stub observation."""
        return {
            "spatial": np.zeros(self._observation_spec.spatial_shape, dtype=np.float32),
            "vector": np.zeros(self._observation_spec.vector_dim, dtype=np.float32),
        }
