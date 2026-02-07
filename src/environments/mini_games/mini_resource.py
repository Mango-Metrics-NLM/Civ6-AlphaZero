"""MiniResource: A simple resource-collection environment for SDK testing.

A 2-player game on a small grid. Each player has one unit that can move
in 4 directions or stay. Resources spawn randomly. The player who collects
the most resources by the step limit wins.

This environment satisfies the StrategicEnvironment protocol.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.constants import RESOURCE_COLLECTION_REWARD
from src.core.types import ObservationSpec

logger = logging.getLogger(__name__)

# Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
_NUM_ACTIONS = 5
_DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]


class MiniResourceEnv:
    """Simple resource-collection environment for testing and curriculum learning.

    State encoding (6 channels):
    - Channel 0: Current player position (1-hot)
    - Channel 1: Opponent position (1-hot)
    - Channel 2: Resource locations
    - Channel 3: Current player score (constant fill, normalized)
    - Channel 4: Opponent score (constant fill, normalized)
    - Channel 5: Turn progress (constant fill, normalized)

    Vector features (4 dims):
    - [0] Current player score (normalized by max_resources)
    - [1] Opponent score (normalized)
    - [2] Turn progress (step / max_steps)
    - [3] Bias term (always 1.0)

    Args:
        board_size: Grid height and width.
        max_steps: Maximum steps before game ends.
        num_resources: Number of resources on the board.
        seed: Optional default random seed.
    """

    _SPATIAL_CHANNELS = 6
    _VECTOR_DIM = 4

    def __init__(
        self,
        board_size: int = 8,
        max_steps: int = 50,
        num_resources: int = 5,
        seed: int | None = None,
    ) -> None:
        self._board_size = board_size
        self._max_steps = max_steps
        self._num_resources = num_resources
        self._default_seed = seed

        self._observation_spec = ObservationSpec(
            spatial_shape=(self._SPATIAL_CHANNELS, board_size, board_size),
            vector_dim=self._VECTOR_DIM,
        )

        # State (initialized in reset)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._positions: list[tuple[int, int]] = [(0, 0), (0, 0)]
        self._scores: list[int] = [0, 0]
        self._resources: set[tuple[int, int]] = set()
        self._current_player: int = 0
        self._step_count: int = 0
        self._terminated: bool = False
        self._truncated: bool = False

    @property
    def observation_spec(self) -> ObservationSpec:
        return self._observation_spec

    @property
    def action_space_size(self) -> int:
        return _NUM_ACTIONS

    def reset(self, *, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
        """Reset to initial state."""
        actual_seed = seed if seed is not None else self._default_seed
        self._rng = np.random.default_rng(actual_seed)

        # Place players at opposite corners
        self._positions = [
            (0, 0),
            (self._board_size - 1, self._board_size - 1),
        ]
        self._scores = [0, 0]
        self._current_player = 0
        self._step_count = 0
        self._terminated = False
        self._truncated = False

        # Place resources randomly (not on player positions)
        self._resources = set()
        occupied = set(self._positions)
        while len(self._resources) < self._num_resources:
            r = int(self._rng.integers(0, self._board_size))
            c = int(self._rng.integers(0, self._board_size))
            if (r, c) not in occupied and (r, c) not in self._resources:
                self._resources.add((r, c))

        logger.debug(
            "MiniResource reset: board=%d, resources=%d, seed=%s",
            self._board_size,
            self._num_resources,
            actual_seed,
        )
        return self.get_observation()

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        """Execute action for current player."""
        if self._terminated or self._truncated:
            raise RuntimeError("Cannot step a terminated environment")

        if not (0 <= action < _NUM_ACTIONS):
            raise ValueError(f"Invalid action {action}, must be in [0, {_NUM_ACTIONS})")

        # Move current player
        player = self._current_player
        row, col = self._positions[player]
        dr, dc = _DIRECTIONS[action]
        new_row = max(0, min(self._board_size - 1, row + dr))
        new_col = max(0, min(self._board_size - 1, col + dc))
        self._positions[player] = (new_row, new_col)

        # Collect resource if present
        reward = 0.0
        if (new_row, new_col) in self._resources:
            self._resources.discard((new_row, new_col))
            self._scores[player] += 1
            reward = RESOURCE_COLLECTION_REWARD

        # Switch player
        self._current_player = 1 - self._current_player
        self._step_count += 1

        # Check termination
        if self._step_count >= self._max_steps or len(self._resources) == 0:
            if self._step_count >= self._max_steps:
                self._truncated = True
            else:
                self._terminated = True

            # Final reward based on who collected more
            if self._scores[0] > self._scores[1]:
                reward = 1.0 if player == 0 else -1.0
            elif self._scores[1] > self._scores[0]:
                reward = 1.0 if player == 1 else -1.0
            else:
                reward = 0.0

        # Clamp reward
        reward = max(-1.0, min(1.0, reward))

        info: dict[str, Any] = {
            "scores": list(self._scores),
            "step": self._step_count,
            "current_player": self._current_player,
        }

        obs = self.get_observation()
        return obs, reward, self._terminated, self._truncated, info

    def get_valid_actions(self) -> NDArray[np.bool_]:
        """All 5 actions are always valid."""
        return np.ones(_NUM_ACTIONS, dtype=np.bool_)

    def get_observation(self) -> dict[str, NDArray[np.float32]]:
        """Build observation from current player's perspective."""
        spatial = np.zeros(
            (self._SPATIAL_CHANNELS, self._board_size, self._board_size),
            dtype=np.float32,
        )

        player = self._current_player
        opponent = 1 - player

        # Channel 0: current player position
        pr, pc = self._positions[player]
        spatial[0, pr, pc] = 1.0

        # Channel 1: opponent position
        opr, opc = self._positions[opponent]
        spatial[1, opr, opc] = 1.0

        # Channel 2: resources
        for r, c in self._resources:
            spatial[2, r, c] = 1.0

        # Channel 3-5: score and progress (constant fill)
        max_res = max(self._num_resources, 1)
        spatial[3, :, :] = self._scores[player] / max_res
        spatial[4, :, :] = self._scores[opponent] / max_res
        spatial[5, :, :] = self._step_count / max(self._max_steps, 1)

        # Vector features
        vector = np.array(
            [
                self._scores[player] / max_res,
                self._scores[opponent] / max_res,
                self._step_count / max(self._max_steps, 1),
                1.0,  # bias
            ],
            dtype=np.float32,
        )

        return {"spatial": spatial, "vector": vector}

    def is_terminal(self) -> bool:
        return self._terminated or self._truncated

    def clone(self) -> MiniResourceEnv:
        """Deep copy the environment."""
        return copy.deepcopy(self)

    def get_symmetries(
        self,
        observation: NDArray[np.float32],
        policy: NDArray[np.float32],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Return identity symmetry (no board symmetries exploited for simplicity)."""
        return [(observation, policy)]

    def __repr__(self) -> str:
        return (
            f"MiniResourceEnv(board_size={self._board_size}, "
            f"max_steps={self._max_steps}, step={self._step_count})"
        )
