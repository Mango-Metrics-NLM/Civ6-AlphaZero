"""TerritoryControl: A 2-player territory capture mini-game.

Players take turns placing markers on a grid to claim territory.
The player who controls the most cells when the board is full wins.
Adjacent cells reinforce control (connected regions score bonus).

Satisfies the StrategicEnvironment protocol.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.types import ObservationSpec

logger = logging.getLogger(__name__)


class TerritoryControlEnv:
    """Territory control environment for curriculum learning.

    State encoding (5 channels):
    - Channel 0: Current player territory (1 = claimed)
    - Channel 1: Opponent territory (1 = claimed)
    - Channel 2: Unclaimed cells (1 = available)
    - Channel 3: Current player score (constant fill, normalized)
    - Channel 4: Turn progress (constant fill, normalized)

    Vector features (4 dims):
    - [0] Current player territory count (normalized)
    - [1] Opponent territory count (normalized)
    - [2] Turn progress (step / total_cells)
    - [3] Bias term (always 1.0)

    Action space: board_size * board_size (one action per cell).
    Invalid actions are cells already claimed.

    Args:
        board_size: Grid height and width.
        seed: Optional default random seed.
    """

    _SPATIAL_CHANNELS = 5
    _VECTOR_DIM = 4

    def __init__(
        self,
        board_size: int = 6,
        seed: int | None = None,
    ) -> None:
        self._board_size = board_size
        self._default_seed = seed
        self._total_cells = board_size * board_size
        self._max_steps = self._total_cells

        self._observation_spec = ObservationSpec(
            spatial_shape=(self._SPATIAL_CHANNELS, board_size, board_size),
            vector_dim=self._VECTOR_DIM,
        )

        # State (initialized in reset)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        # 0 = unclaimed, 1 = player 1, 2 = player 2
        self._board: NDArray[np.int8] = np.zeros(
            (board_size, board_size), dtype=np.int8
        )
        self._current_player: int = 0  # 0 or 1
        self._step_count: int = 0
        self._terminated: bool = False
        self._scores: list[int] = [0, 0]

    @property
    def observation_spec(self) -> ObservationSpec:
        return self._observation_spec

    @property
    def action_space_size(self) -> int:
        return self._total_cells

    def reset(self, *, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
        """Reset to empty board."""
        actual_seed = seed if seed is not None else self._default_seed
        self._rng = np.random.default_rng(actual_seed)

        self._board = np.zeros(
            (self._board_size, self._board_size), dtype=np.int8
        )
        self._current_player = 0
        self._step_count = 0
        self._terminated = False
        self._scores = [0, 0]
        logger.debug(
            "TerritoryControl reset: board=%d, max_steps=%d, seed=%s",
            self._board_size,
            self._max_steps,
            actual_seed,
        )

        return self.get_observation()

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        """Place a marker at the chosen cell."""
        if self._terminated:
            raise RuntimeError("Cannot step a terminated environment")

        if not (0 <= action < self._total_cells):
            raise ValueError(
                f"Invalid action {action}, must be in [0, {self._total_cells})"
            )

        row = action // self._board_size
        col = action % self._board_size

        # Capture who is acting before the player switch
        acting_player = self._current_player

        # Check if cell is valid (unclaimed)
        if self._board[row, col] != 0:
            # Invalid move: current player loses the turn
            reward = -0.1
        else:
            # Claim the cell
            player_marker = acting_player + 1  # 1 or 2
            self._board[row, col] = player_marker
            self._scores[acting_player] += 1
            reward = 0.0

        # Switch player
        self._current_player = 1 - self._current_player
        self._step_count += 1

        # Check termination: board full or no valid moves
        unclaimed = int(np.sum(self._board == 0))
        if unclaimed == 0 or self._step_count >= self._max_steps:
            self._terminated = True

            # Compute final scores with territory bonus
            score_p1 = self._compute_territory_score(1)
            score_p2 = self._compute_territory_score(2)

            # Reward from acting player's perspective (consistent with
            # MiniResourceEnv and TacticalCombatEnv conventions)
            acting_marker = acting_player + 1
            acting_score = score_p1 if acting_marker == 1 else score_p2
            opponent_score = score_p2 if acting_marker == 1 else score_p1

            if acting_score > opponent_score:
                reward = 1.0
            elif opponent_score > acting_score:
                reward = -1.0
            else:
                reward = 0.0

        reward = max(-1.0, min(1.0, reward))

        info: dict[str, Any] = {
            "scores": list(self._scores),
            "step": self._step_count,
            "current_player": self._current_player,
            "unclaimed": unclaimed,
        }

        return self.get_observation(), reward, self._terminated, False, info

    def _compute_territory_score(self, player_marker: int) -> int:
        """Compute territory score: owned cells + bonus for connected regions."""
        owned = int(np.sum(self._board == player_marker))
        # Bonus: largest connected region size
        bonus = self._largest_connected_region(player_marker)
        return owned + bonus

    def _largest_connected_region(self, player_marker: int) -> int:
        """Find the largest connected region of a player via BFS."""
        visited = np.zeros_like(self._board, dtype=bool)
        max_size = 0

        for r in range(self._board_size):
            for c in range(self._board_size):
                if self._board[r, c] == player_marker and not visited[r, c]:
                    size = self._bfs(r, c, player_marker, visited)
                    max_size = max(max_size, size)

        return max_size

    def _bfs(
        self,
        start_r: int,
        start_c: int,
        marker: int,
        visited: NDArray[np.bool_],
    ) -> int:
        """BFS to find connected region size."""
        queue = [(start_r, start_c)]
        visited[start_r, start_c] = True
        size = 0

        while queue:
            r, c = queue.pop(0)
            size += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self._board_size
                    and 0 <= nc < self._board_size
                    and not visited[nr, nc]
                    and self._board[nr, nc] == marker
                ):
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        return size

    def get_valid_actions(self) -> NDArray[np.bool_]:
        """Return mask of unclaimed cells."""
        mask: NDArray[np.bool_] = self._board.flatten() == 0
        return mask

    def get_observation(self) -> dict[str, NDArray[np.float32]]:
        """Build observation from current player's perspective."""
        spatial = np.zeros(
            (self._SPATIAL_CHANNELS, self._board_size, self._board_size),
            dtype=np.float32,
        )

        player_marker = self._current_player + 1
        opponent_marker = 2 - self._current_player

        # Channel 0: current player territory
        spatial[0] = (self._board == player_marker).astype(np.float32)
        # Channel 1: opponent territory
        spatial[1] = (self._board == opponent_marker).astype(np.float32)
        # Channel 2: unclaimed
        spatial[2] = (self._board == 0).astype(np.float32)
        # Channel 3: score normalized
        spatial[3, :, :] = self._scores[self._current_player] / max(self._total_cells, 1)
        # Channel 4: progress
        spatial[4, :, :] = self._step_count / max(self._max_steps, 1)

        vector = np.array(
            [
                self._scores[self._current_player] / max(self._total_cells, 1),
                self._scores[1 - self._current_player] / max(self._total_cells, 1),
                self._step_count / max(self._max_steps, 1),
                1.0,
            ],
            dtype=np.float32,
        )

        return {"spatial": spatial, "vector": vector}

    def is_terminal(self) -> bool:
        return self._terminated

    def clone(self) -> TerritoryControlEnv:
        """Deep copy the environment."""
        return copy.deepcopy(self)

    def get_symmetries(
        self,
        observation: NDArray[np.float32],
        policy: NDArray[np.float32],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Return 4 rotational symmetries + identity."""
        symmetries = [(observation, policy)]
        obs = observation
        pol = policy.reshape(self._board_size, self._board_size)

        for k in range(1, 4):
            rotated_obs = np.rot90(obs, k, axes=(1, 2)).copy()
            rotated_pol = np.rot90(pol, k).flatten().copy()
            symmetries.append((rotated_obs, rotated_pol))

        return symmetries

    def __repr__(self) -> str:
        return (
            f"TerritoryControlEnv(board_size={self._board_size}, "
            f"step={self._step_count}, scores={self._scores})"
        )
