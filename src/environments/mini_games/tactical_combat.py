"""TacticalCombat: A 2-player tactical unit combat mini-game.

Players command units on a grid, moving and attacking each turn.
Units have health and attack values.  The player who eliminates
all enemy units wins.

Satisfies the StrategicEnvironment protocol.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from src.core.types import ObservationSpec

logger = logging.getLogger(__name__)


@dataclass
class Unit:
    """A combat unit on the grid."""

    player: int
    row: int
    col: int
    health: int = 10
    max_health: int = 10
    attack: int = 3
    alive: bool = True

    def take_damage(self, damage: int) -> None:
        self.health = max(0, self.health - damage)
        if self.health == 0:
            self.alive = False


class TacticalCombatEnv:
    """Tactical combat environment for curriculum learning.

    Each player starts with ``num_units`` units placed on their side
    of the board.  On each turn the current player selects one of
    their units and an action (move or attack).

    Action encoding (per unit):
    - 0-3: Move (up/right/down/left)
    - 4: Stay / skip
    Total actions = num_units * 5

    State encoding (7 channels):
    - Channel 0: Current player unit positions
    - Channel 1: Current player unit health (normalized)
    - Channel 2: Opponent unit positions
    - Channel 3: Opponent unit health (normalized)
    - Channel 4: Current player unit count (constant fill, normalized)
    - Channel 5: Opponent unit count (constant fill, normalized)
    - Channel 6: Turn progress (constant fill, normalized)

    Vector features (5 dims):
    - [0] Current player total health (normalized)
    - [1] Opponent total health (normalized)
    - [2] Current player alive units (normalized)
    - [3] Opponent alive units (normalized)
    - [4] Bias term (always 1.0)

    Args:
        board_size: Grid height and width.
        num_units: Number of units per player.
        max_steps: Maximum steps before draw.
        attack_range: Manhattan distance for attacks.
        seed: Optional default random seed.
    """

    _SPATIAL_CHANNELS = 7
    _VECTOR_DIM = 5
    _ACTIONS_PER_UNIT = 5  # up, right, down, left, stay
    _DIRECTIONS: ClassVar[list[tuple[int, int]]] = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

    def __init__(
        self,
        board_size: int = 8,
        num_units: int = 3,
        max_steps: int = 100,
        attack_range: int = 1,
        seed: int | None = None,
    ) -> None:
        self._board_size = board_size
        self._num_units = num_units
        self._max_steps = max_steps
        self._attack_range = attack_range
        self._default_seed = seed

        self._total_actions = num_units * self._ACTIONS_PER_UNIT

        self._observation_spec = ObservationSpec(
            spatial_shape=(self._SPATIAL_CHANNELS, board_size, board_size),
            vector_dim=self._VECTOR_DIM,
        )

        # State (initialized in reset)
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._units: list[list[Unit]] = [[], []]
        self._current_player: int = 0
        self._step_count: int = 0
        self._terminated: bool = False
        self._truncated: bool = False

    @property
    def observation_spec(self) -> ObservationSpec:
        return self._observation_spec

    @property
    def action_space_size(self) -> int:
        return self._total_actions

    def reset(self, *, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
        """Reset: place units on opposite sides of the board."""
        actual_seed = seed if seed is not None else self._default_seed
        self._rng = np.random.default_rng(actual_seed)

        self._current_player = 0
        self._step_count = 0
        self._terminated = False
        self._truncated = False

        # Place player 0 units on top rows
        self._units = [[], []]
        for _i in range(self._num_units):
            col = int(self._rng.integers(0, self._board_size))
            self._units[0].append(Unit(player=0, row=0, col=col))

        # Place player 1 units on bottom rows
        for _i in range(self._num_units):
            col = int(self._rng.integers(0, self._board_size))
            self._units[1].append(
                Unit(player=1, row=self._board_size - 1, col=col)
            )

        logger.debug(
            "TacticalCombat reset: board=%d, units=%d, seed=%s",
            self._board_size,
            self._num_units,
            actual_seed,
        )
        return self.get_observation()

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        """Execute an action: select unit + direction."""
        if self._terminated or self._truncated:
            raise RuntimeError("Cannot step a terminated environment")

        if not (0 <= action < self._total_actions):
            raise ValueError(
                f"Invalid action {action}, must be in [0, {self._total_actions})"
            )

        player = self._current_player
        unit_idx = action // self._ACTIONS_PER_UNIT
        move_idx = action % self._ACTIONS_PER_UNIT

        reward = 0.0
        my_units = [u for u in self._units[player] if u.alive]

        if unit_idx < len(my_units):
            unit = my_units[unit_idx]
            dr, dc = self._DIRECTIONS[move_idx]
            new_row = max(0, min(self._board_size - 1, unit.row + dr))
            new_col = max(0, min(self._board_size - 1, unit.col + dc))
            unit.row = new_row
            unit.col = new_col

            # Check for combat: attack adjacent enemy units
            opponent = 1 - player
            for enemy in self._units[opponent]:
                if not enemy.alive:
                    continue
                dist = abs(enemy.row - new_row) + abs(enemy.col - new_col)
                if dist <= self._attack_range:
                    enemy.take_damage(unit.attack)
                    if not enemy.alive:
                        reward += 0.2  # Reward for eliminating

        # Switch player
        self._current_player = 1 - self._current_player
        self._step_count += 1

        # Check termination
        p0_alive = sum(1 for u in self._units[0] if u.alive)
        p1_alive = sum(1 for u in self._units[1] if u.alive)

        if p0_alive == 0 or p1_alive == 0:
            self._terminated = True
            if p0_alive > 0 and p1_alive == 0:
                reward = 1.0 if player == 0 else -1.0
            elif p1_alive > 0 and p0_alive == 0:
                reward = 1.0 if player == 1 else -1.0
            else:
                reward = 0.0
        elif self._step_count >= self._max_steps:
            self._truncated = True
            # Score based on remaining health
            p0_health = sum(u.health for u in self._units[0] if u.alive)
            p1_health = sum(u.health for u in self._units[1] if u.alive)
            if p0_health > p1_health:
                reward = 0.5 if player == 0 else -0.5
            elif p1_health > p0_health:
                reward = 0.5 if player == 1 else -0.5
            else:
                reward = 0.0

        reward = max(-1.0, min(1.0, reward))

        info: dict[str, Any] = {
            "step": self._step_count,
            "current_player": self._current_player,
            "p0_alive": p0_alive,
            "p1_alive": p1_alive,
        }

        return self.get_observation(), reward, self._terminated, self._truncated, info

    def get_valid_actions(self) -> NDArray[np.bool_]:
        """Valid actions: alive units can perform any of 5 moves."""
        mask = np.zeros(self._total_actions, dtype=np.bool_)
        my_units = [u for u in self._units[self._current_player] if u.alive]
        for i in range(min(len(my_units), self._num_units)):
            start = i * self._ACTIONS_PER_UNIT
            mask[start : start + self._ACTIONS_PER_UNIT] = True
        return mask

    def get_observation(self) -> dict[str, NDArray[np.float32]]:
        """Build observation from current player's perspective."""
        spatial = np.zeros(
            (self._SPATIAL_CHANNELS, self._board_size, self._board_size),
            dtype=np.float32,
        )

        player = self._current_player
        opponent = 1 - player

        # Channel 0-1: current player units
        for unit in self._units[player]:
            if unit.alive:
                spatial[0, unit.row, unit.col] = 1.0
                spatial[1, unit.row, unit.col] = unit.health / unit.max_health

        # Channel 2-3: opponent units
        for unit in self._units[opponent]:
            if unit.alive:
                spatial[2, unit.row, unit.col] = 1.0
                spatial[3, unit.row, unit.col] = unit.health / unit.max_health

        # Channel 4-5: unit counts
        my_alive = sum(1 for u in self._units[player] if u.alive)
        opp_alive = sum(1 for u in self._units[opponent] if u.alive)
        spatial[4, :, :] = my_alive / max(self._num_units, 1)
        spatial[5, :, :] = opp_alive / max(self._num_units, 1)

        # Channel 6: turn progress
        spatial[6, :, :] = self._step_count / max(self._max_steps, 1)

        # Vector features
        my_health = sum(u.health for u in self._units[player] if u.alive)
        opp_health = sum(u.health for u in self._units[opponent] if u.alive)
        max_total = self._num_units * 10

        vector = np.array(
            [
                my_health / max(max_total, 1),
                opp_health / max(max_total, 1),
                my_alive / max(self._num_units, 1),
                opp_alive / max(self._num_units, 1),
                1.0,
            ],
            dtype=np.float32,
        )

        return {"spatial": spatial, "vector": vector}

    def is_terminal(self) -> bool:
        return self._terminated or self._truncated

    def clone(self) -> TacticalCombatEnv:
        """Deep copy the environment."""
        return copy.deepcopy(self)

    def get_symmetries(
        self,
        observation: NDArray[np.float32],
        policy: NDArray[np.float32],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Return identity symmetry (combat has no spatial symmetries)."""
        return [(observation, policy)]

    def __repr__(self) -> str:
        p0_alive = sum(1 for u in self._units[0] if u.alive)
        p1_alive = sum(1 for u in self._units[1] if u.alive)
        return (
            f"TacticalCombatEnv(board={self._board_size}, "
            f"step={self._step_count}, alive=[{p0_alive}, {p1_alive}])"
        )
