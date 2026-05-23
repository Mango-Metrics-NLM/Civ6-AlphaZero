"""Stockfish strength curriculum for chess training.

Three strategies are supported, selectable via :class:`StockfishCurriculumConfig`:

- ``fixed`` — always returns the initial value. Useful for one-off demos
  and reproducible eval matchups.
- ``linear`` — bumps strength by ``step_size`` every ``step_every`` iterations,
  clamped to ``[min_value, max_value]``. Independent of win-rate.
- ``adaptive`` — bumps strength when the agent's rolling win-rate over the
  last ``window_size`` iterations exceeds ``promote_threshold``, and reduces
  it when win-rate drops below ``demote_threshold``. Win-rate is supplied by
  the orchestrator via :meth:`StockfishCurriculum.update`.

The curriculum is intentionally orthogonal to :class:`StockfishOpponent` —
the opponent provides the mechanism (``set_skill`` / ``set_elo``); this
class owns the policy. The orchestrator wires them together.

No values are hardcoded into the orchestrator: every threshold, step, and
window size lives on the dataclass and is configurable via YAML or env.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from alpha_zero.opponents.stockfish_opponent import (
    STOCKFISH_ELO_MAX,
    STOCKFISH_ELO_MIN,
    STOCKFISH_SKILL_MAX,
    STOCKFISH_SKILL_MIN,
)

logger = logging.getLogger(__name__)

CurriculumMode = Literal["fixed", "linear", "adaptive"]
CurriculumTarget = Literal["skill", "elo"]


@dataclass
class StockfishCurriculumConfig:
    """Configuration for :class:`StockfishCurriculum`."""

    #: Which property of the opponent to drive: ``"skill"`` or ``"elo"``.
    target: CurriculumTarget = "skill"
    #: One of ``fixed``, ``linear``, ``adaptive``.
    mode: CurriculumMode = "fixed"
    #: Initial value (skill 0-20 OR elo 1320-3190).
    initial_value: int = 5
    #: Lower and upper clamps. None → use target-appropriate defaults.
    min_value: int | None = None
    max_value: int | None = None
    #: ``linear`` mode only: bump amount per ``step_every`` iterations.
    step_size: int = 1
    #: ``linear`` mode only: bump every N iterations.
    step_every: int = 1
    #: ``adaptive`` mode only: win-rate window size.
    window_size: int = 5
    #: ``adaptive`` mode only: promote skill when rolling win-rate exceeds this.
    promote_threshold: float = 0.6
    #: ``adaptive`` mode only: demote skill when rolling win-rate drops below this.
    demote_threshold: float = 0.3
    #: ``adaptive`` mode only: by how much to step on promote/demote.
    adaptive_step: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "linear", "adaptive"):
            raise ValueError(f"Unknown curriculum mode {self.mode!r}")
        if self.target not in ("skill", "elo"):
            raise ValueError(f"Unknown curriculum target {self.target!r}")
        if not 0.0 <= self.demote_threshold <= self.promote_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= demote_threshold <= promote_threshold <= 1; "
                f"got demote={self.demote_threshold}, promote={self.promote_threshold}"
            )
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.step_every < 1:
            raise ValueError("step_every must be >= 1")
        # Resolve target-appropriate clamps.
        if self.target == "skill":
            self.min_value = self.min_value if self.min_value is not None else STOCKFISH_SKILL_MIN
            self.max_value = self.max_value if self.max_value is not None else STOCKFISH_SKILL_MAX
        else:
            self.min_value = self.min_value if self.min_value is not None else STOCKFISH_ELO_MIN
            self.max_value = self.max_value if self.max_value is not None else STOCKFISH_ELO_MAX
        if not self.min_value <= self.initial_value <= self.max_value:
            raise ValueError(
                f"initial_value {self.initial_value} out of [{self.min_value}, {self.max_value}]"
            )


@dataclass
class StockfishCurriculum:
    """Stateful strength scheduler driven by training iterations and win-rates."""

    config: StockfishCurriculumConfig
    _value: int = field(init=False)
    _iterations: int = field(init=False, default=0)
    _window: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self._value = self.config.initial_value
        self._window = deque(maxlen=self.config.window_size)

    @property
    def current_value(self) -> int:
        return self._value

    @property
    def target(self) -> CurriculumTarget:
        return self.config.target

    def reset(self) -> None:
        """Reset to initial value and clear win-rate history."""
        self._value = self.config.initial_value
        self._iterations = 0
        self._window.clear()

    def update(self, win_rate: float | None = None) -> int:
        """Advance the curriculum by one iteration.

        Args:
            win_rate: Most recent agent win-rate in ``[0, 1]`` against the
                current opponent strength. Ignored when ``mode == "fixed"``
                or ``mode == "linear"``.

        Returns:
            The new strength value (skill or elo).
        """
        self._iterations += 1
        if self.config.mode == "fixed":
            return self._value
        if self.config.mode == "linear":
            if self._iterations % self.config.step_every == 0:
                self._set_value(self._value + self.config.step_size)
            return self._value
        # adaptive
        if win_rate is not None:
            if not 0.0 <= win_rate <= 1.0:
                raise ValueError(f"win_rate {win_rate} must be in [0, 1]")
            self._window.append(win_rate)
        if len(self._window) < self.config.window_size:
            return self._value
        rolling = sum(self._window) / len(self._window)
        if rolling >= self.config.promote_threshold:
            self._set_value(self._value + self.config.adaptive_step)
            self._window.clear()
            logger.info(
                "Curriculum promote: win_rate=%.3f → value=%d",
                rolling,
                self._value,
            )
        elif rolling <= self.config.demote_threshold:
            self._set_value(self._value - self.config.adaptive_step)
            self._window.clear()
            logger.info(
                "Curriculum demote: win_rate=%.3f → value=%d",
                rolling,
                self._value,
            )
        return self._value

    def _set_value(self, candidate: int) -> None:
        assert self.config.min_value is not None
        assert self.config.max_value is not None
        self._value = max(self.config.min_value, min(self.config.max_value, candidate))
