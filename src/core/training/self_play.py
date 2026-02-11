"""Self-play data collection for generating training examples."""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from src.core.config import MCTSConfig
from src.core.constants import (
    DEFAULT_MAX_STEPS,
    DEFAULT_TEMPERATURE_SCHEDULE,
    DIRICHLET_NOISE_STEP_THRESHOLD,
)
from src.core.protocols import StrategicEnvironment
from src.core.types import GameOutcome, GameTrajectory, TrainingExample

logger = logging.getLogger(__name__)

# Type for eval function: (spatial, vector, action_mask) -> (policy, value)
EvalFn = Callable[
    [NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]],
    tuple[NDArray[np.float32], float],
]


class SelfPlayCollector:
    """Collects training data by running self-play games.

    The collector is decoupled from the MCTS engine and environment --
    it accepts them as arguments and orchestrates the game loop.

    Args:
        eval_fn: Neural network evaluation function.
        mcts_config: MCTS configuration.
        max_steps: Maximum steps per game before truncation.
        temperature_schedule: Dict mapping step -> temperature.
            Uses the temperature of the highest key <= current step.
    """

    def __init__(
        self,
        eval_fn: EvalFn,
        mcts_config: MCTSConfig,
        max_steps: int = DEFAULT_MAX_STEPS,
        temperature_schedule: dict[int, float] | None = None,
    ) -> None:
        self._eval_fn = eval_fn
        self._mcts_config = mcts_config
        self._max_steps = max_steps
        self._temperature_schedule = temperature_schedule or dict(DEFAULT_TEMPERATURE_SCHEDULE)

    def update_eval_fn(self, eval_fn: EvalFn) -> None:
        """Replace the evaluation function (e.g. after network weights update)."""
        self._eval_fn = eval_fn

    def _get_temperature(self, step: int) -> float:
        """Get temperature for the given step from the schedule."""
        temp = 1.0
        for threshold, value in sorted(self._temperature_schedule.items()):
            if step >= threshold:
                temp = value
            else:
                break
        return temp

    def play_game(
        self,
        env: StrategicEnvironment,
        add_noise: bool = True,
        seed: int | None = None,
    ) -> GameTrajectory:
        """Play a complete self-play game and collect training examples.

        Args:
            env: Environment instance (will be reset).
            add_noise: Whether to add Dirichlet noise at MCTS root.
            seed: Optional seed for environment reset.

        Returns:
            GameTrajectory with all training examples from the game.
        """
        from src.core.mcts.engine import MCTSEngine, select_action

        logger.debug(
            "Starting self-play game: seed=%s, noise=%s, max_steps=%d",
            seed, add_noise, self._max_steps,
        )
        env.reset(seed=seed)
        engine = MCTSEngine(self._mcts_config, self._eval_fn)

        examples: list[TrainingExample] = []
        step = 0
        outcome = GameOutcome.draw

        while not env.is_terminal() and step < self._max_steps:
            # Run MCTS search
            action_probs, _ = engine.search(
                env,
                add_noise=add_noise and step < DIRICHLET_NOISE_STEP_THRESHOLD,
            )

            # Get temperature and select action
            temperature = self._get_temperature(step)
            action_mask = env.get_valid_actions()
            action = select_action(action_probs, temperature, action_mask)

            # Store training example (value will be filled in later)
            current_obs = env.get_observation()
            examples.append(
                TrainingExample(
                    spatial=current_obs["spatial"].copy(),
                    vector=current_obs["vector"].copy(),
                    action_mask=action_mask.copy(),
                    policy=action_probs.copy(),
                    value=0.0,  # placeholder
                )
            )

            # Take step
            _obs, reward, terminated, truncated, _info = env.step(action)
            step += 1

            if terminated:
                # Assign outcome based on reward
                if reward > 0:
                    outcome = GameOutcome.win
                elif reward < 0:
                    outcome = GameOutcome.loss
                else:
                    outcome = GameOutcome.draw
            elif truncated or step >= self._max_steps:
                outcome = GameOutcome.timeout

        # Backfill values: +1 for winner's moves, -1 for loser's, 0 for draw
        value_map = {
            GameOutcome.win: 1.0,
            GameOutcome.loss: -1.0,
            GameOutcome.draw: 0.0,
            GameOutcome.timeout: 0.0,
        }
        final_value = value_map[outcome]

        # Assign alternating values (negamax convention)
        for i, example in enumerate(examples):
            # Value from perspective of player who made this move
            # Alternates sign for 2-player games
            perspective_value = final_value * ((-1.0) ** (len(examples) - 1 - i))
            example.value = perspective_value

        trajectory = GameTrajectory(
            examples=examples,
            outcome=outcome,
            metadata={"steps": step, "max_steps": self._max_steps},
        )

        logger.info(
            "Game completed: %s in %d steps, %d examples",
            outcome.value,
            step,
            len(examples),
        )

        return trajectory

    def play_games(
        self,
        env_factory: Callable[[], StrategicEnvironment],
        num_games: int,
        seed_offset: int = 0,
    ) -> list[GameTrajectory]:
        """Play multiple self-play games.

        Args:
            env_factory: Callable that creates a new environment instance.
            num_games: Number of games to play.
            seed_offset: Base seed offset for reproducibility.

        Returns:
            List of GameTrajectory instances.
        """
        trajectories = []
        for i in range(num_games):
            env = env_factory()
            trajectory = self.play_game(
                env, seed=seed_offset + i if seed_offset else None
            )
            trajectories.append(trajectory)
        return trajectories
