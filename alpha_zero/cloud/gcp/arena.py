"""
Shared Arena Module for Model Evaluation.

Provides reusable arena logic for running head-to-head matches between two models.
Used by both evaluation_worker.py and post-training tournament scripts.

Usage:
    from alpha_zero.cloud.gcp.arena import ArenaResult, run_arena

    result = run_arena(
        model_a=new_model,
        model_b=old_model,
        num_games=20,
        mcts_type="gumbel",
        mcts_sims=25,
    )
    print(f"Win rate: {result.win_rate:.2%}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class ArenaResult:
    """Result of an arena evaluation between two models."""

    wins: int
    losses: int
    draws: int
    total_games: int

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total_games, 1)

    @property
    def draw_rate(self) -> float:
        return self.draws / max(self.total_games, 1)

    @property
    def loss_rate(self) -> float:
        return self.losses / max(self.total_games, 1)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "win_rate": self.win_rate,
            "draw_rate": self.draw_rate,
            "loss_rate": self.loss_rate,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "total_games": self.total_games,
        }


def run_arena(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    num_games: int,
    mcts_type: str = "gumbel",
    mcts_sims: int = 25,
    gumbel_num_actions: int = 16,
    board_size: int = 8,
    max_turns: int = 30,
    device: torch.device | None = None,
    num_parallel_games: int = 1,
    gameplay_preset: str = "default",
) -> ArenaResult:
    """Play model_a vs. model_b in an arena.

    Alternates starting player each game for fairness. Uses greedy play
    (temperature=0, no noise) for evaluation.

    Args:
        model_a: First model (results are from this model's perspective).
        model_b: Second model (opponent).
        num_games: Number of games to play.
        mcts_type: MCTS algorithm variant ("civ6", "gumbel", or "muzero").
        mcts_sims: Number of MCTS simulations per move.
        gumbel_num_actions: Number of sampled actions for Gumbel MCTS.
        board_size: Size of game board.
        max_turns: Maximum game turns before truncation.
        device: PyTorch device to use. If None, defaults to CPU.
        num_parallel_games: Number of parallel games (accepted for API consistency
            with selfplay_worker; arena currently runs games sequentially).
        gameplay_preset: Environment preset ("default" or "enhanced").

    Returns:
        ArenaResult with wins/losses/draws from model_a's perspective.
    """
    from alpha_zero.core.mcts_factory import create_mcts
    from alpha_zero.envs.env_factory import create_game_env

    if device is None:
        device = torch.device("cpu")

    if num_parallel_games > 1:
        logger.info(
            "Arena: num_parallel_games=%d accepted for API consistency "
            "(arena runs games sequentially for player alternation)",
            num_parallel_games,
        )

    # MuZero has a different search API: (observation, action_mask) vs (env)
    is_muzero = mcts_type == "muzero"
    if is_muzero:
        logger.debug("Arena: using MuZero search path (observation + action_mask)")

    start_time = time.time()
    wins = 0
    losses = 0
    draws = 0

    for game_idx in range(num_games):
        game_start_time = time.time()

        # Alternate which player model_a controls
        a_is_player_0 = game_idx % 2 == 0

        env = create_game_env(
            board_size=board_size,
            max_turns=max_turns,
            gameplay_preset=gameplay_preset,
        )
        mcts_a = create_mcts(
            variant=mcts_type,
            network=model_a,
            device=str(device),
            num_simulations=mcts_sims,
            num_sampled_actions=gumbel_num_actions,
        )
        mcts_b = create_mcts(
            variant=mcts_type,
            network=model_b,
            device=str(device),
            num_simulations=mcts_sims,
            num_sampled_actions=gumbel_num_actions,
        )

        _obs, _info = env.reset()
        done = False
        move_count = 0

        while not done:
            current_is_a = (env.current_player == 0 and a_is_player_0) or (
                env.current_player == 1 and not a_is_player_0
            )
            mcts = mcts_a if current_is_a else mcts_b

            # Greedy play for evaluation (temperature=0, no noise)
            if is_muzero:
                # MuZeroMCTS.search() uses learned dynamics model internally;
                # it requires observation dict + action_mask, not env.
                obs = env.get_observation()
                action_probs, _ = mcts.search(obs, add_noise=False, action_mask=obs["action_mask"])  # type: ignore
            else:
                # Standard MCTS variants (civ6/gumbel) traverse the env directly
                action_probs, _ = mcts.search(env, add_noise=False)
            action = mcts.select_action(action_probs, temperature=0.0)

            _obs, _reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            move_count += 1

        # Determine result from model_a's perspective
        if env.winner is None:
            draws += 1
            result = "draw"
        elif (env.winner == 0 and a_is_player_0) or (env.winner == 1 and not a_is_player_0):
            wins += 1
            result = "win"
        else:
            losses += 1
            result = "loss"

        elapsed = time.time() - game_start_time
        player_str = "P0" if a_is_player_0 else "P1"
        logger.info(
            "Game %d/%d: model_a=%s, result=%s, moves=%d (%.1fs)",
            game_idx + 1,
            num_games,
            player_str,
            result,
            move_count,
            elapsed,
        )

    # Log arena metrics to W&B
    total_elapsed = time.time() - start_time
    try:
        from alpha_zero.utils.experiment_tracker import log_metrics

        log_metrics(
            {
                "arena/total_games": num_games,
                "arena/wins": wins,
                "arena/losses": losses,
                "arena/draws": draws,
                "arena/win_rate": wins / max(num_games, 1),
                "arena/games_per_minute": num_games / max(total_elapsed / 60, 0.001),
            }
        )
    except ImportError:
        logger.debug("experiment tracker not available, skipping arena metrics logging")

    return ArenaResult(
        wins=wins,
        losses=losses,
        draws=draws,
        total_games=num_games,
    )
