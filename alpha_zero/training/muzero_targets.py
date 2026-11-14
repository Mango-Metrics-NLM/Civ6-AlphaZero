"""MuZero training targets for two-player self-play, in the frames the search reads.

One definition for ``scripts/run_muzero_training.py`` and the cloud
``run_muzero_selfplay`` (2026-11-12, maintainer-approved: "outcome-only, board
game style"). Before it the two disagreed with each other and with the search:
the script labelled values from **player 0's** perspective and summed the env's
shaped rewards across player changes, while the cloud path stored mover-relative
outcomes beside unsigned shaped reward targets.

The frames match :mod:`alpha_zero.core.mcts.muzero_tree`:

* **Value** of position ``t``: the final outcome from the perspective of the
  player to move at ``t`` -- what the cloud path already stored as ``rewards``.
* **Reward** of transition ``t``: zero, except the final transition, which
  carries the outcome from the perspective of the player to move *after* it.
  MicroCiv's shaped rewards are dropped on purpose: ``CITY_FOUND_REWARD`` is
  ``+1.0``, as large as a win, so a search computing ``reward + discount *
  value`` would rank founding a city with winning the game, and adding them to
  an outcome value target would count them twice.
* The absorbing state after the game is worth ``0``; callers that store one
  (``run_muzero_training`` keeps ``T + 1`` values) append it themselves.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["outcome_for", "outcome_only_targets"]


def outcome_for(winner: int | None, player: int) -> float:
    """``+1`` if ``player`` won, ``-1`` if another player did, ``0`` for a draw.

    The same rule as ``cloud.gcp.selfplay_worker.outcome_value_for_player``
    (a test holds them equal); duplicated so ``training`` need not import
    ``cloud``.
    """
    if winner is None:
        return 0.0
    return 1.0 if int(winner) == int(player) else -1.0


def outcome_only_targets(
    winner: int | None,
    players: Sequence[int],
    final_to_play: int,
) -> tuple[list[float], list[float]]:
    """Reward and value targets for one finished game.

    Args:
        winner: Winning player, or ``None`` for a draw.
        players: The player to move at each recorded position, in order.
        final_to_play: The player to move after the last action
            (``env.current_player`` once the game is over).

    Returns:
        ``(reward_targets, value_targets)``, each one entry per position.
    """
    rewards = [0.0] * len(players)
    if rewards:
        rewards[-1] = outcome_for(winner, final_to_play)
    values = [outcome_for(winner, player) for player in players]
    return rewards, values
