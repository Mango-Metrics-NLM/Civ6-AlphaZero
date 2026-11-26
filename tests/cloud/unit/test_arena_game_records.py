"""Per-game records on ``ArenaResult`` and playing a subset of a run (LF-D6).

The benchmark needs the outcome of every game, keyed by map pair and seat, to
compute a pair-clustered interval, and it needs to play one shard of a run with
the *same* seeds the full run would use. Both are additive to the deployed
arena: ``ArenaResult`` keeps its fields and ``to_dict`` keeps its keys.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import pytest

from alpha_zero.cloud.gcp.arena import ArenaResult, run_arena
from alpha_zero.evaluation.baseline_agents import GreedyHeuristicAgent, RandomAgent

_SEED = 99


@pytest.fixture(autouse=True)
def _quiet_env() -> Any:
    env_logger = logging.getLogger("alpha_zero.envs.microciv")
    previous = env_logger.level
    env_logger.setLevel(logging.ERROR)
    yield
    env_logger.setLevel(previous)


def _arena(**kwargs: Any) -> ArenaResult:
    return run_arena(
        model_a=None,
        model_b=None,
        num_games=kwargs.pop("num_games", 6),
        board_size=5,
        max_turns=6,
        seed=_SEED,
        player_a_factory=lambda s: GreedyHeuristicAgent(seed=s),
        player_b_factory=lambda s: RandomAgent(seed=s),
        **kwargs,
    )


class _Recording:
    """A player that plays as ``inner`` does and logs every action it returns."""

    def __init__(self, inner: Any, log: list[int]) -> None:
        self._inner = inner
        self._log = log

    def select_action(self, env: Any, observation: Any) -> int:
        action = int(self._inner.select_action(env, observation))
        self._log.append(action)
        return action


class TestGameRecords:
    def test_old_constructor_and_serialisation_unchanged(self) -> None:
        result = ArenaResult(wins=1, losses=2, draws=3, total_games=6)
        assert result.games == []
        assert set(result.to_dict()) == {
            "win_rate",
            "draw_rate",
            "loss_rate",
            "wins",
            "draws",
            "losses",
            "total_games",
            "score_rate",
        }

    def test_one_record_per_game_in_pair_and_seat_order(self) -> None:
        result = _arena()
        assert [(g["pair"], g["seat"]) for g in result.games] == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        ]
        assert {g["result"] for g in result.games} <= {"win", "draw", "loss"}
        outcomes = [g["result"] for g in result.games]
        assert (outcomes.count("win"), outcomes.count("draw"), outcomes.count("loss")) == (
            result.wins,
            result.draws,
            result.losses,
        )

    def test_every_record_carries_the_games_length_and_a_digest_of_its_actions(self) -> None:
        result = _arena()
        assert result.games, "no games were played"
        for game in result.games:
            assert set(game) == {"pair", "seat", "result", "moves", "actions_sha256"}
            assert game["moves"] > 0
            assert re.fullmatch(r"[0-9a-f]{64}", game["actions_sha256"])

    def test_the_digest_is_every_action_of_the_game_in_order(self) -> None:
        """Recomputed here from what the players actually returned, both sides interleaved.

        Negative control: dropping ``actions.update`` from the arena loop, or hashing
        only one side's moves, fails this; so does counting moves any other way.
        """
        logs: list[list[int]] = []

        def player_a(seed: int) -> _Recording:
            logs.append([])  # the arena builds model_a's player first, once per game
            return _Recording(GreedyHeuristicAgent(seed=seed), logs[-1])

        def player_b(seed: int) -> _Recording:
            return _Recording(RandomAgent(seed=seed), logs[-1])

        result = run_arena(
            model_a=None,
            model_b=None,
            num_games=4,
            board_size=5,
            max_turns=6,
            seed=_SEED,
            player_a_factory=player_a,
            player_b_factory=player_b,
        )
        assert len(logs) == len(result.games) == 4
        for game, actions in zip(result.games, logs, strict=True):
            assert game["moves"] == len(actions) > 0
            text = "".join(f"{action}," for action in actions)
            assert game["actions_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert len({game["actions_sha256"] for game in result.games}) > 1, "one digest for all"

    def test_a_seeded_arena_repeats_move_for_move(self) -> None:
        assert _arena().games == _arena().games

    def test_a_subset_replays_those_games_of_the_full_run(self) -> None:
        full = _arena()
        tail = _arena(game_indices=[2, 3, 4, 5])
        head = _arena(game_indices=[0, 1])
        assert head.games + tail.games == full.games
        assert tail.total_games == 4
        assert (head.wins + tail.wins, head.losses + tail.losses) == (full.wins, full.losses)

    def test_rejects_indices_outside_the_run(self) -> None:
        with pytest.raises(ValueError, match="game_indices"):
            _arena(game_indices=[6])
