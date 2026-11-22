"""GM-B8b: the turn limit must not hand back the win that elimination was denied.

Found while the tests for the elimination rule were being written: once the last visible enemy
was gone, the simulated game ran on to the turn limit and ended there with the full win bonus
("more cities than an opponent with nothing in sight"). The fake win was only delayed. It also
existed before GM-B8b for a state with nobody else in it.

And the counter counted every later step of such a line, not the line.
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv
from alpha_zero.bridge.protocol import GameStateData
from alpha_zero.utils.constants import REWARDS
from tests.fixtures.bridge_search_states import city, make_state, unit
from tests.fixtures.bridge_search_translation import convert

END_TURN = 0
WIN = float(REWARDS.WIN_REWARD)


def _env(state: GameStateData, **kw: Any) -> BridgeMCTSEnv:
    return BridgeMCTSEnv(convert(state, 16), **kw)


def _state(*, enemy_unit: bool, own_city: bool, enemy_city: bool) -> GameStateData:
    units = [unit(2, "UNIT_WARRIOR", 0, 5, 5)]
    if enemy_unit:
        units.append(unit(3, "UNIT_WARRIOR", 1, 6, 5, health=1))
    cities = ([city(8, 0, 2, 2)] if own_city else []) + ([city(9, 1, 12, 12)] if enemy_city else [])
    return make_state(size=16, units=units, cities=cities)


def _attack(env: BridgeMCTSEnv) -> int:
    return int(
        env.env.action_space_handler.encode_action(
            {"type": "attack", "unit_idx": 0, "direction": (1, 0)}
        )
    )


def _play_to_the_limit(env: BridgeMCTSEnv) -> tuple[float, list[float]]:
    """End turns until the game ends; the final reward and every reward on the way."""
    rewards: list[float] = []
    for _ in range(60):
        _, reward, terminated, truncated, _ = env.step(END_TURN)
        rewards.append(float(reward))
        if terminated or truncated:
            return float(reward), rewards
    raise AssertionError("the turn limit never ended the game")


class TestNoWinAtTheLimitOverNothingVisible:
    @pytest.mark.parametrize("own_city", [False, True])
    def test_after_the_last_visible_enemy_is_killed(self, own_city: bool) -> None:
        env = _env(
            _state(enemy_unit=True, own_city=own_city, enemy_city=False),
            max_turns=6,
            min_lookahead=3,
        )
        env.step(_attack(env))
        final, rewards = _play_to_the_limit(env)
        assert env.env.done is True, "the turn limit still ends the game"
        assert abs(final) < WIN, f"a win bonus came back at the limit: {rewards}"
        assert env.env.winner is None

    @pytest.mark.parametrize("own_city", [False, True])
    def test_with_nobody_else_seeded_at_all(self, own_city: bool) -> None:
        env = _env(
            _state(enemy_unit=False, own_city=own_city, enemy_city=False),
            max_turns=6,
            min_lookahead=3,
        )
        final, rewards = _play_to_the_limit(env)
        assert abs(final) < WIN, f"a win bonus at the limit with nobody in sight: {rewards}"

    def test_control_a_visible_enemy_city_still_decides_the_limit(self) -> None:
        """With something of the enemy's in sight the simulator's own outcome stands."""
        env = _env(
            _state(enemy_unit=False, own_city=False, enemy_city=True), max_turns=6, min_lookahead=3
        )
        _play_to_the_limit(env)
        assert env.env.winner == 1

    def test_control_more_cities_than_a_visible_enemy_is_still_a_win(self) -> None:
        state = make_state(
            size=16,
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, 12, 12)],
            cities=[city(8, 0, 2, 2)],
        )
        env = _env(state, max_turns=6, min_lookahead=3)
        _play_to_the_limit(env)
        assert env.env.winner == 0, "an enemy with a unit in sight and fewer cities has lost"


class TestTheCounterCountsLinesNotSteps:
    def test_later_steps_of_the_same_line_do_not_count_again(self) -> None:
        env = _env(_state(enemy_unit=True, own_city=False, enemy_city=False))
        env.step(_attack(env))
        assert env.elimination_suppressed == 1
        for _ in range(8):
            env.step(END_TURN)
        assert env.elimination_suppressed == 1

    def test_two_clones_that_each_kill_count_two(self) -> None:
        env = _env(_state(enemy_unit=True, own_city=False, enemy_city=False))
        a, b = env.clone(), env.clone()
        a.step(_attack(a))
        b.step(_attack(b))
        assert env.elimination_suppressed == 2

    def test_a_clone_made_after_the_kill_does_not_count_again(self) -> None:
        env = _env(_state(enemy_unit=True, own_city=False, enemy_city=False))
        env.step(_attack(env))
        later = env.clone()
        later.step(END_TURN)
        assert env.elimination_suppressed == 1
