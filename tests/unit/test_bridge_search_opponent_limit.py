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

    def test_more_cities_than_an_enemy_whose_cities_are_not_in_sight_is_no_win(self) -> None:
        """Review (`sdlc-ml-scientist`): one enemy unit in sight and no enemy city is the common
        case under fog (the capital is out of sight), and "1 city to 0" was still a full win."""
        state = make_state(
            size=16,
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, 12, 12)],
            cities=[city(8, 0, 2, 2)],
        )
        env = _env(state, max_turns=6, min_lookahead=3)
        final, rewards = _play_to_the_limit(env)
        assert env.env.winner is None
        assert abs(final) < WIN, rewards

    def test_control_more_cities_than_an_enemy_city_in_sight_is_still_a_win(self) -> None:
        state = make_state(
            size=16,
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5)],
            cities=[city(8, 0, 2, 2), city(7, 0, 4, 9), city(9, 1, 12, 12)],
        )
        env = _env(state, max_turns=6, min_lookahead=3)
        _play_to_the_limit(env)
        assert env.env.winner == 0

    def test_the_returned_info_agrees_with_the_simulator(self) -> None:
        """Review: the limit branch cleared the winner but returned the info read before."""
        env = _env(
            _state(enemy_unit=True, own_city=True, enemy_city=False), max_turns=6, min_lookahead=3
        )
        env.step(_attack(env))
        info: dict[str, Any] = {}
        for _ in range(60):
            _, _, terminated, truncated, info = env.step(END_TURN)
            if terminated or truncated:
                break
        assert info["winner"] is None
        assert env.get_info()["winner"] is None


class TestLosingEverythingIsALoss:
    def test_a_last_unit_that_takes_the_last_visible_enemy_with_it_has_lost(self) -> None:
        """Review: the simulator calls a mutual kill of both last pieces a draw; for the
        player, whose pieces are all known, it is the end."""
        state = make_state(
            size=16,
            units=[
                unit(2, "UNIT_WARRIOR", 0, 5, 5, health=1),
                unit(3, "UNIT_WARRIOR", 1, 6, 5, health=1),
            ],
        )
        env = _env(state)
        _, _, terminated, _, info = env.step(_attack(env))
        if any(u.player == 0 for u in env.env.units):
            pytest.fail("the scenario did not kill the attacker: adjust the healths")
        assert terminated is True
        assert env.env.winner == 1
        assert info["winner"] == 1
        assert env.elimination_suppressed == 0


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


def _mutual(extra_own: list[Any] | None = None) -> GameStateData:
    """Own and enemy one-health warriors side by side: an attack kills both."""
    return make_state(
        size=16,
        units=[
            unit(2, "UNIT_WARRIOR", 0, 5, 5, health=1),
            unit(3, "UNIT_WARRIOR", 1, 6, 5, health=1),
            *(extra_own or []),
        ],
    )


def _enemy_attack(env: BridgeMCTSEnv) -> int:
    return int(
        env.env.action_space_handler.encode_action(
            {"type": "attack", "unit_idx": 0, "direction": (-1, 0)}
        )
    )


class TestTheLossIsPaid:
    """Review (Copilot, PR #165): a mutual kill was declared a loss but paid nothing."""

    def test_the_acting_player_s_own_attack(self) -> None:
        env = _env(_mutual())
        _, reward, terminated, _, _ = env.step(_attack(env))
        assert not any(u.player == 0 for u in env.env.units), "the attacker survived"
        assert terminated is True and env.env.winner == 1
        # The acting player is still to move after its own attack: it is paid the loss.
        assert env.env.current_player == 0
        assert reward <= -WIN / 2

    def test_the_enemy_s_attack(self) -> None:
        env = _env(_mutual())
        env.step(END_TURN)
        assert env.env.current_player == 1
        _, reward, terminated, _, _ = env.step(_enemy_attack(env))
        if any(u.player == 0 for u in env.env.units) or any(u.player == 1 for u in env.env.units):
            pytest.fail("the enemy attack did not kill both units: adjust the scenario")
        assert terminated is True and env.env.winner == 1
        # The enemy is to move after its own attack: it is paid the win.
        assert env.env.current_player == 1
        assert reward >= WIN / 2

    def test_control_a_loss_the_simulator_already_paid_is_not_paid_twice(self) -> None:
        """The enemy kills the acting player's last unit and survives: the simulator's own
        loss, untouched."""
        state = make_state(
            size=16,
            units=[
                unit(2, "UNIT_WARRIOR", 0, 5, 5, health=1),
                unit(3, "UNIT_WARRIOR", 1, 6, 5),
            ],
        )
        env = _env(state)
        env.step(END_TURN)
        _, reward, terminated, _, _ = env.step(_enemy_attack(env))
        if any(u.player == 0 for u in env.env.units):
            pytest.fail("the enemy attack did not kill the unit: adjust the scenario")
        assert terminated is True and env.env.winner == 1
        assert reward < 2 * WIN


class TestUnitsTheSimulatorLeftOutAreStillAlive:
    """Review (Copilot, PR #165): an own scout has no stand-in in the simulator, so losing
    every own piece IN the simulator is not losing the game while the scout lives."""

    def test_a_mutual_kill_with_an_own_scout_alive_is_not_a_loss(self) -> None:
        env = _env(_mutual([unit(8, "UNIT_SCOUT", 0, 1, 1)]))
        assert env.proxy_stats["own_units_not_represented"] == 1
        _, reward, terminated, _, info = env.step(_attack(env))
        assert not any(u.player == 0 for u in env.env.units), "the attacker survived"
        assert terminated is False
        assert env.env.winner is None and info["winner"] is None
        assert abs(reward) < WIN

    def test_the_enemy_killing_the_last_simulated_unit_is_not_a_loss_either(self) -> None:
        state = make_state(
            size=16,
            units=[
                unit(2, "UNIT_WARRIOR", 0, 5, 5, health=1),
                unit(8, "UNIT_SCOUT", 0, 1, 1),
                unit(3, "UNIT_WARRIOR", 1, 6, 5),
            ],
        )
        env = _env(state)
        env.step(END_TURN)
        _, _, terminated, _, _ = env.step(_enemy_attack(env))
        if any(u.player == 0 for u in env.env.units):
            pytest.fail("the enemy attack did not kill the unit: adjust the scenario")
        assert terminated is False and env.env.winner is None

    def test_a_clone_keeps_the_count(self) -> None:
        env = _env(_mutual([unit(8, "UNIT_SCOUT", 0, 1, 1)])).clone()
        _, _, terminated, _, _ = env.step(_attack(env))
        assert terminated is False

    def test_an_enemy_unit_left_out_does_not_count_as_own(self) -> None:
        env = _env(_mutual([unit(8, "UNIT_SCOUT", 1, 1, 1)]))
        assert env.proxy_stats["own_units_not_represented"] == 0
        _, _, terminated, _, _ = env.step(_attack(env))
        assert terminated is True and env.env.winner == 1


class TestTheLimitKeepsWhatIsKnown:
    """Review round two (Copilot, PR #165): with only an own scout left out, no own city and
    an enemy city in sight, the turn limit's 0-1 count was cleared to a draw. The left-out
    unit prevents a false elimination; it does not erase a known count."""

    def _state(self, enemy_city: bool) -> GameStateData:
        return make_state(
            size=16,
            units=[unit(8, "UNIT_SCOUT", 0, 1, 1)],
            cities=[city(9, 1, 12, 12)] if enemy_city else [],
        )

    def test_a_visible_enemy_city_still_decides_the_limit(self) -> None:
        env = _env(self._state(enemy_city=True), max_turns=6, min_lookahead=3)
        assert env.proxy_stats["own_units_not_represented"] == 1
        _play_to_the_limit(env)
        assert env.env.done is True
        assert env.env.winner == 1

    def test_control_without_an_enemy_city_nothing_is_decided(self) -> None:
        env = _env(self._state(enemy_city=False), max_turns=6, min_lookahead=3)
        final, _ = _play_to_the_limit(env)
        assert env.env.winner is None
        assert abs(final) < WIN


def _three_sided(own_health: int) -> BridgeMCTSEnv:
    """The acting player's one warrior beside an enemy; a third side elsewhere."""
    converted = convert(
        make_state(
            size=16,
            units=[
                unit(2, "UNIT_WARRIOR", 0, 5, 5, health=own_health),
                unit(3, "UNIT_WARRIOR", 1, 6, 5),
                unit(4, "UNIT_WARRIOR", 1, 12, 12),
            ],
        ),
        16,
    )
    third = next(u for u in converted["units"] if u["_id"] == 4)
    third["player"] = 2
    return BridgeMCTSEnv(converted)


class TestMoreThanTwoSides:
    """Review round two (Copilot, PR #165): with three sides the simulator plays on among
    the others after the acting player's last piece is gone, so the search valued a lost
    position with the network instead of seeing the loss."""

    def test_the_simulator_has_three_sides(self) -> None:
        env = _three_sided(own_health=100)
        assert env.env.num_players == 3

    def test_losing_the_last_piece_ends_the_game_as_a_loss(self) -> None:
        env = _three_sided(own_health=1)
        env.step(END_TURN)
        assert env.env.current_player == 1
        _, reward, terminated, _, info = env.step(_enemy_attack(env))
        assert not any(u.player == 0 for u in env.env.units), "the unit survived the attack"
        assert terminated is True and env.env.done is True
        assert env.env.winner not in (None, 0)
        assert info["winner"] == env.env.winner
        assert reward >= WIN / 2, "the enemy, to move after its attack, is paid the win"

    def test_control_while_a_piece_is_left_the_game_goes_on(self) -> None:
        env = _three_sided(own_health=100)
        env.step(END_TURN)
        _, _, terminated, _, _ = env.step(_enemy_attack(env))
        assert any(u.player == 0 for u in env.env.units)
        assert terminated is False and env.env.winner is None

    def test_a_three_sided_state_seeded_with_nothing_of_the_player_s_is_not_ended_at_once(
        self,
    ) -> None:
        """The rule ends the game when the player LOSES its last piece, not when it never
        had one in the simulator (then there is nothing to lose there)."""
        converted = convert(
            make_state(
                size=16,
                units=[unit(3, "UNIT_WARRIOR", 1, 6, 5), unit(4, "UNIT_WARRIOR", 1, 12, 12)],
            ),
            16,
        )
        next(u for u in converted["units"] if u["_id"] == 4)["player"] = 2
        env = BridgeMCTSEnv(converted)
        _, _, terminated, _, _ = env.step(END_TURN)
        assert terminated is False
