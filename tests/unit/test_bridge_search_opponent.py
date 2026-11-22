"""GM-B8b: the search's simulator never lets the acting player win by elimination.

Real ``BridgeMCTSEnv`` objects built from real converted states. The simulator knows only what
the player can see, so "the enemy has nothing left" means "nothing visible is left"; a win by
elimination before the turn limit is therefore taken back (win reward included), on the env and
on every clone. The root player's own elimination and the turn limit are unchanged.
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
#: The per-kill reward, measured on the control (enemy city alive): the only thing a kill pays.
WIN = float(REWARDS.WIN_REWARD)


def _attack(env: BridgeMCTSEnv, unit_idx: int = 0, direction: tuple[int, int] = (1, 0)) -> int:
    return int(
        env.env.action_space_handler.encode_action(
            {"type": "attack", "unit_idx": unit_idx, "direction": direction}
        )
    )


def _env(state: GameStateData, **kw: Any) -> BridgeMCTSEnv:
    return BridgeMCTSEnv(convert(state, 16), **kw)


def _beside_lone_enemy(enemy_city: bool = False) -> GameStateData:
    """Own warrior (5,5) beside a one-health enemy warrior (6,5): the attack kills it."""
    return make_state(
        size=16,
        units=[unit(2, "UNIT_WARRIOR", 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, 6, 5, health=1)],
        cities=[city(9, 1, 12, 12)] if enemy_city else [],
    )


def _two_enemies(enemy_city: bool = False) -> GameStateData:
    """Two own warriors, each beside its own one-health enemy warrior."""
    return make_state(
        size=16,
        units=[
            unit(2, "UNIT_WARRIOR", 0, 5, 5),
            unit(4, "UNIT_WARRIOR", 0, 5, 9),
            unit(3, "UNIT_WARRIOR", 1, 6, 5, health=1),
            unit(6, "UNIT_WARRIOR", 1, 6, 9, health=1),
        ],
        cities=[city(9, 1, 12, 12)] if enemy_city else [],
    )


class TestKillingTheLastVisibleEnemy:
    def test_the_game_does_not_end_and_there_is_no_win_bonus(self) -> None:
        env = _env(_beside_lone_enemy())
        assert env.proxy_stats["no_visible_enemy"] == 0
        attack = _attack(env)
        assert env.get_valid_actions()[attack] > 0
        _, reward, terminated, truncated, info = env.step(attack)
        assert terminated is False and truncated is False
        assert info["winner"] is None and env.env.winner is None
        assert env.env.done is False
        assert [u.player for u in env.env.units] == [0]  # the enemy really is gone
        assert env.elimination_suppressed == 1
        assert reward < WIN

    def test_the_reward_is_the_per_kill_reward_only(self) -> None:
        # Control: with an enemy city alive the same kill never ended the game.
        control = _env(_beside_lone_enemy(enemy_city=True))
        _, control_reward, control_term, _, _ = control.step(_attack(control))
        assert control_term is False and control.elimination_suppressed == 0
        env = _env(_beside_lone_enemy())
        _, reward, _, _, _ = env.step(_attack(env))
        assert reward == pytest.approx(control_reward)
        assert 0.0 < reward < WIN

    def test_the_same_on_a_clone(self) -> None:
        env = _env(_beside_lone_enemy())
        clone = env.clone()
        _, reward, terminated, _, info = clone.step(_attack(clone))
        assert terminated is False and info["winner"] is None and clone.env.done is False
        assert 0.0 < reward < WIN

    def test_two_plies_deep_in_a_clone_chain(self) -> None:
        env = _env(_two_enemies())
        first = env.clone()
        _, r1, t1, _, _ = first.step(_attack(first, 0))
        assert t1 is False and first.elimination_suppressed == 0  # one enemy still stands
        second = first.clone()
        _, r2, t2, _, info = second.step(_attack(second, 1))  # own unit idx 1: the other warrior
        assert t2 is False and info["winner"] is None and second.env.done is False
        assert second.elimination_suppressed == 1
        assert 0.0 < r2 < WIN and r1 == pytest.approx(r2)

    def test_two_plies_deep_matches_the_control_with_an_enemy_city(self) -> None:
        control = _env(_two_enemies(enemy_city=True))
        control.step(_attack(control, 0))
        _, expected, term, _, _ = control.step(_attack(control, 1))
        assert term is False and control.elimination_suppressed == 0
        env = _env(_two_enemies())
        env.step(_attack(env, 0))
        _, reward, _, _, _ = env.step(_attack(env, 1))
        assert reward == pytest.approx(expected)


class TestOwnEliminationIsStillTheEnd:
    @staticmethod
    def _doomed() -> GameStateData:
        return make_state(
            size=16,
            units=[
                unit(2, "UNIT_WARRIOR", 0, 5, 5, health=1),
                unit(3, "UNIT_WARRIOR", 1, 6, 5),
            ],
        )

    def test_the_enemy_kills_our_last_unit_two_plies_deep(self) -> None:
        env = _env(self._doomed())
        child = env.clone()
        child.step(END_TURN)  # the enemy is to move
        grandchild = child.clone()
        assert grandchild.env.current_player == 1
        _, reward, terminated, truncated, info = grandchild.step(_attack(grandchild, 0, (-1, 0)))
        assert terminated is True and truncated is False
        assert info["winner"] == 1 and grandchild.env.done is True
        assert reward >= WIN  # the mover (the enemy) won; for the root player it is a loss
        assert grandchild.elimination_suppressed == 0

    def test_it_is_not_counted_and_does_not_touch_the_env(self) -> None:
        env = _env(self._doomed())
        clone = env.clone()
        clone.step(END_TURN)
        clone.step(_attack(clone, 0, (-1, 0)))
        assert env.env.done is False and env.elimination_suppressed == 0


class TestTurnLimit:
    def test_the_turn_limit_still_ends_the_game(self) -> None:
        env = _env(_beside_lone_enemy(), max_turns=6, min_lookahead=3)
        env.step(_attack(env))
        ended = None
        for _ in range(40):
            _, _, terminated, truncated, _ = env.step(END_TURN)
            if terminated or truncated:
                ended = (terminated, truncated)
                break
        assert ended == (True, True)
        assert env.env.done is True


class TestCounter:
    def test_it_starts_at_zero(self) -> None:
        assert _env(_beside_lone_enemy()).elimination_suppressed == 0

    def test_one_counter_over_the_env_and_all_its_clones(self) -> None:
        env = _env(_beside_lone_enemy())
        a, b = env.clone(), env.clone()
        a.step(_attack(a))
        assert env.elimination_suppressed == a.elimination_suppressed == 1
        assert b.elimination_suppressed == 1
        b.step(_attack(b))
        assert env.elimination_suppressed == a.elimination_suppressed == 2
        assert a.clone().elimination_suppressed == 2

    def test_a_separately_built_env_has_its_own_counter(self) -> None:
        one, other = _env(_beside_lone_enemy()), _env(_beside_lone_enemy())
        one.step(_attack(one))
        assert one.elimination_suppressed == 1 and other.elimination_suppressed == 0

    def test_nobody_seeded_is_suppressed_the_same_way(self) -> None:
        env = _env(make_state(size=16, units=[unit(2, "UNIT_WARRIOR", 0, 5, 5)]))
        _, reward, terminated, _, info = env.step(END_TURN)
        assert terminated is False and info["winner"] is None and abs(reward) < WIN
        assert env.elimination_suppressed == 1

    def test_an_enemy_with_a_city_never_needs_suppression(self) -> None:
        env = _env(_beside_lone_enemy(enemy_city=True))
        env.step(_attack(env))
        env.step(END_TURN)
        env.clone().step(END_TURN)
        assert env.elimination_suppressed == 0


class TestStaysTwoSided:
    @staticmethod
    def _crowd() -> GameStateData:
        state = make_state(
            size=16,
            units=[
                unit(2, "UNIT_WARRIOR", 0, 5, 5),
                unit(3, "UNIT_WARRIOR", 1, 6, 5, health=1),
                unit(4, "UNIT_ARCHER", 4, 9, 9),
            ],
            cities=[city(9, 4, 12, 12)],
        )
        state.observes_others = True
        state.other_players = [
            {"id": 1, "kind": "major", "at_war": True},
            {"id": 4, "kind": "barbarian"},
        ]
        return state

    def test_units_of_several_hostile_owners_share_the_enemy_side(self) -> None:
        converted = convert(self._crowd(), 16)
        assert converted["_others"]["hostile_units"] == 2
        assert converted["_others"]["hostile_cities"] == 1
        env = BridgeMCTSEnv(converted)
        assert env.env.num_players == 2
        assert sorted(u.player for u in env.env.units) == [0, 1, 1]
        assert [c.player for c in env.env.cities] == [1]

    def test_killing_one_of_two_hostile_owners_units_is_not_the_end(self) -> None:
        env = BridgeMCTSEnv(convert(self._crowd(), 16))
        _, _, terminated, _, _ = env.step(_attack(env))
        assert terminated is False and env.elimination_suppressed == 0
