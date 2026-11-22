"""GM-B6a: what ``BridgeMCTSEnv`` adds so a search from a live state has a usable root.

Root restriction, no-visible-enemy, busy cities and the simulator clock.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv
from alpha_zero.bridge.protocol import LegalAction
from alpha_zero.utils.constants import REWARDS
from tests.fixtures.bridge_search_states import city, make_state, unit
from tests.fixtures.bridge_search_translation import build, convert, rich_state

SIM_END_TURN = 0


def _only(env: BridgeMCTSEnv, *indexes: int) -> np.ndarray:
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    for i in indexes:
        mask[i] = 1.0
    return mask


def _set(mask: np.ndarray) -> set[int]:
    return {int(i) for i in np.nonzero(mask)[0]}


def _skip_index(env: BridgeMCTSEnv, unit_idx: int = 0) -> int:
    return int(env.env.action_space_handler.skip_offset + unit_idx)


def _lone_warrior(turn: int = 3, enemy: bool = False, **kw: Any) -> BridgeMCTSEnv:
    units = [unit(2, "UNIT_WARRIOR", 0, 5, 5)]
    if enemy:
        units.append(unit(3, "UNIT_WARRIOR", 1, 12, 12))
    state = make_state(size=16, units=units, turn_id=turn)
    return BridgeMCTSEnv(convert(state, 16), **kw)


class TestOwnIds:
    def test_live_ids_in_simulator_order_skip_what_the_proxy_leaves_out(self) -> None:
        _, env = build(rich_state())
        assert env.own_unit_ids() == [2, 4, 5]  # builder 7 is not in the simulator
        assert env.own_city_ids() == [10]

    def test_ids_follow_the_current_player_only(self) -> None:
        _, env = build(rich_state())
        assert 3 not in env.own_unit_ids()
        assert 11 not in env.own_city_ids()

    def test_ids_agree_with_the_simulator_s_own_lists(self) -> None:
        _, env = build(rich_state(units_first=(4, 2)))
        sim_units = [u.id for u in env.env.units if u.player == env.current_player]
        sim_cities = [c.id for c in env.env.cities if c.player == env.current_player]
        assert env.own_unit_ids() == sim_units == [4, 2, 5]
        assert env.own_city_ids() == sim_cities

    def test_a_clone_reports_the_same_ids(self) -> None:
        _, env = build(rich_state())
        clone = env.clone()
        assert clone.own_unit_ids() == env.own_unit_ids()
        assert clone.own_city_ids() == env.own_city_ids()


class TestRestrictRoot:
    def test_the_root_mask_is_the_simulator_mask_and_the_restriction(self) -> None:
        _, env = build(rich_state())
        full = env.get_valid_actions().copy()
        skip = _skip_index(env)
        assert full[skip] > 0 and full[SIM_END_TURN] > 0 and len(_set(full)) > 3
        env.restrict_root(_only(env, SIM_END_TURN, skip))
        assert _set(env.get_valid_actions()) == {SIM_END_TURN, skip}
        assert _set(env.get_observation()["action_mask"]) == {SIM_END_TURN, skip}

    def test_the_restriction_cannot_add_an_action_the_simulator_forbids(self) -> None:
        _, env = build(rich_state())
        full = env.get_valid_actions()
        forbidden = int(np.nonzero(full == 0)[0][0])
        env.restrict_root(np.ones(env.action_space.n, dtype=np.float32))
        assert _set(env.get_valid_actions()) == _set(full)
        env2 = build(rich_state())[1]
        env2.restrict_root(_only(env2, SIM_END_TURN, forbidden))
        assert _set(env2.get_valid_actions()) == {SIM_END_TURN}

    def test_a_step_lifts_the_restriction(self) -> None:
        _, env = build(rich_state())
        _, free = build(rich_state())
        env.restrict_root(_only(env, SIM_END_TURN, _skip_index(env)))
        env.step(_skip_index(env))
        free.step(_skip_index(free))
        assert _set(env.get_valid_actions()) == _set(free.get_valid_actions())
        assert len(_set(env.get_valid_actions())) > 2

    def test_a_clone_carries_the_restriction(self) -> None:
        _, env = build(rich_state())
        env.restrict_root(_only(env, SIM_END_TURN))
        clone = env.clone()
        assert _set(clone.get_valid_actions()) == {SIM_END_TURN}
        assert _set(clone.get_observation()["action_mask"]) == {SIM_END_TURN}

    def test_a_step_on_a_clone_lifts_it_for_that_clone_only(self) -> None:
        _, env = build(rich_state())
        skip = _skip_index(env)
        env.restrict_root(_only(env, SIM_END_TURN, skip))
        first, second = env.clone(), env.clone()
        first.step(skip)
        assert len(_set(first.get_valid_actions())) > 2
        assert _set(second.get_valid_actions()) == {SIM_END_TURN, skip}
        assert _set(env.get_valid_actions()) == {SIM_END_TURN, skip}
        assert _set(env.clone().get_valid_actions()) == {SIM_END_TURN, skip}

    def test_a_clone_of_a_stepped_clone_is_unrestricted(self) -> None:
        _, env = build(rich_state())
        env.restrict_root(_only(env, SIM_END_TURN, _skip_index(env)))
        child = env.clone()
        child.step(_skip_index(env))
        grandchild = child.clone()
        assert _set(grandchild.get_valid_actions()) == _set(child.get_valid_actions())
        assert len(_set(grandchild.get_valid_actions())) > 2

    def test_no_restriction_leaves_the_mask_alone(self) -> None:
        _, env = build(rich_state())
        assert _set(env.get_observation()["action_mask"]) == _set(env.get_valid_actions())
        assert len(_set(env.get_valid_actions())) > 3


class TestNoVisibleEnemy:
    def test_stepping_does_not_end_the_game_for_want_of_an_enemy(self) -> None:
        env = _lone_warrior(enemy=False)
        _, reward, terminated, truncated, info = env.step(SIM_END_TURN)
        assert terminated is False and truncated is False
        assert info["winner"] is None
        assert env.env.done is False
        assert env.env.winner is None
        assert abs(reward) < REWARDS.WIN_REWARD  # no win bonus

    def test_a_search_can_keep_going_for_several_turns(self) -> None:
        env = _lone_warrior(enemy=False)
        for _ in range(6):
            _, _, terminated, _, _ = env.step(SIM_END_TURN)
            assert terminated is False
        assert env.env.done is False

    def test_children_of_a_clone_are_not_terminal_either(self) -> None:
        env = _lone_warrior(enemy=False)
        _, reward, terminated, _, _ = env.clone().step(SIM_END_TURN)
        assert terminated is False and abs(reward) < REWARDS.WIN_REWARD
        # and a clone of the stepped clone, which is what a deeper simulation does
        child = env.clone()
        child.step(SIM_END_TURN)
        _, _, terminated, _, _ = child.clone().step(SIM_END_TURN)
        assert terminated is False

    def test_the_turn_limit_still_ends_it(self) -> None:
        # live turn 3 < max_turns - lookahead = 4: sim turn 3; the turn counter advances
        # every second end_turn and ends the game when it reaches max_turns = 5.
        env = _lone_warrior(turn=3, enemy=False, max_turns=5, min_lookahead=1)
        assert env.turn == 3
        results = [env.step(SIM_END_TURN) for _ in range(4)]
        assert [r[2] for r in results] == [False, False, False, True]
        assert results[-1][3] is True  # truncated
        assert env.env.done is True

    def test_the_flag_is_reported(self) -> None:
        assert _lone_warrior(enemy=False).proxy_stats["no_visible_enemy"] == 1
        assert _lone_warrior(enemy=True).proxy_stats["no_visible_enemy"] == 0

    def test_an_enemy_city_alone_counts_as_visible(self) -> None:
        state = make_state(
            size=16, units=[unit(2, "UNIT_WARRIOR", 0, 5, 5)], cities=[city(9, 1, 12, 12)]
        )
        env = BridgeMCTSEnv(convert(state, 16))
        assert env.proxy_stats["no_visible_enemy"] == 0
        _, _, terminated, _, _ = env.step(SIM_END_TURN)
        assert terminated is False  # one city each side survives; nothing was eliminated

    def test_an_eliminated_visible_enemy_still_terminates(self) -> None:
        # Enemy warrior with 1 health next to ours: the attack kills it and the game is won.
        state = make_state(
            size=16,
            units=[
                unit(2, "UNIT_WARRIOR", 0, 5, 5),
                unit(3, "UNIT_WARRIOR", 1, 6, 5, health=1),
            ],
        )
        env = BridgeMCTSEnv(convert(state, 16))
        assert env.proxy_stats["no_visible_enemy"] == 0
        attack = env.env.action_space_handler.encode_action(
            {"type": "attack", "unit_idx": 0, "direction": (1, 0)}
        )
        assert env.get_valid_actions()[attack] > 0
        _, reward, terminated, _, info = env.step(attack)
        assert terminated is True
        assert info["winner"] == 0
        assert reward >= REWARDS.WIN_REWARD
        assert env.env.done is True

    def test_a_clone_keeps_the_flag_and_the_behaviour(self) -> None:
        clone = _lone_warrior(enemy=False).clone()
        assert clone.proxy_stats["no_visible_enemy"] == 1
        assert clone.step(SIM_END_TURN)[2] is False

    def test_a_visible_enemy_that_is_eliminated_in_a_clone_still_terminates(self) -> None:
        state = make_state(
            size=16,
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, 6, 5, health=1)],
        )
        env = BridgeMCTSEnv(convert(state, 16))
        attack = env.env.action_space_handler.encode_action(
            {"type": "attack", "unit_idx": 0, "direction": (1, 0)}
        )
        assert env.clone().step(attack)[2] is True


class TestBusyCity:
    @staticmethod
    def _env(producing: Any) -> BridgeMCTSEnv:
        own = city(10, 0, 5, 5)
        own.current_production = producing
        state = make_state(
            size=16,
            units=[unit(3, "UNIT_WARRIOR", 1, 12, 12)],
            cities=[own],
            legal_actions=[LegalAction("end_turn")],
        )
        return BridgeMCTSEnv(convert(state, 16))

    @staticmethod
    def _produce_actions(env: BridgeMCTSEnv) -> set[int]:
        handler = env.env.action_space_handler
        return {i for i in _set(env.get_valid_actions()) if i >= handler.produce_offset}

    def test_a_city_with_production_is_busy(self) -> None:
        env = self._env("UNIT_WARRIOR")
        assert self._produce_actions(env) == set()
        assert env.proxy_stats["busy_cities"] == 1

    @pytest.mark.parametrize("idle", [None, ""])
    def test_a_city_with_none_or_empty_production_is_idle(self, idle: Any) -> None:
        env = self._env(idle)
        assert len(self._produce_actions(env)) == 3  # warrior, settler, archer
        assert env.proxy_stats["busy_cities"] == 0

    def test_only_the_busy_city_is_busy(self) -> None:
        busy, idle = city(10, 0, 3, 3), city(11, 0, 9, 9)
        busy.current_production = "BUILDING_MONUMENT"
        state = make_state(size=16, units=[unit(3, "UNIT_WARRIOR", 1, 14, 14)], cities=[busy, idle])
        env = BridgeMCTSEnv(convert(state, 16))
        handler = env.env.action_space_handler
        produce = sorted(i for i in _set(env.get_valid_actions()) if i >= handler.produce_offset)
        # City index 0 (id 10) is busy; only index 1 (id 11) offers its three units.
        first = handler.produce_offset + 1 * handler.num_unit_types
        assert produce == [first, first + 1, first + 2]
        assert env.proxy_stats["busy_cities"] == 1
        assert env.own_city_ids() == [10, 11]

    def test_busy_cities_survive_a_clone(self) -> None:
        env = self._env("UNIT_WARRIOR")
        assert self._produce_actions(env.clone()) == set()
        assert env.clone().proxy_stats["busy_cities"] == 1


class TestClock:
    @staticmethod
    def _env(live_turn: int, max_turns: int = 30, **kw: Any) -> BridgeMCTSEnv:
        state = make_state(
            size=16,
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, 12, 12)],
            turn_id=live_turn,
        )
        return BridgeMCTSEnv(convert(state, 16), max_turns=max_turns, **kw)

    @pytest.mark.parametrize(
        ("live", "sim", "clamped"),
        [
            (0, 0, 0),
            (5, 5, 0),
            (19, 19, 0),
            (20, 20, 0),  # max_turns - min_lookahead: the boundary is not clamped
            (21, 20, 1),
            (29, 20, 1),
            (30, 20, 1),
            (400, 20, 1),
        ],
    )
    def test_default_lookahead_is_ten_turns(self, live: int, sim: int, clamped: int) -> None:
        env = self._env(live)
        assert env.turn == sim
        assert env.env.turn == sim
        assert env.proxy_stats["sim_turn"] == sim
        assert env.proxy_stats["turn_clamped"] == clamped

    def test_a_live_turn_past_the_horizon_is_a_playable_root(self) -> None:
        env = self._env(400)
        assert env.env.done is False
        _, _, terminated, truncated, _ = env.step(_skip_index(env))
        assert terminated is False and truncated is False

    def test_the_search_has_turns_ahead(self) -> None:
        env = self._env(400)
        for _ in range(2 * 5):  # five full turns of end_turn from both sides
            _, _, terminated, _, _ = env.step(SIM_END_TURN)
            assert terminated is False
        assert env.turn == 25

    def test_the_limit_is_reached_after_the_lookahead(self) -> None:
        env = self._env(400)  # sim turn 20; the 30-turn limit is ten of our turns away
        ended_at = None
        for step in range(1, 40):
            if env.step(SIM_END_TURN)[2]:
                ended_at = step
                break
        assert ended_at == 20  # ten rounds of two end_turns

    @pytest.mark.parametrize(
        ("lookahead", "live", "sim"),
        [(0, 25, 25), (5, 25, 25), (5, 26, 25), (25, 25, 5), (30, 25, 0), (99, 25, 0)],
    )
    def test_min_lookahead_is_capped_at_max_turns(
        self, lookahead: int, live: int, sim: int
    ) -> None:
        env = self._env(live, min_lookahead=lookahead)
        assert env.turn == sim
        assert env.proxy_stats["sim_turn"] == sim
        assert env.proxy_stats["turn_clamped"] == int(live > sim)

    def test_never_below_zero(self) -> None:
        env = self._env(7, max_turns=5, min_lookahead=9)
        assert env.turn == 0
        assert env.proxy_stats["sim_turn"] == 0
        assert env.proxy_stats["turn_clamped"] == 1

    def test_a_clone_keeps_the_clock_stats(self) -> None:
        clone = self._env(400).clone()
        assert clone.proxy_stats["sim_turn"] == 20
        assert clone.proxy_stats["turn_clamped"] == 1
        assert clone.turn == 20

    def test_the_new_stats_sit_beside_the_old_ones(self) -> None:
        stats = self._env(5).proxy_stats
        assert {"units", "units_not_represented", "impassable_tiles"} <= set(stats)
        assert {"no_visible_enemy", "busy_cities", "sim_turn", "turn_clamped"} <= set(stats)
