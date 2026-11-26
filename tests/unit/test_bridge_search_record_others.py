"""GM-B8b: the ``others`` and ``elimination_suppressed`` keys of the decision record."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from alpha_zero.bridge.mcts_policy import Civ6MctsPolicy
from alpha_zero.bridge.protocol import GameStateData, LegalAction
from alpha_zero.core.civ6_mcts import GumbelMCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from tests.fixtures.bridge_search_states import city, make_state, unit
from tests.fixtures.bridge_search_translation import convert

OTHERS_KEYS = {
    "observes_others",
    "hostile_units",
    "hostile_cities",
    "left_out_not_hostile",
    "left_out_unknown",
}


@pytest.fixture(scope="module")
def policy() -> Civ6MctsPolicy:
    network = MicroCivNetwork(board_size=16, action_space_size=376)
    config = GumbelMCTSConfig(num_simulations=8, num_sampled_actions=4)
    return Civ6MctsPolicy(network, config, device="cpu", max_turns=30)


def _state(observes: bool | None) -> GameStateData:
    state = make_state(
        size=16,
        units=[
            unit(2, "UNIT_WARRIOR", 0, 5, 5),
            unit(3, "UNIT_WARRIOR", 1, 12, 12),  # at war: hostile
            unit(4, "UNIT_SCOUT", 5, 14, 2),  # a peaceful minor: left out
            unit(6, "UNIT_SCOUT", 7, 3, 14),  # the mod said nothing: unknown
        ],
        cities=[city(9, 1, 13, 13)],
        legal_actions=[LegalAction("end_turn"), LegalAction("skip", unit_id=2)],
    )
    state.observes_others = observes
    if observes is not None:
        state.other_players = [
            {"id": 1, "kind": "major", "at_war": True},
            {"id": 5, "kind": "minor", "at_war": False},
        ]
    return state


def _run(policy: Civ6MctsPolicy, state: GameStateData) -> dict[str, Any]:
    return asyncio.run(policy.select_command(convert(state, 16), decision_seed=3))[1]


class TestOthersInTheRecord:
    def test_the_counts_are_the_converters_and_plain_ints(self, policy: Civ6MctsPolicy) -> None:
        record = _run(policy, _state(True))
        assert set(record["others"]) == OTHERS_KEYS
        assert record["others"]["hostile_units"] == 1
        assert record["others"]["hostile_cities"] == 1
        assert record["others"]["left_out_not_hostile"] == 1
        assert record["others"]["left_out_unknown"] == 1
        assert record["others"]["observes_others"] == 1
        for key, value in record["others"].items():
            assert type(value) is int, key  # no numpy, no bool

    def test_an_older_mod_state_has_none(self, policy: Civ6MctsPolicy) -> None:
        record = _run(policy, _state(None))
        assert record["others"] is None
        assert "others" in record

    def test_a_state_without_the_key_has_none(self, policy: Civ6MctsPolicy) -> None:
        converted = convert(_state(True), 16)
        del converted["_others"]
        _, record = asyncio.run(policy.select_command(converted, decision_seed=3))
        assert record["others"] is None

    def test_observes_others_false_is_reported_as_zero(self, policy: Civ6MctsPolicy) -> None:
        record = _run(policy, _state(False))
        assert record["others"]["observes_others"] == 0
        assert record["others"]["hostile_units"] == 1  # the entries still classify the owners

    def test_the_record_serialises(self, policy: Civ6MctsPolicy) -> None:
        for observes in (True, False, None):
            record = _run(policy, _state(observes))
            assert json.loads(json.dumps(record))["others"] == record["others"]


class TestSuppressedInTheRecord:
    def test_an_int_present_on_a_search(self, policy: Civ6MctsPolicy) -> None:
        record = _run(policy, _state(True))
        assert record["fallback"] is None
        assert type(record["elimination_suppressed"]) is int
        assert record["elimination_suppressed"] >= 0

    def test_zero_on_a_fallback_before_any_search(self, policy: Civ6MctsPolicy) -> None:
        state = _state(True)
        state.legal_actions = []  # nothing translatable, so no search runs
        record = _run(policy, state)
        assert record["fallback"] is not None
        assert record["elimination_suppressed"] == 0
        assert record["others"] is not None
        json.dumps(record)

    def test_it_is_the_envs_count_after_the_search(self, policy: Civ6MctsPolicy) -> None:
        # A lone own warrior and nobody else: any step the search simulates past the root
        # would end the game by elimination, so the count is positive.
        state = make_state(
            size=16,
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5)],
            legal_actions=[LegalAction("end_turn"), LegalAction("skip", unit_id=2)],
        )
        record = _run(policy, state)
        assert record["elimination_suppressed"] > 0
