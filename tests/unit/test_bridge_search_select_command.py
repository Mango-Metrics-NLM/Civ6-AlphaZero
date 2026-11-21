"""GM-B6a: ``Civ6MctsPolicy.select_command``, un-mocked.

A real ``MicroCivNetwork``, a real ``GumbelMCTS``, the real ``StateConverter`` and the real
translator. The only check on the command is the independent ``is_offered`` in the fixtures.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv, Civ6MctsPolicy
from alpha_zero.bridge.protocol import LegalAction
from alpha_zero.core.civ6_mcts import GumbelMCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from tests.fixtures.bridge_search_states import make_state, unit
from tests.fixtures.bridge_search_translation import BOARD, convert, is_offered, rich_state

SIM_ACTIONS = 376
SIMS = 8
RECORD_FIELDS = {
    "sim_turn",
    "turn_clamped",
    "root_actions",
    "untranslatable",
    "chosen_action",
    "chosen_sim",
    "command",
    "policy_mass",
    "root_value",
    "search",
    "proxy",
    "seed",
    "fallback",
}


@pytest.fixture(scope="module")
def policy() -> Civ6MctsPolicy:
    network = MicroCivNetwork(board_size=BOARD, action_space_size=SIM_ACTIONS)
    config = GumbelMCTSConfig(num_simulations=SIMS, num_sampled_actions=4)
    return Civ6MctsPolicy(network, config, device="cpu", max_turns=30)


def _run(
    policy: Civ6MctsPolicy, state: Any, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    return asyncio.run(policy.select_command(convert(state), **kwargs))


class TestChosenCommand:
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_the_command_is_one_the_game_offered(self, policy: Civ6MctsPolicy, seed: int) -> None:
        state = rich_state()
        command, record = _run(policy, state, decision_seed=seed)
        assert is_offered(command, state.legal_actions), command
        assert record["fallback"] is None

    def test_the_record_describes_the_decision(self, policy: Civ6MctsPolicy) -> None:
        state = rich_state()
        command, record = _run(policy, state, decision_seed=7)
        assert RECORD_FIELDS <= set(record)
        assert record["command"] == command
        assert record["root_actions"] == sorted(record["root_actions"])
        assert record["chosen_action"] in record["root_actions"]
        assert len(record["root_actions"]) > 1
        handler = BridgeMCTSEnv(convert(state)).env.action_space_handler
        decoded = handler.decode_action(record["chosen_action"])
        assert record["chosen_sim"]["type"] == decoded["type"]
        assert 0.0 < record["policy_mass"] <= 1.0
        assert isinstance(record["root_value"], float)
        assert set(record["search"]) >= {"root_evals", "sim_evals"}
        assert record["search"]["sim_evals"] > 0
        assert record["seed"] == 7
        assert record["proxy"]["units"] == 4  # builder 7 left out of five own and enemy units
        assert record["proxy"]["units_not_represented"] == 1
        assert record["sim_turn"] == 3 and record["turn_clamped"] == 0
        assert isinstance(record["untranslatable"], dict)
        assert sum(record["untranslatable"].values()) > 0

    def test_the_record_is_json_serialisable(self, policy: Civ6MctsPolicy) -> None:
        command, record = _run(policy, rich_state(), decision_seed=3)
        json.dumps(record)
        json.dumps(command)
        assert json.loads(json.dumps(record))["command"] == command

    def test_the_chosen_action_is_the_search_s_own_winner(self, policy: Civ6MctsPolicy) -> None:
        """Run the search again on the restricted root and compare with its halving winner.

        The acting search reports a one-hot on the halving winner. The default
        completed-Q target gives mass to root children the search never
        visited, so "the largest mass" could be an unsearched action (found by
        the independent review; this test first pinned the largest mass).
        """
        state = rich_state()
        converted = convert(state)
        _, record = _run(policy, state, decision_seed=11)
        env = BridgeMCTSEnv(converted, max_turns=30)
        mask = np.zeros(SIM_ACTIONS, dtype=np.float32)
        mask[record["root_actions"]] = 1.0
        env.restrict_root(mask)
        search = type(policy.mcts)(
            network=policy.network,
            config=GumbelMCTSConfig(
                num_simulations=SIMS, num_sampled_actions=4, seed=11, policy_target="one_hot"
            ),
            device="cpu",
        )
        probs, _ = search.search(env)
        winners = np.nonzero(probs)[0].tolist()
        assert len(winners) == 1, "premise: a one-hot names exactly the halving winner"
        assert winners[0] in record["root_actions"]
        assert record["chosen_action"] == winners[0]
        assert record["policy_mass"] == 1.0
        assert record["policy_mass"] == pytest.approx(float(probs[record["chosen_action"]]))

    def test_a_live_turn_past_the_horizon_is_still_searched(self, policy: Civ6MctsPolicy) -> None:
        state = rich_state()
        state.turn_id = 900
        command, record = _run(policy, state, decision_seed=2)
        assert record["turn_clamped"] == 1 and record["sim_turn"] == 20
        assert record["fallback"] is None
        assert is_offered(command, state.legal_actions)

    def test_a_state_with_no_visible_enemy_is_searched(self, policy: Civ6MctsPolicy) -> None:
        state = make_state(
            size=16,
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5)],
            legal_actions=[
                LegalAction("end_turn"),
                LegalAction("move", unit_id=2, targets=[(5, 6), (6, 5)]),
                LegalAction("skip", unit_id=2),
            ],
        )
        network = MicroCivNetwork(board_size=16, action_space_size=SIM_ACTIONS)
        p = Civ6MctsPolicy(network, GumbelMCTSConfig(num_simulations=SIMS, num_sampled_actions=4))
        command, record = asyncio.run(p.select_command(convert(state, 16), decision_seed=4))
        assert is_offered(command, state.legal_actions)
        assert record["proxy"]["no_visible_enemy"] == 1
        assert record["fallback"] is None


class TestDecisionSeed:
    def test_same_seed_same_state_same_command_and_record(self, policy: Civ6MctsPolicy) -> None:
        first = _run(policy, rich_state(), decision_seed=21)
        second = _run(policy, rich_state(), decision_seed=21)
        assert first[0] == second[0]
        assert json.dumps(first[1], sort_keys=True) == json.dumps(second[1], sort_keys=True)

    def test_the_seed_does_not_depend_on_what_ran_before(self, policy: Civ6MctsPolicy) -> None:
        first = _run(policy, rich_state(), decision_seed=21)
        _run(policy, rich_state(), decision_seed=99)
        _run(policy, rich_state())
        again = _run(policy, rich_state(), decision_seed=21)
        assert json.dumps(first[1], sort_keys=True) == json.dumps(again[1], sort_keys=True)

    def test_seeds_change_the_search_not_the_offered_set(self, policy: Civ6MctsPolicy) -> None:
        state = rich_state()
        records = [_run(policy, state, decision_seed=s)[1] for s in range(6)]
        assert len({tuple(r["root_actions"]) for r in records}) == 1
        assert {r["seed"] for r in records} == set(range(6))

    def test_without_a_seed_the_policy_still_answers(self, policy: Civ6MctsPolicy) -> None:
        state = rich_state()
        command, record = _run(policy, state)
        assert is_offered(command, state.legal_actions)
        assert record["seed"] is None or isinstance(record["seed"], int)


class TestFallbackToEndTurn:
    def test_nothing_translatable_is_an_end_turn_with_a_reason(
        self, policy: Civ6MctsPolicy
    ) -> None:
        state = rich_state()
        command, record = _run(policy, state, allowed=lambda c: False, decision_seed=1)
        assert command == {"action_type": "end_turn"}
        assert isinstance(record["fallback"], str) and record["fallback"]
        assert record["chosen_action"] is None
        assert record["root_actions"] == []
        assert record["command"] == command
        json.dumps(record)

    def test_nothing_offered_is_an_end_turn_with_a_reason(self, policy: Civ6MctsPolicy) -> None:
        state = make_state(width=84, height=54, units=[unit(2, "UNIT_WARRIOR", 0, 40, 27)])
        state.legal_actions = []
        command, record = _run(policy, state, decision_seed=1)
        assert command == {"action_type": "end_turn"}
        assert isinstance(record["fallback"], str) and record["fallback"]

    def test_allowed_that_keeps_only_end_turn_gives_end_turn(self, policy: Civ6MctsPolicy) -> None:
        state = rich_state()
        seen: list[dict[str, Any]] = []

        def only_end(command: dict[str, Any]) -> bool:
            seen.append(command)
            return command["action_type"] == "end_turn"

        command, record = _run(policy, state, allowed=only_end, decision_seed=1)
        assert command == {"action_type": "end_turn"}
        assert record["root_actions"] == [0]
        assert record["chosen_action"] == 0
        assert record["fallback"] is None
        assert len(seen) > 1, "every candidate was put to the filter"

    def test_a_ruled_out_command_is_never_returned(self, policy: Civ6MctsPolicy) -> None:
        state = rich_state()
        banned = {"unit_skip", "unit_move"}
        for seed in range(4):
            command, _ = _run(
                policy, state, allowed=lambda c: c["action_type"] not in banned, decision_seed=seed
            )
            assert command["action_type"] not in banned


class TestSelectActionUnchanged:
    def test_select_action_still_returns_a_simulator_index(self, policy: Civ6MctsPolicy) -> None:
        state = rich_state()
        converted = convert(state)
        mask = BridgeMCTSEnv(converted, max_turns=30).get_valid_actions()
        index, diagnostics = asyncio.run(policy.select_action({}, mask, converted))
        assert isinstance(index, int) and 0 <= index < SIM_ACTIONS
        assert mask[index] > 0
        assert set(diagnostics) == {"value_estimate", "policy_kind", "mcts_sims"}
        assert diagnostics["mcts_sims"] == float(SIMS)

    def test_select_action_without_a_state_still_raises(self, policy: Civ6MctsPolicy) -> None:
        with pytest.raises(ValueError, match="converted_state"):
            asyncio.run(policy.select_action({}, np.ones(SIM_ACTIONS), None))
