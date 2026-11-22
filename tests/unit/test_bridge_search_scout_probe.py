"""GM-B8b zero-game probe: the real search beside a lone visible hostile unit.

Own warrior next to a one-health hostile archer, no enemy city. (A real scout would not do:
the proxy leaves scouts out of the simulator, which makes it "no visible enemy", a case the
old rule already covered.) Without the elimination rule, killing the unit reads as winning the
whole game. An untrained network cannot be asked to prefer anything, so this asserts only what
is robust (the search completes, returns an offered command, and the rule fired), and prints
how often the attack is chosen as a first measurement.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv, Civ6MctsPolicy
from alpha_zero.bridge.protocol import GameStateData, LegalAction
from alpha_zero.core.civ6_mcts import GumbelMCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from tests.fixtures.bridge_search_states import make_state, unit
from tests.fixtures.bridge_search_translation import convert, is_offered

SEEDS = [1, 2, 3, 4, 5, 6]
POSITIONS = [(5, 5), (8, 3), (3, 9)]  # own warrior; the scout is one tile to the east


def _state(own: tuple[int, int], *, with_moves: bool) -> GameStateData:
    x, y = own
    legal = [
        LegalAction("end_turn"),
        LegalAction("skip", unit_id=2),
        LegalAction("attack", unit_id=2, targets=[(x + 1, y)]),
    ]
    if with_moves:
        legal.append(LegalAction("move", unit_id=2, targets=[(x - 1, y), (x, y + 1), (x, y - 1)]))
    state = make_state(
        size=16,
        units=[
            unit(2, "UNIT_WARRIOR", 0, x, y),
            unit(3, "UNIT_ARCHER", 9, x + 1, y, health=1),
        ],
        legal_actions=legal,
    )
    state.observes_others = True
    state.other_players = [{"id": 9, "kind": "barbarian"}]
    return state


@pytest.fixture(scope="module")
def policy() -> Civ6MctsPolicy:
    network = MicroCivNetwork(board_size=16, action_space_size=376)
    config = GumbelMCTSConfig(num_simulations=12, num_sampled_actions=8)
    return Civ6MctsPolicy(network, config, device="cpu", max_turns=30)


def _run(
    policy: Civ6MctsPolicy, state: GameStateData, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    return asyncio.run(policy.select_command(convert(state, 16), decision_seed=seed))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("own", POSITIONS)
def test_the_search_completes_offers_a_command_and_the_rule_fires(
    policy: Civ6MctsPolicy, own: tuple[int, int], seed: int
) -> None:
    # A handful of root actions, all within the sampled set, so the
    # attack is always simulated: the rule is what stood between the search and a fake win.
    state = _state(own, with_moves=False)
    command, record = _run(policy, state, seed)
    assert record["fallback"] is None
    assert is_offered(command, state.legal_actions), command
    assert record["search"]["sim_evals"] > 0
    assert record["others"]["hostile_units"] == 1 and record["others"]["hostile_cities"] == 0
    handler = BridgeMCTSEnv(convert(state, 16)).env.action_space_handler
    attack = handler.encode_action({"type": "attack", "unit_idx": 0, "direction": (1, 0)})
    assert attack in record["root_actions"]
    assert len(record["root_actions"]) <= 8  # all within the sampled set: the attack is simulated
    assert record["elimination_suppressed"] > 0


def test_print_how_often_the_attack_is_chosen(
    policy: Civ6MctsPolicy, capsys: pytest.CaptureFixture[str]
) -> None:
    chosen: dict[bool, list[int]] = {False: [], True: []}
    suppressed: list[int] = []
    for with_moves in (False, True):
        for own in POSITIONS:
            for seed in SEEDS:
                state = _state(own, with_moves=with_moves)
                command, record = _run(policy, state, seed)
                assert is_offered(command, state.legal_actions)
                chosen[with_moves].append(int(command["action_type"] == "unit_attack"))
                suppressed.append(record["elimination_suppressed"])
    with capsys.disabled():
        for with_moves, hits in chosen.items():
            print(
                f"\n[GM-B8b probe] root has moves={with_moves}: attack chosen "
                f"{sum(hits)}/{len(hits)}",
                end="",
            )
        print(
            f"\n[GM-B8b probe] suppressed per search: min {min(suppressed)} max {max(suppressed)}"
        )
    assert len(suppressed) == 2 * len(POSITIONS) * len(SEEDS)
