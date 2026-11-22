"""The decision seed reaches the simulator too (review of PR #162).

MicroCiv draws by itself when a settler founds a city on its own. The decision
seed reached the search and not the simulator, so two searches of one state
with one seed could still differ once a rollout got that far.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from alpha_zero.bridge import mcts_policy
from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv, Civ6MctsPolicy
from alpha_zero.core.civ6_mcts import GumbelMCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from tests.fixtures.bridge_search_translation import BOARD, convert, rich_state

SIM_ACTIONS = 376


def _policy(seed: int | None = None) -> Civ6MctsPolicy:
    network = MicroCivNetwork(board_size=BOARD, action_space_size=SIM_ACTIONS)
    config = GumbelMCTSConfig(num_simulations=8, num_sampled_actions=4, seed=seed)
    return Civ6MctsPolicy(network, config, device="cpu", max_turns=30)


@pytest.fixture
def simulator_seeds(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    """The seed every simulator built by ``mcts_policy`` was given."""
    seen: list[int | None] = []
    real = mcts_policy.MicroCivEnv

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("seed"))
        return real(*args, **kwargs)

    monkeypatch.setattr(mcts_policy, "MicroCivEnv", spy)
    return seen


class TestSimulatorSeed:
    def test_same_seed_same_simulator_draws(self) -> None:
        state = convert(rich_state())
        first = BridgeMCTSEnv.from_bridge_state(state, seed=5).env._rng.random(8)
        second = BridgeMCTSEnv.from_bridge_state(state, seed=5).env._rng.random(8)
        other = BridgeMCTSEnv.from_bridge_state(state, seed=6).env._rng.random(8)
        assert first.tolist() == second.tolist()
        assert first.tolist() != other.tolist()

    def test_a_clone_carries_the_simulator_s_generator(self) -> None:
        env = BridgeMCTSEnv.from_bridge_state(convert(rich_state()), seed=5)
        assert env.clone().env._rng.random(4).tolist() == env.env._rng.random(4).tolist()

    def test_the_decision_seed_seeds_the_simulator(self, simulator_seeds: list[int | None]) -> None:
        policy = _policy()
        for decision_seed in (21, 21, 22):
            asyncio.run(policy.select_command(convert(rich_state()), decision_seed=decision_seed))
        assert None not in simulator_seeds
        assert simulator_seeds[0] == simulator_seeds[1] != simulator_seeds[2]

    def test_the_configured_seed_is_used_when_no_decision_seed_is_given(
        self, simulator_seeds: list[int | None]
    ) -> None:
        asyncio.run(_policy(seed=9).select_command(convert(rich_state())))
        assert simulator_seeds and simulator_seeds[0] is not None

    def test_the_simulator_s_stream_is_not_the_search_s(
        self, simulator_seeds: list[int | None]
    ) -> None:
        _, record = asyncio.run(_policy().select_command(convert(rich_state()), decision_seed=21))
        assert record["seed"] == 21
        assert simulator_seeds[0] != 21

    def test_without_any_seed_the_simulator_stays_unseeded(
        self, simulator_seeds: list[int | None]
    ) -> None:
        asyncio.run(_policy().select_command(convert(rich_state())))
        assert simulator_seeds == [None]
