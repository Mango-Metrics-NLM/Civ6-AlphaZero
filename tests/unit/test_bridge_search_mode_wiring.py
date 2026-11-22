"""What the live bridge's search mode (``--mcts``) is today (GM-B5, Part A).

These are characterisation tests: they pin, by measurement, the shapes that cannot meet
when ``Civ6MctsPolicy`` is wired as ``run_civ6_bridge`` wires it, and what that does to a
running game. They pass before the refusal lands and after it, because the refusal stops
``run_civ6_bridge`` from building the policy and does not change the policy itself.

Nothing here is mocked: the converters, the networks, the search and the agent loop are the
real ones. The only fake is the transport, which records what the loop sends.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from alpha_zero import civ6_local
from alpha_zero.bridge.action_converter import ActionConverter
from alpha_zero.bridge.agent_loop import AgentLoopConfig, BridgeAgentLoop
from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv, Civ6MctsPolicy
from alpha_zero.bridge.protocol import AckMessage, LegalAction
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_actions import ActionCategory, ActionSpaceConfig
from alpha_zero.core.civ6_mcts import GumbelMCTS, GumbelMCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.envs.civ6_encoder import EncoderConfig
from tests.fixtures.bridge_search_states import make_state, unit

#: Searches here are tiny on purpose; the point is the shapes, not the play.
SIMS = 4
#: The simulator's action space (``MicroCivEnv.action_space``) as measured: 376.
SIM_ACTIONS = 376
#: Own warrior and the move the game offers it: (5, 5) -> (5, 6).
OWN_X, OWN_Y = 5, 5


class _Live:
    """The pieces ``run_civ6_bridge`` builds for a search-mode config, un-mocked."""

    def __init__(self) -> None:
        self.cfg = civ6_local.Civ6LocalConfig(
            mode="civ6", use_mcts=True, num_simulations=SIMS, device="cpu"
        )
        self.state_converter = StateConverter(
            board_size=self.cfg.board_size, research_registry=False
        )
        self.action_converter = ActionConverter(
            board_size=self.cfg.board_size,
            action_space_config=ActionSpaceConfig(
                board_size=self.cfg.board_size,
                max_units=20,
                max_cities=5,
                num_production_items=20,
            ),
            research_registry=False,
        )
        self.network = civ6_local._build_network_for_bridge(
            self.cfg, action_space_size=self.action_converter.action_space_size
        )
        self.policy = civ6_local._build_bridge_policy(self.cfg, self.network)


@pytest.fixture(scope="module")
def live() -> _Live:
    return _Live()


def _state() -> Any:
    """One own warrior, one far enemy warrior, a move and an end of turn on offer."""
    return make_state(
        player_id=0,
        size=16,
        units=[
            unit(2, "UNIT_WARRIOR", 0, OWN_X, OWN_Y),
            unit(3, "UNIT_WARRIOR", 1, 14, 14),
        ],
        legal_actions=[
            LegalAction(action_type="end_turn", unit_id=0, city_id=0),
            LegalAction(action_type="move", unit_id=2, city_id=0, targets=[(OWN_X, OWN_Y + 1)]),
        ],
    )


class _RecordingBridge:
    """Hands the loop one state and records every command it sends, by route."""

    def __init__(self, state: Any) -> None:
        self._state = state
        self._served = False
        self._msg_id = 100
        self.acked: list[Any] = []  # commands sent through send_and_wait_with_retry
        self.fallback: list[Any] = []  # commands sent through send_command

    async def wait_for_state(self, timeout: float, require_new: bool = True) -> Any:
        assert not self._served, "the loop asked for a second state in a one-turn run"
        self._served = True
        return self._state

    def next_msg_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def send_and_wait_with_retry(self, command: Any) -> AckMessage:
        self.acked.append(command)
        return AckMessage(msg_id=command.msg_id, success=True)

    async def send_command(self, command: Any) -> None:
        self.fallback.append(command)

    def check_heartbeat(self) -> bool:
        return True


class TestPolicyAsWired:
    """A1 and A2: the search policy as the live mode builds it fails on every turn."""

    def test_policy_is_the_search_policy(self, live: _Live) -> None:
        assert isinstance(live.policy, Civ6MctsPolicy)

    def test_select_action_raises_on_channel_mismatch(self, live: _Live) -> None:
        state = _state()
        converted = live.state_converter.convert(state)
        mask = live.action_converter.generate_action_mask(state.legal_actions, converted)
        spatial, vector = live.state_converter.convert_and_encode(state)
        obs = {"spatial": spatial.numpy(), "vector": vector.numpy(), "action_mask": mask}

        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(live.policy.select_action(obs, mask, converted))

        message = str(excinfo.value)
        assert "84 channels" in message
        assert "13 channels" in message

    def test_loop_sends_only_an_end_turn(self, live: _Live) -> None:
        state = _state()
        converted = live.state_converter.convert(state)
        mask = live.action_converter.generate_action_mask(state.legal_actions, converted)
        # The turn really does offer something besides ending it, so ending it is a loss.
        assert int(mask.sum()) >= 2

        bridge = _RecordingBridge(state)
        loop = BridgeAgentLoop(
            bridge=bridge,  # type: ignore[arg-type]
            policy=live.policy,
            state_converter=live.state_converter,
            action_converter=live.action_converter,
            config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=2.0),
        )
        rc = asyncio.run(loop.run())

        assert rc == loop.EXIT_OK
        # Since GM-B6b the loop asks this policy for a command; the search still
        # fails on the full-game network, which is now counted, and the turn is
        # ended through the ordinary send path instead of the blind fallback.
        sent = bridge.acked + bridge.fallback
        assert len(sent) == 1
        assert [c.action_type for c in sent[0].commands] == ["end_turn"]
        stats = loop.stats()
        assert stats["last_error"] is not None
        assert "84 channels" in stats["last_error"]
        assert stats["search_failures"] == 1
        assert stats["search_decisions"] == 1


class TestShapesThatCannotMeet:
    """A3: the two interfaces the search crosses do not have the same sizes."""

    def test_observation_channels_differ(self, live: _Live) -> None:
        converted = live.state_converter.convert(_state())
        sim_obs = BridgeMCTSEnv.from_bridge_state(converted).get_observation()
        live_spatial, live_vector = live.state_converter.convert_and_encode(_state())

        assert sim_obs["spatial"].shape == (13, 16, 16)
        assert sim_obs["vector"].shape == (64,)
        assert tuple(live_spatial.shape) == (EncoderConfig().spatial_channels, 16, 16)
        assert EncoderConfig().spatial_channels == 84
        assert tuple(live_vector.shape) == (EncoderConfig().vector_dim,)
        assert live_vector.shape[0] != sim_obs["vector"].shape[0]
        # The live network's first layer takes the encoder's channels, not the simulator's.
        assert live.network.spatial_encoder.conv_in.weight.shape[1] == 84

    def test_action_space_sizes_differ(self, live: _Live) -> None:
        converted = live.state_converter.convert(_state())
        env = BridgeMCTSEnv.from_bridge_state(converted)

        assert env.action_space.n == SIM_ACTIONS
        assert live.action_converter.action_space_size == 88562
        assert live.action_converter.action_space_size != env.action_space.n


@pytest.fixture(scope="module")
def searched(live: _Live) -> dict[str, Any]:
    state = _state()
    converted = live.state_converter.convert(state)
    env = BridgeMCTSEnv.from_bridge_state(converted)
    network = MicroCivNetwork(board_size=16, action_space_size=SIM_ACTIONS)
    search = GumbelMCTS(
        network=network,
        config=GumbelMCTSConfig(num_simulations=SIMS, num_sampled_actions=SIMS),
        device="cpu",
    )
    policy, _value = search.search(env)
    return {
        "state": state,
        "converted": converted,
        "sim_mask": env.get_valid_actions(),
        "policy": policy,
        "index": int(np.argmax(policy)),
    }


class TestSearchWithASimulatorShapedNetwork:
    """A4: even with a network that fits the simulator, the index means two things."""

    def test_search_returns_a_simulator_legal_index(self, searched: dict[str, Any]) -> None:
        assert searched["policy"].shape == (SIM_ACTIONS,)
        assert 0 <= searched["index"] < SIM_ACTIONS
        assert searched["sim_mask"][searched["index"]] == 1.0
        # No mass off the simulator's legal set.
        off = searched["policy"][searched["sim_mask"] == 0]
        assert float(off.sum()) == 0.0

    def test_end_turn_is_index_zero_in_both_by_coincidence(
        self, live: _Live, searched: dict[str, Any]
    ) -> None:
        # Measured: the contract expected the simulator's END_TURN to decode to something
        # else in the live space. It does not: both spaces put END_TURN first.
        sim_end = 0
        live_end = live.action_converter.action_space.category_offsets[ActionCategory.END_TURN]
        assert live_end == sim_end
        mask = live.action_converter.generate_action_mask(
            searched["state"].legal_actions, searched["converted"]
        )
        assert mask[sim_end] == 1.0

    def test_every_other_simulator_move_is_a_different_live_action(
        self, live: _Live, searched: dict[str, Any]
    ) -> None:
        """The simulator's move indexes (unit slot x 8 directions) are not the live ones.

        The live space addresses an absolute target tile; the simulator a direction.
        For each simulator move the position it would reach is compared with the target
        the live decoder reads out of the very same index.
        """
        sim_handler = BridgeMCTSEnv.from_bridge_state(
            searched["converted"]
        ).env.action_space_handler
        legal_moves = [
            int(i)
            for i in np.nonzero(searched["sim_mask"])[0]
            if sim_handler.move_offset <= i < sim_handler.attack_offset
        ]
        assert legal_moves, "the state offers the simulator warrior some moves"
        for index in legal_moves:
            decoded = sim_handler.decode_action(index)
            dx, dy = decoded["direction"]
            sim_target = (OWN_X + dx, OWN_Y + dy)
            live_action = live.action_converter.action_space.decode(index)
            assert live_action.target != sim_target, (
                f"index {index}: the simulator moves to {sim_target}, the live decode is "
                f"{live_action}"
            )

    def test_the_live_mask_marks_the_move_at_another_index(
        self, live: _Live, searched: dict[str, Any]
    ) -> None:
        converted = searched["converted"]
        live_mask = live.action_converter.generate_action_mask(
            searched["state"].legal_actions, converted
        )
        sim_handler = BridgeMCTSEnv.from_bridge_state(converted).env.action_space_handler
        sim_index = sim_handler.encode_action({"type": "move", "unit_idx": 0, "direction": (0, 1)})
        assert searched["sim_mask"][sim_index] == 1.0
        assert live_mask[sim_index] == 0.0, "the simulator's move index is not the live move"
        live_indexes = {int(i) for i in np.nonzero(live_mask)[0]}
        assert live_indexes != {int(i) for i in np.nonzero(searched["sim_mask"])[0]}
