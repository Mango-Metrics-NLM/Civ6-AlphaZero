"""GM-B6b (E): ``alpha_zero.bridge.policy_factory``.

The moved builders are checked through both names (``civ6_local``'s old ones delegate), and
``build_search_policy`` against a real checkpoint written by the project's own save path
(``Civ6Network.save(path, metadata=build_provenance(...))``).
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch

from alpha_zero import civ6_local
from alpha_zero.bridge import policy_factory
from alpha_zero.bridge.agent_loop import GreedyNetworkPolicy
from alpha_zero.bridge.mcts_policy import Civ6MctsPolicy
from alpha_zero.bridge.policy_factory import (
    SearchPolicy,
    build_bridge_policy,
    build_network_for_bridge,
    build_search_policy,
)
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.envs.civ6_encoder import EncoderConfig
from alpha_zero.training_civ6 import TrainingConfig
from alpha_zero.utils.checkpoint import build_provenance
from alpha_zero.utils.constants import DEFAULT_ACTION_SPACE_SIZE
from tests.fixtures import lua_commands as c
from tests.fixtures.bridge_search_translation import is_offered

SIM_ACTIONS = 376


def local_config(**kwargs: Any) -> civ6_local.Civ6LocalConfig:
    kwargs.setdefault("mode", "civ6")
    kwargs.setdefault("board_size", 8)
    kwargs.setdefault("device", "cpu")
    return civ6_local.Civ6LocalConfig(**kwargs)


def write_checkpoint(
    path: Path, *, board: int, horizon: int | None, seed: int = 3, metadata: Any = "provenance"
) -> MicroCivNetwork:
    """A simulator network with distinctive weights, saved the way training saves one."""
    torch.manual_seed(seed)
    network = MicroCivNetwork(board_size=board, action_space_size=SIM_ACTIONS)
    if metadata == "provenance":
        run = TrainingConfig(board_size=board, max_turns=horizon if horizon is not None else 30)
        metadata = build_provenance(run, git_sha=None)
        if horizon is None:
            del metadata["max_turns"]
    network.save(str(path), metadata=metadata)
    return network


def same_weights(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class TestMovedNetworkBuilder:
    def test_both_names_build_the_same_network(self) -> None:
        cfg = local_config()
        torch.manual_seed(1)
        moved = build_network_for_bridge(cfg)
        torch.manual_seed(1)
        old = civ6_local._build_network_for_bridge(cfg)
        assert type(moved) is type(old) is MicroCivNetwork
        assert same_weights(moved.state_dict(), old.state_dict())

    def test_it_takes_the_full_encoder_s_shape_and_the_default_action_space(self) -> None:
        network = build_network_for_bridge(local_config(board_size=8))
        encoder = EncoderConfig()
        assert network.config.spatial_channels == encoder.spatial_channels == 84
        assert network.config.vector_dim == encoder.vector_dim
        assert network.config.spatial_size == 8
        assert network.config.action_space_size == DEFAULT_ACTION_SPACE_SIZE

    def test_the_action_space_can_be_given_through_either_name(self) -> None:
        cfg = local_config()
        assert build_network_for_bridge(cfg, 321).config.action_space_size == 321
        assert build_network_for_bridge(cfg, action_space_size=322).config.action_space_size == 322
        assert civ6_local._build_network_for_bridge(cfg, 323).config.action_space_size == 323

    def test_a_checkpoint_file_is_loaded_by_both_names(self, tmp_path: Path) -> None:
        cfg0 = local_config()
        torch.manual_seed(11)
        source = build_network_for_bridge(cfg0)
        path = tmp_path / "net.pt"
        torch.save({"model_state_dict": source.state_dict()}, path)
        cfg = local_config(checkpoint_path=str(path))
        torch.manual_seed(99)
        moved = build_network_for_bridge(cfg)
        torch.manual_seed(98)
        old = civ6_local._build_network_for_bridge(cfg)
        assert same_weights(moved.state_dict(), source.state_dict())
        assert same_weights(old.state_dict(), source.state_dict())

    def test_a_missing_checkpoint_falls_back_to_synthetic_weights_as_before(
        self, tmp_path: Path
    ) -> None:
        cfg = local_config(checkpoint_path=str(tmp_path / "nowhere.pt"))
        assert isinstance(build_network_for_bridge(cfg), MicroCivNetwork)
        assert isinstance(civ6_local._build_network_for_bridge(cfg), MicroCivNetwork)


class TestMovedPolicyBuilder:
    def test_the_greedy_policy_is_the_default(self) -> None:
        cfg = local_config()
        network = build_network_for_bridge(cfg)
        for build in (build_bridge_policy, civ6_local._build_bridge_policy):
            policy = build(cfg, network)
            assert isinstance(policy, GreedyNetworkPolicy)
            assert policy.network is network

    @pytest.mark.parametrize(("sims", "sampled"), [(50, 16), (16, 16), (8, 8), (3, 3)])
    def test_the_search_policy_takes_its_budget_from_the_config(
        self, sims: int, sampled: int
    ) -> None:
        cfg = local_config(use_mcts=True, num_simulations=sims)
        network = build_network_for_bridge(cfg)
        for build in (build_bridge_policy, civ6_local._build_bridge_policy):
            policy = build(cfg, network)
            assert isinstance(policy, Civ6MctsPolicy)
            assert policy.network is network
            assert policy.config.num_simulations == sims
            assert policy.config.num_sampled_actions == sampled
            assert policy.device == "cpu"


class TestBuildSearchPolicyWithACheckpoint:
    def test_the_network_is_the_saved_one_for_the_recorded_board(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        saved = write_checkpoint(path, board=6, horizon=20)
        result = build_search_policy(path)
        assert isinstance(result, SearchPolicy)
        assert result.trained is True
        assert result.board_size == 6
        assert result.max_turns == 20
        assert isinstance(result.policy, Civ6MctsPolicy)
        network = result.policy.network
        assert same_weights(network.state_dict(), saved.state_dict())
        other = MicroCivNetwork(board_size=6, action_space_size=SIM_ACTIONS)
        assert not same_weights(other.state_dict(), saved.state_dict()), "premise: init differs"
        assert network.training is False

    def test_the_policy_horizon_is_the_recorded_one(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        write_checkpoint(path, board=6, horizon=17)
        assert build_search_policy(path).policy.max_turns == 17

    def test_the_network_takes_the_simulator_s_observation_at_the_recorded_board(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "ckpt.pt"
        write_checkpoint(path, board=6, horizon=20)
        network = build_search_policy(path).policy.network
        assert network.config.spatial_channels == 13
        assert network.config.spatial_size == 6
        assert network.config.action_space_size == SIM_ACTIONS
        with torch.no_grad():
            policy, value = network(torch.zeros(1, 13, 6, 6), torch.zeros(1, 64))
        assert tuple(policy.shape) == (1, SIM_ACTIONS)
        assert value.shape[0] == 1

    def test_a_string_path_works_too(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        write_checkpoint(path, board=6, horizon=20)
        assert build_search_policy(str(path)).trained is True

    def test_the_seed_the_budget_and_the_device_reach_the_policy(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        write_checkpoint(path, board=6, horizon=20)
        result = build_search_policy(path, device="cpu", num_simulations=40, seed=5)
        assert result.policy.config.seed == 5
        assert result.policy.config.num_simulations == 40
        assert result.policy.config.num_sampled_actions == 16
        assert result.policy.device == "cpu"
        small = build_search_policy(path, num_simulations=6).policy
        assert small.config.num_simulations == small.config.num_sampled_actions == 6
        default = build_search_policy(path).policy
        assert default.config.num_simulations == 16
        assert default.config.seed is None

    def test_the_policy_searches_a_stand_in_state_at_the_recorded_board(
        self, tmp_path: Path
    ) -> None:
        """A 16x16 checkpoint: the converter at the recorded board feeds the search it built."""
        path = tmp_path / "ckpt.pt"
        write_checkpoint(path, board=16, horizon=30)
        result = build_search_policy(path, num_simulations=8, seed=2)
        runtime = c.loaded_turn()
        raw = c.states(runtime)[0]
        from alpha_zero.bridge.protocol import StateMessage

        state = StateMessage.from_dict(raw).state
        converted = StateConverter(board_size=result.board_size).convert(state)
        command, record = asyncio.run(result.policy.select_command(converted, decision_seed=1))
        assert record["fallback"] is None
        assert is_offered(command, state.legal_actions)

    @pytest.mark.parametrize("metadata", [None, {"board_size": 6}])
    def test_a_checkpoint_that_records_no_horizon_is_refused_naming_it(
        self, tmp_path: Path, metadata: Any
    ) -> None:
        path = tmp_path / "ckpt.pt"
        write_checkpoint(path, board=6, horizon=20, metadata=metadata)
        with pytest.raises(ValueError, match="max_turns"):
            build_search_policy(path)

    def test_the_same_checkpoint_with_a_horizon_is_accepted(self, tmp_path: Path) -> None:
        """Control: the refusals above are about the horizon and nothing else in the file."""
        path = tmp_path / "ckpt.pt"
        write_checkpoint(path, board=6, horizon=20, metadata={"board_size": 6, "max_turns": 20})
        assert build_search_policy(path).max_turns == 20

    def test_a_provenance_without_the_horizon_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.pt"
        write_checkpoint(path, board=6, horizon=None)
        with pytest.raises(ValueError, match="max_turns"):
            build_search_policy(path)

    def test_a_path_that_is_not_a_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            build_search_policy(tmp_path)
        with pytest.raises(ValueError):
            build_search_policy(tmp_path / "nowhere.pt")


class TestBuildSearchPolicyWithoutACheckpoint:
    def test_it_is_refused_by_default(self) -> None:
        with pytest.raises(ValueError):
            build_search_policy()
        with pytest.raises(ValueError):
            build_search_policy(board_size=6, max_turns=20)

    def test_allowing_untrained_is_not_enough_without_both_sizes(self) -> None:
        with pytest.raises(ValueError):
            build_search_policy(allow_untrained=True)
        with pytest.raises(ValueError):
            build_search_policy(allow_untrained=True, board_size=6)
        with pytest.raises(ValueError):
            build_search_policy(allow_untrained=True, max_turns=20)

    def test_untrained_with_both_sizes_is_a_fresh_simulator_network(self) -> None:
        result = build_search_policy(
            allow_untrained=True, board_size=6, max_turns=20, num_simulations=8, seed=4
        )
        assert result.trained is False
        assert (result.board_size, result.max_turns) == (6, 20)
        assert result.policy.max_turns == 20
        assert result.policy.config.seed == 4
        network = result.policy.network
        assert network.config.spatial_channels == 13
        assert network.config.spatial_size == 6
        assert network.config.action_space_size == SIM_ACTIONS
        with torch.no_grad():
            policy, _ = network(torch.zeros(1, 13, 6, 6), torch.zeros(1, 64))
        assert tuple(policy.shape) == (1, SIM_ACTIONS)

    def test_search_policy_is_a_frozen_record(self) -> None:
        result = build_search_policy(allow_untrained=True, board_size=6, max_turns=20)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.trained = True  # type: ignore[misc]


class TestRunBridgeStillRefusesTheSearch:
    def test_run_civ6_bridge_does_not_call_build_search_policy(self) -> None:
        source = Path(civ6_local.__file__).read_text(encoding="utf-8")
        assert "build_search_policy" not in source

    def test_the_old_names_are_still_there(self) -> None:
        assert callable(civ6_local._build_bridge_policy)
        assert callable(civ6_local._build_network_for_bridge)
        assert policy_factory.build_search_policy is build_search_policy
