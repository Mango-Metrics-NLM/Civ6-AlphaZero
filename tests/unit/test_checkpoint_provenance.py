"""Checkpoint provenance (LF-A15 / LF-A16).

A checkpoint trained at one game horizon and read at another is not a broken
file: it loads, plays and produces a number. These tests pin the three things
that make the number trustworthy -- the metadata survives ``weights_only=True``,
a horizon mismatch is refused where checkpoints are read, and legacy files
(which carry no metadata) are tolerated with a warning rather than rejected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from alpha_zero import training_civ6
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.training_civ6 import (
    TrainingConfig,
    build_parser,
    config_from_args,
    infer_start_iteration,
    resume_network_weights,
)
from alpha_zero.utils.checkpoint import (
    HorizonMismatch,
    build_provenance,
    check_provenance,
    load_network,
)

BOARD = 6
ACTIONS = 376
CHANNELS = 13
PRIMITIVES = (str, int, float, bool, type(None))
EXPECTED_KEYS = {
    "board_size",
    "max_turns",
    "num_players",
    "mcts",
    "gumbel_k",
    "gumbel_sims",
    "c_scale",
    "policy_target",
    "bootstrap",
    "seed",
    "git_sha",
    "env_version",
}


def _network() -> MicroCivNetwork:
    return MicroCivNetwork(
        board_size=BOARD,
        action_space_size=ACTIONS,
        num_res_blocks=1,
        num_filters=8,
        spatial_channels=CHANNELS,
    )


def _provenance(max_turns: int = 30, **overrides: Any) -> dict[str, Any]:
    config = TrainingConfig(board_size=BOARD, max_turns=max_turns, seed=7, **overrides)
    return build_provenance(config, git_sha="abc123")


class TestBuildProvenance:
    def test_has_exactly_the_documented_keys_and_only_primitives(self) -> None:
        meta = _provenance()
        assert set(meta) == EXPECTED_KEYS
        assert all(isinstance(v, PRIMITIVES) for v in meta.values())

    def test_records_the_resolved_search_config_not_raw_args(self) -> None:
        args = build_parser().parse_args(
            [
                "--mcts",
                "gumbel",
                "--gumbel_k",
                "8",
                "--gumbel_sims",
                "24",
                "--c_scale",
                "0.5",
                "--policy_target",
                "one_hot",
                "--bootstrap",
                "none",
                "--max_turns",
                "45",
                "--seed",
                "3",
            ]
        )
        meta = build_provenance(config_from_args(args, "ckpt"), git_sha=None)
        assert meta["mcts"] == "gumbel"
        assert meta["gumbel_k"] == 8
        assert meta["gumbel_sims"] == 24
        assert meta["c_scale"] == 0.5
        assert meta["policy_target"] == "one_hot"
        assert meta["bootstrap"] == "none"
        assert meta["max_turns"] == 45
        assert meta["seed"] == 3
        assert meta["git_sha"] is None

    def test_defaults_describe_the_puct_simple_bootstrap_run(self) -> None:
        meta = _provenance()
        assert meta["mcts"] == "civ6"
        assert meta["bootstrap"] == "simple_mcts"
        assert isinstance(meta["env_version"], int)
        assert isinstance(meta["policy_target"], str)


class TestMetadataRoundTrip:
    def test_survives_weights_only_load(self, tmp_path: Path) -> None:
        path = tmp_path / "m.pt"
        meta = _provenance()
        _network().save(str(path), metadata=meta)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert loaded["metadata"] == meta

    def test_save_without_metadata_writes_no_metadata_key(self, tmp_path: Path) -> None:
        path = tmp_path / "legacy.pt"
        _network().save(str(path))
        assert "metadata" not in torch.load(path, map_location="cpu", weights_only=True)

    def test_non_primitive_metadata_is_refused_at_save(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="primitive"):
            _network().save(str(tmp_path / "bad.pt"), metadata={"cfg": object()})


class TestResumeHorizon:
    def test_max_turns_mismatch_raises_on_resume(self, tmp_path: Path) -> None:
        path = tmp_path / "c.pt"
        _network().save(str(path), metadata=_provenance(max_turns=30))
        with pytest.raises(HorizonMismatch, match="30"):
            resume_network_weights(_network(), str(path), max_turns=60, board_size=BOARD)

    def test_matching_horizon_resumes(self, tmp_path: Path) -> None:
        path = tmp_path / "c.pt"
        source = _network()
        source.save(str(path), metadata=_provenance(max_turns=30))
        target = _network()
        resume_network_weights(target, str(path), max_turns=30, board_size=BOARD)
        for a, b in zip(source.state_dict().values(), target.state_dict().values(), strict=True):
            assert torch.equal(a, b)

    def test_legacy_file_warns_and_passes(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "legacy.pt"
        _network().save(str(path))
        with caplog.at_level(logging.WARNING):
            resume_network_weights(_network(), str(path), max_turns=60, board_size=BOARD)
        assert any("no provenance" in r.getMessage().lower() for r in caplog.records)

    def test_horizon_mismatch_is_a_value_error(self) -> None:
        assert issubclass(HorizonMismatch, ValueError)


class TestLoadNetworkHorizon:
    def test_max_turns_mismatch_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "c.pt"
        _network().save(str(path), metadata=_provenance(max_turns=30))
        with pytest.raises(HorizonMismatch):
            load_network(path, device="cpu", expected_max_turns=60)

    def test_matching_horizon_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "c.pt"
        _network().save(str(path), metadata=_provenance(max_turns=30))
        assert load_network(path, device="cpu", expected_max_turns=30) is not None

    def test_not_asking_for_a_horizon_does_not_check(self, tmp_path: Path) -> None:
        path = tmp_path / "c.pt"
        _network().save(str(path), metadata=_provenance(max_turns=30))
        assert load_network(path, device="cpu") is not None

    def test_legacy_file_warns_and_passes(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "legacy.pt"
        _network().save(str(path))
        with caplog.at_level(logging.WARNING):
            load_network(path, device="cpu", expected_max_turns=60)
        assert any("no provenance" in r.getMessage().lower() for r in caplog.records)


class TestCheckProvenance:
    def test_none_metadata_is_legacy(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            check_provenance(None, expected_max_turns=10, source="x.pt")
        assert caplog.records

    def test_no_expectation_is_silent(self) -> None:
        check_provenance({"max_turns": 5}, expected_max_turns=None, source="x.pt")


class TestBenchmarkLoader:
    def test_benchmark_loader_refuses_a_horizon_mismatch(self, tmp_path: Path) -> None:
        from alpha_zero.evaluation.benchmark import _load_network

        path = tmp_path / "c.pt"
        _network().save(str(path), metadata=_provenance(max_turns=30))
        with pytest.raises(HorizonMismatch):
            _load_network(path, "cpu", expected_max_turns=60)


def _tiny_config(tmp_path: Path, **overrides: Any) -> TrainingConfig:
    base: dict[str, Any] = {
        "board_size": 4,
        "max_turns": 3,
        "num_simulations": 2,
        "num_res_blocks": 1,
        "num_filters": 8,
        "num_games_per_iteration": 1,
        "num_iterations": 1,
        "min_replay_size": 10**9,  # never train: these tests are about iteration 0
        "save_interval": 10**9,
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "log_dir": str(tmp_path / "log"),
        "seed": 1,
        "device": "cpu",
    }
    base.update(overrides)
    return TrainingConfig(**base)


class TestBootstrapSwitch:
    def test_default_bootstrap_is_simple_mcts(self) -> None:
        assert build_parser().parse_args([]).bootstrap == "simple_mcts"
        assert TrainingConfig().bootstrap == "simple_mcts"

    def test_invalid_bootstrap_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--bootstrap", "nope"])

    @pytest.mark.parametrize(("bootstrap", "expected"), [("simple_mcts", True), ("none", False)])
    def test_iteration_zero_uses_simple_mcts_only_when_asked(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        bootstrap: str,
        expected: bool,
    ) -> None:
        seen: list[bool] = []

        def fake(*_a: Any, use_simple_mcts: bool = False, **_k: Any) -> list[Any]:
            seen.append(use_simple_mcts)
            return []

        monkeypatch.setattr(training_civ6, "run_self_play", fake)
        training_civ6.train(_tiny_config(tmp_path, bootstrap=bootstrap))
        assert seen == [expected]

    def test_bootstrap_none_never_constructs_simple_mcts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[int] = []
        real = training_civ6.SimpleMCTS

        class Spy(real):  # type: ignore[valid-type, misc]
            def __init__(self, *a: Any, **k: Any) -> None:
                built.append(1)
                super().__init__(*a, **k)

        monkeypatch.setattr(training_civ6, "SimpleMCTS", Spy)
        training_civ6.train(_tiny_config(tmp_path, bootstrap="none"))
        assert built == []

        training_civ6.train(_tiny_config(tmp_path, bootstrap="simple_mcts"))
        assert built, "the spy must observe the default path or it proves nothing"


class TestCheckpointZero:
    def test_checkpoint_zero_carries_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(training_civ6, "run_self_play", lambda *a, **k: [])
        config = _tiny_config(tmp_path)
        training_civ6.train(config)
        ckpt = Path(config.checkpoint_dir)
        zero = torch.load(ckpt / "checkpoint_0.pt", map_location="cpu", weights_only=True)
        assert zero["metadata"]["bootstrap"] == "simple_mcts"
        assert zero["metadata"]["max_turns"] == config.max_turns
        # No training ran (replay buffer never filled), so latest == checkpoint_0
        # in weights; and latest carries provenance too.
        latest = torch.load(ckpt / "latest.pt", map_location="cpu", weights_only=True)
        assert latest["metadata"] == zero["metadata"]
        for k, v in zero["state_dict"].items():
            assert torch.equal(v, latest["state_dict"][k])

    def test_checkpoint_zero_is_saved_before_any_self_play(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _tiny_config(tmp_path)
        existed: list[bool] = []

        def fake(*_a: Any, **_k: Any) -> list[Any]:
            existed.append((Path(config.checkpoint_dir) / "checkpoint_0.pt").exists())
            return []

        monkeypatch.setattr(training_civ6, "run_self_play", fake)
        training_civ6.train(config)
        assert existed == [True]

    def test_name_resolves_to_loop_index_zero(self) -> None:
        # checkpoint_<N>.pt resumes at index N; for N == 0 that is "from the top".
        assert infer_start_iteration("d/checkpoint_0.pt") == 0

    def test_resuming_from_checkpoint_zero_does_not_rewrite_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(training_civ6, "run_self_play", lambda *a, **k: [])
        config = _tiny_config(tmp_path)
        training_civ6.train(config)
        zero_path = Path(config.checkpoint_dir) / "checkpoint_0.pt"
        before = zero_path.read_bytes()

        # A different seed would initialise different weights, so a rewrite
        # would change the bytes.
        resumed = _tiny_config(tmp_path, resume_from=str(zero_path), seed=99)
        training_civ6.train(resumed)
        assert zero_path.read_bytes() == before

    def test_start_iteration_above_zero_does_not_write_checkpoint_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(training_civ6, "run_self_play", lambda *a, **k: [])
        config = _tiny_config(tmp_path, num_iterations=3, start_iteration=2)
        training_civ6.train(config)
        assert not (Path(config.checkpoint_dir) / "checkpoint_0.pt").exists()
