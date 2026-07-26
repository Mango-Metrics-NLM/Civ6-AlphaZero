"""Contract tests for :func:`alpha_zero.utils.checkpoint.load_network`.

This repo writes checkpoints in more than one shape, and consumers used to
hardcode whichever one their author happened to have in front of them. That is
how the web UI ended up unable to load a checkpoint produced by the documented
training entrypoint. ``load_network`` is the single reader, so the shapes it
must accept are pinned here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from alpha_zero.core.civ6_actions import MicroCivActionSpace
from alpha_zero.core.civ6_network import MicroCivNetwork, NetworkConfig
from alpha_zero.utils.checkpoint import (
    compute_weight_checksum,
    load_network,
    safe_load_checkpoint,
    save_checkpoint,
)

BOARD_SIZE = 8
TINY = {"num_res_blocks": 2, "num_filters": 8}


def _action_space_size(board_size: int = BOARD_SIZE) -> int:
    return MicroCivActionSpace(board_size=board_size).total_size


def make_network(board_size: int = BOARD_SIZE, num_players: int = 2) -> MicroCivNetwork:
    """Build a small network; non-default so wrong-architecture loads fail."""
    kwargs = dict(TINY)
    if num_players > 2:
        kwargs["spatial_channels"] = 3 + 4 * num_players + 2
    return MicroCivNetwork(
        board_size=board_size,
        action_space_size=_action_space_size(board_size),
        num_players=num_players,
        **kwargs,
    )


class TestWriterFormats:
    """Every shape written anywhere in the repo must load."""

    def test_civ6_network_save_format(self, tmp_path: Path) -> None:
        """{"config", "state_dict"} — what training_civ6 actually writes."""
        net = make_network()
        path = tmp_path / "a.pt"
        net.save(str(path))

        loaded = load_network(path)

        assert compute_weight_checksum(loaded) == pytest.approx(
            compute_weight_checksum(net), rel=1e-6
        )

    def test_orchestrator_model_state_dict_format(self, tmp_path: Path) -> None:
        """{"model_state_dict", "config"} — what the checkpointer writes."""
        net = make_network()
        path = tmp_path / "b.pt"
        save_checkpoint(path, net.state_dict(), config=net.config)

        loaded = load_network(path)

        assert compute_weight_checksum(loaded) == pytest.approx(
            compute_weight_checksum(net), rel=1e-6
        )

    def test_bare_state_dict_with_explicit_arch(self, tmp_path: Path) -> None:
        """A raw state dict carries no config, so the caller must supply one."""
        net = make_network()
        path = tmp_path / "c.pt"
        torch.save(net.state_dict(), path)

        loaded = load_network(
            path,
            board_size=BOARD_SIZE,
            action_space_size=_action_space_size(),
            **TINY,
        )

        assert compute_weight_checksum(loaded) == pytest.approx(
            compute_weight_checksum(net), rel=1e-6
        )


class TestReturnedNetworkContract:
    """Callers rely on placement and mode; search classes do neither."""

    def test_returned_network_is_in_eval_mode(self, tmp_path: Path) -> None:
        path = tmp_path / "a.pt"
        make_network().save(str(path))

        assert not load_network(path).training

    def test_returned_network_is_on_requested_device(self, tmp_path: Path) -> None:
        path = tmp_path / "a.pt"
        make_network().save(str(path))

        loaded = load_network(path, device="cpu")

        assert next(loaded.parameters()).device.type == "cpu"

    def test_unavailable_device_falls_back(self, tmp_path: Path) -> None:
        """NG-4: requesting CUDA on a CPU-only host must still return a model."""
        if torch.cuda.is_available():
            pytest.skip("requires a CPU-only host")
        path = tmp_path / "a.pt"
        make_network().save(str(path))

        loaded = load_network(path, device="cuda")

        assert next(loaded.parameters()).device.type == "cpu"


class TestNPlayerCheckpoints:
    """The regression that motivated putting num_players on NetworkConfig."""

    @pytest.mark.parametrize("num_players", [3, 4])
    def test_n_player_checkpoint_round_trips(self, tmp_path: Path, num_players: int) -> None:
        """Without num_players in the config this raised a size mismatch.

        MicroCivNetwork swaps in an NPlayerValueHead when num_players > 2, but
        NetworkConfig did not record it — so reconstructing from a saved config
        built a scalar head and load_state_dict failed on value_head.fc2.weight.
        """
        net = make_network(num_players=num_players)
        path = tmp_path / f"n{num_players}.pt"
        net.save(str(path))

        loaded = load_network(path)

        assert loaded.num_players == num_players
        assert type(loaded.value_head) is type(net.value_head)
        assert compute_weight_checksum(loaded) == pytest.approx(
            compute_weight_checksum(net), rel=1e-6
        )

    def test_two_player_default_is_unchanged(self, tmp_path: Path) -> None:
        """Checkpoints predating the field must still load as 2-player."""
        net = make_network()
        path = tmp_path / "old.pt"
        net.save(str(path))

        # Simulate a pre-change checkpoint by dropping the new key.
        blob = safe_load_checkpoint(path).data
        blob["config"].pop("num_players", None)
        torch.save(blob, path)

        loaded = load_network(path)

        assert loaded.num_players == 2


class TestConfigSchemaDiscrimination:
    """Two unrelated dataclasses in this repo are both named NetworkConfig."""

    def test_orchestrator_config_does_not_crash_the_loader(self, tmp_path: Path) -> None:
        """The orchestrator schema lacks spatial_size/action_space_size.

        Passing it to the core NetworkConfig constructor raises TypeError on
        'num_heads'. The loader must recognise the foreign shape and say so.
        """
        from alpha_zero.orchestrator.config import NetworkConfig as OrchestratorNetworkConfig

        net = make_network()
        path = tmp_path / "orch.pt"
        save_checkpoint(path, net.state_dict(), config=OrchestratorNetworkConfig())

        with pytest.raises(ValueError, match="no usable architecture config"):
            load_network(path)

    def test_orchestrator_config_loads_with_explicit_arch(self, tmp_path: Path) -> None:
        from alpha_zero.orchestrator.config import NetworkConfig as OrchestratorNetworkConfig

        net = make_network()
        path = tmp_path / "orch.pt"
        save_checkpoint(path, net.state_dict(), config=OrchestratorNetworkConfig())

        loaded = load_network(
            path,
            board_size=BOARD_SIZE,
            action_space_size=_action_space_size(),
            **TINY,
        )

        assert compute_weight_checksum(loaded) == pytest.approx(
            compute_weight_checksum(net), rel=1e-6
        )

    def test_unknown_config_keys_are_ignored(self, tmp_path: Path) -> None:
        """A checkpoint from a newer build must not break this one."""
        net = make_network()
        path = tmp_path / "future.pt"
        config = dataclasses.asdict(net.config)
        config["some_future_field"] = "value"
        save_checkpoint(path, net.state_dict(), config=config)

        loaded = load_network(path)

        assert loaded.config.spatial_size == BOARD_SIZE


class TestErrorPaths:
    """Failures must name the problem instead of guessing."""

    def test_missing_config_raises_with_guidance(self, tmp_path: Path) -> None:
        path = tmp_path / "bare.pt"
        torch.save(make_network().state_dict(), path)

        with pytest.raises(ValueError, match="no usable architecture config"):
            load_network(path)

    def test_board_size_mismatch_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "a.pt"
        make_network(board_size=8).save(str(path))

        with pytest.raises(ValueError, match="board_size"):
            load_network(path, expected_board_size=16)

    def test_matching_board_size_is_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "a.pt"
        make_network(board_size=8).save(str(path))

        assert load_network(path, expected_board_size=8) is not None

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_network(tmp_path / "nope.pt")


class TestSaveCheckpointConfigCoercion:
    """A dataclass config must not force the unsafe torch.load path."""

    def test_dataclass_config_is_stored_as_primitives(self, tmp_path: Path) -> None:
        """NetworkConfig is not a registered safe global.

        Storing the instance itself makes weights_only=True fail, so every
        later load silently falls back to weights_only=False.
        """
        path = tmp_path / "cfg.pt"
        save_checkpoint(path, make_network().state_dict(), config=NetworkConfig())

        result = safe_load_checkpoint(path)

        assert isinstance(result.data["config"], dict)
        assert result.loaded_safely, "dataclass config forced the unsafe load path"

    def test_dict_config_passes_through(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.pt"
        save_checkpoint(path, make_network().state_dict(), config={"spatial_size": 8})

        assert safe_load_checkpoint(path).data["config"] == {"spatial_size": 8}
