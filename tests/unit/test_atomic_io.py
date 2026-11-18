"""A killed process never leaves half a file behind (``alpha_zero.utils.atomic_io``).

The re-run rule (``docs/EVALUATION.md`` section 7.1, "After a VOID") compares what
an interrupted attempt left with what its re-run produced. A checkpoint or a
``result.json`` torn by the interruption itself could not be read, so it could
not be shown to reproduce, and an innocent interruption would cost the run its
PASS. These tests pin the two writes that matter, and the helper under them.

Negative controls (each planted, observed failing, then restored):

1. ``write_atomically`` writing straight to the target ->
   ``test_a_writer_that_dies_leaves_the_old_content``,
   ``test_a_writer_that_dies_on_a_new_file_leaves_nothing``,
   ``test_a_leftover_temporary_file_is_overwritten`` and
   ``test_a_checkpoint_save_killed_midway_keeps_the_previous_checkpoint`` fail.
2. ``create_atomically`` using ``os.replace`` (which overwrites) ->
   ``test_create_refuses_a_target_that_appeared_while_writing`` and
   ``test_create_works_and_still_refuses_on_a_filesystem_without_hard_links`` fail.
3. ``Civ6Network.save`` calling ``torch.save(payload, path)`` directly ->
   ``test_a_checkpoint_save_killed_midway_keeps_the_previous_checkpoint`` fails.
4. The benchmark writing ``result.json`` with ``open(target, "x")`` ->
   ``test_a_benchmark_killed_while_writing_leaves_no_result`` fails.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from alpha_zero.core import civ6_network
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.evaluation import benchmark as bench
from alpha_zero.utils import atomic_io

OLD, NEW = "old content\n", "new content\n"
_TINY_BENCHMARK = ["--candidate", "heuristic", "--opponents=random", "--n-games", "2"]
_TINY_BENCHMARK += ["--seed", "5", "--board-size", "5", "--max-turns", "4", "--device", "cpu"]


class Killed(BaseException):
    """Stands in for the process dying: not an ``Exception``, like ``KeyboardInterrupt``."""


def _half_then_die(tmp: Path) -> None:
    tmp.write_text(NEW[: len(NEW) // 2], encoding="utf-8")
    raise Killed


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# ---------------------------------------------------------------------------
# write_atomically: replace, or leave the target exactly as it was
# ---------------------------------------------------------------------------


def test_write_replaces_the_target_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "record.json"
    target.write_text(OLD, encoding="utf-8")
    atomic_io.write_text_atomically(target, NEW)
    assert target.read_text(encoding="utf-8") == NEW
    assert _names(tmp_path) == ["record.json"]


def test_a_writer_that_dies_leaves_the_old_content(tmp_path: Path) -> None:
    target = tmp_path / "record.json"
    target.write_text(OLD, encoding="utf-8")
    with pytest.raises(Killed):
        atomic_io.write_atomically(target, _half_then_die)
    assert target.read_text(encoding="utf-8") == OLD, "a torn write must not reach the target"
    assert _names(tmp_path) == ["record.json"]


def test_a_writer_that_dies_on_a_new_file_leaves_nothing(tmp_path: Path) -> None:
    with pytest.raises(Killed):
        atomic_io.write_atomically(tmp_path / "record.json", _half_then_die)
    assert _names(tmp_path) == []


def test_a_leftover_temporary_file_is_overwritten(tmp_path: Path) -> None:
    """A process that really died left its ``.tmp`` behind; the next write must not trip on it."""
    target = tmp_path / "record.json"
    atomic_io.temporary_path(target).write_text("half of an earlier attempt", encoding="utf-8")
    atomic_io.write_text_atomically(target, NEW)
    assert target.read_text(encoding="utf-8") == NEW and _names(tmp_path) == ["record.json"]


def test_the_temporary_file_sits_beside_the_target(tmp_path: Path) -> None:
    """Same directory, so the same filesystem: the rename is atomic only then."""
    tmp = atomic_io.temporary_path(tmp_path / "ckpt" / "checkpoint_5.pt")
    assert tmp.parent == tmp_path / "ckpt"
    assert tmp.name == "checkpoint_5.pt" + atomic_io.TEMPORARY_SUFFIX
    assert not tmp.name.endswith(".pt"), "a leftover must not look like a checkpoint"


# ---------------------------------------------------------------------------
# create_atomically: the same, and never overwrite
# ---------------------------------------------------------------------------


def test_create_writes_a_new_file(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    atomic_io.create_text_atomically(target, NEW)
    assert target.read_text(encoding="utf-8") == NEW and _names(tmp_path) == ["result.json"]


def test_create_refuses_an_existing_target_and_leaves_it_untouched(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text(OLD, encoding="utf-8")
    with pytest.raises(FileExistsError, match="never overwritten"):
        atomic_io.create_text_atomically(target, NEW)
    assert target.read_text(encoding="utf-8") == OLD and _names(tmp_path) == ["result.json"]


def _write_while_another_process_creates(target: Path) -> Any:
    def write(tmp: Path) -> None:
        tmp.write_text(NEW, encoding="utf-8")
        target.write_text(OLD, encoding="utf-8")  # the other process got there first

    return write


def test_create_refuses_a_target_that_appeared_while_writing(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    with pytest.raises(FileExistsError):
        atomic_io.create_atomically(target, _write_while_another_process_creates(target))
    assert target.read_text(encoding="utf-8") == OLD, "the other process's file must survive"
    assert _names(tmp_path) == ["result.json"]


def _no_hard_links(src: Any, dst: Any) -> None:
    raise OSError("hard links are not supported on this filesystem")


def test_create_works_and_still_refuses_on_a_filesystem_without_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "link", _no_hard_links)
    target = tmp_path / "result.json"
    atomic_io.create_text_atomically(target, NEW)
    assert target.read_text(encoding="utf-8") == NEW and _names(tmp_path) == ["result.json"]

    other = tmp_path / "other.json"
    with pytest.raises(FileExistsError):
        atomic_io.create_atomically(other, _write_while_another_process_creates(other))
    assert other.read_text(encoding="utf-8") == OLD


def test_create_killed_while_writing_leaves_nothing(tmp_path: Path) -> None:
    with pytest.raises(Killed):
        atomic_io.create_atomically(tmp_path / "result.json", _half_then_die)
    assert _names(tmp_path) == []


# ---------------------------------------------------------------------------
# The two writes the re-run rule depends on
# ---------------------------------------------------------------------------


def _network(seed: int) -> MicroCivNetwork:
    torch.manual_seed(seed)
    return MicroCivNetwork(
        board_size=6, action_space_size=376, num_res_blocks=1, num_filters=8, spatial_channels=13
    )


def _state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state: dict[str, torch.Tensor] = payload["state_dict"]
    return state


def test_a_checkpoint_is_saved_whole_and_loads_back(tmp_path: Path) -> None:
    network = _network(1)
    path = tmp_path / "checkpoint_5.pt"
    network.save(str(path), metadata={"seed": 7})
    saved = _state(path)
    assert saved.keys() == network.state_dict().keys() and saved
    assert all(torch.equal(saved[k], v) for k, v in network.state_dict().items())
    assert _names(tmp_path) == ["checkpoint_5.pt"]


def test_a_checkpoint_save_killed_midway_keeps_the_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "latest.pt"
    first = _network(1)
    first.save(str(path))
    before = path.read_bytes()

    def torn_save(payload: Any, target: Any) -> None:
        Path(target).write_bytes(b"PK\x03\x04 half a checkpoint")
        raise Killed

    monkeypatch.setattr(civ6_network.torch, "save", torn_save)
    with pytest.raises(Killed):
        _network(2).save(str(path))

    assert path.read_bytes() == before, "the previous checkpoint must survive a killed save"
    assert _names(tmp_path) == ["latest.pt"]
    monkeypatch.undo()
    assert all(torch.equal(_state(path)[k], v) for k, v in first.state_dict().items())


def test_a_benchmark_killed_while_writing_leaves_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``result.json`` means "the arm never finished", so the same ``--out`` can be re-run."""
    logging.getLogger("alpha_zero.envs.microciv").setLevel(logging.ERROR)
    out = tmp_path / "arm"

    def torn_text(tmp: Path, text: str, encoding: str) -> None:
        tmp.write_text(text[: len(text) // 2], encoding=encoding)
        raise Killed

    monkeypatch.setattr(atomic_io, "_write_text", torn_text)
    with pytest.raises(Killed):
        bench.main([*_TINY_BENCHMARK, "--out", str(out)])
    assert not (out / "result.json").exists(), "a torn result.json would read as a finished arm"
    assert [p.name for p in out.iterdir()] == []

    monkeypatch.undo()
    assert bench.main([*_TINY_BENCHMARK, "--out", str(out)]) == 0
    envelope = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert envelope["results"][0]["games"], "the re-run wrote a whole result"
    assert sorted(p.name for p in out.iterdir()) == ["report.md", "result.json"]
