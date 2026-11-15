"""Tests for ``alpha_zero.evaluation.reproduction``: the "After a VOID" comparison.

The module answers one question -- does a seeded re-run reproduce the attempt it
replaces? -- so each test below is a property the answer depends on: tensors are
compared and not files, wall-clock fields are ignored and nothing else is, a
missing or unreadable component is never a pass, and "nothing to compare" is not
"reproduced".

Negative controls (2026-11-14, re-run against the final code). Each defect was
planted in ``alpha_zero/evaluation/reproduction.py``, the planted text was read
back from disk, the listed tests were seen to FAIL, and the file was restored and
checked byte-identical (sha256) before the next one:

1.  ``compare_digests`` skips checkpoints (the ``_compare_checkpoints`` call
    removed) -> ``test_one_changed_tensor_is_named_and_nothing_else_is``,
    ``test_many_differing_tensors_are_counted_and_capped``,
    ``test_tensors_only_one_side_holds_are_a_difference``,
    ``test_a_component_missing_from_the_rerun_is_a_difference[checkpoint]``,
    ``test_what_only_the_rerun_holds_is_ignored``,
    ``test_an_unreadable_rerun_checkpoint_is_a_difference``,
    ``test_cli_differs_lists_the_differences_and_exits_1`` and
    ``test_cli_runs_as_a_module_and_propagates_its_status`` failed.
2.  ``checkpoint_digest`` hashes the file's bytes instead of its tensors ->
    ``test_checkpoint_digest_ignores_file_name_and_metadata`` failed: two files
    holding the same network "differ". So did 28 other tests here, because the
    synthetic re-run carries different checkpoint metadata. **Not** the real
    cross-process check (``test_two_processes_reproduce_on_cpu`` in
    ``tests/scripts/test_check_determinism.py`` passed under this defect): with
    this torch the bytes of two identical runs are equal, the archive carries no
    timestamp. File hashing is wrong because of the file stem and the metadata,
    not because identical runs differ.
3.  ``tensor_digest`` leaves dtype and shape out of the hash ->
    ``test_tensor_digest_covers_dtype_and_shape`` failed.
4.  ``checkpoint_digest`` loads with ``strict_security=False`` ->
    ``test_a_checkpoint_is_never_loaded_unsafely`` failed (the environment opt-in
    was honoured and the file was unpickled).
5.  The "no tensors" guard removed ->
    ``test_a_checkpoint_without_tensors_is_unreadable_not_equal`` failed.
6.  ``ITERATION_RECORD_EXCLUDED_FIELDS = ()`` ->
    ``test_only_the_excluded_fields_are_dropped``,
    ``test_wall_clock_fields_alone_are_not_a_difference``,
    ``test_identical_runs_reproduce`` and 23 other tests here failed (every
    comparison of the synthetic attempt with its re-run), as did the real check.
7.  Game records not compared (``if False and ...`` in ``_compare_arena``) ->
    ``test_one_changed_game_is_named_by_arm_and_index`` failed.
8.  Iteration records not compared (the ``_compare_iterations`` call removed) ->
    ``test_one_changed_iteration_field_is_named``, the four
    ``test_identity_and_result_fields_are_compared`` cases,
    ``test_a_repeated_nan_is_not_a_difference_and_a_type_change_is``,
    ``test_a_component_missing_from_the_rerun_is_a_difference[iteration]`` and
    ``test_what_only_the_rerun_holds_is_ignored`` failed.
9.  A component missing from the candidate is skipped (``continue`` without a
    difference, in all three ``_compare_*`` loops) -> all three
    ``test_a_component_missing_from_the_rerun_is_a_difference`` cases,
    ``test_what_only_the_rerun_holds_is_ignored``,
    ``test_an_unreadable_rerun_checkpoint_is_a_difference`` and
    ``test_cli_runs_as_a_module_and_propagates_its_status`` failed.
10. An unreadable reference component is dropped silently (the ``unreadable``
    differences removed) ->
    ``test_an_unreadable_attempt_checkpoint_is_a_difference`` and the three
    ``test_an_unparseable_arena_result_is_a_difference`` cases failed.
11. The CLI calls zero compared components a pass (the ``compared == 0`` branch
    removed) -> ``test_cli_nothing_to_compare_exits_2`` failed.
12. The comparison made symmetric (candidate-only components reported) ->
    ``test_what_only_the_rerun_holds_is_ignored`` and six others failed. That one
    guards the documented direction, not a safety property.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Any

import pytest
import torch
import yaml

from alpha_zero.evaluation import reproduction as rp
from alpha_zero.evaluation.a1_registration import run_directory
from alpha_zero.training.run_record import ITERATIONS_FILE
from alpha_zero.utils.safe_load import UNSAFE_CHECKPOINT_ENV_VAR, UnsafeCheckpointError

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 20261114
ARM = "primary"
#: Per-game outcomes of the synthetic primary arm (two seat-swapped pairs).
RESULTS = ("win", "draw", "loss", "win")
#: Iterations the synthetic run trained; it holds checkpoints 0..ITERATIONS.
ITERATIONS = 2


# ---------------------------------------------------------------------------
# Synthetic run directories in the run-card layout
# ---------------------------------------------------------------------------


def state(seed: int) -> dict[str, torch.Tensor]:
    """A small ``state_dict`` with the dtypes a real one has (float32 and a 0-dim int64)."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "conv.weight": torch.randn(4, 3, 3, 3, generator=generator),
        "conv.bias": torch.randn(4, generator=generator),
        "bn.running_mean": torch.randn(4, generator=generator),
        "bn.num_batches_tracked": torch.tensor(seed + 1),
        "head.weight": torch.randn(5, 4, generator=generator),
    }


def write_checkpoint(
    path: Path, tensors: dict[str, torch.Tensor], *, metadata: dict[str, Any] | None = None
) -> Path:
    """A checkpoint shaped like ``MicroCivNetwork.save`` writes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": {"board_size": 6}, "state_dict": tensors, "metadata": metadata or {}}
    torch.save(payload, path)
    return path


def game_records(results: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"pair": i // 2, "seat": i % 2, "result": r} for i, r in enumerate(results)]


def write_arena(
    run_dir: Path, arm: str, results: tuple[str, ...], *, stamp: str = "a", seed: int = SEED
) -> Path:
    """A ``result.json`` shaped like the benchmark's; ``stamp`` moves only the non-results."""
    out = run_dir / rp.ARENA_DIR_NAME / arm
    out.mkdir(parents=True, exist_ok=True)
    file_hash = hashlib.sha256(stamp.encode()).hexdigest()
    generated = f"2026-11-14T00:00:0{len(stamp)}+00:00 ({stamp})"
    envelope = {
        "benchmark_schema_version": 2,
        "protocol": {
            "board_size": 6,
            "max_turns": 6,
            "num_simulations": 4,
            "gumbel_k": 4,
            "n_games": len(results),
            "mcts_type": "gumbel",
            "gameplay_preset": "default",
            "paired_seats": True,
        },
        "candidate": {"kind": "checkpoint", "id": f"checkpoint:{file_hash}"},
        "shard": None,
        "evidence": {"seed": seed, "generated_utc": generated, "command": f"--out {out}"},
        "results": [
            {
                "opponent_id": f"checkpoint:{file_hash[::-1]}",
                "timestamp": generated,
                "num_games": len(results),
                "games": game_records(results),
            }
        ],
    }
    target = out / rp.ARENA_RESULT_FILE
    target.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return target


def iteration_record(iteration: int, *, stamp: str = "a") -> dict[str, Any]:
    """One ``iterations.jsonl`` line with the fields ``training_civ6`` writes."""
    return {
        "run_id": f"a1-20261111-{stamp}",
        "seed": SEED,
        "git_sha": "b03878f3b1b58f3efef8508b33805c24dfedbf66",
        "started_at": f"2026-11-14T03:23:4{len(stamp)}Z ({stamp})",
        "resumed_from": None,
        "start_iteration": 0,
        "iteration": iteration,
        "self_play": {
            "games": 2,
            "examples": 30 + iteration,
            "seconds": 0.25 + len(stamp) + iteration,
            "simple_mcts": False,
        },
        "train": {"policy_loss": 1.5 / iteration, "total_loss": 2.5 / iteration, "value_loss": 1.0},
        "prequential": {"n_games": 2, "value_mse": 0.99, "generator": "network"},
        "eval": None,
        "checkpoint": f"checkpoint_{iteration}.pt",
    }


def write_iterations(run_dir: Path, records: list[dict[str, Any]]) -> Path:
    target = run_dir / rp.CHECKPOINT_DIR_NAME / ITERATIONS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), "utf-8")
    return target


def checkpoint_path(run_dir: Path, number: int) -> Path:
    return run_dir / rp.CHECKPOINT_DIR_NAME / f"checkpoint_{number}.pt"


def build_run(run_dir: Path, *, stamp: str = "a") -> Path:
    """A complete synthetic run: checkpoints 0..2, one arena arm, two iteration records.

    Two runs built with different ``stamp`` values are an attempt and a faithful
    re-run: the same tensors, games and losses, and everything that legitimately
    differs between two executions (run id, wall-clock fields, file hashes,
    checkpoint metadata) different.
    """
    for number in range(ITERATIONS + 1):
        write_checkpoint(checkpoint_path(run_dir, number), state(number), metadata={"run": stamp})
    write_arena(run_dir, ARM, RESULTS, stamp=stamp)
    write_iterations(run_dir, [iteration_record(i, stamp=stamp) for i in range(1, ITERATIONS + 1)])
    return run_dir


def edit_json(path: Path, change: Any) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def edit_iteration(run_dir: Path, iteration: int, change: Any) -> None:
    target = run_dir / rp.CHECKPOINT_DIR_NAME / ITERATIONS_FILE
    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["iteration"] == iteration:
            change(record)
    write_iterations(run_dir, records)


@pytest.fixture
def runs(tmp_path: Path) -> tuple[Path, Path]:
    """``(attempt, rerun)``: a run and a faithful re-run of it."""
    return build_run(tmp_path / "attempt", stamp="a"), build_run(tmp_path / "rerun", stamp="bb")


def differences(attempt: Path, rerun: Path) -> list[str]:
    return rp.compare_digests(rp.run_digest(attempt), rp.run_digest(rerun))


# ---------------------------------------------------------------------------
# checkpoint_digest: tensors, not files
# ---------------------------------------------------------------------------


def test_checkpoint_digest_names_every_tensor(tmp_path: Path) -> None:
    tensors = state(3)
    digest = rp.checkpoint_digest(write_checkpoint(tmp_path / "checkpoint_3.pt", tensors))
    assert list(digest) == list(tensors)
    assert all(len(value) == hashlib.sha256().digest_size * 2 for value in digest.values())
    assert len(set(digest.values())) == len(tensors), "distinct tensors must hash apart"


def test_checkpoint_digest_reads_the_key_the_trainer_writes(tmp_path: Path) -> None:
    """Bound to the real writer: ``MicroCivNetwork.save`` puts the weights under ``state_dict``."""
    from alpha_zero.core.civ6_network import MicroCivNetwork

    torch.manual_seed(SEED)
    network = MicroCivNetwork(board_size=6, action_space_size=32, num_res_blocks=1, num_filters=8)
    path = tmp_path / "checkpoint_0.pt"
    network.save(str(path), metadata={"seed": SEED})
    digest = rp.checkpoint_digest(path)
    assert set(digest) == set(network.state_dict())
    for key, tensor in network.state_dict().items():
        assert digest[key] == rp.tensor_digest(tensor), key


def test_checkpoint_digest_ignores_file_name_and_metadata(tmp_path: Path) -> None:
    """The same network in two files that differ byte for byte has one digest."""
    first = write_checkpoint(tmp_path / "checkpoint_5.pt", state(5), metadata={"git_sha": "a" * 40})
    second = write_checkpoint(tmp_path / "latest.pt", state(5), metadata={"git_sha": "b" * 40})
    assert first.read_bytes() != second.read_bytes(), "the premise: the files differ"
    assert rp.checkpoint_digest(first) == rp.checkpoint_digest(second)


def test_checkpoint_digest_sees_one_changed_element(tmp_path: Path) -> None:
    tensors = state(1)
    before = rp.checkpoint_digest(write_checkpoint(tmp_path / "a.pt", tensors))
    tensors["conv.weight"][0, 0, 0, 0] += 1e-6
    after = rp.checkpoint_digest(write_checkpoint(tmp_path / "b.pt", tensors))
    assert [key for key in before if before[key] != after[key]] == ["conv.weight"]


def test_tensor_digest_covers_dtype_and_shape() -> None:
    """Equal bytes are not equal tensors."""
    flat = torch.arange(6, dtype=torch.float32)
    assert rp.tensor_digest(flat.reshape(2, 3)) != rp.tensor_digest(flat.reshape(3, 2))
    assert rp.tensor_digest(torch.zeros(4, dtype=torch.int32)) != rp.tensor_digest(
        torch.zeros(4, dtype=torch.float32)
    )
    assert rp.tensor_digest(flat) == rp.tensor_digest(flat.clone())


def test_tensor_digest_handles_scalars_empties_bools_and_views() -> None:
    for tensor in (torch.tensor(7), torch.empty(0), torch.tensor([True, False])):
        assert len(rp.tensor_digest(tensor)) == 64
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    # A non-contiguous view hashes as the values it shows, not as its storage.
    assert rp.tensor_digest(base.t()) == rp.tensor_digest(base.t().contiguous().clone())
    assert rp.tensor_digest(base.t()) != rp.tensor_digest(base)
    # NaN payloads and signed zeros are bits like any other.
    assert rp.tensor_digest(torch.tensor([0.0])) != rp.tensor_digest(torch.tensor([-0.0]))
    assert rp.tensor_digest(torch.tensor([float("nan")])) == rp.tensor_digest(
        torch.tensor([float("nan")])
    )


class _NotATensor:
    """A pickled object the tensor-only unpickler refuses."""


def test_a_checkpoint_is_never_loaded_unsafely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not even with the process-wide opt-in: a comparison must not run a file's code."""
    path = tmp_path / "checkpoint_1.pt"
    torch.save({"state_dict": state(1), "extra": _NotATensor()}, path)
    monkeypatch.setenv(UNSAFE_CHECKPOINT_ENV_VAR, "1")
    with pytest.raises(UnsafeCheckpointError):
        rp.checkpoint_digest(path)


def test_a_checkpoint_without_tensors_is_unreadable_not_equal(tmp_path: Path) -> None:
    """Two empty tensor maps would otherwise compare equal."""
    path = tmp_path / "checkpoint_1.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="no tensors"):
        rp.checkpoint_digest(path)
    with pytest.raises(FileNotFoundError):
        rp.checkpoint_digest(tmp_path / "absent.pt")


# ---------------------------------------------------------------------------
# run_digest: the run-card layout
# ---------------------------------------------------------------------------


def test_run_digest_reads_the_run_card_layout(tmp_path: Path) -> None:
    run = build_run(tmp_path / "run")
    # Files a run directory also holds, none of which is a component.
    write_checkpoint(run / rp.CHECKPOINT_DIR_NAME / "latest.pt", state(ITERATIONS))
    (run / rp.ARENA_DIR_NAME / "unfinished").mkdir()
    (run / "run.json").write_text("{}", encoding="utf-8")
    digest = rp.run_digest(run)
    assert list(digest[rp.COMPONENT_CHECKPOINTS]) == [
        "checkpoint_0.pt",
        "checkpoint_1.pt",
        "checkpoint_2.pt",
    ]
    assert list(digest[rp.COMPONENT_ARENAS]) == [ARM]
    assert digest[rp.COMPONENT_ARENAS][ARM]["rows"][0]["games"] == game_records(RESULTS)
    assert list(digest[rp.COMPONENT_ITERATIONS]) == ["1", "2"]
    assert digest[rp.UNREADABLE] == {}
    assert rp.component_counts(digest) == {"checkpoints": 3, "arenas": 1, "iterations": 2}
    assert rp.compared_components(digest) == 6
    assert rp.compare_digests(json.loads(json.dumps(digest)), digest) == [], "plain JSON data"


def test_checkpoints_are_ordered_by_number_not_by_name(tmp_path: Path) -> None:
    for number in (10, 2, 0):
        write_checkpoint(checkpoint_path(tmp_path, number), state(number))
    assert list(rp.run_digest(tmp_path)[rp.COMPONENT_CHECKPOINTS]) == [
        "checkpoint_0.pt",
        "checkpoint_2.pt",
        "checkpoint_10.pt",
    ]


def test_missing_components_are_absent_and_an_empty_run_offers_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    digest = rp.run_digest(empty)
    assert rp.component_counts(digest) == {"checkpoints": 0, "arenas": 0, "iterations": 0}
    assert rp.compared_components(digest) == 0
    only_checkpoint = tmp_path / "partial"
    write_checkpoint(checkpoint_path(only_checkpoint, 0), state(0))
    assert rp.component_counts(rp.run_digest(only_checkpoint)) == {
        "checkpoints": 1,
        "arenas": 0,
        "iterations": 0,
    }
    with pytest.raises(NotADirectoryError):
        rp.run_digest(tmp_path / "absent")


def test_layout_constants_match_the_registered_card_and_the_trainer() -> None:
    """``<D>\\ckpt`` and ``<D>\\arena\\<arm>`` are read from the A1 card, not assumed."""
    from alpha_zero import training_civ6

    card = yaml.safe_load((REPO_ROOT / "configs/run_cards/a1.yaml").read_text(encoding="utf-8"))
    run_dir = PureWindowsPath(run_directory(card["card_id"]))
    trained = 0
    arenas = 0
    for invocation in card["invocations"]:
        argv = invocation["argv"]
        if invocation["runner"] == "training_civ6":
            checkpoint_dir = PureWindowsPath(argv[argv.index("--checkpoint_dir") + 1])
            assert checkpoint_dir == run_dir / rp.CHECKPOINT_DIR_NAME
            trained += 1
        else:
            out = PureWindowsPath(argv[argv.index("--out") + 1])
            assert out.parent == run_dir / rp.ARENA_DIR_NAME, invocation["name"]
            arenas += 1
    assert trained and arenas, "the card has no training command or no arena arm to read"
    assert rp.CHECKPOINT_FILE_RE.pattern == training_civ6._CHECKPOINT_ITERATION_RE.pattern
    assert rp.ARENA_RESULT_FILE == "result.json"


def test_only_the_excluded_fields_are_dropped(tmp_path: Path) -> None:
    record = iteration_record(1)
    write_iterations(tmp_path, [record])
    digested = rp.run_digest(tmp_path)[rp.COMPONENT_ITERATIONS]["1"]
    expected = json.loads(json.dumps(record))
    del expected["started_at"], expected["run_id"], expected["self_play"]["seconds"]
    assert digested == expected
    assert set(rp.ITERATION_RECORD_EXCLUDED_FIELDS) == {
        ("started_at",),
        ("run_id",),
        ("self_play", "seconds"),
    }


def test_the_last_line_for_an_iteration_is_the_one_compared(tmp_path: Path) -> None:
    first, second = iteration_record(1), iteration_record(1)
    second["train"]["total_loss"] = 9.0
    second["supersedes_earlier_record"] = True
    write_iterations(tmp_path, [first, second])
    digested = rp.run_digest(tmp_path)[rp.COMPONENT_ITERATIONS]
    assert list(digested) == ["1"]
    assert digested["1"]["train"]["total_loss"] == 9.0
    assert digested["1"]["supersedes_earlier_record"] is True


# ---------------------------------------------------------------------------
# compare_digests
# ---------------------------------------------------------------------------


def test_identical_runs_reproduce(runs: tuple[Path, Path]) -> None:
    """A faithful re-run differs in run id, clocks, file hashes and metadata -- and passes."""
    attempt, rerun = runs
    assert differences(attempt, rerun) == []
    assert differences(rerun, attempt) == []
    assert rp.compared_components(rp.run_digest(attempt)) == 6


def test_one_changed_tensor_is_named_and_nothing_else_is(runs: tuple[Path, Path]) -> None:
    attempt, rerun = runs
    tensors = state(1)
    tensors["head.weight"][2, 1] += 1e-6
    write_checkpoint(checkpoint_path(rerun, 1), tensors)
    found = differences(attempt, rerun)
    assert len(found) == 1, found
    assert "checkpoint_1.pt" in found[0] and "head.weight" in found[0]
    assert "1 of 5 tensors differ" in found[0]
    assert "conv.weight" not in found[0]


def test_many_differing_tensors_are_counted_and_capped(runs: tuple[Path, Path]) -> None:
    attempt, rerun = runs
    many = {f"layer{i}.weight": torch.full((2,), float(i)) for i in range(9)}
    write_checkpoint(checkpoint_path(attempt, 1), many)
    write_checkpoint(checkpoint_path(rerun, 1), {k: v + 1 for k, v in many.items()})
    (found,) = differences(attempt, rerun)
    assert "9 of 9 tensors differ" in found
    named = [name for name in many if name in found]
    assert len(named) == rp.MAX_NAMED_DIFFERENCES
    assert f"(+{9 - rp.MAX_NAMED_DIFFERENCES} more)" in found


def test_tensors_only_one_side_holds_are_a_difference(runs: tuple[Path, Path]) -> None:
    attempt, rerun = runs
    tensors = state(2)
    tensors["extra.bias"] = torch.zeros(2)
    del tensors["conv.bias"]
    write_checkpoint(checkpoint_path(rerun, 2), tensors)
    found = differences(attempt, rerun)
    assert len(found) == 2, found
    assert any("only in the reference: conv.bias" in d for d in found)
    assert any("only in the candidate: extra.bias" in d for d in found)


def test_one_changed_game_is_named_by_arm_and_index(runs: tuple[Path, Path]) -> None:
    attempt, rerun = runs
    changed = (*RESULTS[:2], "win", *RESULTS[3:])
    assert changed != RESULTS
    write_arena(rerun, ARM, changed, stamp="bb")
    (found,) = differences(attempt, rerun)
    assert f"arena {ARM}" in found
    assert "game index 2" in found and "1 of 4 games differ" in found
    assert '"result": "loss"' in found and '"result": "win"' in found


def _stamp_actions(run: Path, extra_move_in_game: int | None) -> None:
    """Give every game record a length and an action digest, as the arena writes them."""
    path = run / rp.ARENA_DIR_NAME / ARM / rp.ARENA_RESULT_FILE
    envelope = json.loads(path.read_text(encoding="utf-8"))
    for index, game in enumerate(envelope["results"][0]["games"]):
        actions = f"{index},1,2," + ("3," if index == extra_move_in_game else "")
        game["moves"] = actions.count(",")
        game["actions_sha256"] = hashlib.sha256(actions.encode()).hexdigest()
    path.write_text(json.dumps(envelope), encoding="utf-8")


def test_other_moves_with_the_same_outcome_are_a_difference(runs: tuple[Path, Path]) -> None:
    """Win/draw/loss alone cannot see a game that was played differently (PR #138 review).

    Negative control: with the records cut back to ``{pair, seat, result}``, which is
    what the arena wrote before it recorded ``moves`` and ``actions_sha256``, the two
    runs below compare equal.
    """
    attempt, rerun = runs
    _stamp_actions(attempt, None)
    _stamp_actions(rerun, None)
    assert differences(attempt, rerun) == []
    _stamp_actions(rerun, 1)
    (found,) = differences(attempt, rerun)
    assert f"arena {ARM}" in found
    assert "game index 1" in found and "1 of 4 games differ" in found
    assert "actions_sha256" in found and '"moves": 4' in found


def test_arena_shape_differences_are_reported(runs: tuple[Path, Path]) -> None:
    attempt, rerun = runs
    write_arena(rerun, ARM, RESULTS[:2], stamp="bb", seed=SEED + 1)
    found = differences(attempt, rerun)
    assert any("protocol differs" in d and "n_games" in d for d in found), found
    assert any("seed differs" in d for d in found), found
    assert any("4 games, the candidate 2" in d for d in found), found


def test_opponent_file_hashes_are_not_compared_but_opponent_kinds_are(
    runs: tuple[Path, Path],
) -> None:
    attempt, rerun = runs
    result = rerun / rp.ARENA_DIR_NAME / ARM / rp.ARENA_RESULT_FILE
    # Already differing file hashes passed in test_identical_runs_reproduce; a
    # different opponent altogether does not.
    edit_json(result, lambda e: e["results"][0].update(opponent_id="random"))
    (found,) = differences(attempt, rerun)
    assert "the candidate's opponent is random" in found


def test_one_changed_iteration_field_is_named(runs: tuple[Path, Path]) -> None:
    attempt, rerun = runs
    edit_iteration(rerun, 2, lambda r: r["train"].update(total_loss=1.2500001))
    (found,) = differences(attempt, rerun)
    assert "iteration record 2" in found and "train.total_loss" in found
    assert "1.25" in found and "1.2500001" in found
    assert "iteration record 1" not in found


@pytest.mark.parametrize("field", ["git_sha", "seed", "resumed_from", "checkpoint"])
def test_identity_and_result_fields_are_compared(runs: tuple[Path, Path], field: str) -> None:
    """Only the clocks and the run id are exempt: a different commit or seed is a difference."""
    attempt, rerun = runs
    edit_iteration(rerun, 1, lambda r: r.update({field: "changed"}))
    (found,) = differences(attempt, rerun)
    assert "iteration record 1" in found and field in found


def test_wall_clock_fields_alone_are_not_a_difference(runs: tuple[Path, Path]) -> None:
    attempt, rerun = runs

    def retime(record: dict[str, Any]) -> None:
        record["started_at"] = "2027-01-01T00:00:00Z"
        record["run_id"] = "a1-20261111-r2"
        record["self_play"]["seconds"] = 12345.678

    for iteration in range(1, ITERATIONS + 1):
        edit_iteration(rerun, iteration, retime)
    assert differences(attempt, rerun) == []


def test_a_repeated_nan_is_not_a_difference_and_a_type_change_is(runs: tuple[Path, Path]) -> None:
    attempt, rerun = runs
    for run in (attempt, rerun):
        edit_iteration(run, 1, lambda r: r["train"].update(value_loss=float("nan")))
    assert differences(attempt, rerun) == []
    edit_iteration(rerun, 2, lambda r: r["self_play"].update(games=2.0))
    (found,) = differences(attempt, rerun)
    assert "self_play.games" in found


@pytest.mark.parametrize("component", ["checkpoint", "arena", "iteration"])
def test_a_component_missing_from_the_rerun_is_a_difference(
    runs: tuple[Path, Path], component: str
) -> None:
    attempt, rerun = runs
    if component == "checkpoint":
        checkpoint_path(rerun, 1).unlink()
        expected = "checkpoint checkpoint_1.pt: missing from the candidate"
    elif component == "arena":
        (rerun / rp.ARENA_DIR_NAME / ARM / rp.ARENA_RESULT_FILE).unlink()
        expected = f"arena {ARM}: missing from the candidate"
    else:
        write_iterations(rerun, [iteration_record(1, stamp="bb")])
        expected = "iteration record 2: missing from the candidate"
    assert differences(attempt, rerun) == [expected]


def test_what_only_the_rerun_holds_is_ignored(runs: tuple[Path, Path]) -> None:
    """A re-run finishes more than the attempt it reproduces; the reverse is a difference."""
    attempt, rerun = runs
    write_checkpoint(checkpoint_path(rerun, 3), state(3))
    write_arena(rerun, "symmetry", RESULTS, stamp="bb")
    write_iterations(rerun, [iteration_record(i, stamp="bb") for i in range(1, ITERATIONS + 2)])
    assert differences(attempt, rerun) == []
    reverse = differences(rerun, attempt)
    assert len(reverse) == 3, reverse
    assert all("missing from the candidate" in d for d in reverse)


def test_an_unreadable_attempt_checkpoint_is_a_difference(runs: tuple[Path, Path]) -> None:
    """A checkpoint torn by the interruption cannot be shown to reproduce: never a pass."""
    attempt, rerun = runs
    torn = checkpoint_path(attempt, 2)
    torn.write_bytes(torn.read_bytes()[:100])
    digest = rp.run_digest(attempt)
    assert "checkpoint_2.pt" not in digest[rp.COMPONENT_CHECKPOINTS]
    assert list(digest[rp.UNREADABLE]) == ["checkpoints/checkpoint_2.pt"]
    (found,) = rp.compare_digests(digest, rp.run_digest(rerun))
    assert "checkpoints/checkpoint_2.pt" in found and "unreadable in the reference" in found


def test_an_unreadable_rerun_checkpoint_is_a_difference(runs: tuple[Path, Path]) -> None:
    attempt, rerun = runs
    checkpoint_path(rerun, 2).write_bytes(b"not a checkpoint")
    (found,) = differences(attempt, rerun)
    assert "checkpoint checkpoint_2.pt: unreadable in the candidate" in found


@pytest.mark.parametrize("content", ["{not json", "{}", '{"results": [{"opponent_id": "random"}]}'])
def test_an_unparseable_arena_result_is_a_difference(runs: tuple[Path, Path], content: str) -> None:
    """Torn JSON, no rows, and a row without per-game records are all "cannot verify"."""
    attempt, rerun = runs
    (attempt / rp.ARENA_DIR_NAME / ARM / rp.ARENA_RESULT_FILE).write_text(content, "utf-8")
    (found,) = differences(attempt, rerun)
    assert f"arenas/{ARM}" in found and "unreadable in the reference" in found


def test_a_torn_last_record_line_is_skipped_and_the_rest_compared(runs: tuple[Path, Path]) -> None:
    """``iterations.jsonl`` is torn mid-line by an interruption; the complete lines still count."""
    attempt, rerun = runs
    target = attempt / rp.CHECKPOINT_DIR_NAME / ITERATIONS_FILE
    lines = target.read_text(encoding="utf-8").splitlines()
    target.write_text(lines[0] + "\n" + lines[1][:40], encoding="utf-8")
    digest = rp.run_digest(attempt)
    assert list(digest[rp.COMPONENT_ITERATIONS]) == ["1"]
    assert rp.compare_digests(digest, rp.run_digest(rerun)) == []


def test_digests_of_another_schema_version_never_compare(runs: tuple[Path, Path]) -> None:
    attempt, rerun = runs
    old = rp.run_digest(attempt)
    old[rp.SCHEMA_KEY] = rp.DIGEST_SCHEMA_VERSION + 1
    (found,) = rp.compare_digests(old, rp.run_digest(rerun))
    assert "schema versions" in found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_reproduced_exits_0_and_writes_json(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    attempt, rerun = runs
    out = tmp_path / "nested" / "outcome.json"
    assert rp.main(["compare", str(attempt), str(rerun), "--out", str(out)]) == rp.EXIT_REPRODUCED
    assert capsys.readouterr().out.splitlines()[0] == rp.STATUS_REPRODUCED
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == rp.STATUS_REPRODUCED
    assert payload["components_compared"] == 6
    assert payload["components"] == {"checkpoints": 3, "arenas": 1, "iterations": 2}
    assert payload["differences"] == []
    assert payload["attempt_dir"] == str(attempt) and payload["rerun_dir"] == str(rerun)


def test_cli_differs_lists_the_differences_and_exits_1(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    attempt, rerun = runs
    write_checkpoint(checkpoint_path(rerun, 1), state(99))
    out = tmp_path / "outcome.json"
    assert rp.main(["compare", str(attempt), str(rerun), "--out", str(out)]) == rp.EXIT_DIFFERS
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == rp.STATUS_DIFFERS
    assert any("checkpoint_1.pt" in line for line in printed[1:])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == rp.STATUS_DIFFERS and len(payload["differences"]) == 1


def test_cli_nothing_to_compare_exits_2(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An attempt that left nothing behind is not reproduced by anything."""
    _, rerun = runs
    empty = tmp_path / "left-nothing"
    empty.mkdir()
    out = tmp_path / "outcome.json"
    code = rp.main(["compare", str(empty), str(rerun), "--out", str(out)])
    assert code == rp.EXIT_NOTHING_TO_COMPARE == 2
    assert capsys.readouterr().out.splitlines()[0] == rp.STATUS_NOTHING_TO_COMPARE
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == rp.STATUS_NOTHING_TO_COMPARE
    assert payload["components_compared"] == 0


def test_cli_missing_directory_exits_2(
    runs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    attempt, _ = runs
    code = rp.main(["compare", str(attempt), str(tmp_path / "typo")])
    assert code == rp.EXIT_NOTHING_TO_COMPARE
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0] == rp.STATUS_NOTHING_TO_COMPARE
    assert "typo" in captured.err


def test_cli_runs_as_a_module_and_propagates_its_status(runs: tuple[Path, Path]) -> None:
    """``python -m alpha_zero.evaluation.reproduction compare`` is the documented entry point."""
    attempt, rerun = runs
    checkpoint_path(rerun, 0).unlink()
    completed = subprocess.run(
        [sys.executable, "-m", rp.__name__, "compare", str(attempt), str(rerun)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == rp.EXIT_DIFFERS, completed.stderr[-2000:]
    assert completed.stdout.splitlines()[0] == rp.STATUS_DIFFERS
    assert "checkpoint_0.pt: missing from the candidate" in completed.stdout
