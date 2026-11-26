"""``iterations.jsonl``: run identity, rotation, resume marking, crash safety (T10 review I2).

Before this, every record was appended to one file with no run identity:

* a second fresh run into the same directory (the default ``checkpoints/civ6``)
  appended its ``1, 2, ...`` after the first run's;
* a resume appended the resumed iterations a second time, indistinguishably;
* a shorter re-run left the earlier run's higher-numbered ``eval_<k>.json`` beside
  its own.

Also covered here: the fail-soft prequential path (M1) and the torn-final-line
reader rule (M8).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from alpha_zero import training_civ6
from alpha_zero.evaluation.scoring import ScoreSummary, summarize
from alpha_zero.training import run_record
from alpha_zero.training.run_record import ITERATIONS_FILE, IterationLog, read_iteration_records
from alpha_zero.training_civ6 import TrainingConfig

IDENTITY_KEYS = ("run_id", "seed", "git_sha", "started_at", "resumed_from", "start_iteration")


def _config(tmp_path: Path, **overrides: Any) -> TrainingConfig:
    base: dict[str, Any] = {
        "board_size": 4,
        "max_turns": 3,
        "num_simulations": 2,
        "num_res_blocks": 1,
        "num_filters": 8,
        "num_games_per_iteration": 1,
        "num_iterations": 2,
        "min_replay_size": 10**9,
        "save_interval": 1,
        "eval_interval": 1,
        "eval_games": 1,
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "log_dir": str(tmp_path / "log"),
        "seed": 1,
        "device": "cpu",
    }
    base.update(overrides)
    return TrainingConfig(**base)


def _summary(*_a: Any, **_k: Any) -> ScoreSummary:
    return summarize(1, 0, 0)


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real games: empty self-play and a canned evaluation."""
    monkeypatch.setattr(training_civ6, "run_self_play", lambda *a, **k: [])
    monkeypatch.setattr(training_civ6, "evaluate_against_random", _summary)


def _records(ckpt: Path) -> list[dict[str, Any]]:
    return read_iteration_records(ckpt / ITERATIONS_FILE)


class TestFreshRuns:
    def test_every_record_carries_the_run_identity(self, tmp_path: Path, fast: None) -> None:
        config = _config(tmp_path, run_id="run-a")
        training_civ6.train(config)
        records = _records(Path(config.checkpoint_dir))
        assert [r["iteration"] for r in records] == [1, 2]
        for record in records:
            for key in IDENTITY_KEYS:
                assert key in record, key
            assert record["run_id"] == "run-a"
            assert record["seed"] == 1
            assert record["resumed_from"] is None
            assert record["start_iteration"] == 0
        assert len({r["started_at"] for r in records}) == 1

    def test_second_fresh_run_rotates_the_first_aside(self, tmp_path: Path, fast: None) -> None:
        first = _config(tmp_path, run_id="run-a", num_iterations=3)
        training_civ6.train(first)
        ckpt = Path(first.checkpoint_dir)
        assert {p.name for p in ckpt.glob("eval_*.json")} == {
            "eval_1.json",
            "eval_2.json",
            "eval_3.json",
        }

        training_civ6.train(_config(tmp_path, run_id="run-b", num_iterations=1))

        # The live directory holds only run b: its record and its own eval file.
        assert [(r["run_id"], r["iteration"]) for r in _records(ckpt)] == [("run-b", 1)]
        assert {p.name for p in ckpt.glob("eval_*.json")} == {"eval_1.json"}
        # Run a is kept, never deleted, in one previous_<timestamp>/ directory.
        (previous,) = sorted(ckpt.glob("previous_*"))
        assert [r["run_id"] for r in _records(previous)] == ["run-a"] * 3
        assert {p.name for p in previous.glob("eval_*.json")} == {
            "eval_1.json",
            "eval_2.json",
            "eval_3.json",
        }

    def test_rotation_never_overwrites_an_earlier_rotation(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        for n in range(2):
            (ckpt / ITERATIONS_FILE).write_text(json.dumps({"n": n}) + "\n", "utf-8")
            run_record.rotate_previous_run(ckpt, stamp="20261111T000000Z")
        rotated = sorted(ckpt.glob("previous_*"))
        assert len(rotated) == 2
        assert sorted(_records(p)[0]["n"] for p in rotated) == [0, 1]


class TestResume:
    def test_resumed_iterations_are_marked_not_silently_duplicated(
        self, tmp_path: Path, fast: None
    ) -> None:
        first = _config(tmp_path, run_id="run-a", num_iterations=3)
        training_civ6.train(first)
        ckpt = Path(first.checkpoint_dir)

        resumed = _config(
            tmp_path,
            run_id="run-a2",
            num_iterations=3,
            resume_from=str(ckpt / "checkpoint_1.pt"),
        )
        training_civ6.train(resumed)

        records = _records(ckpt)
        assert not list(ckpt.glob("previous_*")), "a resume must not rotate the run aside"
        assert [r["iteration"] for r in records] == [1, 2, 3, 2, 3]
        original, again = records[:3], records[3:]
        assert all(r["resumed_from"] is None for r in original)
        assert not any(r.get("supersedes_earlier_record") for r in original)
        for record in again:
            assert record["resumed_from"] == str(ckpt / "checkpoint_1.pt")
            assert record["start_iteration"] == 1
            assert record["supersedes_earlier_record"] is True
        # The two processes partition cleanly.
        assert {r["run_id"] for r in original} == {"run-a"}
        assert {r["run_id"] for r in again} == {"run-a2"}


class TestPrequentialFailures:
    def test_fail_soft_path_records_logs_error_once_and_summarises(
        self, tmp_path: Path, fast: None, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        def boom(*_a: Any, **_k: Any) -> dict[str, Any]:
            raise RuntimeError("diagnostic exploded")

        monkeypatch.setattr(training_civ6, "prequential_value_metrics", boom)
        config = _config(tmp_path)
        with caplog.at_level(logging.INFO, logger=training_civ6.__name__):
            training_civ6.train(config)  # completes: recorded, not gated

        records = _records(Path(config.checkpoint_dir))
        assert all("diagnostic exploded" in r["prequential"]["error"] for r in records)
        failures = [r for r in caplog.records if "Prequential diagnostics failed" in r.message]
        assert [r.levelno for r in failures] == [logging.ERROR, logging.WARNING]
        assert "prequential failures: 2/2" in caplog.text

    def test_clean_run_summarises_zero_failures(
        self, tmp_path: Path, fast: None, caplog: Any
    ) -> None:
        with caplog.at_level(logging.INFO, logger=training_civ6.__name__):
            training_civ6.train(_config(tmp_path))
        assert "prequential failures: 0/2" in caplog.text


class TestCrashSafety:
    def test_reader_skips_a_torn_final_line(self, tmp_path: Path) -> None:
        path = tmp_path / ITERATIONS_FILE
        path.write_text(json.dumps({"iteration": 1}) + "\n" + '{"iteration": 2, "tr', "utf-8")
        assert read_iteration_records(path) == [{"iteration": 1}]

    def test_each_append_is_one_write_then_flush_and_fsync(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synced: list[int] = []
        monkeypatch.setattr(run_record.os, "fsync", lambda fd: synced.append(fd))
        log = IterationLog(tmp_path, identity={"run_id": "r"}, fresh=True)
        log.append({"iteration": 1})
        log.append({"iteration": 2})
        assert len(synced) == 2
        assert [r["iteration"] for r in _records(tmp_path)] == [1, 2]

    def test_a_resume_after_a_torn_line_still_marks_duplicates(self, tmp_path: Path) -> None:
        path = tmp_path / ITERATIONS_FILE
        path.write_text(json.dumps({"iteration": 1}) + "\n" + '{"iter', "utf-8")
        log = IterationLog(tmp_path, identity={}, fresh=False)
        log.append({"iteration": 1})
        log.append({"iteration": 2})
        appended = _records(tmp_path)[-2:]
        assert appended[0]["supersedes_earlier_record"] is True
        assert "supersedes_earlier_record" not in appended[1]
