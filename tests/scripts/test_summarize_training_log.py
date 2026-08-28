"""Tests for scripts/summarize_training_log.py.

The summarizer exists so a local training run leaves committable evidence
behind. These tests pin the parsing contract against a realistic log, including
the partial-iteration case, since a killed run is exactly when the artifact
matters most.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "summarize_training_log.py"
_spec = importlib.util.spec_from_file_location("summarize_training_log", _SCRIPT)
assert _spec and _spec.loader
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)


SAMPLE_LOG = """\
2026-08-28 02:13:50,275 [INFO] Network parameters: 570,367
2026-08-28 02:13:50,275 [INFO] Action space size: 376
2026-08-28 02:13:50,275 [INFO] Players: 2 | spatial channels: 13 | value head: scalar | scoring: winner_take_all
2026-08-28 02:13:50,275 [INFO] Device: cpu
2026-08-28 02:13:51,437 [INFO] Iteration 1/2
Self-play:   0%|          | 0/20 [00:00<?, ?it/s]\r2026-08-28 02:15:59,027 [INFO] Self-play: 2568 examples from 20 games (127.6s)
2026-08-28 02:16:00,000 [INFO] Epoch 1: loss=3.1416, policy=2.7183, value=0.4233
2026-08-28 02:16:45,154 [INFO] Training complete (46.1s)
2026-08-28 02:17:06,770 [INFO] Win rate vs random: 5.0%
2026-08-28 02:17:06,777 [INFO] Iteration 2/2
2026-08-28 02:19:14,637 [INFO] Self-play: 2555 examples from 20 games (127.9s)
2026-08-28 02:20:00,000 [INFO] Epoch 1: loss=2.5000, policy=2.0000, value=0.5000
2026-08-28 02:20:01,000 [INFO] Training complete (44.0s)
2026-08-28 02:20:30,000 [INFO] Win rate vs random: 55.0%
EXIT=0

real\t6m40.0s
"""


class TestParseLog:
    @pytest.fixture
    def parsed(self) -> dict:
        return summarize.parse_log(SAMPLE_LOG)

    def test_header_fields(self, parsed: dict) -> None:
        assert parsed["network_parameters"] == 570367  # comma stripped
        assert parsed["action_space_size"] == 376
        assert parsed["device"] == "cpu"
        assert parsed["num_players"] == 2
        assert parsed["spatial_channels"] == 13
        assert parsed["scoring_method"] == "winner_take_all"

    def test_both_iterations_parsed(self, parsed: dict) -> None:
        assert [i["iteration"] for i in parsed["iterations"]] == [1, 2]

    def test_selfplay_metrics(self, parsed: dict) -> None:
        first = parsed["iterations"][0]
        assert first["selfplay_examples"] == 2568
        assert first["selfplay_games"] == 20
        assert first["selfplay_seconds"] == pytest.approx(127.6)

    def test_tqdm_carriage_returns_do_not_hide_a_line(self, parsed: dict) -> None:
        # The self-play line is emitted after a \r-terminated progress bar.
        assert "selfplay_examples" in parsed["iterations"][0]

    def test_win_rate_is_stored_as_a_fraction(self, parsed: dict) -> None:
        assert parsed["iterations"][0]["win_rate_vs_random"] == pytest.approx(0.05)
        assert parsed["iterations"][1]["win_rate_vs_random"] == pytest.approx(0.55)

    def test_epoch_losses(self, parsed: dict) -> None:
        epochs = parsed["iterations"][0]["epochs"]
        assert epochs[0]["total_loss"] == pytest.approx(3.1416)
        assert epochs[0]["value_loss"] == pytest.approx(0.4233)

    def test_exit_code_and_wall_clock(self, parsed: dict) -> None:
        assert parsed["exit_code"] == 0
        assert parsed["wall_clock_seconds"] == pytest.approx(400.0)

    def test_partial_final_iteration_is_kept(self) -> None:
        # A run killed mid-iteration must still report what it got, rather than
        # silently dropping the incomplete entry.
        truncated = SAMPLE_LOG.split("2026-08-28 02:19:14,637")[0]
        parsed = summarize.parse_log(truncated)
        assert len(parsed["iterations"]) == 2
        assert "selfplay_examples" not in parsed["iterations"][1]
        assert parsed["exit_code"] is None


class TestConfigParsing:
    def test_numbers_are_coerced(self) -> None:
        cfg = summarize._parse_config(["board_size=8", "lr=0.001", "mode=fast"])
        assert cfg == {"board_size": 8, "lr": 0.001, "mode": "fast"}

    def test_empty_key_is_ignored(self) -> None:
        assert summarize._parse_config(["=8"]) == {}


class TestCli:
    def test_writes_both_artifacts(self, tmp_path: Path) -> None:
        log = tmp_path / "run.log"
        log.write_text(SAMPLE_LOG)
        out = tmp_path / "evidence"
        rc = summarize.main(
            [str(log), "--out", str(out), "--title", "T", "--config", "board_size=8"]
        )
        assert rc == 0
        data = json.loads((out / "run_result.json").read_text())
        assert data["provenance"]["config"] == {"board_size": 8}
        assert data["provenance"]["python_version"]
        report = (out / "run_report.md").read_text()
        assert "# T" in report
        # The convention requires this section; a report without it is not evidence.
        assert "## Scope and limitations" in report

    def test_missing_log_is_an_error(self, tmp_path: Path) -> None:
        assert summarize.main([str(tmp_path / "nope.log"), "--out", str(tmp_path / "o")]) == 1

    def test_log_without_iterations_is_an_error(self, tmp_path: Path) -> None:
        log = tmp_path / "empty.log"
        log.write_text("nothing useful here\n")
        assert summarize.main([str(log), "--out", str(tmp_path / "o")]) == 1
