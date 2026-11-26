"""Unit tests for helpers extracted during PLR0915 decomposition of cloud entrypoints."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.training_worker import (
    _init_optional_training_tools,
    _maybe_resume_from_checkpoint,
)
from alpha_zero.cloud.gcp.vertex_orchestrator import (
    _apply_architecture_preset_to_args,
    _check_circuit_breaker,
    _detect_learning_stall,
)


class TestVertexOrchestratorExtractedHelpers:
    """Verify helper behavior preserved from vertex_orchestrator.main() split."""

    def test_apply_architecture_preset_to_args_noop_and_populated(self) -> None:
        args_none = Namespace(architecture_preset=None, num_res_blocks=4)
        _apply_architecture_preset_to_args(args_none)
        assert args_none.num_res_blocks == 4

        args_small = Namespace(architecture_preset="small")
        _apply_architecture_preset_to_args(args_small)
        assert args_small.num_res_blocks > 0
        assert args_small.num_filters > 0
        assert args_small.num_heads > 0

    def test_check_circuit_breaker_threshold_and_alert(self) -> None:
        tracker = MagicMock()
        tracker.is_enabled = True

        assert _check_circuit_breaker(2, 3, 4, tracker) is False
        assert _check_circuit_breaker(5, 0, 4, tracker) is False
        tracker.alert.assert_not_called()

        assert _check_circuit_breaker(3, 3, 4, tracker, phase_label="eval") is True
        tracker.alert.assert_called_once()
        alert_kwargs = tracker.alert.call_args.kwargs
        assert "(eval)" in alert_kwargs["text"]
        assert "iteration 4" in alert_kwargs["text"]

    def test_detect_learning_stall_skips_without_eval_and_alerts_on_stall(self) -> None:
        tracker = MagicMock()
        tracker.is_enabled = True
        history: list[float] = []

        stalled, trend = _detect_learning_stall(
            Namespace(evaluation_games=0, stall_threshold=0.01, stall_lookback=3),
            iteration=1,
            model_version=1,
            win_rate=0.5,
            win_rate_history=history,
            wandb_logger=tracker,
        )
        assert (stalled, trend) == (False, 0.0)
        assert history == []

        mock_diag = MagicMock()
        mock_diag.detect_learning_stall.return_value = True
        mock_diag.compute_win_rate_trend.return_value = -0.02
        with patch(
            "alpha_zero.training.diagnostics.TrainingDiagnostics",
            return_value=mock_diag,
        ):
            stalled, trend = _detect_learning_stall(
                Namespace(evaluation_games=20, stall_threshold=0.01, stall_lookback=2),
                iteration=3,
                model_version=2,
                win_rate=0.45,
                win_rate_history=history,
                wandb_logger=tracker,
            )
        assert stalled is True
        assert trend == pytest.approx(-0.02)
        assert history == [0.45]
        tracker.alert.assert_called_once()


class TestTrainingWorkerExtractedHelpers:
    """Verify helper behavior preserved from training_worker.main() split."""

    @pytest.mark.parametrize(
        ("resume_from", "expected_step"),
        [
            ("42", 42),
            ("gs://bucket/training_step_150.pt", 150),
            ("gs://bucket/training_step_notanint.pt", None),
        ],
    )
    def test_maybe_resume_from_checkpoint_parses_steps(
        self, resume_from: str, expected_step: int | None
    ) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        args = Namespace(resume_from=resume_from, dry_run=False)
        config = GCPConfig()

        mock_storage = MagicMock()
        mock_storage.load_checkpoint.return_value = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": 99,
            "epoch": 4,
        }
        with patch(
            "alpha_zero.cloud.gcp.training_worker.GCSCheckpointStorage",
            return_value=mock_storage,
        ):
            epoch, step = _maybe_resume_from_checkpoint(args, config, model, optimizer, True)

        mock_storage.load_checkpoint.assert_called_once_with(step=expected_step)
        assert (epoch, step) == (4, 99)

    def test_init_optional_training_tools_cosine_fallback_decay(self) -> None:
        args = Namespace(
            lr_schedule_type="cosine",
            lr_initial=0.002,
            lr_final=0.0001,
            lr_warmup_steps=5,
            lr_decay_steps=0,
            epochs=3,
            diagnostic_window=10,
        )
        lr_scheduler, diagnostics, report, profiler = _init_optional_training_tools(
            args, num_batches=7, is_main_process=True
        )
        assert lr_scheduler is not None
        assert diagnostics is not None
        assert report is not None
        assert profiler is not None
