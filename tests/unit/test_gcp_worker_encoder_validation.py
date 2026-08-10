"""Tests for validate_encoder_state wiring in GCP workers.

Verifies that the evaluation and selfplay workers call
``validate_encoder_state`` after ``load_state_dict(strict=False)``
to detect silently cross-loaded encoder weights.

PR #107 follow-up: the validator was created but not wired into
the production load paths. These tests pin the wiring contract.
"""

from __future__ import annotations

import logging

import pytest
import torch

from alpha_zero.utils.checkpoint import (
    EncoderMismatchError,
    validate_encoder_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _DummyModel(torch.nn.Module):
    """Minimal model with a patch_embed-style encoder for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.ModuleDict(
            {
                "patch_embed": torch.nn.ModuleDict(
                    {
                        "proj": torch.nn.Linear(4, 4),
                    }
                ),
            }
        )
        self.value_head = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(x)


class _DummyHexModel(torch.nn.Module):
    """Minimal model with hex_crystalline-style encoder keys."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.ModuleDict(
            {
                "hex_conv": torch.nn.ModuleDict(
                    {
                        "hex_kernel": torch.nn.Linear(4, 4),
                        "crystalline_lattice": torch.nn.Linear(4, 4),
                    }
                ),
            }
        )
        self.value_head = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(x)


# ---------------------------------------------------------------------------
# Tests: validate_encoder_state contract
# ---------------------------------------------------------------------------


class TestValidateEncoderStateContract:
    """Test the validate_encoder_state function directly."""

    def test_matched_checkpoint_passes_silently(self) -> None:
        """When model and checkpoint agree, no error or warning."""
        model = _DummyModel()
        keys = list(model.state_dict().keys())
        assert validate_encoder_state(model, "patch_embed", checkpoint_keys=keys, strict=True)

    def test_cross_loaded_checkpoint_raises_strict(self) -> None:
        """A hex checkpoint loaded into a patch_embed model raises on strict."""
        model = _DummyModel()
        hex_model = _DummyHexModel()
        hex_keys = list(hex_model.state_dict().keys())

        with pytest.raises(EncoderMismatchError):
            validate_encoder_state(model, "hex_crystalline", checkpoint_keys=hex_keys, strict=True)

    def test_cross_loaded_checkpoint_warns_nonstrict(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A hex checkpoint loaded into a patch_embed model logs error, returns False."""
        model = _DummyModel()
        hex_model = _DummyHexModel()
        hex_keys = list(hex_model.state_dict().keys())

        with caplog.at_level(logging.ERROR):
            result = validate_encoder_state(
                model, "hex_crystalline", checkpoint_keys=hex_keys, strict=False
            )
        assert result is False

    def test_no_checkpoint_keys_degrades_gracefully(self, caplog: pytest.LogCaptureFixture) -> None:
        """Without checkpoint_keys, only build-time assertion runs."""
        model = _DummyModel()
        with caplog.at_level(logging.DEBUG):
            result = validate_encoder_state(model, "patch_embed", strict=False)
        assert result is True
        assert "without checkpoint_keys" in caplog.text


# ---------------------------------------------------------------------------
# Tests: GCP worker wiring
# ---------------------------------------------------------------------------


class TestEvaluationWorkerEncoderValidation:
    """Verify that evaluation_worker.load_model calls validate_encoder_state."""

    def test_load_model_calls_validate(self) -> None:
        """The validate_encoder_state call must happen after strict=False load."""
        # We can't easily call load_model without GCS, but we can verify the
        # import and function exist in the module source.
        import inspect

        import alpha_zero.cloud.gcp.evaluation_worker as ew

        source = inspect.getsource(ew.load_model)

        assert "validate_encoder_state" in source, (
            "evaluation_worker.load_model must call validate_encoder_state after strict=False load"
        )
        assert "classify_encoder_keys" in source, (
            "evaluation_worker.load_model must use classify_encoder_keys "
            "to infer encoder type from checkpoint"
        )


class TestSelfplayWorkerEncoderValidation:
    """Verify that selfplay_worker load functions call validate_encoder_state."""

    def test_load_model_contains_validate(self) -> None:
        """The Transformer load path must call validate_encoder_state."""
        import inspect

        import alpha_zero.cloud.gcp.selfplay_worker as sw

        source = inspect.getsource(sw.load_model)

        assert "validate_encoder_state" in source, (
            "selfplay_worker.load_model must call validate_encoder_state after strict=False load"
        )

    def test_load_muzero_model_contains_validate(self) -> None:
        """The MuZero load path must call validate_encoder_state."""
        import inspect

        import alpha_zero.cloud.gcp.selfplay_worker as sw

        source = inspect.getsource(sw.load_muzero_model)

        assert "validate_encoder_state" in source, (
            "selfplay_worker.load_muzero_model must call validate_encoder_state "
            "after strict=False load"
        )
        assert "classify_encoder_keys" in source, (
            "selfplay_worker.load_muzero_model must use classify_encoder_keys "
            "to infer encoder type from checkpoint"
        )
