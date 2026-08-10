"""Tests for encoder-identity validation on loaded checkpoints.

Several load paths in this repository call ``load_state_dict(strict=False)``,
which turns an architecture mismatch into a warning. A hex checkpoint loaded
into a patch-embed network therefore produces a model whose encoder is randomly
initialised — it still runs, still plays, and still reports plausible-looking
metrics, so the failure is invisible exactly where it matters most: during an
evaluation whose whole purpose is comparing encoders.

``validate_encoder_state`` closes that hole by checking the weights themselves
rather than trusting metadata.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from alpha_zero.cloud.gcp.training_worker import create_model
from alpha_zero.utils.checkpoint import (
    EncoderMismatchError,
    classify_encoder_keys,
    validate_encoder_state,
)


def _model(encoder_type: str) -> nn.Module:
    return create_model(board_size=8, encoder_type=encoder_type, num_res_blocks=2, num_heads=4)


@pytest.mark.parametrize("encoder_type", ["patch_embed", "hex_crystalline"])
def test_matching_encoder_passes(encoder_type: str) -> None:
    """A model validated against its own encoder type is accepted."""
    assert validate_encoder_state(_model(encoder_type), encoder_type) is True


def test_hex_weights_rejected_as_patch_embed() -> None:
    """The mismatch that silently degrades an evaluation is caught."""
    with pytest.raises(EncoderMismatchError) as excinfo:
        validate_encoder_state(_model("hex_crystalline"), "patch_embed")

    message = str(excinfo.value)
    assert "patch_embed" in message
    assert "hex_crystalline" in message


def test_patch_embed_weights_rejected_as_hex() -> None:
    """The reverse mismatch is caught too."""
    with pytest.raises(EncoderMismatchError) as excinfo:
        validate_encoder_state(_model("patch_embed"), "hex_crystalline")

    message = str(excinfo.value)
    assert "hex_crystalline" in message
    assert "patch_embed" in message


def test_error_message_explains_the_consequence() -> None:
    """The message has to say why this is fatal, not just that it happened."""
    with pytest.raises(EncoderMismatchError, match="randomly initialised"):
        validate_encoder_state(_model("patch_embed"), "hex_crystalline")


def test_non_strict_mode_reports_instead_of_raising() -> None:
    """Callers that want to decide for themselves get a boolean."""
    assert validate_encoder_state(_model("patch_embed"), "hex_crystalline", strict=False) is False


def test_non_strict_mode_logs_an_error(caplog: pytest.LogCaptureFixture) -> None:
    """A tolerated mismatch is still loud in the logs."""
    with caplog.at_level("ERROR"):
        validate_encoder_state(_model("patch_embed"), "hex_crystalline", strict=False)

    assert "Encoder mismatch" in caplog.text


def test_unknown_encoder_type_is_a_programming_error() -> None:
    """A typo in the expected type must not silently pass validation."""
    with pytest.raises(ValueError, match="Unknown encoder type"):
        validate_encoder_state(_model("patch_embed"), "no_such_encoder")


def test_validation_works_on_a_bare_encoder_module() -> None:
    """Validation is structural, so it also works outside the full network."""
    from alpha_zero.hexcrystal import HexEncoderConfig, HexSpatialEncoder

    encoder = HexSpatialEncoder(HexEncoderConfig(in_channels=13, spatial_size=8))

    assert validate_encoder_state(encoder, "hex_crystalline") is True


class TestCheckpointAwareValidation:
    """The failure the docstring promises to catch: a lenient cross-load.

    ``load_state_dict(strict=False)`` leaves the target's encoder randomly
    initialised and returns normally. Inspecting only the model cannot see this
    — ``state_dict()`` keys describe the module tree that was *built*, not what
    was loaded into it — so the checkpoint's keys have to be part of the check.
    """

    def test_lenient_cross_load_is_detected(self) -> None:
        target = _model("patch_embed")
        source = _model("hex_crystalline")

        incompatible = target.load_state_dict(source.state_dict(), strict=False)
        assert incompatible.missing_keys, "test setup no longer produces a partial load"

        with pytest.raises(EncoderMismatchError, match="randomly initialised"):
            validate_encoder_state(
                target, "patch_embed", checkpoint_keys=source.state_dict().keys()
            )

    def test_model_only_check_cannot_see_it(self) -> None:
        """Documents the limitation that motivates ``checkpoint_keys``.

        Without the checkpoint the call is a build-time assertion, and this test
        pins that so the weaker form is never mistaken for the real check.
        """
        target = _model("patch_embed")
        target.load_state_dict(_model("hex_crystalline").state_dict(), strict=False)

        assert validate_encoder_state(target, "patch_embed") is True

    def test_matching_checkpoint_passes(self) -> None:
        model = _model("hex_crystalline")

        assert (
            validate_encoder_state(
                model, "hex_crystalline", checkpoint_keys=model.state_dict().keys()
            )
            is True
        )

    def test_error_names_the_checkpoint_encoder(self) -> None:
        target = _model("patch_embed")
        source = _model("hex_crystalline")

        with pytest.raises(EncoderMismatchError) as excinfo:
            validate_encoder_state(
                target, "patch_embed", checkpoint_keys=source.state_dict().keys()
            )

        assert "hex_crystalline" in str(excinfo.value)


class TestKeyClassification:
    """Classification is scoped to the encoder subtree, not the whole model."""

    def test_classifies_each_encoder(self) -> None:
        assert classify_encoder_keys(_model("patch_embed").state_dict()) == "patch_embed"
        assert classify_encoder_keys(_model("hex_crystalline").state_dict()) == "hex_crystalline"

    def test_unrelated_module_named_like_the_encoder_is_ignored(self) -> None:
        """A future ``*crystalline*`` module elsewhere must not false-positive."""
        keys = [
            "encoder.patch_embed.proj.weight",
            "some_other_crystalline_thing.weight",
        ]

        assert classify_encoder_keys(keys) == "patch_embed"

    def test_bare_encoder_state_dict_is_classified(self) -> None:
        """Keys with no network prefix are all treated as encoder keys."""
        assert classify_encoder_keys(["hex_conv_stack.0.conv.weight"]) == "hex_crystalline"
        assert classify_encoder_keys(["proj.weight"]) == "patch_embed"
