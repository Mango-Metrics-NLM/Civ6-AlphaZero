"""Tests for the Hugging Face upload tool.

Publishing is irreversible in practice ΓÇö once weights are publicly fetched you
cannot unring that bell ΓÇö so the safety properties matter more than the happy
path and are tested first: nothing uploads without an explicit flag, nothing
uploads without a credential, and nothing pickled ever ships.

Every test injects a fake Hub API. No test touches the network; a test that did
would be a bug in the tool, and ``test_dry_run_makes_no_api_calls`` is written to
catch exactly that.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("huggingface_hub", reason="requires the 'hub' extra")
pytest.importorskip("safetensors", reason="requires the 'hub' extra")

import torch

from alpha_zero.cloud.gcp.training_worker import create_model

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "upload_encoder_to_hub.py"

_spec = importlib.util.spec_from_file_location("upload_encoder_to_hub", _SCRIPT)
assert _spec is not None and _spec.loader is not None
uploader = importlib.util.module_from_spec(_spec)
sys.modules["upload_encoder_to_hub"] = uploader
_spec.loader.exec_module(uploader)


def _checkpoint(tmp_path: Path, encoder_type: str) -> Path:
    """Write a minimal but genuine checkpoint for the given encoder."""
    import dataclasses
    from enum import Enum

    model = create_model(board_size=8, encoder_type=encoder_type, num_res_blocks=2, num_heads=4)
    raw = dataclasses.asdict(model.config)
    model_config = {k: (v.value if isinstance(v, Enum) else v) for k, v in raw.items()}
    model_config["encoder_type"] = encoder_type

    path = tmp_path / f"{encoder_type}.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": model_config}, path)
    return path


@pytest.fixture
def hex_checkpoint(tmp_path: Path) -> Path:
    return _checkpoint(tmp_path, "hex_crystalline")


@pytest.fixture
def ready_card(tmp_path: Path) -> Path:
    """A model card with no unsubstituted evidence placeholder.

    The repository's real card intentionally carries the placeholder until an
    A/B run fills it, and ``verify_staged`` refuses to publish that. These tests
    are about the tool's behaviour, not the card's current state, so they supply
    their own.
    """
    card = tmp_path / "ready_card.md"
    card.write_text("# Hex encoder\n\nResults: 100 games, 50% score rate.\n")
    return card


@pytest.fixture
def fake_api() -> MagicMock:
    """A stand-in Hub API that records calls and reaches no network."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class TestPublicationSafety:
    """Uploading must be deliberate, credentialed, and never pickled."""

    def test_dry_run_is_the_default(
        self, hex_checkpoint: Path, ready_card: Path, tmp_path: Path, fake_api: MagicMock
    ) -> None:
        exit_code = uploader.main(
            [
                "--checkpoint",
                str(hex_checkpoint),
                "--repo-id",
                "someone/whatever",
                "--staging-dir",
                str(tmp_path / "staging"),
                "--model-card",
                str(ready_card),
            ],
            api=fake_api,
        )

        assert exit_code == 0
        assert (tmp_path / "staging" / "model.safetensors").exists()

    def test_dry_run_makes_no_api_calls(
        self, hex_checkpoint: Path, ready_card: Path, tmp_path: Path, fake_api: MagicMock
    ) -> None:
        """The absence of the flag must mean the absence of network effects."""
        uploader.main(
            [
                "--checkpoint",
                str(hex_checkpoint),
                "--repo-id",
                "someone/whatever",
                "--staging-dir",
                str(tmp_path / "staging"),
                "--model-card",
                str(ready_card),
            ],
            api=fake_api,
        )

        assert fake_api.create_repo.call_count == 0
        assert fake_api.upload_folder.call_count == 0

    def test_publish_without_a_token_fails_before_staging(
        self,
        hex_checkpoint: Path,
        ready_card: Path,
        tmp_path: Path,
        fake_api: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A missing credential is caught up front, not after doing work."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        exit_code = uploader.main(
            [
                "--checkpoint",
                str(hex_checkpoint),
                "--repo-id",
                "someone/whatever",
                "--staging-dir",
                str(tmp_path / "staging"),
                "--model-card",
                str(ready_card),
                "--publish",
            ],
            api=fake_api,
        )

        assert exit_code == 1
        assert fake_api.upload_folder.call_count == 0
        assert not (tmp_path / "staging").exists(), "staging ran despite the missing token"

    def test_publish_with_a_token_uploads_once(
        self,
        hex_checkpoint: Path,
        ready_card: Path,
        tmp_path: Path,
        fake_api: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "test-token")

        exit_code = uploader.main(
            [
                "--checkpoint",
                str(hex_checkpoint),
                "--repo-id",
                "someone/whatever",
                "--staging-dir",
                str(tmp_path / "staging"),
                "--model-card",
                str(ready_card),
                "--publish",
            ],
            api=fake_api,
        )

        assert exit_code == 0
        assert fake_api.create_repo.call_count == 1
        assert fake_api.upload_folder.call_count == 1

    def test_repo_creation_is_idempotent(
        self,
        hex_checkpoint: Path,
        ready_card: Path,
        tmp_path: Path,
        fake_api: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-running updates in place instead of failing on an existing repo."""
        monkeypatch.setenv("HF_TOKEN", "test-token")

        uploader.main(
            [
                "--checkpoint",
                str(hex_checkpoint),
                "--repo-id",
                "someone/whatever",
                "--staging-dir",
                str(tmp_path / "staging"),
                "--model-card",
                str(ready_card),
                "--publish",
            ],
            api=fake_api,
        )

        assert fake_api.create_repo.call_args.kwargs["exist_ok"] is True

    def test_pickled_weights_are_refused(self, tmp_path: Path) -> None:
        """A ``.pt`` in the payload aborts the upload.

        Loading a pickle executes arbitrary code on the consumer's machine, so
        this is a hard refusal rather than a warning.
        """
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "model.safetensors").write_bytes(b"x")
        (staging / "config.json").write_text('{"embed_dim": 64}')
        (staging / "README.md").write_text("card")
        (staging / "model.pt").write_bytes(b"pickle")

        with pytest.raises(uploader.UploadError, match="pickled"):
            uploader.verify_staged(staging)

    def test_incomplete_payload_is_refused(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "model.safetensors").write_bytes(b"x")

        with pytest.raises(uploader.UploadError, match="missing"):
            uploader.verify_staged(staging)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestEncoderExtraction:
    """Only a genuine hex checkpoint may be published."""

    def test_hex_checkpoint_yields_a_hub_encoder(self, hex_checkpoint: Path) -> None:
        from alpha_zero.hexcrystal.hub_mixin import HexSpatialEncoderForHub

        encoder = uploader.extract_encoder(hex_checkpoint)

        assert isinstance(encoder, HexSpatialEncoderForHub)

    def test_extracted_weights_match_the_checkpoint(self, hex_checkpoint: Path) -> None:
        """Extraction must carry the trained weights, not re-initialise them."""
        # weights_only=True: the payload is tensors plus a primitive model_config,
        # so there is no reason to permit arbitrary unpickling here.
        payload = torch.load(hex_checkpoint, map_location="cpu", weights_only=True)
        encoder = uploader.extract_encoder(hex_checkpoint)

        prefix = "encoder.patch_embed."
        expected = {
            k[len(prefix) :]: v for k, v in payload["state_dict"].items() if k.startswith(prefix)
        }
        actual = encoder.state_dict()

        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            assert torch.allclose(value, actual[key], atol=1e-6), f"weight drift in {key}"

    def test_patch_embed_checkpoint_is_rejected(self, tmp_path: Path) -> None:
        """Publishing the wrong architecture must fail loudly, not silently."""
        wrong = _checkpoint(tmp_path, "patch_embed")

        with pytest.raises(uploader.UploadError, match="hex_crystalline"):
            uploader.extract_encoder(wrong)

    def test_checkpoint_without_model_config_is_rejected(self, tmp_path: Path) -> None:
        """A checkpoint of unknown provenance cannot be published."""
        path = tmp_path / "bare.pt"
        torch.save({"state_dict": {}}, path)

        with pytest.raises(uploader.UploadError, match="model_config"):
            uploader.extract_encoder(path)

    def test_missing_checkpoint_exits_non_zero(self, tmp_path: Path, fake_api: MagicMock) -> None:
        exit_code = uploader.main(
            ["--checkpoint", str(tmp_path / "nope.pt"), "--repo-id", "a/b"], api=fake_api
        )

        assert exit_code == 1


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


class TestStaging:
    """The staged payload is what the world sees; it must be complete."""

    @pytest.fixture
    def encoder(self, hex_checkpoint: Path) -> Any:
        return uploader.extract_encoder(hex_checkpoint)

    def test_staging_writes_the_expected_artifacts(self, encoder: Any, tmp_path: Path) -> None:
        staged = uploader.stage_artifacts(encoder, tmp_path / "s", None, None)

        names = {p.name for p in staged}
        assert "model.safetensors" in names
        assert "config.json" in names

    def test_config_round_trips_into_the_encoder_config(self, encoder: Any, tmp_path: Path) -> None:
        from alpha_zero.hexcrystal import HexEncoderConfig

        uploader.stage_artifacts(encoder, tmp_path / "s", None, None)
        payload = json.loads((tmp_path / "s" / "config.json").read_text())

        assert HexEncoderConfig.from_dict(payload).in_channels == 13

    def test_model_card_replaces_the_placeholder_readme(self, encoder: Any, tmp_path: Path) -> None:
        """An unedited auto-generated README must never reach a public page."""
        card = tmp_path / "card.md"
        card.write_text("# Real model card\n")

        uploader.stage_artifacts(encoder, tmp_path / "s", card, None)

        assert (tmp_path / "s" / "README.md").read_text() == "# Real model card\n"

    def test_evidence_table_is_spliced_into_the_card(self, encoder: Any, tmp_path: Path) -> None:
        """Published numbers come from the committed evidence, not by hand."""
        card = tmp_path / "card.md"
        card.write_text("Results:\n\n<!-- AB_SUMMARY_TABLE -->\n")
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        (evidence / "summary_table.md").write_text("| Games | 800 |\n")

        uploader.stage_artifacts(encoder, tmp_path / "s", card, evidence)

        published = (tmp_path / "s" / "README.md").read_text()
        assert "| Games | 800 |" in published
        assert "AB_SUMMARY_TABLE" not in published

    def test_staging_is_rebuilt_from_scratch(self, encoder: Any, tmp_path: Path) -> None:
        """A stale file from a previous run must not be republished."""
        staging = tmp_path / "s"
        staging.mkdir()
        (staging / uploader._STAGING_MARKER).touch()
        (staging / "leftover.txt").write_text("stale")

        uploader.stage_artifacts(encoder, staging, None, None)

        assert not (staging / "leftover.txt").exists()

    def test_staging_refuses_a_directory_it_does_not_own(
        self, encoder: Any, tmp_path: Path
    ) -> None:
        """A populated directory without our marker is never deleted.

        Guards against ``--staging-dir docs`` (or ``.``) destroying real work:
        the rmtree runs on the default path, before --publish is considered.
        """
        precious = tmp_path / "precious"
        precious.mkdir()
        (precious / "important.txt").write_text("do not delete")

        with pytest.raises(uploader.UploadError, match="not empty"):
            uploader.stage_artifacts(encoder, precious, None, None)

        assert (precious / "important.txt").exists()

    def test_unsubstituted_evidence_placeholder_blocks_publication(
        self, encoder: Any, tmp_path: Path
    ) -> None:
        """A card still holding the A/B marker must not be published.

        Otherwise the public page asserts an evaluation with no numbers.
        """
        card = tmp_path / "card.md"
        card.write_text("Results:\n\n<!-- AB_SUMMARY_TABLE -->\n")
        staging = tmp_path / "s"

        uploader.stage_artifacts(encoder, staging, card, None)

        with pytest.raises(uploader.UploadError, match="placeholder"):
            uploader.verify_staged(staging)

    def test_nested_pickle_is_refused(self, tmp_path: Path) -> None:
        """A pickle one directory down is still caught."""
        staging = tmp_path / "s"
        (staging / "nested").mkdir(parents=True)
        (staging / "model.safetensors").write_bytes(b"x")
        (staging / "config.json").write_text('{"embed_dim": 64}')
        (staging / "README.md").write_text("card")
        (staging / "nested" / "sneaky.ckpt").write_bytes(b"pickle")

        with pytest.raises(uploader.UploadError, match="pickled"):
            uploader.verify_staged(staging)
