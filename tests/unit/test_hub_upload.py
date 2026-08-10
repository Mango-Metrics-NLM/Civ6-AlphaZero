"""Unit tests for alpha_zero.hub.upload.

Tests cover:
- upload_model_to_hub dry-run flow
- Protocol enforcement (HubUploadable)
- repo_id validation
- Model card auto-generation
- HfApi mocking for publish flow
"""

from pathlib import Path
from unittest import mock

import pytest

# Guard: skip entire module if hub extra is missing.
hf_hub = pytest.importorskip("huggingface_hub", reason="requires hub extra")

from alpha_zero.hub.model_card import ModelCardMetadata
from alpha_zero.hub.upload import HubUploadable, upload_model_to_hub


class _FakeModel:
    """Minimal model that satisfies HubUploadable protocol."""

    def save_pretrained(self, save_directory, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "model.safetensors").write_text("fake-weights")
        (Path(save_directory) / "config.json").write_text('{"test": true}')


class _NotAModel:
    """Object that does NOT satisfy HubUploadable."""

    pass


class TestHubUploadableProtocol:
    """Tests for the HubUploadable protocol."""

    def test_fake_model_is_uploadable(self):
        assert isinstance(_FakeModel(), HubUploadable)

    def test_non_model_is_not_uploadable(self):
        assert not isinstance(_NotAModel(), HubUploadable)


class TestUploadModelDryRun:
    """Tests for upload_model_to_hub in dry-run mode."""

    def test_dry_run_saves_locally(self, tmp_path):
        model = _FakeModel()
        save_dir = tmp_path / "output"
        result = upload_model_to_hub(
            model=model,
            repo_id="test-org/test-model",
            dry_run=True,
            save_directory=save_dir,
        )
        assert result.exists()
        assert (result / "model.safetensors").exists()
        assert (result / "README.md").exists()

    def test_dry_run_generates_model_card(self, tmp_path):
        model = _FakeModel()
        meta = ModelCardMetadata(model_name="test-card-model")
        result = upload_model_to_hub(
            model=model,
            repo_id="org/repo",
            metadata=meta,
            dry_run=True,
            save_directory=tmp_path / "out",
        )
        card = (result / "README.md").read_text(encoding="utf-8")
        assert "test-card-model" in card

    def test_dry_run_does_not_call_hf_api(self, tmp_path):
        model = _FakeModel()
        with mock.patch("alpha_zero.hub.upload.HfApi") as mock_api:
            upload_model_to_hub(
                model=model,
                repo_id="org/repo",
                dry_run=True,
                save_directory=tmp_path / "out",
            )
            mock_api.assert_not_called()

    def test_auto_metadata_when_none(self, tmp_path):
        model = _FakeModel()
        result = upload_model_to_hub(
            model=model,
            repo_id="org/repo",
            metadata=None,
            dry_run=True,
            save_directory=tmp_path / "out",
        )
        card = (result / "README.md").read_text(encoding="utf-8")
        assert "org/repo" in card

    def test_metadata_repo_id_set_automatically(self, tmp_path):
        model = _FakeModel()
        meta = ModelCardMetadata(model_name="test")
        upload_model_to_hub(
            model=model,
            repo_id="org/repo",
            metadata=meta,
            dry_run=True,
            save_directory=tmp_path / "out",
        )
        assert meta.repo_id == "org/repo"


class TestUploadValidation:
    """Tests for input validation."""

    def test_rejects_non_uploadable_model(self, tmp_path):
        with pytest.raises(TypeError, match="save_pretrained"):
            upload_model_to_hub(
                model=_NotAModel(),  # type: ignore[arg-type]
                repo_id="org/repo",
                dry_run=True,
                save_directory=tmp_path,
            )

    def test_rejects_malformed_repo_id(self, tmp_path):
        model = _FakeModel()
        with pytest.raises(ValueError, match="org/repo"):
            upload_model_to_hub(
                model=model,
                repo_id="no-slash-here",
                dry_run=True,
                save_directory=tmp_path,
            )


class TestUploadPublish:
    """Tests for the actual publish path (HfApi mocked)."""

    def test_publish_calls_hf_api(self, tmp_path):
        model = _FakeModel()
        with mock.patch("alpha_zero.hub.upload.HfApi") as MockApi:
            mock_instance = MockApi.return_value
            upload_model_to_hub(
                model=model,
                repo_id="org/repo",
                dry_run=False,
                save_directory=tmp_path / "out",
            )
            mock_instance.create_repo.assert_called_once_with(
                repo_id="org/repo", exist_ok=True, private=False,
            )
            mock_instance.upload_folder.assert_called_once()

    def test_publish_private_repo(self, tmp_path):
        model = _FakeModel()
        with mock.patch("alpha_zero.hub.upload.HfApi") as MockApi:
            mock_instance = MockApi.return_value
            upload_model_to_hub(
                model=model,
                repo_id="org/repo",
                dry_run=False,
                private=True,
                save_directory=tmp_path / "out",
            )
            mock_instance.create_repo.assert_called_once_with(
                repo_id="org/repo", exist_ok=True, private=True,
            )

    def test_publish_custom_commit_message(self, tmp_path):
        model = _FakeModel()
        with mock.patch("alpha_zero.hub.upload.HfApi") as MockApi:
            mock_instance = MockApi.return_value
            upload_model_to_hub(
                model=model,
                repo_id="org/repo",
                dry_run=False,
                commit_message="custom message",
                save_directory=tmp_path / "out",
            )
            call_kwargs = mock_instance.upload_folder.call_args
            assert call_kwargs[1]["commit_message"] == "custom message"
