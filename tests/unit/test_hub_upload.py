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

from alpha_zero.hub.model_card import ModelCardMetadata  # noqa: E402
from alpha_zero.hub.upload import HubUploadable, upload_model_to_hub  # noqa: E402


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
        with mock.patch("huggingface_hub.HfApi") as mock_api:
            upload_model_to_hub(
                model=model,
                repo_id="org/repo",
                dry_run=True,
                save_directory=tmp_path / "out",
            )
            # HfApi should never be instantiated during a dry run.
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
        with mock.patch("huggingface_hub.HfApi") as MockApi:
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
        with mock.patch("huggingface_hub.HfApi") as MockApi:
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
        with mock.patch("huggingface_hub.HfApi") as MockApi:
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

    def test_auto_temp_dir_created_when_save_directory_none(self, tmp_path):
        """When save_directory=None, a temp dir must be created and returned."""
        model = _FakeModel()
        result = upload_model_to_hub(
            model=model,
            repo_id="org/repo",
            dry_run=True,
            save_directory=None,
        )
        # Should be an absolute path that exists (dry-run keeps it).
        assert result.is_absolute()
        assert result.exists()

    def test_auto_temp_dir_cleaned_up_after_publish(self, tmp_path):
        """When save_directory=None and dry_run=False, temp dir is cleaned after upload."""
        model = _FakeModel()
        captured: list[Path] = []

        original_save = model.save_pretrained

        def capturing_save(path: str) -> None:
            captured.append(Path(path))
            original_save(path)

        model.save_pretrained = capturing_save  # type: ignore[method-assign]

        with mock.patch("huggingface_hub.HfApi") as MockApi:
            mock_instance = MockApi.return_value
            result = upload_model_to_hub(
                model=model,
                repo_id="org/repo",
                dry_run=False,
                save_directory=None,  # auto-creates temp dir
            )
            mock_instance.create_repo.assert_called_once()
            mock_instance.upload_folder.assert_called_once()

        # After successful publish, the auto-created dir should be gone.
        assert not result.exists(), (
            "Auto-created temp dir should be removed after successful upload"
        )


class TestHubUploadableProtocolLate:

    def test_protocol_importable(self):
        """Import guard on HfApi must not block protocol import."""
        # This was M2 — HfApi used to be at module level.
        from alpha_zero.hub.upload import HubUploadable

        assert HubUploadable is not None

    def test_isinstance_check(self):
        class GoodModel:
            def save_pretrained(self, save_directory: str, **kw) -> None:
                pass

        class BadModel:
            pass

        from alpha_zero.hub.upload import HubUploadable

        assert isinstance(GoodModel(), HubUploadable)
        assert not isinstance(BadModel(), HubUploadable)
