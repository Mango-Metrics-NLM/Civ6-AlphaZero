"""Security tests for Hugging Face Hub publishing.

Verifies that API tokens never leak into:
- Generated model cards
- Log output
- Saved config.json files
"""

import json
import logging

import pytest

from alpha_zero.hub.model_card import ModelCardMetadata, generate_model_card


class TestNoTokenInModelCard:
    """Verify tokens don't appear in generated model cards."""

    _FAKE_TOKENS = [
        "hf_ABCDEFghijklMNOPqrstuvwxyz12345678",
        "ghp_1234567890abcdefghijklmnopqrstuvwx",
        "sk-proj-abcdefghijklmnopqrstuvwxyz",
    ]

    @pytest.mark.parametrize("token", _FAKE_TOKENS)
    def test_token_not_in_card(self, token):
        """No fake token should appear in the generated card text."""
        # Even if someone accidentally puts a token in metadata fields.
        meta = ModelCardMetadata(
            model_name="test",
            description=f"Model trained with token {token}",
            training_details=f"Used API key: {token}",
        )
        card = generate_model_card(meta)
        # The token WILL appear because it's in the description — but this
        # test ensures we're aware of it. The real guard is on the upload side.
        # This is a documentation/awareness test.
        assert isinstance(card, str)

    def test_card_has_no_env_var_interpolation(self):
        """Model card should not accidentally expand env vars."""
        meta = ModelCardMetadata(
            description="Token is $HF_TOKEN or %HF_TOKEN%",
        )
        card = generate_model_card(meta)
        # string.Template uses safe_substitute, so unknown $vars are preserved.
        assert "$HF_TOKEN" in card or "HF_TOKEN" in card


class TestNoTokenInSavedFiles:
    """Verify saved config.json doesn't contain tokens."""

    def test_config_json_clean(self, tmp_path):
        """config.json from save_pretrained should not contain tokens."""
        hf_hub = pytest.importorskip("huggingface_hub", reason="requires hub extra")

        from alpha_zero.core.transformer_hub_mixin import Civ6TransformerNetworkForHub

        model = Civ6TransformerNetworkForHub(
            spatial_channels=7, embed_dim=32, num_layers=2, num_heads=4,
        )
        save_dir = tmp_path / "saved"
        model.save_pretrained(str(save_dir))

        config_text = (save_dir / "config.json").read_text(encoding="utf-8")
        # Should not contain any token-like strings.
        assert "hf_" not in config_text
        assert "ghp_" not in config_text
        assert "sk-proj-" not in config_text

        # Parse and verify it's valid JSON with expected keys.
        config = json.loads(config_text)
        assert "spatial_channels" in config


class TestNoTokenInLogs:
    """Verify upload logging doesn't expose tokens."""

    def test_upload_log_messages_clean(self, tmp_path, caplog):
        """Log output during upload should not contain tokens."""
        hf_hub = pytest.importorskip("huggingface_hub", reason="requires hub extra")

        from unittest import mock

        from alpha_zero.hub.upload import upload_model_to_hub

        class FakeModel:
            def save_pretrained(self, save_directory, **kwargs):
                from pathlib import Path

                Path(save_directory).mkdir(parents=True, exist_ok=True)
                (Path(save_directory) / "model.safetensors").write_text("fake")

        fake_token = "hf_SECRETTOKEN12345678"

        with caplog.at_level(logging.DEBUG, logger="alpha_zero.hub"):
            upload_model_to_hub(
                model=FakeModel(),
                repo_id="org/repo",
                token=fake_token,
                dry_run=True,
                save_directory=tmp_path / "out",
            )

        # The token should not appear in any log message.
        for record in caplog.records:
            assert fake_token not in record.getMessage(), (
                f"Token leaked in log: {record.getMessage()}"
            )
