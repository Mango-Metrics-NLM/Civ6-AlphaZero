"""Integration test for the full hub publish pipeline.

Tests the complete flow: create model → save → generate card → verify
all artefacts. No network calls — everything uses tmp_path.
"""

import json

import pytest
import torch

# Guard: skip if hub extra is missing.
hf_hub = pytest.importorskip("huggingface_hub", reason="requires hub extra")

from alpha_zero.hub.model_card import ModelCardMetadata  # noqa: E402
from alpha_zero.hub.upload import upload_model_to_hub  # noqa: E402


class TestTransformerPublishPipeline:
    """End-to-end pipeline: Transformer → save → card → verify."""

    def test_full_pipeline(self, tmp_path):
        from alpha_zero.core.transformer_hub_mixin import Civ6TransformerNetworkForHub

        torch.manual_seed(42)
        model = Civ6TransformerNetworkForHub(
            spatial_channels=7, embed_dim=32, num_layers=2, num_heads=4,
        )

        metadata = ModelCardMetadata(
            model_name="civ6-transformer-test",
            description="Integration test model.",
            architecture="Civ6TransformerNetworkForHub",
            num_parameters=str(sum(p.numel() for p in model.parameters())),
            import_path="alpha_zero.core.transformer_hub_mixin",
            class_name="Civ6TransformerNetworkForHub",
            repo_id="test-org/civ6-transformer-test",
            tags=("integration-test",),
        )

        save_dir = upload_model_to_hub(
            model=model,
            repo_id="test-org/civ6-transformer-test",
            metadata=metadata,
            dry_run=True,
            save_directory=tmp_path / "publish",
        )

        # Verify all expected artefacts.
        assert (save_dir / "model.safetensors").exists()
        assert (save_dir / "config.json").exists()
        assert (save_dir / "README.md").exists()

        # Verify config.json round-trips.
        config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
        assert config["embed_dim"] == 32
        assert config["num_layers"] == 2

        # Verify model card content.
        card = (save_dir / "README.md").read_text(encoding="utf-8")
        assert "civ6-transformer-test" in card
        assert "Civ6TransformerNetworkForHub" in card
        assert "integration-test" in card

        # Verify reload produces identical weights.
        reloaded = Civ6TransformerNetworkForHub.from_pretrained(str(save_dir))
        for (n_a, p_a), (n_b, p_b) in zip(
            model.named_parameters(), reloaded.named_parameters(), strict=True,
        ):
            assert n_a == n_b
            assert torch.allclose(p_a, p_b, atol=1e-6)


class TestMuZeroPublishPipeline:
    """End-to-end pipeline: MuZero → save → card → verify."""

    def test_full_pipeline(self, tmp_path):
        from alpha_zero.core.muzero_hub_mixin import MuZeroNetworkForHub

        torch.manual_seed(42)
        model = MuZeroNetworkForHub(
            spatial_channels=7, spatial_size=8, num_res_blocks=2, num_filters=32,
        )

        metadata = ModelCardMetadata(
            model_name="civ6-muzero-test",
            description="Integration test MuZero model.",
            tags=("integration-test",),
        )

        save_dir = upload_model_to_hub(
            model=model,
            repo_id="test-org/civ6-muzero-test",
            metadata=metadata,
            dry_run=True,
            save_directory=tmp_path / "publish",
        )

        assert (save_dir / "model.safetensors").exists()
        assert (save_dir / "config.json").exists()
        assert (save_dir / "README.md").exists()

        card = (save_dir / "README.md").read_text(encoding="utf-8")
        assert "civ6-muzero-test" in card

        reloaded = MuZeroNetworkForHub.from_pretrained(str(save_dir))
        for (n_a, p_a), (n_b, p_b) in zip(
            model.named_parameters(), reloaded.named_parameters(), strict=True,
        ):
            assert n_a == n_b
            assert torch.allclose(p_a, p_b, atol=1e-6)


class TestDatasetExportPipeline:
    """End-to-end pipeline: create data → export → verify."""

    def test_elo_export_pipeline(self, tmp_path):
        """Full ELO export: create source data → export → verify JSONL."""
        import json

        from alpha_zero.hub.datasets import DatasetExportConfig, EloDatasetExporter

        # Create source data.
        source = tmp_path / "results"
        source.mkdir()
        (source / "elo_ratings.json").write_text(
            json.dumps([{"model": "v1", "elo": 1200}]),
            encoding="utf-8",
        )

        config = DatasetExportConfig(
            source_dir=source,
            output_dir=tmp_path / "export",
            description="Integration test ELO dataset",
        )
        exporter = EloDatasetExporter(config)
        out = exporter.export_to_dir()

        assert (out / "elo_results.jsonl").exists()
        info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
        assert info["num_records"] == 1
        assert info["description"] == "Integration test ELO dataset"
