"""Unit tests for alpha_zero.hub.datasets.

Tests cover:
- EloDatasetExporter file scanning, loading, and export
- ReplayDatasetExporter file scanning and export
- DatasetExportConfig defaults
- Edge cases (empty dirs, max_samples)
"""

import json

import pytest

from alpha_zero.hub.datasets import (
    DatasetExportConfig,
    EloDatasetExporter,
    ReplayDatasetExporter,
)


class TestDatasetExportConfig:
    """Tests for DatasetExportConfig."""

    def test_defaults(self):
        config = DatasetExportConfig()
        assert config.dry_run is True
        assert config.max_samples is None
        assert config.repo_id == ""

    def test_custom_values(self):
        config = DatasetExportConfig(
            repo_id="org/dataset",
            max_samples=100,
            description="Custom dataset",
        )
        assert config.repo_id == "org/dataset"
        assert config.max_samples == 100


class TestEloDatasetExporter:
    """Tests for EloDatasetExporter."""

    @pytest.fixture()
    def elo_source(self, tmp_path):
        """Create a temp directory with sample ELO files."""
        source = tmp_path / "results"
        source.mkdir()

        # JSON file.
        (source / "elo_ratings.json").write_text(
            json.dumps([
                {"model": "v1", "elo": 1200, "games": 50},
                {"model": "v2", "elo": 1350, "games": 100},
            ]),
            encoding="utf-8",
        )

        # CSV file.
        (source / "tournament_results.csv").write_text(
            "model,wins,losses,draws\nv1,20,25,5\nv2,30,15,5\n",
            encoding="utf-8",
        )

        return source

    def test_scan_finds_elo_files(self, elo_source):
        config = DatasetExportConfig(source_dir=elo_source)
        exporter = EloDatasetExporter(config)
        files = exporter._scan_elo_files()
        assert len(files) == 2

    def test_export_creates_jsonl(self, elo_source, tmp_path):
        config = DatasetExportConfig(
            source_dir=elo_source,
            output_dir=tmp_path / "export",
        )
        exporter = EloDatasetExporter(config)
        out = exporter.export_to_dir()

        assert (out / "elo_results.jsonl").exists()
        assert (out / "dataset_info.json").exists()

        # Verify JSONL content.
        records = []
        with (out / "elo_results.jsonl").open(encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        # 2 JSON records + 2 CSV rows = 4 total.
        assert len(records) == 4

    def test_export_with_max_samples(self, elo_source, tmp_path):
        config = DatasetExportConfig(
            source_dir=elo_source,
            output_dir=tmp_path / "export",
            max_samples=2,
        )
        exporter = EloDatasetExporter(config)
        out = exporter.export_to_dir()

        records = []
        with (out / "elo_results.jsonl").open(encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        assert len(records) == 2

    def test_export_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        config = DatasetExportConfig(
            source_dir=empty,
            output_dir=tmp_path / "export",
        )
        exporter = EloDatasetExporter(config)
        out = exporter.export_to_dir()

        records = []
        with (out / "elo_results.jsonl").open(encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        assert len(records) == 0

    def test_dataset_info_metadata(self, elo_source, tmp_path):
        config = DatasetExportConfig(
            source_dir=elo_source,
            output_dir=tmp_path / "export",
            description="Test ELO dataset",
        )
        exporter = EloDatasetExporter(config)
        out = exporter.export_to_dir()

        info = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
        assert info["description"] == "Test ELO dataset"
        assert info["num_records"] == 4

    def test_kwargs_construction(self, elo_source, tmp_path):
        exporter = EloDatasetExporter(
            source_dir=elo_source,
            output_dir=tmp_path / "export",
        )
        out = exporter.export_to_dir()
        assert out.exists()

    def test_push_to_hub_requires_datasets_lib(self, elo_source):
        config = DatasetExportConfig(
            source_dir=elo_source,
            repo_id="org/dataset",
            dry_run=False,
        )
        exporter = EloDatasetExporter(config)
        # This test verifies the import guard fires if datasets isn't installed.
        # If datasets IS installed, it tests the dry-run path.
        try:
            import datasets  # noqa: F401

            config.dry_run = True
            exporter.push_to_hub()  # Should not raise with dry_run=True
        except ImportError:
            with pytest.raises(ImportError, match="datasets"):
                exporter.push_to_hub()


class TestReplayDatasetExporter:
    """Tests for ReplayDatasetExporter."""

    @pytest.fixture()
    def replay_source(self, tmp_path):
        """Create a temp directory with sample replay files."""
        source = tmp_path / "replays"
        source.mkdir()

        # JSONL replay file.
        with (source / "episode_replay.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"turn": 1, "action": "move", "reward": 0.5}) + "\n")
            f.write(json.dumps({"turn": 2, "action": "build", "reward": 1.0}) + "\n")

        # JSON trajectory file.
        (source / "trajectory_data.json").write_text(
            json.dumps([{"episode": 0, "total_reward": 5.0}]),
            encoding="utf-8",
        )

        return source

    def test_scan_finds_replay_files(self, replay_source):
        config = DatasetExportConfig(source_dir=replay_source)
        exporter = ReplayDatasetExporter(config)
        files = exporter._scan_replay_files()
        assert len(files) == 2

    def test_export_creates_jsonl(self, replay_source, tmp_path):
        config = DatasetExportConfig(
            source_dir=replay_source,
            output_dir=tmp_path / "export",
        )
        exporter = ReplayDatasetExporter(config)
        out = exporter.export_to_dir()

        assert (out / "replays.jsonl").exists()
        records = []
        with (out / "replays.jsonl").open(encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        # 2 JSONL records + 1 JSON record = 3 total.
        assert len(records) == 3

    def test_export_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        config = DatasetExportConfig(
            source_dir=empty,
            output_dir=tmp_path / "export",
        )
        exporter = ReplayDatasetExporter(config)
        out = exporter.export_to_dir()
        assert (out / "replays.jsonl").exists()
