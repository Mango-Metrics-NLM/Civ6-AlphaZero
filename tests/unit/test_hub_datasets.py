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


class TestEloExporterPushToHub:
    """Tests for EloDatasetExporter.push_to_hub dry_run path."""

    @pytest.fixture()
    def elo_source(self, tmp_path):
        source = tmp_path / "results"
        source.mkdir()
        (source / "elo_ratings.json").write_text(
            '[{"model": "v1", "elo": 1200}]', encoding="utf-8",
        )
        return source

    def test_push_to_hub_dry_run_no_network(self, elo_source, tmp_path):
        """push_to_hub with dry_run=True should not actually push."""
        try:
            from datasets import Dataset  # noqa: F401
        except ImportError:
            pytest.skip("datasets library not installed")

        from unittest import mock

        config = DatasetExportConfig(
            source_dir=elo_source,
            output_dir=tmp_path / "export",
            repo_id="org/test-dataset",
            dry_run=True,
        )
        exporter = EloDatasetExporter(config)

        with mock.patch("datasets.Dataset.push_to_hub") as mock_push:
            exporter.push_to_hub()
            mock_push.assert_not_called()

    def test_push_to_hub_missing_repo_id_raises(self, elo_source, tmp_path):
        """push_to_hub with no repo_id configured must raise ValueError."""
        try:
            import datasets  # noqa: F401
        except ImportError:
            pytest.skip("datasets library not installed")

        config = DatasetExportConfig(
            source_dir=elo_source,
            output_dir=tmp_path / "export",
            repo_id="",  # Explicitly empty
        )
        exporter = EloDatasetExporter(config)

        with pytest.raises(ValueError, match="repo_id"):
            exporter.push_to_hub()


class TestReplayExporterPushToHub:
    """Tests for ReplayDatasetExporter.push_to_hub."""

    @pytest.fixture()
    def replay_source(self, tmp_path):
        source = tmp_path / "replays"
        source.mkdir()
        with (source / "episode_replay.jsonl").open("w", encoding="utf-8") as f:
            f.write('{"turn": 1, "action": "move"}\n')
        return source

    def test_push_to_hub_dry_run(self, replay_source, tmp_path):
        """ReplayDatasetExporter.push_to_hub with dry_run=True should not push."""
        try:
            from datasets import Dataset  # noqa: F401
        except ImportError:
            pytest.skip("datasets library not installed")

        from unittest import mock

        config = DatasetExportConfig(
            source_dir=replay_source,
            output_dir=tmp_path / "export",
            repo_id="org/replays",
            dry_run=True,
        )
        exporter = ReplayDatasetExporter(config)

        with mock.patch("datasets.Dataset.push_to_hub") as mock_push:
            exporter.push_to_hub()
            mock_push.assert_not_called()

    def test_push_to_hub_missing_repo_id_raises(self, replay_source, tmp_path):
        """ReplayDatasetExporter.push_to_hub with no repo_id must raise ValueError."""
        try:
            import datasets  # noqa: F401
        except ImportError:
            pytest.skip("datasets library not installed")

        config = DatasetExportConfig(
            source_dir=replay_source,
            output_dir=tmp_path / "export",
            repo_id="",
        )
        exporter = ReplayDatasetExporter(config)

        with pytest.raises(ValueError, match="repo_id"):
            exporter.push_to_hub()


class TestSingleDictJsonReplay:
    """Test the single-dict JSON replay branch (line 232-233)."""

    def test_single_dict_json_loaded(self, tmp_path):
        source = tmp_path / "replays"
        source.mkdir()
        # Single dict (not a list) — this exercises the else branch in export
        (source / "trajectory_data.json").write_text(
            '{"episode": 0, "total_reward": 5.0}', encoding="utf-8",
        )
        config = DatasetExportConfig(
            source_dir=source,
            output_dir=tmp_path / "export",
        )
        exporter = ReplayDatasetExporter(config)
        out = exporter.export_to_dir()

        records = []
        with (out / "replays.jsonl").open(encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        assert len(records) == 1
        assert records[0]["episode"] == 0


class TestConfigurableScanPatterns:
    """Test that scan patterns are configurable via DatasetExportConfig."""

    def test_custom_elo_patterns(self, tmp_path):
        """Custom ELO scan pattern should only match configured names."""
        source = tmp_path / "data"
        source.mkdir()
        # File with standard name — should NOT be matched.
        (source / "elo_ratings.json").write_text("[]", encoding="utf-8")
        # File with custom name — should be matched.
        (source / "custom_ranking.json").write_text('[{"rank": 1}]', encoding="utf-8")

        config = DatasetExportConfig(
            source_dir=source,
            elo_scan_patterns=("*custom_ranking*",),
        )
        exporter = EloDatasetExporter(config)
        files = exporter._scan_elo_files()
        assert len(files) == 1
        assert "custom_ranking" in files[0].name

    def test_custom_replay_patterns(self, tmp_path):
        """Custom replay scan pattern should only match configured names."""
        source = tmp_path / "data"
        source.mkdir()
        (source / "episode_replay.jsonl").write_text("", encoding="utf-8")
        (source / "game_log.jsonl").write_text('{"turn": 1}\n', encoding="utf-8")

        config = DatasetExportConfig(
            source_dir=source,
            replay_scan_patterns=("*game_log*",),
        )
        exporter = ReplayDatasetExporter(config)
        files = exporter._scan_replay_files()
        assert len(files) == 1
        assert "game_log" in files[0].name


class TestOutputDirDefault:
    """Verify output_dir default is absolute (M4 fix)."""

    def test_default_output_dir_is_absolute(self):
        config = DatasetExportConfig()
        assert config.output_dir.is_absolute(), (
            "DatasetExportConfig.output_dir default must be an absolute path"
        )

    def test_description_from_constants(self):
        from alpha_zero.hub.constants import DEFAULT_DATASET_DESCRIPTION

        config = DatasetExportConfig()
        assert config.description == DEFAULT_DATASET_DESCRIPTION
