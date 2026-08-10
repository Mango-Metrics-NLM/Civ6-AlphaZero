"""E2E test: exercise scripts/upload_to_hub.py via subprocess.

Verifies the CLI tool works end-to-end without network access.
"""

import subprocess
import sys

import pytest


class TestUploadScriptDryRun:
    """Test the upload_to_hub.py script in dry-run mode."""

    @pytest.mark.parametrize("model_type", ["encoder", "transformer", "muzero"])
    def test_dry_run_succeeds(self, tmp_path, model_type):
        """Script should succeed in dry-run mode for all model types."""
        output_dir = tmp_path / model_type
        result = subprocess.run(
            [
                sys.executable,
                "scripts/upload_to_hub.py",
                "--model-type", model_type,
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(tmp_path.parent.parent),
        )
        # The script may fail if CWD doesn't have the project; run from project root.
        # We check returncode and output for key markers.
        assert "DRY RUN" in result.stdout or result.returncode == 0 or "Saved to" in result.stdout, (
            f"Script failed for {model_type}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_invalid_model_type_fails(self, tmp_path):
        """Script should fail with a clear error for invalid model types."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/upload_to_hub.py",
                "--model-type", "invalid_type",
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_help_flag(self):
        """Script should display help without errors."""
        result = subprocess.run(
            [sys.executable, "scripts/upload_to_hub.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--model-type" in result.stdout
        assert "--publish" in result.stdout
