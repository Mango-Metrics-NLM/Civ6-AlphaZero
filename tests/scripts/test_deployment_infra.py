"""Tests for launch_muzero_training.ps1 helper and setup_env.ps1.

Tests validate the setup_env PowerShell script logic by testing
the Python-side env var loading and validation it depends upon.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"


class TestSetupEnvScript:
    """Verify scripts/setup_env.ps1 exists and is well-formed."""

    def test_script_exists(self) -> None:
        assert (SCRIPTS_DIR / "setup_env.ps1").exists()

    def test_script_has_env_file_param(self) -> None:
        content = (SCRIPTS_DIR / "setup_env.ps1").read_text()
        assert "EnvFile" in content
        assert ".env" in content

    def test_script_validates_critical_vars(self) -> None:
        content = (SCRIPTS_DIR / "setup_env.ps1").read_text()
        assert "CIV6AZ_VERTEX_PROJECT" in content
        assert "CIV6AZ_GCS_BUCKET" in content
        assert "CIV6AZ_CONTAINER_IMAGE_URI" in content


class TestLaunchMuZeroScript:
    """Verify scripts/launch_muzero_training.ps1 exists and is well-formed."""

    def test_script_exists(self) -> None:
        assert (SCRIPTS_DIR / "launch_muzero_training.ps1").exists()

    def test_script_has_dry_run_param(self) -> None:
        content = (SCRIPTS_DIR / "launch_muzero_training.ps1").read_text()
        assert "DryRun" in content

    def test_script_uses_algorithm_muzero(self) -> None:
        content = (SCRIPTS_DIR / "launch_muzero_training.ps1").read_text()
        assert "--algorithm=muzero" in content

    def test_script_calls_orchestrator_not_entrypoint(self) -> None:
        """Ensure launch script invokes orchestrator (local), not entrypoint (container)."""
        content = (SCRIPTS_DIR / "launch_muzero_training.ps1").read_text()
        assert "vertex_orchestrator" in content
        assert "entrypoint" not in content.lower() or "entrypoint" in content.split("#")[0] is False

    def test_script_sources_setup_env(self) -> None:
        content = (SCRIPTS_DIR / "launch_muzero_training.ps1").read_text()
        assert "setup_env.ps1" in content

    def test_script_has_configurable_parameters(self) -> None:
        content = (SCRIPTS_DIR / "launch_muzero_training.ps1").read_text()
        for param in ["Iterations", "Workers", "GamesPerWorker", "Epochs", "AgentId"]:
            assert param in content, f"Missing parameter: {param}"


class TestGcloudIgnore:
    """Verify .gcloudignore excludes large directories."""

    def test_gcloudignore_exists(self) -> None:
        assert (ROOT / ".gcloudignore").exists()

    def test_excludes_wandb(self) -> None:
        content = (ROOT / ".gcloudignore").read_text()
        assert "wandb/" in content

    def test_excludes_models(self) -> None:
        content = (ROOT / ".gcloudignore").read_text()
        assert "models/" in content

    def test_excludes_git(self) -> None:
        content = (ROOT / ".gcloudignore").read_text()
        assert ".git/" in content

    def test_excludes_large_binaries(self) -> None:
        content = (ROOT / ".gcloudignore").read_text()
        assert "*.pt" in content
        assert "*.pth" in content


class TestDockerIgnore:
    """Verify .dockerignore excludes large directories."""

    def test_dockerignore_exists(self) -> None:
        assert (ROOT / ".dockerignore").exists()

    def test_excludes_wandb(self) -> None:
        content = (ROOT / ".dockerignore").read_text()
        assert "wandb/" in content

    def test_excludes_tests(self) -> None:
        content = (ROOT / ".dockerignore").read_text()
        assert "tests/" in content


class TestEnvFile:
    """Verify .env has required GCP configuration."""

    def test_env_file_exists(self) -> None:
        assert (ROOT / ".env").exists()

    def test_has_vertex_project(self) -> None:
        content = (ROOT / ".env").read_text()
        assert "CIV6AZ_VERTEX_PROJECT=" in content

    def test_has_gcs_bucket(self) -> None:
        content = (ROOT / ".env").read_text()
        assert "CIV6AZ_GCS_BUCKET=" in content

    def test_has_container_image(self) -> None:
        content = (ROOT / ".env").read_text()
        assert "CIV6AZ_CONTAINER_IMAGE_URI=" in content

    def test_has_cloudsdk_update_check_disabled(self) -> None:
        content = (ROOT / ".env").read_text()
        assert "CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK=1" in content

    def test_container_tag_is_v0_15_0(self) -> None:
        content = (ROOT / ".env").read_text()
        assert "v0.15.0" in content
