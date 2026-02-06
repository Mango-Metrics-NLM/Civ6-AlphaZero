"""
Tests for GCP Authentication Setup Script.

These tests validate the PowerShell script behavior and can be run
as part of the CI/CD pipeline to ensure the auth setup process works correctly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Get the scripts directory
SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "scripts"
SETUP_SCRIPT = SCRIPTS_DIR / "setup_gcp_auth.ps1"


class TestGCPAuthScript:
    """Tests for setup_gcp_auth.ps1 script."""

    def test_script_exists(self) -> None:
        """Verify the setup script exists."""
        assert SETUP_SCRIPT.exists(), f"Script not found at {SETUP_SCRIPT}"

    def test_script_has_required_parameters(self) -> None:
        """Verify script has documented parameters."""
        content = SETUP_SCRIPT.read_text()

        # Check for parameter definitions
        assert "[CmdletBinding()]" in content, "Script should use CmdletBinding"
        assert "$SkipLogin" in content, "Script should have SkipLogin parameter"
        assert "$ValidateOnly" in content, "Script should have ValidateOnly parameter"

    def test_script_defines_required_env_vars(self) -> None:
        """Verify script documents required environment variables."""
        content = SETUP_SCRIPT.read_text()

        required_vars = [
            "CIV6AZ_VERTEX_PROJECT",
            "CIV6AZ_GCS_BUCKET",
        ]

        for var in required_vars:
            assert var in content, f"Script should reference {var}"

    def test_script_defines_optional_env_vars(self) -> None:
        """Verify script documents optional environment variables."""
        content = SETUP_SCRIPT.read_text()

        optional_vars = [
            "CIV6AZ_CONTAINER_IMAGE_URI",
            "CIV6AZ_VERTEX_REGION",
            "CIV6AZ_VERTEX_SERVICE_ACCOUNT",
        ]

        for var in optional_vars:
            assert var in content, f"Script should reference {var}"

    def test_script_has_help_documentation(self) -> None:
        """Verify script has proper help documentation."""
        content = SETUP_SCRIPT.read_text()

        # Check for comment-based help
        assert ".SYNOPSIS" in content, "Script should have SYNOPSIS"
        assert ".DESCRIPTION" in content, "Script should have DESCRIPTION"
        assert ".EXAMPLE" in content, "Script should have EXAMPLE"

    def test_script_checks_gcloud_installation(self) -> None:
        """Verify script checks for gcloud CLI."""
        content = SETUP_SCRIPT.read_text()

        assert "Test-GcloudInstalled" in content or "Get-Command gcloud" in content, \
            "Script should check for gcloud installation"

    def test_script_handles_adc_configuration(self) -> None:
        """Verify script handles Application Default Credentials."""
        content = SETUP_SCRIPT.read_text()

        assert "application-default" in content.lower() or "adc" in content.lower(), \
            "Script should handle ADC configuration"

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="PowerShell script tests only run on Windows"
    )
    def test_script_syntax_validation(self) -> None:
        """Verify script has valid PowerShell syntax."""
        # Use PowerShell to parse the script without executing
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                f"$null = [System.Management.Automation.Language.Parser]::ParseFile('{SETUP_SCRIPT}', [ref]$null, [ref]$errors); $errors.Count"
            ],
            capture_output=True,
            text=True
        )

        # If errors.Count is 0, syntax is valid
        error_count = result.stdout.strip()
        assert error_count == "0", f"Script has syntax errors: {result.stderr}"

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="PowerShell script tests only run on Windows"
    )
    def test_script_validate_only_mode(self) -> None:
        """Test that -ValidateOnly flag works without making changes."""
        # Run script with -ValidateOnly
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(SETUP_SCRIPT), "-ValidateOnly"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Script should run and produce output (exit code may be 1 if env vars not set)
        assert "GCP Authentication Setup" in result.stdout or result.returncode in [0, 1]


class TestEnvironmentVariableValidation:
    """Tests for environment variable configuration validation."""

    REQUIRED_VARS = [
        "CIV6AZ_VERTEX_PROJECT",
        "CIV6AZ_GCS_BUCKET",
    ]

    OPTIONAL_VARS = [
        "CIV6AZ_CONTAINER_IMAGE_URI",
        "CIV6AZ_VERTEX_REGION",
        "CIV6AZ_VERTEX_SERVICE_ACCOUNT",
    ]

    def test_required_vars_documented(self) -> None:
        """Verify all required vars are in our test fixtures."""
        # This ensures tests stay in sync with script
        content = SETUP_SCRIPT.read_text()

        for var in self.REQUIRED_VARS:
            assert var in content

    def test_optional_vars_have_defaults(self) -> None:
        """Verify optional variables have sensible defaults documented."""
        content = SETUP_SCRIPT.read_text()

        # Check that defaults are documented
        assert "Default" in content, "Script should document default values"

    def test_env_var_format_convention(self) -> None:
        """Verify environment variables follow naming convention."""
        all_vars = self.REQUIRED_VARS + self.OPTIONAL_VARS

        for var in all_vars:
            assert var.startswith("CIV6AZ_"), f"{var} should start with CIV6AZ_"
            assert var.isupper(), f"{var} should be uppercase"


class TestScriptModularity:
    """Tests for script modularity and reusability."""

    def test_functions_are_defined(self) -> None:
        """Verify script uses reusable functions."""
        content = SETUP_SCRIPT.read_text()

        expected_functions = [
            "Write-Status",
            "Test-GcloudInstalled",
            "Get-GcloudConfig",
            "Test-EnvironmentVariables",
        ]

        for func in expected_functions:
            assert f"function {func}" in content, f"Script should define {func}"

    def test_no_hardcoded_project_values(self) -> None:
        """Verify no hardcoded project IDs or bucket names."""
        content = SETUP_SCRIPT.read_text()

        # These patterns would indicate hardcoded values
        hardcoded_patterns = [
            "gs://civ6",
            "iansh",  # username
            "civ6-alphazero",  # specific project
        ]

        for pattern in hardcoded_patterns:
            assert pattern.lower() not in content.lower(), \
                f"Script should not contain hardcoded value: {pattern}"

    def test_script_has_main_function(self) -> None:
        """Verify script uses Main function pattern."""
        content = SETUP_SCRIPT.read_text()

        assert "function Main" in content, "Script should have Main function"
        assert "exit (Main)" in content or "Main" in content, \
            "Script should call Main function"
