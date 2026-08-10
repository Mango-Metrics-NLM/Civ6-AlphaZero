"""Smoke test: hub import guard fires when hub extra is missing.

This test runs in a subprocess with huggingface_hub blocked from import,
verifying that the import guard in each hub mixin produces a clear
ImportError with installation instructions.
"""

import subprocess
import sys

import pytest


# Each (module_path, expected_message_fragment) pair to test.
_GUARDED_MODULES = [
    ("alpha_zero.hexcrystal.hub_mixin", "hub"),
    ("alpha_zero.core.transformer_hub_mixin", "hub"),
    ("alpha_zero.core.muzero_hub_mixin", "hub"),
]


class TestHubImportGuard:
    """Verify ImportError when hub extra is not installed."""

    @pytest.mark.parametrize("module_path,fragment", _GUARDED_MODULES)
    def test_import_guard_fires(self, module_path, fragment):
        """Importing hub modules without huggingface_hub raises ImportError."""
        code = (
            f"import sys; "
            f"sys.modules['huggingface_hub'] = None; "
            f"try:\n"
            f"    import {module_path}\n"
            f"    sys.exit(1)\n"
            f"except ImportError as e:\n"
            f"    assert '{fragment}' in str(e).lower(), f'Bad message: {{e}}'\n"
            f"    sys.exit(0)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Import guard did not fire for {module_path}:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
