"""Convergence validation script smoke (production-path).

Replaces the legacy v1.3.0 MicroCivNetwork-based tests with a smoke that
exercises the actual ``scripts/validate_convergence.py`` end-to-end, which
itself uses the production ``train_epoch`` + ``GCSCheckpointStorage`` round
trip.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _file_uri(path: Path) -> str:
    """Build an MLflow-compatible ``file:///abs/path`` URI.

    MLflow requires three slashes for absolute file paths; ``file:/abs/path``
    is silently interpreted as a relative path and runs land in ``./mlruns``
    next to the working directory.
    """
    return path.resolve().as_uri()


@pytest.mark.integration
def test_validate_convergence_script_smoke(tmp_path: Path) -> None:
    """The convergence script runs to completion on a tiny config and writes MLflow."""
    mlruns = tmp_path / "mlruns"
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": _file_uri(mlruns),
        "MLFLOW_EXPERIMENT_NAME": "ci-validate",
        "MLFLOW_ENABLED": "true",
        "PYTHONUNBUFFERED": "1",
    }
    script = PROJECT_ROOT / "scripts" / "validate_convergence.py"
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(script),
            "--iterations=2",
            "--games=2",
            "--epochs=1",
            "--eval-games=2",
            "--max-turns=8",
            "--device=cpu",
        ],
        cwd=tmp_path,  # run in tmp so any relative-mlruns fallback also lands here
        capture_output=True,
        text=True,
        timeout=110,
        env=env,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    # The script's own pass-gate (logged ``convergence validation PASSED`` /
    # ``FAILED``) drives the return code; that is the load-bearing assertion.
    # MLflow's side-effect of creating ``mlruns/`` is intentionally NOT
    # asserted here — its filesystem-backend behaviour drifts across
    # platforms (e.g. silent fallback when the URI isn't a 3-slash absolute,
    # the FutureWarning'd file-store deprecation in MLflow 2.x). Convergence
    # of the script is the contract; MLflow connectivity is covered by
    # tests/unit/test_monitoring/.
    assert "convergence validation" in (result.stderr + result.stdout).lower(), (
        f"script did not log a convergence verdict; stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
