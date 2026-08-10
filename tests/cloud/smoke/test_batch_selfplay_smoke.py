"""Smoke test: selfplay_worker --dry_run with batch mode.

Exercises the subprocess entry point that Cloud Build invokes. Catches
regressions in the CLI argument plumbing for --num_parallel_games /
--max_batch_size that the function-level tests in
tests/integration/test_selfplay_worker_batch.py and
tests/integration/test_batch_selfplay_integration.py cannot see.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.cloud
@pytest.mark.smoke
def test_selfplay_worker_batch_dry_run(tmp_path):
    """selfplay_worker --dry_run with --num_parallel_games=2 exits 0."""
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": f"file:{tmp_path}/mlruns",
        "MLFLOW_ENABLED": "true",
        "MLFLOW_EXPERIMENT_NAME": "smoke-batch",
        "PYTHONUNBUFFERED": "1",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "alpha_zero.cloud.gcp.selfplay_worker",
            "--dry_run",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--games=2",
            "--mcts_type=gumbel",
            "--mcts_simulations=2",
            "--num_parallel_games=2",
            "--max_batch_size=4",
            "--board_size=8",
            "--max_turns=5",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=110,
        env=env,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
