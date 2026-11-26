"""The live-GCS test module needs an explicit opt-in and never sees the bucket.

``tests/e2e/test_gcs_live.py`` once read ``CIV6AZ_GCS_BUCKET`` at import time,
before the function-scoped strip fixture could run. With that variable and
Application Default Credentials on a workstation, an ordinary suite run could
upload to and delete from the real bucket.

The subprocess runs here are hermetic on purpose: credentials are pointed at
empty directories and HTTP is routed to a dead port, so a regression fails the
assertion instead of reaching GCS.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import LIVE_GCS_OPT_IN_ENV, scrub_cloud_environment

ROOT = Path(__file__).resolve().parents[2]
SENTINEL = "civ6-sentinel-bucket-never-real"


def _hermetic_env(tmp_path: Path) -> dict[str, str]:
    empty = tmp_path / "empty"
    empty.mkdir()
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GOOGLE_", "VERTEX_", "CIV6AZ_GCS", "CIV6AZ_LIVE"))
    }
    env.update(
        {
            "CIV6AZ_GCS_BUCKET": SENTINEL,
            "CLOUDSDK_CONFIG": str(empty),
            "NO_GCE_CHECK": "True",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
        }
    )
    return env


def test_scrub_removes_cloud_variables_without_opt_in() -> None:
    environ = {
        "CIV6AZ_GCS_BUCKET": "b",
        "GOOGLE_APPLICATION_CREDENTIALS": "/x",
        "VERTEX_REGION": "r",
        "PATH": "/bin",
    }
    removed = scrub_cloud_environment(environ)
    assert environ == {"PATH": "/bin"}
    assert sorted(removed) == [
        "CIV6AZ_GCS_BUCKET",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "VERTEX_REGION",
    ]


def test_scrub_keeps_variables_under_opt_in() -> None:
    environ = {LIVE_GCS_OPT_IN_ENV: "1", "CIV6AZ_GCS_BUCKET": "b"}
    assert scrub_cloud_environment(environ) == []
    assert environ["CIV6AZ_GCS_BUCKET"] == "b"


@pytest.mark.parametrize("value", ["0", "true", "yes", ""])
def test_only_exactly_one_opts_in(value: str) -> None:
    environ = {LIVE_GCS_OPT_IN_ENV: value, "CIV6AZ_GCS_BUCKET": "b"}
    scrub_cloud_environment(environ)
    assert "CIV6AZ_GCS_BUCKET" not in environ


def test_live_module_is_skipped_and_bucket_unseen_without_opt_in(tmp_path: Path) -> None:
    code = (
        "import os, sys, pytest\n"
        "rc = pytest.main(['tests/e2e/test_gcs_live.py', '-q', '-rs', '-p', 'no:cacheprovider'])\n"
        "print('BUCKET_AFTER=' + repr(os.environ.get('CIV6AZ_GCS_BUCKET')))\n"
        "sys.exit(int(rc))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_hermetic_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=110,
        check=False,
    )
    out = proc.stdout + proc.stderr
    assert "BUCKET_AFTER=None" in out, out
    assert "skipped" in out and LIVE_GCS_OPT_IN_ENV in out, out
    assert "passed" not in out and "failed" not in out and "error" not in out.lower(), out
    assert proc.returncode in (0, 5), out  # 5: the skipped module collects no tests
