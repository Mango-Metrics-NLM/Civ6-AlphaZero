"""The deployed images must run Python >= 3.11, and the build must say so loudly.

The images are assembled with ``PYTHONPATH=/app`` rather than ``pip install``, so
``requires-python = ">=3.11"`` in ``pyproject.toml`` is never enforced inside them. Until
2026-11-11 the base was ``nvcr.io/nvidia/pytorch:24.09-py3`` (Python 3.10) while the
Dockerfile comment claimed 3.12, and the Cloud Build guard only printed a warning. This
gate pins three things:

1. Both Dockerfiles build from one NGC PyTorch tag, at or above the first tag verified to
   ship Python >= 3.11 (26.10 ships 3.12). A later tag passes; an older one fails.
2. Every Cloud Build config's ``check-python-version`` step *exits non-zero* on a Python
   below 3.11 -- executed, not grepped, so a report-only rewrite is caught.
3. Every ``pip install -r`` in the Dockerfiles carries the torch constraint, so pip cannot
   replace NGC's CUDA-matched torch build with a PyPI wheel.

Each predicate is a pure function so the negative controls drive the same code.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCKERFILES = ("dockerfiles/Dockerfile.vertex", "dockerfiles/Dockerfile.inference")
CLOUD_BUILD_CONFIGS = (
    "cloudbuild.yaml",
    "dockerfiles/cloudbuild.yaml",
    "infra/cloudbuild-gameplay.yaml",
)
#: First NGC PyTorch tag verified (release notes) to ship Python >= 3.11 -- 26.10 is 3.12.
MIN_NGC_TAG = (26, 10)
CHECK_STEP_ID = "check-python-version"
TORCH_CONSTRAINT = "-c /tmp/torch-constraint.txt"

_NGC_FROM = re.compile(r"^FROM\s+nvcr\.io/nvidia/pytorch:(\d+)\.(\d+)-py3\b", re.MULTILINE)
_PIP_INSTALL_R = re.compile(r"pip install[^\n&]*-r\s+\S+")


def ngc_tags(text: str) -> list[tuple[int, int]]:
    return [(int(y), int(m)) for y, m in _NGC_FROM.findall(text)]


def check_step_script(doc: dict) -> str | None:
    for step in doc.get("steps") or []:
        if step.get("id") == CHECK_STEP_ID:
            return str(step["args"][-1])
    return None


def script_exit_code(script: str, version: tuple[int, int]) -> int:
    """Run a check script as if under ``version`` (the step's own ``import sys`` sees it)."""
    # A namedtuple, not a plain tuple: a script that reads ``.major`` must not crash, or a
    # report-only step would exit non-zero for the wrong reason and pass this gate.
    prelude = (
        "import collections, sys\n"
        "VI = collections.namedtuple('VI', 'major minor micro releaselevel serial')\n"
        f"sys.version_info = VI({version[0]}, {version[1]}, 0, 'final', 0)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + script], capture_output=True, check=False
    ).returncode


def unconstrained_pip_installs(text: str) -> list[str]:
    return [m for m in _PIP_INSTALL_R.findall(text) if TORCH_CONSTRAINT not in m]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


class TestBaseImage:
    def test_dockerfiles_share_one_ngc_tag_at_or_above_the_floor(self) -> None:
        tags = {rel: ngc_tags(_read(rel)) for rel in DOCKERFILES}
        assert all(len(t) == 1 for t in tags.values()), f"expected one NGC FROM each: {tags}"
        distinct = {t[0] for t in tags.values()}
        assert len(distinct) == 1, f"Dockerfiles disagree on the base tag: {tags}"
        (tag,) = distinct
        assert tag >= MIN_NGC_TAG, f"NGC {tag} predates {MIN_NGC_TAG}, the Python 3.11+ floor"

    def test_every_pip_install_keeps_ngc_torch(self) -> None:
        found = {rel: _PIP_INSTALL_R.findall(_read(rel)) for rel in DOCKERFILES}
        assert all(found.values()), f"no pip install parsed -- vacuous: {found}"
        bad = {rel: unconstrained_pip_installs(_read(rel)) for rel in DOCKERFILES}
        assert not {k: v for k, v in bad.items() if v}, bad

    def test_cloud_build_check_fails_below_311_and_passes_at_312(self) -> None:
        for rel in CLOUD_BUILD_CONFIGS:
            script = check_step_script(yaml.safe_load(_read(rel)))
            assert script, f"{rel}: no {CHECK_STEP_ID} step -- the check would be vacuous"
            assert script_exit_code(script, (3, 10)) != 0, f"{rel}: 3.10 does not fail the build"
            assert script_exit_code(script, (3, 12)) == 0, f"{rel}: 3.12 fails the build"


class TestNegativeControls:
    def test_old_base_is_caught(self) -> None:
        assert ngc_tags("FROM nvcr.io/nvidia/pytorch:24.09-py3\n") == [(24, 9)]
        assert (24, 9) < MIN_NGC_TAG

    def test_report_only_check_is_caught(self) -> None:
        # The step as it stood before the base moved: warns, reads v.major, exits 0.
        warn = (
            'import sys; v = sys.version_info; print("python", sys.version); '
            'v >= (3, 11) or print("WARNING: image Python", f"{v.major}.{v.minor}")'
        )
        assert script_exit_code(warn, (3, 10)) == 0

    def test_unconstrained_install_is_caught(self) -> None:
        text = "RUN pip install --no-cache-dir -r requirements-container.txt\n"
        assert unconstrained_pip_installs(text)
        assert not unconstrained_pip_installs(text.replace("-r", f"{TORCH_CONSTRAINT} -r"))
