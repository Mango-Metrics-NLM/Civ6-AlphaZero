"""``test_guard_decision_json.py`` must import and build its PATH on Windows (PR #171 review).

CI is Linux, so this simulates Windows in a child interpreter: ``os.name`` and ``sys.platform``
say Windows, ``os.fsdecode`` raises on a byte that is not UTF-8 (as it does there), and
``Path.symlink_to`` raises (no symlink privilege). The module must import, leave the raw-byte
case out, and build its restricted PATH without a symbolic link.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import os, sys, pathlib, tempfile
import pytest  # noqa: F401  (imported before the platform is faked)
import tests.governance._guard_harness  # noqa: F401
real_fsdecode = os.fsdecode
def fake_fsdecode(b):
    if isinstance(b, bytes):
        b.decode("utf-8")  # raises on 0xff, as on Windows
    return real_fsdecode(b)
def no_symlink(self, *a, **k):
    raise OSError("symbolic link privilege not held")
real_name = os.name
import shutil
found = {t: shutil.which(t) for t in ("git", "cat")}  # resolved before the platform is faked
shutil.which = lambda tool, *a, **k: found.get(tool)
os.fsdecode = fake_fsdecode
pathlib.Path.symlink_to = no_symlink
os.name = "nt"
sys.platform = "win32"
import tests.governance.test_guard_decision_json as m
os.name = real_name  # the module reads it at import; restore host os.name before Path(d)
assert "raw_byte" not in m.ODD_NAMES, m.ODD_NAMES
with tempfile.TemporaryDirectory() as d:
    path = m._bin_dir(pathlib.Path(d), ("git", "cat"))
assert path, path
print("ok")
"""


def test_the_decision_json_tests_import_and_build_a_path_on_windows() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")
