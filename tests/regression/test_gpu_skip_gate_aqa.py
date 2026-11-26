"""AQA tests for the opt-in ``CIV6AZ_REQUIRE_GPU`` skip-to-failure pytest gate.

Covers:
(a) Unset ``CIV6AZ_REQUIRE_GPU`` leaves all skips untouched (CPU-only CI safe).
(b) Invalid ``CIV6AZ_REQUIRE_GPU`` (``0``, ``-1``, non-integer) fails with usage error.
(c) Skip-conversion logic turns CUDA/GPU skips into failures, preserves non-GPU skips,
    and respects custom ``CIV6AZ_GPU_SKIP_PATTERN`` without requiring live GPUs.
(d) Excessive ``CIV6AZ_REQUIRE_GPU=99`` aborts session start with exit code 4 naming
    the observed device count.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fixtures.gpu_skip_gate import (
    DEFAULT_GPU_SKIP_PATTERN,
    GPU_SKIP_PATTERN_ENV,
    REQUIRE_GPU_ENV,
    apply_skip_gate,
    extract_skip_reason,
    required_gpu_count,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE_TEST = """from pytest import skip

def test_gpu_skip():
    skip("CUDA not available")

def test_other_skip():
    skip("needs network")
"""


def _run_gate_subprocess(
    tmp_path: Path, env_overrides: dict[str, str | None]
) -> subprocess.CompletedProcess[str]:
    test_file = tmp_path / "test_probe.py"
    test_file.write_text(_PROBE_TEST, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.pop(GPU_SKIP_PATTERN_ENV, None)
    env.pop("PYTEST_ADDOPTS", None)
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.fixtures.gpu_skip_gate",
            str(test_file),
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--rootdir",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_unset_gate_leaves_skips_untouched(tmp_path: Path) -> None:
    """Case (a): with CIV6AZ_REQUIRE_GPU unset, both skips remain skipped."""
    proc = _run_gate_subprocess(tmp_path, {REQUIRE_GPU_ENV: None})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "2 skipped" in proc.stdout, proc.stdout


@pytest.mark.parametrize("bad_value", ["0", "-1", "abc", "1.5"])
def test_invalid_gate_value_errors(tmp_path: Path, bad_value: str) -> None:
    """Case (b): invalid CIV6AZ_REQUIRE_GPU raises UsageError and exits with error."""
    with pytest.raises(pytest.UsageError, match=REQUIRE_GPU_ENV):
        required_gpu_count({REQUIRE_GPU_ENV: bad_value})

    proc = _run_gate_subprocess(tmp_path, {REQUIRE_GPU_ENV: bad_value})
    assert proc.returncode == int(pytest.ExitCode.USAGE_ERROR), proc.stdout + proc.stderr
    assert REQUIRE_GPU_ENV in proc.stdout + proc.stderr


def test_skip_gate_conversion_and_custom_pattern() -> None:
    """Case (c): CUDA/GPU skips fail, unrelated skips stay skipped, custom pattern works."""
    gpu_rep = SimpleNamespace(
        outcome="skipped",
        wasxfail=None,
        longrepr=("test_x.py", 10, "Skipped: CUDA not available"),
    )
    assert apply_skip_gate(
        gpu_rep,
        nodeid="test_x.py::test_gpu",
        environ={REQUIRE_GPU_ENV: "2"},
    )
    assert gpu_rep.outcome == "failed"
    assert "CIV6AZ_REQUIRE_GPU=2" in str(gpu_rep.longrepr)
    assert "CUDA not available" in str(gpu_rep.longrepr)

    other_rep = SimpleNamespace(
        outcome="skipped",
        wasxfail=None,
        longrepr=("test_x.py", 12, "Skipped: needs network"),
    )
    assert not apply_skip_gate(
        other_rep,
        nodeid="test_x.py::test_net",
        environ={REQUIRE_GPU_ENV: "2"},
    )
    assert other_rep.outcome == "skipped"

    cpu_only_rep = SimpleNamespace(
        outcome="skipped",
        wasxfail=None,
        longrepr=("test_x.py", 14, "Skipped: requires a CPU-only host"),
    )
    assert not apply_skip_gate(
        cpu_only_rep,
        nodeid="test_x.py::test_cpu",
        environ={REQUIRE_GPU_ENV: "2"},
    )

    custom_rep = SimpleNamespace(
        outcome="skipped",
        wasxfail=None,
        longrepr=("test_x.py", 16, "Skipped: custom_nccl unavailable"),
    )
    assert apply_skip_gate(
        custom_rep,
        nodeid="test_x.py::test_custom",
        environ={REQUIRE_GPU_ENV: "1", GPU_SKIP_PATTERN_ENV: r"custom_nccl"},
    )
    assert custom_rep.outcome == "failed"

    empty_pattern_rep = SimpleNamespace(
        outcome="skipped",
        wasxfail=None,
        longrepr=("test_x.py", 18, "Skipped: needs network"),
    )
    assert not apply_skip_gate(
        empty_pattern_rep,
        nodeid="test_x.py::test_empty_pat",
        environ={REQUIRE_GPU_ENV: "2", GPU_SKIP_PATTERN_ENV: ""},
    )
    assert empty_pattern_rep.outcome == "skipped"
    assert extract_skip_reason(None) == ""
    assert DEFAULT_GPU_SKIP_PATTERN == r"cuda|gpu|device"


def test_excessive_gpu_requirement_exits_with_device_count(tmp_path: Path) -> None:
    """Case (d): requesting more GPUs than visible fails session start with ExitCode 4."""
    proc = _run_gate_subprocess(tmp_path, {REQUIRE_GPU_ENV: "99"})
    assert proc.returncode == int(pytest.ExitCode.USAGE_ERROR), proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "device_count=0" in combined, combined
    assert "99" in combined, combined
