"""Opt-in pytest plugin enforcing that CUDA/GPU skips fail on GPU hosts.

Activated by ``CIV6AZ_REQUIRE_GPU=<n>`` (positive integer). When unset or empty
the plugin takes no action so CPU-only CI environments continue to skip CUDA
tests normally. When active:

1. Session startup fails with ``pytest.UsageError`` / ``ExitCode.USAGE_ERROR`` if
   ``torch.cuda.device_count() < n``.
2. Any skipped test (non-``wasxfail``) whose skip reason matches
   ``CIV6AZ_GPU_SKIP_PATTERN`` (default ``r"cuda|gpu|device"``, case-insensitive)
   is converted into a failure so silent hardware skips cannot mask GPU defects.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Generator, Mapping
from typing import Any

import pytest

logger = logging.getLogger(__name__)

REQUIRE_GPU_ENV = "CIV6AZ_REQUIRE_GPU"
GPU_SKIP_PATTERN_ENV = "CIV6AZ_GPU_SKIP_PATTERN"
DEFAULT_GPU_SKIP_PATTERN = r"cuda|gpu|device"


def required_gpu_count(environ: Mapping[str, str] | None = None) -> int | None:
    """Parse ``CIV6AZ_REQUIRE_GPU`` from *environ*, or return ``None`` if unset/empty."""
    env = os.environ if environ is None else environ
    raw = env.get(REQUIRE_GPU_ENV, "").strip()
    if not raw:
        return None
    try:
        count = int(raw)
    except ValueError as exc:
        raise pytest.UsageError(
            f"{REQUIRE_GPU_ENV}={raw!r} must be a positive integer >= 1"
        ) from exc
    if count < 1:
        raise pytest.UsageError(f"{REQUIRE_GPU_ENV}={raw!r} must be >= 1")
    return count


def compile_skip_pattern(environ: Mapping[str, str] | None = None) -> re.Pattern[str]:
    """Compile the GPU skip reason regex from ``CIV6AZ_GPU_SKIP_PATTERN`` or default."""
    env = os.environ if environ is None else environ
    raw = env.get(GPU_SKIP_PATTERN_ENV, "").strip() or DEFAULT_GPU_SKIP_PATTERN
    try:
        return re.compile(raw, re.IGNORECASE)
    except re.error as exc:
        raise pytest.UsageError(f"Invalid regex in {GPU_SKIP_PATTERN_ENV}={raw!r}: {exc}") from exc


def extract_skip_reason(longrepr: Any) -> str:
    """Extract human-readable skip text from pytest ``report.longrepr``."""
    if isinstance(longrepr, tuple) and len(longrepr) >= 3:
        text = str(longrepr[2])
    elif longrepr is None:
        return ""
    else:
        text = str(longrepr)
    if text.startswith("Skipped: "):
        return text[len("Skipped: ") :]
    return text


def apply_skip_gate(
    rep: Any,
    *,
    nodeid: str,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Convert a GPU-related skip report to ``failed`` when ``CIV6AZ_REQUIRE_GPU`` is active.

    Returns ``True`` when the report outcome was converted to ``failed``.
    """
    if getattr(rep, "outcome", None) != "skipped" or getattr(rep, "wasxfail", None):
        return False
    required = required_gpu_count(environ)
    if required is None:
        return False
    reason = extract_skip_reason(getattr(rep, "longrepr", None))
    pattern = compile_skip_pattern(environ)
    if not reason or not pattern.search(reason):
        return False
    logger.debug(
        "%s=%d converted GPU skip on %s to failure: %s",
        REQUIRE_GPU_ENV,
        required,
        nodeid,
        reason,
    )
    rep.outcome = "failed"
    rep.longrepr = f"{REQUIRE_GPU_ENV}={required}: a GPU skip is not a pass: {reason}"
    return True


def pytest_configure(config: pytest.Config) -> None:
    """Validate gate env vars at configure time when ``CIV6AZ_REQUIRE_GPU`` is set."""
    required = required_gpu_count(os.environ)
    if required is None:
        return
    compile_skip_pattern(os.environ)
    logger.debug("%s=%d gate enabled for session", REQUIRE_GPU_ENV, required)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail immediately at session start if fewer than ``CIV6AZ_REQUIRE_GPU`` GPUs are visible."""
    required = required_gpu_count(os.environ)
    if required is None:
        return
    import torch

    visible = torch.cuda.device_count()
    logger.debug("%s=%d checked torch.cuda.device_count()=%d", REQUIRE_GPU_ENV, required, visible)
    if visible < required:
        pytest.exit(
            f"{REQUIRE_GPU_ENV}={required} requires >= {required} CUDA device(s), "
            f"found device_count={visible}",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[object]
) -> Generator[None, Any, None]:
    """Fail any CUDA/GPU skip when ``CIV6AZ_REQUIRE_GPU`` is active."""
    outcome = yield
    rep = outcome.get_result()
    apply_skip_gate(rep, nodeid=item.nodeid, environ=os.environ)
