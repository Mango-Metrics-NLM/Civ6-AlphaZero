"""Run pytest with IPv4-only DNS to bypass Windows IPv6 routing hang.

Root cause: Windows IPv6 routing is broken (WinError 10051), causing
`import torch` and other network-touching libraries to hang indefinitely.
This script patches socket.getaddrinfo BEFORE any imports occur.

Usage:
    python scripts/run_tests_ipv4.py [pytest args...]

Examples:
    python scripts/run_tests_ipv4.py tests/ -v --tb=short
    python scripts/run_tests_ipv4.py tests/cloud/unit/ -v
    python scripts/run_tests_ipv4.py tests/ --cov=alpha_zero --cov-report=html
"""

from __future__ import annotations

# ───────────────────────────────────────────────────────────────────────
# IPv4 enforcement — MUST be the very first thing before ANY imports
# ───────────────────────────────────────────────────────────────────────
import socket
from typing import Any

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(
    *args: Any, **kwargs: Any,
) -> list[tuple[socket.AddressFamily, int, int, str, tuple[str, int]]]:
    """Filter DNS results to IPv4 only, preventing IPv6 hangs."""
    results = _orig_getaddrinfo(*args, **kwargs)
    return [r for r in results if r[0] == socket.AF_INET]


socket.getaddrinfo = _ipv4_only_getaddrinfo  # type: ignore[assignment]

# Also disable torch distributed backends that try network ops at import
import os

os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
os.environ.setdefault("NCCL_DEBUG", "OFF")
os.environ.setdefault("GLOO_LOG_LEVEL", "OFF")
os.environ.setdefault("TORCHELASTIC_LOG_LEVEL", "OFF")
# Disable torch hub network access
os.environ.setdefault("TORCH_HOME", os.path.join(os.environ.get("TEMP", "."), "torch_home"))
# Disable any telemetry/update checks
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
os.environ.setdefault("NO_PROXY", "*")

# ───────────────────────────────────────────────────────────────────────
# Now safe to import everything
# ───────────────────────────────────────────────────────────────────────
import sys

def main() -> int:
    """Run pytest with forwarded CLI arguments."""
    import pytest  # noqa: E402 — must be after IPv4 patch

    # Forward all CLI args to pytest
    args = sys.argv[1:] if len(sys.argv) > 1 else ["tests/", "-v", "--tb=short"]
    print(f"[run_tests_ipv4] IPv4 patch active, running: pytest {' '.join(args)}")
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
