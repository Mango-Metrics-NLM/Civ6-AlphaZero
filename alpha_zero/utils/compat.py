"""Python version compatibility shims.

Provides re-exports that normalize differences between Python versions,
allowing the codebase to work across Python 3.10+ (container) and 3.12+ (dev).

The shim dates from the container base nvcr.io/nvidia/pytorch:24.09-py3
(Python 3.10); datetime.UTC was added in 3.11. The base is now 26.10-py3
(Python 3.12) and requires-python is >=3.11, so the 3.10 branch is retained
only for compatibility with existing imports.

Usage:
    from alpha_zero.utils.compat import UTC
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):  # noqa: UP036
    from datetime import UTC
else:
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

__all__ = ["UTC"]
