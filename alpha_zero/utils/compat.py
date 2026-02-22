"""Python version compatibility shims.

Provides re-exports that normalize differences between Python versions,
allowing the codebase to work across Python 3.10+ (container) and 3.12+ (dev).

The container base image (nvcr.io/nvidia/pytorch:24.09-py3) uses Python 3.10,
while the dev environment uses Python 3.12+. datetime.UTC was added in 3.11.

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
