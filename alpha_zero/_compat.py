"""Python version compatibility shim.

This module patches ``datetime.UTC`` for Python < 3.11, where it does not
exist natively. ``UTC`` was added in Python 3.11 (PEP 615 follow-up).

Import this module BEFORE any ``from datetime import UTC`` to ensure
backward compatibility with Python 3.10 containers.

Usage (at package level ``__init__.py`` or entry point):
    import alpha_zero._compat  # noqa: F401
"""

from __future__ import annotations

import datetime
import sys

if sys.version_info < (3, 11):  # noqa: UP036  -- intentional: this module exists for <3.11 support
    # Python < 3.11: datetime.UTC does not exist — patch it in
    if not hasattr(datetime, "UTC"):
        datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]  # noqa: UP017
