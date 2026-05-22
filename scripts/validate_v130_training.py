"""Deprecated. Use ``scripts/validate_convergence.py`` instead."""
from __future__ import annotations

import sys
import warnings

warnings.warn(
    "validate_v130_training.py is deprecated; use validate_convergence.py.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    from scripts.validate_convergence import main
    sys.exit(main())
