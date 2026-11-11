#!/usr/bin/env python3
"""Run a run card: price it, pin its SHA, isolate it from GCP, run it, record run.json.

Thin CLI over :mod:`alpha_zero.local.run_card`; see that module for the contract.

Usage::

    python scripts/run_card.py configs/run_cards/smoke.yaml --price_only
    python scripts/run_card.py configs/run_cards/smoke.yaml --data_root E:\\civ6az-data

Exit codes: 0 COMPLETED (or priced), 1 FAILED, 2 refused, 130 VOID (interrupted).
"""

from __future__ import annotations

import sys
from pathlib import Path

# The driver must be THIS checkout's, never a stale editable install of another one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_zero.local.run_card import main

if __name__ == "__main__":
    sys.exit(main())
