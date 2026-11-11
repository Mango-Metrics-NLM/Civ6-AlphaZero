#!/usr/bin/env python3
"""Run the fixed benchmark protocol; write ``result.json`` and ``report.md``.

Thin wrapper over :func:`alpha_zero.evaluation.benchmark.main` so the protocol,
the envelope and the CLI live in the package (and are tested there) while the
script stays the operator-facing entry point alongside its siblings.

Usage:
    python scripts/benchmark_suite.py --checkpoint checkpoints/civ6/latest.pt \\
        --opponents random,heuristic,simple_mcts --seed 42 --out docs/evidence/bench_x/
"""

from __future__ import annotations

import sys

from alpha_zero.evaluation.benchmark import main

if __name__ == "__main__":
    sys.exit(main())
