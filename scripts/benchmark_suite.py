#!/usr/bin/env python3
"""Run the fixed benchmark protocol; write ``result.json`` and ``report.md``.

Thin wrapper over :func:`alpha_zero.evaluation.benchmark.main` so the protocol,
the envelope and the CLI live in the package (and are tested there) while the
script stays the operator-facing entry point alongside its siblings.

Usage:
    python scripts/benchmark_suite.py --checkpoint checkpoints/civ6/latest.pt \\
        --opponents random,heuristic,simple_mcts --seed 42 --out docs/evidence/bench_x/
    python scripts/benchmark_suite.py ... --shard 1/2 --out s1/   # then --shard 2/2
    python scripts/benchmark_suite.py --merge s1/ s2/ --out merged/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run THIS checkout's package, never a stale editable install of another one: the
# script's own directory, not the repository root, heads sys.path (as in run_card.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpha_zero.evaluation.benchmark import main

if __name__ == "__main__":
    sys.exit(main())
