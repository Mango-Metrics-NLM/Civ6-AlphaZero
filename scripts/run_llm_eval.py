#!/usr/bin/env python3
"""Run the langfuse-eval-harness over Civ6-AlphaZero's StrategyAdvisor.

Thin wrapper around the ``eval-harness`` CLI that:
  * degrades gracefully (exit 0, "not applicable") when the optional ``[eval]``
    extra is not installed, so it is safe to wire into CI unconditionally;
  * propagates the harness exit code (0 = gate pass, 1 = gate fail) when it is.

By default the StrategyAdvisor target runs OFFLINE with a deterministic stub LLM
(no API keys) — a contract/serialization regression gate. Pass ``--online`` to
drive a real provider (sets ``CIV6AZ_LLM_EVAL_ONLINE=1``); that requires
OPENAI/ANTHROPIC credentials in the environment.

With no --config it gates BOTH StrategyAdvisor modes (analyze + report). Pass
--config one or more times to run a specific subset.

Usage:
    python scripts/run_llm_eval.py [--config PATH ...] [--online] [-- <extra harness args>]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

# By default both StrategyAdvisor modes are gated: analyze (AnalysisResult) and
# report (GameReport). Pass one or more --config to override.
DEFAULT_CONFIGS = [
    "configs/eval/strategy_advisor.yaml",
    "configs/eval/game_report.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        metavar="PATH",
        help=(
            "Eval-harness config YAML; repeatable. "
            f"Default (when omitted): {', '.join(DEFAULT_CONFIGS)}."
        ),
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Drive a real LLM provider (sets CIV6AZ_LLM_EVAL_ONLINE=1; needs API keys).",
    )
    args, passthrough = parser.parse_known_args()

    if importlib.util.find_spec("eval_harness") is None:
        print(
            "[run_llm_eval] langfuse-eval-harness is not installed; "
            "skipping the StrategyAdvisor eval harness (not applicable).\n"
            "               Install it with:  pip install -e '.[eval]'",
            file=sys.stderr,
        )
        return 0

    configs = args.configs or list(DEFAULT_CONFIGS)

    env = dict(os.environ)
    if args.online:
        env["CIV6AZ_LLM_EVAL_ONLINE"] = "1"
    else:
        env.pop("CIV6AZ_LLM_EVAL_ONLINE", None)

    # Run each config in turn; the gate fails fast on the first non-zero exit so
    # CI surfaces the first regression. A clean run requires every config to pass.
    for config in configs:
        config_path = Path(config)
        if not config_path.is_file():
            print(f"[run_llm_eval] config file not found: {config_path}", file=sys.stderr)
            return 2

        # Invoke via the current interpreter's ``-m`` rather than the bare
        # ``eval-harness`` console script: ``sys.executable`` always exists, so
        # this avoids a FileNotFoundError when the script's bin dir is not on
        # PATH even though the package is importable (find_spec confirmed it is).
        cmd = [sys.executable, "-m", "eval_harness.cli", "run", "--config", str(config_path)]
        if not args.online:
            # --offline keeps the harness's Langfuse client in-memory (no network).
            cmd.append("--offline")
        cmd.extend(passthrough)

        print(f"[run_llm_eval] $ {' '.join(cmd)}")
        completed = subprocess.run(cmd, env=env, check=False)
        if completed.returncode != 0:
            return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
