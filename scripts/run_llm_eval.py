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

Usage:
    python scripts/run_llm_eval.py [--config PATH] [--online] [-- <extra harness args>]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = "configs/eval/strategy_advisor.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Eval-harness config YAML (default: {DEFAULT_CONFIG}).",
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
            "skipping the LLM eval gate (not applicable).\n"
            "               Install it with:  pip install -e '.[eval]'",
            file=sys.stderr,
        )
        return 0

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"[run_llm_eval] config file not found: {config_path}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    if args.online:
        env["CIV6AZ_LLM_EVAL_ONLINE"] = "1"
    else:
        env.pop("CIV6AZ_LLM_EVAL_ONLINE", None)

    cmd = ["eval-harness", "run", "--config", str(config_path)]
    if not args.online:
        # --offline keeps the harness's Langfuse client in-memory (no network).
        cmd.append("--offline")
    cmd.extend(passthrough)

    print(f"[run_llm_eval] $ {' '.join(cmd)}")
    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
