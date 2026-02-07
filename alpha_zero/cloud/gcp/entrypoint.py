"""
Container entrypoint that routes to the correct worker based on --mode flag.

Vertex AI passes all args to the container. This script parses --mode
and dispatches to the appropriate worker module.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    # Parse --mode while leaving remaining args for the target worker's argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", default="selfplay")
    args, remaining_args = parser.parse_known_args()
    mode = args.mode

    # Replace sys.argv for the target module's argparse
    sys.argv = [sys.argv[0]] + remaining_args

    if mode == "selfplay":
        from alpha_zero.cloud.gcp.selfplay_worker import main as worker_main

        return worker_main()
    elif mode == "training":
        from alpha_zero.cloud.gcp.training_worker import main as worker_main

        return worker_main()
    elif mode == "evaluation":
        from alpha_zero.cloud.gcp.evaluation_worker import main as worker_main

        return worker_main()
    else:
        print(
            f"Unknown mode: {mode}. Use 'selfplay', 'training', or 'evaluation'.", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
