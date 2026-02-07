"""
Container entrypoint that routes to the correct worker based on --mode flag.

Vertex AI passes all args to the container. This script parses --mode
and dispatches to the appropriate worker module.
"""

from __future__ import annotations

import sys


def main() -> int:
    # Find --mode in args
    mode = "selfplay"
    remaining_args = []

    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
        elif arg == "--mode" and remaining_args:
            # Handle --mode value form (shouldn't happen but be safe)
            pass
        else:
            remaining_args.append(arg)

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
