"""
Entry point for the Vertex AI CLI.

Usage:
    python -m alpha_zero.cloud.gcp --help
    python -m alpha_zero.cloud.gcp init --tier budget
    python -m alpha_zero.cloud.gcp run --iterations 100
"""

import sys

from alpha_zero.cloud.gcp.cli import main

if __name__ == "__main__":
    sys.exit(main())
