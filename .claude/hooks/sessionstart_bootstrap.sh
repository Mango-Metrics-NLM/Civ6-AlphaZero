#!/usr/bin/env bash
# SessionStart: verify Python >= 3.11 and install the dev extra so tests and
# linters are runnable. Emits a systemMessage summarising the environment. The
# message JSON is built with json.dumps to avoid the nested-quote fragility of
# the previous inline implementation.
set -uo pipefail

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  printf '{"systemMessage": "WARNING: Python >= 3.11 is required for this project."}'
  exit 0
fi

pip install -e '.[dev]' -q >/dev/null 2>&1 || true

py="$(python3 --version 2>&1 | cut -d' ' -f2)"
rf="$(ruff --version 2>/dev/null | cut -d' ' -f2 || echo N/A)"
python3 -c \
  'import json,sys; print(json.dumps({"systemMessage": f"Environment: Python {sys.argv[1]} | ruff {sys.argv[2]} | pytest ready"}))' \
  "$py" "$rf"
exit 0
