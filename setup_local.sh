#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

"$PYTHON" -c '
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ required")
'

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"

echo "Installation complete."
.venv/bin/triplet --help
