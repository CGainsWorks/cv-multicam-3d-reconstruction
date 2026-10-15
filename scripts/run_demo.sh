#!/usr/bin/env bash
# The one command that works on a clean machine.
#
#   ./scripts/run_demo.sh
#
# Sets up a virtual environment if there is not one already, then renders a
# miniature capture and runs calibration, reconstruction and 6-DoF pose
# estimation over it, checking every number against the ground truth it
# generated.  About four minutes.  Nothing is downloaded except Python
# packages.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-.venv}"
if [ -x "$VENV/bin/python" ]; then
  VPY="$VENV/bin/python"
elif [ -x "$VENV/Scripts/python.exe" ]; then
  VPY="$VENV/Scripts/python.exe"
else
  echo "==> no virtual environment yet; running scripts/setup.sh"
  ./scripts/setup.sh
  if [ -x "$VENV/bin/python" ]; then VPY="$VENV/bin/python"; else VPY="$VENV/Scripts/python.exe"; fi
fi

mkdir -p results
exec "$VPY" scripts/demo.py --out results/demo.json "$@"
