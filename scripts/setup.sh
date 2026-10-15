#!/usr/bin/env bash
# Virtual environment, dependencies, then the tests.
#
#   ./scripts/setup.sh
#
# Nothing is downloaded except Python packages: every image this repository
# uses is rendered from a seed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi

version="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$version" in
  3.1[0-9]|3.[2-9][0-9]) ;;
  *)
    echo "Python 3.10 or newer is required; found $version via $PYTHON." >&2
    echo "Set PYTHON=/path/to/python3.12 and try again." >&2
    exit 1
    ;;
esac

echo "==> virtual environment in $VENV (Python $version)"
"$PYTHON" -m venv "$VENV"

if [ -x "$VENV/bin/python" ]; then
  VPY="$VENV/bin/python"
else
  VPY="$VENV/Scripts/python.exe"   # Windows layout
fi

echo "==> dependencies"
"$VPY" -m pip install --quiet --upgrade pip

# Torch from the CPU index unless a CUDA build is wanted.  The default PyPI
# wheel pulls the whole CUDA runtime, which is well over a gigabyte and is
# wasted on a machine without an NVIDIA card.  Set TORCH_INDEX to override, for
# example TORCH_INDEX=https://download.pytorch.org/whl/cu121
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
"$VPY" -m pip install --quiet torch --index-url "$TORCH_INDEX"
"$VPY" -m pip install --quiet -r requirements.txt

echo "==> tests"
"$VPY" -m pytest tests/ -q

cat <<EOF

Done.  Activate with:

    source $VENV/bin/activate       # or  $VENV\\Scripts\\activate  on Windows

Then:

    make demo          the whole pipeline on a miniature capture, about 4 minutes
    make data          render the full capture, about 20 minutes
    make experiments   every number the README quotes, about 40 minutes after that
EOF
