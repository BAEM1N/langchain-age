#!/usr/bin/env bash
# Build smoke test — verifies the wheel installs and imports correctly
# in a fresh virtual environment, isolated from the source tree.
#
# Usage:
#   bash scripts/smoke_build.sh
#
# Checks:
#   1. python -m build succeeds (clean dist/)
#   2. Base install: import langchain_age works from site-packages
#   3. [vector] extra: import AGEVector works
#   4. [graph] extra: import AGEGraph works
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "=== Cleaning dist/ ==="
rm -rf dist/
echo "  OK"

echo "=== Building wheel ==="
"$PYTHON" -m pip install --quiet build
"$PYTHON" -m build --quiet
echo "  OK"

# Pick the freshly built wheel (only one after clean)
WHEEL=$(ls -t dist/langchain_age-*.whl | head -1)
WHEEL_ABS="$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"
echo "  Wheel: $WHEEL_ABS"

run_smoke() {
    local label="$1"
    local install_spec="$2"
    local import_cmd="$3"
    local tmpdir
    tmpdir=$(mktemp -d)

    echo ""
    echo "=== Smoke: $label ==="
    "$PYTHON" -m venv "$tmpdir/venv"
    "$tmpdir/venv/bin/python" -m pip install --quiet --upgrade pip
    "$tmpdir/venv/bin/python" -m pip install --quiet "$install_spec"

    # Run import from tmpdir with -I (isolated mode) to prevent
    # the source tree from shadowing the installed package.
    (
        cd "$tmpdir"
        "./venv/bin/python" -I -c "
$import_cmd
import langchain_age
path = langchain_age.__file__
assert 'site-packages' in path, f'Imported from source tree, not wheel: {path}'
print(f'  path={path}')
"
    )
    echo "  OK"

    rm -rf "$tmpdir"
}

# Base install (no extras)
run_smoke "base" \
    "$WHEEL_ABS" \
    "import langchain_age; print(f'  version={langchain_age.__version__}')"

# [vector] extra
run_smoke "vector" \
    "${WHEEL_ABS}[vector]" \
    "from langchain_age import AGEVector, DistanceStrategy; print(f'  AGEVector={AGEVector}')"

# [graph] extra — requires git dep, may be slow
run_smoke "graph" \
    "${WHEEL_ABS}[graph]" \
    "from langchain_age import AGEGraph; print(f'  AGEGraph={AGEGraph}')"

echo ""
echo "=== All smoke tests passed ==="
