#!/usr/bin/env bash
# Build smoke test — verifies the wheel installs and imports correctly
# in a fresh virtual environment.
#
# Usage:
#   bash scripts/smoke_build.sh
#
# Checks:
#   1. python -m build succeeds
#   2. Base install: import langchain_age works
#   3. [vector] extra: import AGEVector works
#   4. [graph] extra: import AGEGraph works
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "=== Building wheel ==="
"$PYTHON" -m pip install --quiet build
"$PYTHON" -m build --quiet
echo "  OK: dist/ created"

WHEEL=$(ls dist/langchain_age-*.whl | head -1)
echo "  Wheel: $WHEEL"

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
    "$tmpdir/venv/bin/python" -c "$import_cmd"
    echo "  OK"

    rm -rf "$tmpdir"
}

# Base install (no extras)
run_smoke "base" \
    "$WHEEL" \
    "import langchain_age; print(f'  version={langchain_age.__version__}')"

# [vector] extra
run_smoke "vector" \
    "${WHEEL}[vector]" \
    "from langchain_age import AGEVector, DistanceStrategy; print(f'  AGEVector={AGEVector}')"

# [graph] extra — requires git dep, may be slow
run_smoke "graph" \
    "${WHEEL}[graph]" \
    "from langchain_age import AGEGraph; print(f'  AGEGraph={AGEGraph}')"

echo ""
echo "=== All smoke tests passed ==="
