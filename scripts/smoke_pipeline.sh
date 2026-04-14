#!/usr/bin/env bash
#
# Build the static library and the gtest binary, then run the
# tests. Does NOT require SCL or hiredis at runtime; the test
# binary links only against the pure compute units.
#
# Usage:
#   bash scripts/smoke_pipeline.sh
#
# Exits non-zero on any failure. Used by CI and for local
# pre-push sanity.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"

echo "[1/3] configure"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Debug

echo "[2/3] build"
cmake --build "$BUILD" -j 2

echo "[3/3] test"
ctest --test-dir "$BUILD" --output-on-failure

echo "all green."
