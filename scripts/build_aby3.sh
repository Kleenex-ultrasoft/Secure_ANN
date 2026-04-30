#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/aby3 [build_dir]" >&2
  exit 1
fi

ABY3_ROOT="$1"
BUILD_DIR="${2:-$ABY3_ROOT/build}"
LIBOTE_PREFIX="${LIBOTE_PREFIX:-$ABY3_ROOT/libOTe/out/install/linux}"
JOBS="${JOBS:-8}"

if [[ ! -f "$LIBOTE_PREFIX/lib/cmake/libOTe/libOTeConfig.cmake" ]]; then
  echo "libOTe install prefix not found: $LIBOTE_PREFIX" >&2
  echo "Build libOTe first so the install tree exists (lib/cmake/libOTe)." >&2
  echo "Typical flow: cd /path/to/aby3 && python3 build.py --setup && python3 build.py" >&2
  echo "Or set LIBOTE_PREFIX to a custom install prefix." >&2
  exit 1
fi

cmake -S "$ABY3_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_PREFIX_PATH="$LIBOTE_PREFIX" \
  -DlibOTe_DIR="$LIBOTE_PREFIX/lib/cmake/libOTe" \
  -DcryptoTools_DIR="$LIBOTE_PREFIX/lib/cmake/cryptoTools" \
  -DCOPROTO_NO_SYSTEM_PATH=ON

cmake --build "$BUILD_DIR" --target hnsecw_search_aby3 -j"$JOBS"
