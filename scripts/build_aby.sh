#!/usr/bin/env bash
# Builds hnsecw_cli: clones ABY at a fixed commit, applies the patches in
# 2pc/patches, adds the HNSecW sources as an ABY example and compiles.
#
#   ABY_ROOT=/path/to/ABY bash scripts/build_aby.sh
#
# The binary ends up in $ABY_ROOT/build/bin/hnsecw_cli.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ABY_ROOT=${ABY_ROOT:-$REPO/build/ABY}
ABY_COMMIT=88fed3ef6789580cac342201e135a358a083ca36

if [ ! -d "$ABY_ROOT/.git" ]; then
  mkdir -p "$(dirname "$ABY_ROOT")"
  git clone https://github.com/encryptogroup/ABY.git "$ABY_ROOT"
fi
cd "$ABY_ROOT"
git checkout -q "$ABY_COMMIT"
git submodule update --init --recursive

apply_once() {  # dir patch
  if git -C "$1" apply --reverse --check "$2" 2>/dev/null; then
    echo "already applied: $(basename "$2")"
  else
    git -C "$1" apply "$2"
    echo "applied: $(basename "$2")"
  fi
}
apply_once "$ABY_ROOT" "$REPO/2pc/patches/aby.patch"
apply_once "$ABY_ROOT/extern/ENCRYPTO_utils" "$REPO/2pc/patches/encrypto_utils.patch"

mkdir -p src/examples/hnsecw
cp "$REPO"/2pc/src/hnsecw/* src/examples/hnsecw/
grep -q 'add_subdirectory(hnsecw)' src/examples/CMakeLists.txt ||
  echo 'add_subdirectory(hnsecw)' >> src/examples/CMakeLists.txt

# ABY relies on transitive includes that newer compilers no longer provide.
cmake -B build -S . -DABY_BUILD_EXE=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS='-include cstdlib -include cstdint'
cmake --build build -j"$(nproc)" --target hnsecw_cli
echo "built $ABY_ROOT/build/bin/hnsecw_cli"
