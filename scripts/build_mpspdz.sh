#!/usr/bin/env bash
# Builds the MP-SPDZ party for the 3PC program: clones MP-SPDZ at a fixed
# commit, applies 3pc/patches/mpspdz.patch,
# compiles replicated-ring-party.x and creates the party certificates.
#
#   MPSPDZ_ROOT=/path/to/MP-SPDZ bash scripts/build_mpspdz.sh
#
# Packages needed on Ubuntu: automake build-essential clang cmake git
# libboost-dev libboost-iostreams-dev libboost-thread-dev libgmp-dev
# libntl-dev libsodium-dev libssl-dev libtool python3
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
MPSPDZ_ROOT=${MPSPDZ_ROOT:-$REPO/build/MP-SPDZ}
MPSPDZ_COMMIT=d39bd91c08801cb195012d6df5267eac3a9c5ef1

if [ ! -d "$MPSPDZ_ROOT/.git" ]; then
  mkdir -p "$(dirname "$MPSPDZ_ROOT")"
  git clone https://github.com/data61/MP-SPDZ.git "$MPSPDZ_ROOT"
fi
cd "$MPSPDZ_ROOT"
git checkout -q "$MPSPDZ_COMMIT"
git submodule update --init --recursive

PATCH=$REPO/3pc/patches/mpspdz.patch
if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "already applied: $(basename "$PATCH")"
else
  git apply "$PATCH"
  echo "applied: $(basename "$PATCH")"
fi

make -j"$(nproc)" replicated-ring-party.x
[ -f Player-Data/P0.pem ] || Scripts/setup-ssl.sh 3
echo "built $MPSPDZ_ROOT/replicated-ring-party.x"
