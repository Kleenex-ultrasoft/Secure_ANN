#!/usr/bin/env bash
set -euo pipefail

MP_SPDZ=${MP_SPDZ:?set MP_SPDZ to MP-SPDZ root}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR" && pwd)

N=1024
ENTRY_WORDS=1
WORD_BITS=32
ACCESSES=10
IMPL="optimal"
PROTOCOL="ring"
RING_BITS=64
LOG_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2;;
    --entry-words) ENTRY_WORDS="$2"; shift 2;;
    --word-bits) WORD_BITS="$2"; shift 2;;
    --accesses) ACCESSES="$2"; shift 2;;
    --impl) IMPL="$2"; shift 2;;
    --protocol) PROTOCOL="$2"; shift 2;;
    --ring-bits) RING_BITS="$2"; shift 2;;
    --log-tag) LOG_TAG="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

cp "$ROOT_DIR/Programs/Source/oram_bench.mpc" "$MP_SPDZ/Programs/Source/"

cd "$MP_SPDZ"
BIN="replicated-ring-party.x"
if [[ "$PROTOCOL" == "ring" ]]; then
  ORAM_N="$N" ORAM_ENTRY_WORDS="$ENTRY_WORDS" ORAM_WORD_BITS="$WORD_BITS" ORAM_ACCESSES="$ACCESSES" ORAM_IMPL="$IMPL" ./compile.py -R "$RING_BITS" oram_bench
else
  ORAM_N="$N" ORAM_ENTRY_WORDS="$ENTRY_WORDS" ORAM_WORD_BITS="$WORD_BITS" ORAM_ACCESSES="$ACCESSES" ORAM_IMPL="$IMPL" ./compile.py -B 1 oram_bench
  BIN="replicated-bin-party.x"
fi

LOG_PREFIX=${LOG_PREFIX:-}
LOG_SUFFIX=${LOG_SUFFIX:-}
BENCH=${BENCH:-}
LOGPROT=${LOGPROT:-}
GDB_PLAYER=${GDB_PLAYER:-}
prefix=${prefix:-}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH:-}
SPDZROOT="$MP_SPDZ"
PLAYERS=3
set +e
source Scripts/run-common.sh
set -e

TAG="${LOG_TAG:-}oram_N${N}_w${ENTRY_WORDS}_acc${ACCESSES}_${IMPL}_"
LOG_PREFIX="$TAG" \
  ORAM_N="$N" ORAM_ENTRY_WORDS="$ENTRY_WORDS" ORAM_WORD_BITS="$WORD_BITS" \
  ORAM_ACCESSES="$ACCESSES" ORAM_IMPL="$IMPL" \
  run_player "$BIN" oram_bench
