#!/usr/bin/env bash
set -euo pipefail

MP_SPDZ=${MP_SPDZ:?set MP_SPDZ to MP-SPDZ root}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

CFG=""
VECS=""
LAYER=0
QUERIES=""
NUM_QUERIES=1
OUTPUT_MODE="id"
REVEAL_OUTPUT=0
ORAM_IMPL="optimal"
PROTOCOL="ring"
RING_BITS=64
TEXT_INPUT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg) CFG="$2"; shift 2;;
    --vecs) VECS="$2"; shift 2;;
    --layer) LAYER="$2"; shift 2;;
    --queries) QUERIES="$2"; shift 2;;
    --num-queries) NUM_QUERIES="$2"; shift 2;;
    --output) OUTPUT_MODE="$2"; shift 2;;
    --reveal) REVEAL_OUTPUT=1; shift;;
    --oram) ORAM_IMPL="$2"; shift 2;;
    --protocol) PROTOCOL="$2"; shift 2;;
    --ring-bits) RING_BITS="$2"; shift 2;;
    --text) TEXT_INPUT=1; shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$CFG" || -z "$VECS" ]]; then
  echo "usage: run_sort_3pc.sh --cfg cfg.json --vecs vecs.npy [--queries q.npy]" >&2
  exit 1
fi

GEN_ARGS=(--cfg "$CFG" --vecs "$VECS" --layer "$LAYER" --out-dir "$MP_SPDZ/Player-Data" --num-queries "$NUM_QUERIES")
if [[ -n "$QUERIES" ]]; then
  GEN_ARGS+=(--queries "$QUERIES")
fi
if [[ "$TEXT_INPUT" -eq 1 ]]; then
  GEN_ARGS+=(--text)
fi

python3 "$ROOT_DIR/gen_sort_inputs.py" "${GEN_ARGS[@]}"

cp "$ROOT_DIR/Programs/Source/sort_3pc.mpc" "$MP_SPDZ/Programs/Source/"

cd "$MP_SPDZ"
if [[ "$PROTOCOL" == "ring" ]]; then
  SORT_CFG="$CFG" SORT_NUM_QUERIES="$NUM_QUERIES" SORT_OUTPUT="$OUTPUT_MODE" SORT_ORAM_IMPL="$ORAM_IMPL" SORT_REVEAL="$REVEAL_OUTPUT" ./compile.py -R "$RING_BITS" sort_3pc
  LOG_PREFIX="${LOG_PREFIX:-}"
  LOG_SUFFIX="${LOG_SUFFIX:-}"
  BENCH="${BENCH:-}"
  LOGPROT="${LOGPROT:-}"
  GDB_PLAYER="${GDB_PLAYER:-}"
  prefix="${prefix:-}"
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}"
  SPDZROOT="$MP_SPDZ"
  PLAYERS=3
  set +e
  source Scripts/run-common.sh
  set -e
  export SORT_CFG="$CFG"
  export SORT_NUM_QUERIES="$NUM_QUERIES"
  export SORT_OUTPUT="$OUTPUT_MODE"
  export SORT_ORAM_IMPL="$ORAM_IMPL"
  if [[ "$REVEAL_OUTPUT" -eq 1 ]]; then
    export SORT_REVEAL=1
  else
    unset SORT_REVEAL
  fi
  run_player replicated-ring-party.x sort_3pc
elif [[ "$PROTOCOL" == "bin" ]]; then
  SORT_CFG="$CFG" SORT_NUM_QUERIES="$NUM_QUERIES" SORT_OUTPUT="$OUTPUT_MODE" SORT_ORAM_IMPL="$ORAM_IMPL" SORT_REVEAL="$REVEAL_OUTPUT" ./compile.py -B 1 sort_3pc
  LOG_PREFIX="${LOG_PREFIX:-}"
  LOG_SUFFIX="${LOG_SUFFIX:-}"
  BENCH="${BENCH:-}"
  LOGPROT="${LOGPROT:-}"
  GDB_PLAYER="${GDB_PLAYER:-}"
  prefix="${prefix:-}"
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}"
  SPDZROOT="$MP_SPDZ"
  PLAYERS=3
  set +e
  source Scripts/run-common.sh
  set -e
  export SORT_CFG="$CFG"
  export SORT_NUM_QUERIES="$NUM_QUERIES"
  export SORT_OUTPUT="$OUTPUT_MODE"
  export SORT_ORAM_IMPL="$ORAM_IMPL"
  if [[ "$REVEAL_OUTPUT" -eq 1 ]]; then
    export SORT_REVEAL=1
  else
    unset SORT_REVEAL
  fi
  run_player replicated-bin-party.x sort_3pc
else
  echo "unknown --protocol: $PROTOCOL (expected ring or bin)" >&2
  exit 1
fi
