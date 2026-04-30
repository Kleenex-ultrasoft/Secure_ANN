#!/usr/bin/env bash
set -euo pipefail

CFG=""
SHARES=""
OUT=""
QUERIES=""
NUM_QUERIES=1
ENTRY_COUNT=0
MODE="single"
OUTPUT_MODE="id"
HOST="127.0.0.1"
PORT=1313
PROGRESS=0
PROFILE=0
SEED=0
BIN_PATH="${ABY3_BIN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg) CFG="$2"; shift 2;;
    --shares) SHARES="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --queries) QUERIES="$2"; shift 2;;
    --num-queries) NUM_QUERIES="$2"; shift 2;;
    --entry-count) ENTRY_COUNT="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --output) OUTPUT_MODE="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --progress) PROGRESS=1; shift;;
    --profile) PROFILE=1; shift;;
    --seed) SEED="$2"; shift 2;;
    --bin) BIN_PATH="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$CFG" || -z "$SHARES" || -z "$OUT" ]]; then
  echo "missing --cfg/--shares/--out" >&2
  exit 1
fi
if [[ "$MODE" == "single" && "$NUM_QUERIES" -gt 1 ]]; then
  echo "single mode requires --num-queries 1 (use --mode multi or --mode batch)" >&2
  exit 1
fi

if [[ -z "$BIN_PATH" ]]; then
  if [[ -n "${ABY3_ROOT:-}" ]]; then
    BIN_PATH="${ABY3_ROOT}/build/frontend/hnsecw_search_aby3"
  elif [[ -n "${ABY3:-}" ]]; then
    BIN_PATH="${ABY3}/build/frontend/hnsecw_search_aby3"
  fi
fi
if [[ -z "$BIN_PATH" ]]; then
  echo "set ABY3_BIN or pass --bin /path/to/hnsecw_search_aby3" >&2
  exit 1
fi
if [[ ! -x "$BIN_PATH" ]]; then
  echo "hnsecw_search_aby3 not found or not executable: $BIN_PATH" >&2
  exit 1
fi

python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pack_queries_aby3.py" \
  --cfg "$CFG" \
  --shares-dir "$SHARES" \
  --out-dir "$SHARES" \
  --queries "$QUERIES" \
  --num-queries "$NUM_QUERIES" \
  --seed "$SEED"

mkdir -p "$OUT/p0" "$OUT/p1" "$OUT/p2"

COMMON_ARGS=(-cfg "$CFG" -shares "$SHARES" -out "$OUT" -mode "$MODE" -num_queries "$NUM_QUERIES" -output "$OUTPUT_MODE" -host "$HOST" -port "$PORT")
if [[ "$ENTRY_COUNT" -gt 0 ]]; then
  COMMON_ARGS+=(-entry_count "$ENTRY_COUNT")
fi
if [[ "$PROGRESS" -eq 1 ]]; then
  COMMON_ARGS+=(-progress)
fi
if [[ "$PROFILE" -eq 1 ]]; then
  COMMON_ARGS+=(-profile)
fi

"$BIN_PATH" -party 0 "${COMMON_ARGS[@]}" &
PID0=$!
"$BIN_PATH" -party 1 "${COMMON_ARGS[@]}" &
PID1=$!
"$BIN_PATH" -party 2 "${COMMON_ARGS[@]}" &
PID2=$!

wait "$PID0" "$PID1" "$PID2"
