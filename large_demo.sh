#!/usr/bin/env bash
#
# HNSecW (2PC ABY) demo on SIFT-100K with a real SIFT query.
#
# No flags needed — running `bash large_demo.sh` from the repo root will
# pick up the SIFT-100K M=16 ef=16 index, the ABY binary built under
# $HNSECW_BUILD/ABY/build/bin/hnsecw_cli, and SIFT's first query
# (~/hnsecw_build/datasets/sift_query.npy[0]).  Override any of:
#   --npz PATH        index .npz (default: sift100k_M16_ef16.npz)
#   --aby-bin PATH    hnsecw_cli (default: $HNSECW_BUILD/ABY/build/bin/hnsecw_cli)
#   --query-npy PATH  uint8 (Q,D) query vectors (default: sift_query.npy)
#   --query-idx N     which row of --query-npy to use (default: 0)
#   --port N          base port (default: 47000)
#   --topk N          top-K (default: 10)
#   --python PATH     python with numpy
#
# The demo prints, per layer:
#   [layer L] online latency(s)=...  comm(MB)=...
# and at the end:
#   [mpc total] online latency(s)=...  comm(MB)=...
#   Top-1 / MPC top-10 / Ground-truth top-10 / recall@10

set -euo pipefail

HNSECW_BUILD=${HNSECW_BUILD:-$HOME/hnsecw_build}
NPZ=${NPZ:-$HNSECW_BUILD/results/sift100k_M16_ef16.npz}
ABY_BIN=${ABY_BIN:-$HNSECW_BUILD/ABY/build/bin/hnsecw_cli}
QUERY_NPY=${QUERY_NPY:-$HNSECW_BUILD/datasets/sift_query.npy}
QUERY_IDX=${QUERY_IDX:-0}
PORT=47000
MODE=single
NUM_QUERIES=1
TOPK=10
LOG_DIR=/tmp/hnsecw_large_demo
PYTHON=${PYTHON:-$HNSECW_BUILD/../miniforge3/envs/hnsecw/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --npz)         NPZ="$2"; shift 2;;
    --aby-bin)     ABY_BIN="$2"; shift 2;;
    --query-npy)   QUERY_NPY="$2"; shift 2;;
    --query-idx)   QUERY_IDX="$2"; shift 2;;
    --port)        PORT="$2"; shift 2;;
    --mode)        MODE="$2"; shift 2;;
    --num-queries) NUM_QUERIES="$2"; shift 2;;
    --topk)        TOPK="$2"; shift 2;;
    --log-dir)     LOG_DIR="$2"; shift 2;;
    --python)      PYTHON="$2"; shift 2;;
    -h|--help)     sed -n '1,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

[[ -f "$NPZ" ]]       || { echo "NPZ not found: $NPZ" >&2; exit 1; }
[[ -x "$ABY_BIN" ]]   || { echo "hnsecw_cli not found: $ABY_BIN" >&2; exit 1; }
[[ -f "$QUERY_NPY" ]] || { echo "query npy not found: $QUERY_NPY" >&2; exit 1; }

HERE=$(cd "$(dirname "$0")" && pwd)
DATA_DIR=/tmp/hnsecw_large_demo_data
mkdir -p "$DATA_DIR" "$LOG_DIR"

# Clip the chosen real query row into a uint8 (1, D) .npy so
# build_hnsecw_data_file.py --query-file can ingest it directly.
PICKED_Q=/tmp/hnsecw_demo_query_uint8.npy
"$PYTHON" - <<PYEOF
import numpy as np
q = np.load("$QUERY_NPY")
row = q[$QUERY_IDX].astype(np.float32)
row = np.clip(row, 0, 255).astype(np.uint8).reshape(1, -1)
np.save("$PICKED_Q", row)
PYEOF

echo "[demo] Building per-layer data files from $NPZ"
L_COUNT=$("$PYTHON" -c "import numpy as np, json; d=np.load('$NPZ', allow_pickle=True); print(json.loads(str(d['meta_json'][0]))['L'])")
for ((L=0; L<L_COUNT; L++)); do
  "$PYTHON" "$HERE/2pc/bench/build_hnsecw_data_file.py" \
    --npz "$NPZ" --layer $L --num_queries $NUM_QUERIES \
    --out "$DATA_DIR/layer${L}.bin" \
    --query-file "$PICKED_Q" 2>&1 | head -1
done

echo
echo "[demo] Running HNSecW (mode=$MODE, port=$PORT)"
HNSECW_DATA_DIR="$DATA_DIR" "$PYTHON" "$HERE/2pc/bench/run_mpc_layer_search.py" \
  --npz "$NPZ" --aby_bin "$ABY_BIN" \
  --addr 127.0.0.1 --port "$PORT" \
  --mode "$MODE" --protocol b2y \
  --num_queries "$NUM_QUERIES" \
  --debug_tag 0 \
  --log_dir "$LOG_DIR" 2>&1 | tail -8

echo
echo "============================================================"
echo "[demo] Top-K + recall summary"
echo "============================================================"
"$PYTHON" - <<PYEOF
import struct, numpy as np, json, os
NPZ = "$NPZ"
LOG = "$LOG_DIR"
TOPK = $TOPK
NUM_Q = $NUM_QUERIES
PICKED_Q = "$PICKED_Q"

d = np.load(NPZ, allow_pickle=True)
meta = json.loads(str(d["meta_json"][0]))
v = d["vecs_0"]
N, D = v.shape
print(f"  Dataset: {NPZ}")
print(f"  N={N} D={D} L={meta['L']} M={meta['M']} ef={meta['ef_base']} tau={meta['tau_base']}")

entry_out = f"{LOG}/entry_chain/layer0_entry_out.bin"
topk_file = f"{LOG}/entry_chain/layer0_entry_out.bin.topk"
with open(entry_out, "rb") as f:
    mpc_top1 = list(struct.unpack(f"<{NUM_Q}I", f.read(NUM_Q * 4)))

mpc_topk = []
if os.path.exists(topk_file):
    with open(topk_file, "rb") as f:
        data = f.read()
    n = len(data) // 4
    mpc_topk = list(struct.unpack(f"<{n}I", data))

q = np.load(PICKED_Q)[0].astype(np.int64)
diff = v.astype(np.int64) - q
dist = (diff**2).sum(axis=1)
pt_top = np.argsort(dist)[:TOPK]
mpc_top = mpc_topk[:TOPK] if mpc_topk else mpc_top1[:1]
overlap = set(mpc_top) & set(pt_top.tolist())
recall = len(overlap) / TOPK if TOPK else 0
print(f"\n  MPC top-{TOPK}:           {mpc_top[:TOPK]}")
print(f"  Ground-truth top-{TOPK}:  {pt_top.tolist()}")
print(f"  Overlap: {sorted(overlap)}")
print(f"  recall@{TOPK} = {recall:.2f}")
PYEOF
echo "============================================================"
