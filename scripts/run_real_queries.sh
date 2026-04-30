#!/usr/bin/env bash
# Run HNSecW 2PC ABY with REAL queries + ground truth, compute recall@10.
# v3: avoid sourcing miniforge activate (which collides with $@); just use absolute python path.
set -euo pipefail

NPZ="${1:-/home/fahong/hnsecw_build/results/sift_1m.npz}"
QUERY_NPY="${2:-/home/fahong/hnsecw_build/datasets/sift_query.npy}"
GT_NPY="${3:-/home/fahong/hnsecw_build/datasets/sift_gt.npy}"
NUM_Q="${4:-5}"
PORT="${5:-48001}"
LOG_DIR="${6:-/tmp/hnsecw_real}"

PY=/home/fahong/miniforge3/bin/python3

mkdir -p "$LOG_DIR/entry_chain"
ABY_BIN=/home/fahong/hnsecw_build/ABY/build/bin/hnsecw_cli
HERE=/home/fahong/hnsecw_build/HNSecW

L=$($PY -c "
import numpy as np, json
d = np.load('$NPZ', allow_pickle=True)
print(json.loads(str(d['meta_json'][0]))['L'])
")
echo "[real] L=$L NPZ=$NPZ NUM_Q=$NUM_Q"

for li in $(seq 0 $((L-1))); do
    DAT="$LOG_DIR/layer${li}.bin"
    $PY "$HERE/2pc/bench/build_hnsecw_data_file.py" \
        --npz "$NPZ" --layer "$li" --num_queries "$NUM_Q" \
        --out "$DAT" --query-file "$QUERY_NPY"
done

HNSECW_DATA_DIR="$LOG_DIR" $PY "$HERE/2pc/bench/run_mpc_layer_search.py" \
    --npz "$NPZ" --aby_bin "$ABY_BIN" \
    --addr 127.0.0.1 --port "$PORT" \
    --mode single --num_queries "$NUM_Q" \
    --rtt_ms 1 --bw_mbps 4000 \
    --log_dir "$LOG_DIR" 2>&1 | tail -8

echo
NPZ_E="$NPZ" QGT_E="$GT_NPY" TOPK_FILE_E="$LOG_DIR/entry_chain/layer0_entry_out.bin.topk" NUM_Q_E="$NUM_Q" \
$PY << 'PY'
import numpy as np, json, struct, os
NPZ = os.environ["NPZ_E"]; QGT = os.environ["QGT_E"]
TOPK_FILE = os.environ["TOPK_FILE_E"]; NUM_Q = int(os.environ["NUM_Q_E"])

d = np.load(NPZ, allow_pickle=True)
v = d["vecs_0"]; N, D = v.shape
gt = np.load(QGT)

if not os.path.exists(TOPK_FILE):
    print(f"NO topk file at {TOPK_FILE}"); raise SystemExit(1)
with open(TOPK_FILE, "rb") as f: data = f.read()
mpc_ids = list(struct.unpack(f"<{len(data)//4}I", data))

last_q = NUM_Q - 1
gt_top10 = gt[last_q, :10].tolist()
mpc_top10 = [x for x in mpc_ids if x < N][:10]
overlap = set(mpc_top10) & set(gt_top10)
print(f"=== REAL query #{last_q} on {NPZ} ===")
print(f"  MPC top-10:          {mpc_top10}")
print(f"  Ground truth top-10: {gt_top10}")
print(f"  Overlap: {sorted(overlap)}")
print(f"  recall@10 = {len(overlap)/10:.2f}")
PY
