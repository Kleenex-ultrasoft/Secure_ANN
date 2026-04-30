#!/usr/bin/env bash
# Multi-query handling via N independent single-mode invocations.
#
# Why this exists: ABY's IKNP-OT extension exhibits cumulative state drift
# when many ABYParty objects are created within a single process, matching
# the open framework issue
#   https://github.com/encryptogroup/ABY/issues/114
# ("Erroneous Multiplication Results", maintainer-acknowledged by @lenerd
# as a multiplication-triple generation bug tied to OT-extension batching).
# The related thread-unsafe IKNP base-OT
#   https://github.com/encryptogroup/ABY/issues/152
# is also acknowledged by the maintainers and unfixed.
#
# Inside a single query (T iterations), per-iter ABYParty recreate keeps
# the OT pool fresh enough to stay clear of #114's threshold.  Across
# queries, even per-iter recreate accumulates allocator state in the same
# process, so the cleanest mitigation is a fresh process per query --
# i.e., this wrapper.  Each query writes its own log_dir/q<i>/ tree so
# Python aggregation and recall scoring stay simple.
#
# Usage:
#   NPZ=/path/to.npz QUERY=/path/to_query.npy GT_NPY=/path/to_gt.npy \
#     NUM_Q=2 PORT_BASE=52000 ./scripts/run_multi_via_single.sh

set -euo pipefail

NPZ=${NPZ:?NPZ required}
QUERY=${QUERY:?QUERY .npy required}
GT_NPY=${GT_NPY:-}
NUM_Q=${NUM_Q:-1}
PORT_BASE=${PORT_BASE:-52000}
OUT_DIR=${OUT_DIR:-/tmp/hnsecw_multi_via_single}
THREADS=${THREADS:-2}
RTT_MS=${RTT_MS:-1}
BW_MBPS=${BW_MBPS:-4000}

PY=${PY:-/home/fahong/miniforge3/bin/python3}
ABY_BIN=${ABY_BIN:-$HOME/hnsecw_build/ABY/build/bin/hnsecw_cli}
HERE=${HERE:-$(cd "$(dirname "$0")/.." && pwd)}

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

L=$($PY -c "import numpy as np, json; d = np.load('$NPZ', allow_pickle=True); print(json.loads(str(d['meta_json'][0]))['L'])")
echo "[wrapper] NPZ=$NPZ L=$L NUM_Q=$NUM_Q THREADS=$THREADS"

for q in $(seq 0 $((NUM_Q-1))); do
    echo "==================== Query $q ===================="
    LOG_DIR="$OUT_DIR/q$q"
    mkdir -p "$LOG_DIR/entry_chain"
    PORT=$((PORT_BASE + q*10))

    for li in $(seq 0 $((L-1))); do
        DAT="$LOG_DIR/layer${li}.bin"
        $PY "$HERE/2pc/bench/build_hnsecw_data_file.py" \
            --npz "$NPZ" --layer "$li" --num_queries 1 \
            --query-file "$QUERY" --query-index "$q" \
            --out "$DAT" >/dev/null
    done

    HNSECW_DATA_DIR="$LOG_DIR" $PY "$HERE/2pc/bench/run_mpc_layer_search.py" \
        --npz "$NPZ" --aby_bin "$ABY_BIN" \
        --addr 127.0.0.1 --port "$PORT" \
        --mode single --num_queries 1 \
        --threads "$THREADS" \
        --rtt_ms "$RTT_MS" --bw_mbps "$BW_MBPS" \
        --log_dir "$LOG_DIR" 2>&1 | tail -3 | sed "s/^/  [q$q] /"
done

if [[ -n "$GT_NPY" ]]; then
    echo
    echo "==================== Recall vs GT ===================="
    NUM_Q_E=$NUM_Q OUT_E=$OUT_DIR NPZ_E=$NPZ GT_E=$GT_NPY \
    $PY << 'PY'
import numpy as np, struct, os
NPZ = os.environ["NPZ_E"]; GT_NPY = os.environ["GT_E"]
NUM_Q = int(os.environ["NUM_Q_E"]); OUT = os.environ["OUT_E"]
gt = np.load(GT_NPY)
d = np.load(NPZ, allow_pickle=True)
N = d["vecs_0"].shape[0]
hits1 = 0
hits10 = 0
total_lat = 0.0
total_comm = 0.0
for q in range(NUM_Q):
    fp = f"{OUT}/q{q}/entry_chain/layer0_entry_out.bin.topk"
    if not os.path.exists(fp):
        print(f"q{q}: NO topk file at {fp}")
        continue
    with open(fp, "rb") as f:
        data = f.read()
    mpc = list(struct.unpack(f"<{len(data)//4}I", data))
    mpc10 = [x for x in mpc if x < N][:10]
    gt10 = gt[q, :10].tolist()
    overlap = set(mpc10) & set(gt10)
    in_gt_top10 = mpc10[0] in gt10 if mpc10 else False
    print(f"q{q}: top1={mpc10[0] if mpc10 else 'NONE'} "
          f"in_gt_top10={in_gt_top10} overlap10@10={len(overlap)}")
    if in_gt_top10:
        hits1 += 1
    if overlap:
        hits10 += 1
    log = f"{OUT}/q{q}/layer0_client.log"
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                if "Total Online" in line:
                    s = line.split("latency(s)=")[1].split()[0]
                    total_lat += float(s)
                    s2 = line.split("comm(MB)=")[1].split()[0]
                    total_comm += float(s2)
                    break
print(f"\nrecall@1_in_top10 = {hits1}/{NUM_Q}")
print(f"recall@10        = {hits10}/{NUM_Q}")
print(f"total online latency(s) = {total_lat:.1f}")
print(f"total online comm(MB)   = {total_comm:.1f}")
PY
fi
