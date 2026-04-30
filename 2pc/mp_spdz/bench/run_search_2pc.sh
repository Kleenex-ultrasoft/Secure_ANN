#!/usr/bin/env bash
# Run a base-layer search for HNSecW under MP-SPDZ semi2k 2PC.
#
# Usage:
#   MODE={single|multi|batch} \
#   NPZ=...npz QUERY=...npy QUERY_INDEX=0 NUM_Q=1 \
#     ./2pc/mp_spdz/bench/run_search_2pc.sh

set -euo pipefail

MODE=${MODE:-single}                 # single | multi | batch
NPZ=${NPZ:?"NPZ=path/to/build_npz.py output is required"}
QUERY=${QUERY:?"QUERY=path/to/queries.npy is required"}
QUERY_INDEX=${QUERY_INDEX:-0}
NUM_Q=${NUM_Q:-1}
LAYER=${LAYER:-0}
# K (top-K to evaluate; default 10).  The MPC kernel returns LW pairs
# where LW = ef = k+10 per the paper; we then evaluate Recall@K.
K=${K:-10}

case "$MODE" in
    single) PROG=hnsecw_search_2pc;        EXPECTED_Q=1 ;;
    multi)  PROG=hnsecw_search_2pc_multi;  EXPECTED_Q=$NUM_Q ;;
    batch)  PROG=hnsecw_search_2pc_batch;  EXPECTED_Q=$NUM_Q ;;
    *) echo "MODE must be single|multi|batch (got: $MODE)" >&2; exit 1 ;;
esac

if [[ "$MODE" == "single" && "$NUM_Q" -ne 1 ]]; then
    echo "single mode requires NUM_Q=1; for NUM_Q>1 use MODE=multi or MODE=batch" >&2
    exit 1
fi

MPSPDZ_ROOT=${MPSPDZ_ROOT:-$HOME/MP-SPDZ}
PARTY_BIN=${PARTY_BIN:-$MPSPDZ_ROOT/semi2k-party.x}
HOST=${HOST:-127.0.0.1}
PY=${PY:-python3}

HERE=$(cd "$(dirname "$0")" && pwd)
SRC_DIR=$(cd "$HERE/.." && pwd)

mkdir -p "$MPSPDZ_ROOT/Programs/Source"
cp "$SRC_DIR/Programs/Source/${PROG}.mpc" \
   "$MPSPDZ_ROOT/Programs/Source/${PROG}.mpc"

mkdir -p "$MPSPDZ_ROOT/Player-Data"
INPUT_PREFIX="$MPSPDZ_ROOT/Player-Data/Input-Binary"
WITH_ENTRIES=()
if [[ "$MODE" != "single" ]]; then
    WITH_ENTRIES=(--with-entries)
fi
"$PY" "$SRC_DIR/bench/gen_inputs.py" \
    --npz "$NPZ" \
    --query-npy "$QUERY" \
    --query-index "$QUERY_INDEX" \
    --num-queries "$EXPECTED_Q" \
    --layer "$LAYER" \
    "${WITH_ENTRIES[@]}" \
    --out "${INPUT_PREFIX}-Binary-P0-0" >/dev/null
: > "${INPUT_PREFIX}-Binary-P1-0"

read N D M T LW ENTRY < <("$PY" - <<PY
import json
import math
import numpy as np

d = np.load("$NPZ", allow_pickle=True)
meta = json.loads(str(d["meta_json"][0]))
layer = $LAYER
L = int(meta.get("L", 1))
vec = d[f"vecs_{layer}"]
neigh = d[f"neigh_{layer}"]
N, D = vec.shape
M = neigh.shape[1]
ef = int(meta.get("ef_base", 16)) if layer == 0 else 1
tau = (int(meta.get("tau_base", 4)) if layer == 0
       else int(meta.get("tau_upper", 4)))
depth_extra = (max(1, int(math.ceil(math.log2(max(2.0, N * D / max(1, M))))))
               if layer == 0 else 0)
T = ef + tau + depth_extra
LW = ef

# Per-query entry into the target layer = HNSW upper-layer descent done
# in plaintext.  Start at node 0 of the top layer, greedy hop until no
# closer neighbour, then descend one layer, repeat, until reaching the
# target layer.  For layer 0 (base) this matches ABY's entry_chain.bin.
queries = np.load("$QUERY")
q = queries[$QUERY_INDEX].astype(np.int64)
cur = 0
for ll in range(L - 1, layer, -1):
    v = d[f"vecs_{ll}"].astype(np.int64)
    g = d[f"neigh_{ll}"].astype(np.int64)
    while True:
        # neighbours of cur at layer ll plus cur itself
        cand = list(g[cur]) + [cur]
        cand = [c for c in cand if 0 <= c < v.shape[0]]
        dists = [int(((v[c] - q) ** 2).sum()) for c in cand]
        best = cand[int(np.argmin(dists))]
        if best == cur:
            break
        cur = best
    # Map cur from layer ll into the layer below via the down_{ll} index
    if ll - 1 >= layer and f"down_{ll}" in d.files:
        cur = int(d[f"down_{ll}"][cur])
ENTRY = int(cur)

print(N, D, M, T, LW, ENTRY)
PY
)
cd "$MPSPDZ_ROOT"
HNSW_N=$N HNSW_D=$D HNSW_M=$M HNSW_T=$T HNSW_LW=$LW HNSW_Q=$EXPECTED_Q \
HNSW_ENTRY=$ENTRY \
    ./compile.py -R 64 -O "$PROG" > /tmp/compile_2pc_${MODE}.log 2>&1

"$PARTY_BIN" -N 2 -p 0 -h "$HOST" -v -IF "$INPUT_PREFIX" \
    "$PROG" > /tmp/p0_2pc_${MODE}.log 2>&1 &
"$PARTY_BIN" -N 2 -p 1 -h "$HOST" -v -IF "$INPUT_PREFIX" \
    "$PROG" > /tmp/p1_2pc_${MODE}.log 2>&1

# Compute recall@LW vs FAISS GT (if GT_NPY supplied) and emit only the
# headline numbers to stdout: recall and online time.
"$PY" - <<PY
import os, re
import numpy as np

LOG = "/tmp/p0_2pc_${MODE}.log"
NPZ_PATH = "$NPZ"
QUERY_NPY = "$QUERY"
QUERY_INDEX = ${QUERY_INDEX}
NUM_Q = ${EXPECTED_Q}
LW = ${LW}
K  = ${K}
LAYER = ${LAYER}

with open(LOG) as f:
    text = f.read()

m = re.search(r"Spent\s+([\d.]+)\s+seconds.*on the online phase", text)
online_s = float(m.group(1)) if m else float("nan")

re_single = re.compile(r"^W\[(\d+)\]=\((\d+),(\d+)\)$", re.M)
re_multi = re.compile(r"^W\[(\d+)\]\[(\d+)\]=\((\d+),(\d+)\)$", re.M)
multi_hits = re_multi.findall(text)
if multi_hits:
    W_ids = [[int(d) for (q, i, dist, d) in multi_hits if int(q) == qi]
             for qi in range(NUM_Q)]
else:
    flat = [int(d) for (i, dist, d) in re_single.findall(text)]
    W_ids = [flat]

# Plaintext top-LW oracle on NPZ's actual base-layer vectors (= "ground
# truth" in the same id space the MPC output lives in).
data = np.load(NPZ_PATH, allow_pickle=True)
vec = data[f"vecs_{LAYER}"].astype(np.int64)
queries_full = np.load(QUERY_NPY).astype(np.int64)
qs = queries_full[QUERY_INDEX:QUERY_INDEX + NUM_Q]

hits = []
for qi, q in enumerate(qs):
    dists = np.sum((vec - q) ** 2, axis=1)
    gt_top = np.argsort(dists, kind="stable")[:K].tolist()
    overlap = len(set(W_ids[qi][:K]) & set(gt_top))
    hits.append(overlap)

avg_pct = 100.0 * sum(hits) / (K * len(hits))
suffix = f"  (avg over {NUM_Q} queries)" if NUM_Q > 1 else ""
print(f"Recall@{K}        = {avg_pct:.2f}%{suffix}")
print(f"Online latency  = {online_s / NUM_Q:.3f} s/query{suffix}")
PY
