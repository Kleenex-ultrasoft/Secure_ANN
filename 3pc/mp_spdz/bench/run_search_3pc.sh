#!/usr/bin/env bash
# Run a base-layer HNSecW search under MP-SPDZ replicated-ring (3PC).
# Same revealed-fetch design as 2pc/mp_spdz/Programs/Source/.
#
# Usage:
#   MODE={single|multi|batch} \
#   NPZ=...npz QUERY=...npy QUERY_INDEX=0 NUM_Q=1 K=10 \
#     ./3pc/mp_spdz/bench/run_search_3pc.sh

set -euo pipefail

MODE=${MODE:-single}
NPZ=${NPZ:?"NPZ=path/to/build_npz.py output is required"}
QUERY=${QUERY:?"QUERY=path/to/queries.npy is required"}
QUERY_INDEX=${QUERY_INDEX:-0}
NUM_Q=${NUM_Q:-1}
LAYER=${LAYER:-0}
K=${K:-10}

case "$MODE" in
    single) PROG=hnsecw_search_3pc;        EXPECTED_Q=1 ;;
    multi)  PROG=hnsecw_search_3pc_multi;  EXPECTED_Q=$NUM_Q ;;
    batch)  PROG=hnsecw_search_3pc_batch;  EXPECTED_Q=$NUM_Q ;;
    *) echo "MODE must be single|multi|batch" >&2; exit 1 ;;
esac
[[ "$MODE" == "single" && "$NUM_Q" -ne 1 ]] && {
    echo "single mode requires NUM_Q=1" >&2; exit 1; }

MPSPDZ_ROOT=${MPSPDZ_ROOT:-$HOME/MP-SPDZ}
PARTY_BIN=${PARTY_BIN:-$MPSPDZ_ROOT/replicated-ring-party.x}
HOST=${HOST:-127.0.0.1}
PY=${PY:-python3}

HERE=$(cd "$(dirname "$0")" && pwd)
SRC_DIR=$(cd "$HERE/.." && pwd)

cp "$SRC_DIR/Programs/Source/${PROG}.mpc" \
   "$MPSPDZ_ROOT/Programs/Source/${PROG}.mpc"

# Stage inputs via the 2PC gen script and (re)link to the default
# Player-Data prefix that replicated-ring-party.x reads from.
GEN="$(cd "$HERE/../../../2pc/mp_spdz/bench" && pwd)/gen_inputs.py"
mkdir -p "$MPSPDZ_ROOT/Player-Data"
INPUT_PREFIX="$MPSPDZ_ROOT/Player-Data/Input-Binary"
WITH_ENTRIES=()
[[ "$MODE" != "single" ]] && WITH_ENTRIES=(--with-entries)
"$PY" "$GEN" --npz "$NPZ" --query-npy "$QUERY" \
    --query-index "$QUERY_INDEX" --num-queries "$EXPECTED_Q" \
    --layer "$LAYER" "${WITH_ENTRIES[@]}" \
    --out "${INPUT_PREFIX}-Binary-P0-0" >/dev/null
: > "${INPUT_PREFIX}-Binary-P1-0"
: > "${INPUT_PREFIX}-Binary-P2-0"

rm -f "$MPSPDZ_ROOT/Player-Data/Input-Binary-P0-0" \
      "$MPSPDZ_ROOT/Player-Data/Input-Binary-P1-0" \
      "$MPSPDZ_ROOT/Player-Data/Input-Binary-P2-0"
ln -s "${INPUT_PREFIX}-Binary-P0-0" \
      "$MPSPDZ_ROOT/Player-Data/Input-Binary-P0-0"
ln -s "${INPUT_PREFIX}-Binary-P1-0" \
      "$MPSPDZ_ROOT/Player-Data/Input-Binary-P1-0"
ln -s "${INPUT_PREFIX}-Binary-P2-0" \
      "$MPSPDZ_ROOT/Player-Data/Input-Binary-P2-0"

# Compute T, LW, per-query entry chain in plaintext upfront.
PARAMS_FILE=$(mktemp)
"$PY" - "$NPZ" "$QUERY" "$QUERY_INDEX" "$EXPECTED_Q" "$LAYER" \
    > "$PARAMS_FILE" <<'PY'
import json, math, sys
import numpy as np
npz, qfile, qstart, nq, layer = sys.argv[1:6]
qstart = int(qstart); nq = int(nq); layer = int(layer)
d = np.load(npz, allow_pickle=True)
meta = json.loads(str(d["meta_json"][0]))
L = int(meta.get("L", 1))
vec = d["vecs_%d" % layer]
neigh = d["neigh_%d" % layer]
N, D = vec.shape
M = neigh.shape[1]
ef = int(meta.get("ef_base", 16)) if layer == 0 else 1
tau = (int(meta.get("tau_base", 4)) if layer == 0
       else int(meta.get("tau_upper", 4)))
depth_extra = (max(1, int(math.ceil(math.log2(max(2.0, N * D / max(1, M))))))
               if layer == 0 else 0)
T = ef + tau + depth_extra
LW = ef
qs = np.load(qfile)[qstart:qstart + nq].astype(np.int64)
entries = []
for q in qs:
    cur = 0
    for ll in range(L - 1, layer, -1):
        v = d["vecs_%d" % ll].astype(np.int64)
        g = d["neigh_%d" % ll].astype(np.int64)
        while True:
            cand = list(g[cur]) + [cur]
            cand = [c for c in cand if 0 <= c < v.shape[0]]
            dists = [int(((v[c] - q) ** 2).sum()) for c in cand]
            best = cand[int(np.argmin(dists))]
            if best == cur:
                break
            cur = best
        if ll - 1 >= layer and ("down_%d" % ll) in d.files:
            cur = int(d["down_%d" % ll][cur])
    entries.append(int(cur))
print("%d %d %d %d %d %d" % (N, D, M, T, LW, entries[0]))
PY
read N D M T LW ENTRY < "$PARAMS_FILE"
rm -f "$PARAMS_FILE"

cd "$MPSPDZ_ROOT"
HNSW_N=$N HNSW_D=$D HNSW_M=$M HNSW_T=$T HNSW_LW=$LW HNSW_Q=$EXPECTED_Q \
HNSW_ENTRY=$ENTRY \
    ./compile.py -R 64 -O "$PROG" > /tmp/compile_3pc_${MODE}.log 2>&1

# Launch 3 parties (replicated-ring is fixed at N=3; positional first).
# `-v` enables the per-phase breakdown (online vs preprocessing).
"$PARTY_BIN" 0 "$PROG" -h "$HOST" -v > /tmp/p0_3pc_${MODE}.log 2>&1 &
"$PARTY_BIN" 1 "$PROG" -h "$HOST" -v > /tmp/p1_3pc_${MODE}.log 2>&1 &
"$PARTY_BIN" 2 "$PROG" -h "$HOST" -v > /tmp/p2_3pc_${MODE}.log 2>&1 &
wait

# Compute Recall@K against an in-NPZ plaintext oracle.  Print only the
# headline numbers.
"$PY" - "$NPZ" "$QUERY" "$QUERY_INDEX" "$EXPECTED_Q" "$LAYER" \
        "$LW" "$K" "$MODE" <<'PY'
import os, re, sys
import numpy as np
npz, qfile, qstart, nq, layer, lw, k, mode = sys.argv[1:9]
qstart = int(qstart); nq = int(nq); layer = int(layer)
lw = int(lw); k = int(k)
log = "/tmp/p0_3pc_%s.log" % mode
with open(log) as f:
    text = f.read()
m = re.search(r"Spent\s+([\d.]+)\s+seconds.*on the online phase", text)
online_s = float(m.group(1)) if m else float("nan")
re_single = re.compile(r"^W\[(\d+)\]=\((\d+),(\d+)\)$", re.M)
re_multi = re.compile(r"^W\[(\d+)\]\[(\d+)\]=\((\d+),(\d+)\)$", re.M)
mh = re_multi.findall(text)
if mh:
    W_ids = [[int(d) for (q, i, dist, d) in mh if int(q) == qi]
             for qi in range(nq)]
else:
    W_ids = [[int(d) for (i, dist, d) in re_single.findall(text)]]
data = np.load(npz, allow_pickle=True)
vec = data["vecs_%d" % layer].astype(np.int64)
qs = np.load(qfile)[qstart:qstart + nq].astype(np.int64)
hits = []
for qi, q in enumerate(qs):
    dists = np.sum((vec - q) ** 2, axis=1)
    gt_top = np.argsort(dists, kind="stable")[:k].tolist()
    hits.append(len(set(W_ids[qi][:k]) & set(gt_top)))
avg_pct = 100.0 * sum(hits) / (k * len(hits))
suffix = "  (avg over %d queries)" % nq if nq > 1 else ""
print("Recall@%d        = %.2f%%%s" % (k, avg_pct, suffix))
print("Online latency  = %.3f s/query%s" % (online_s / nq, suffix))
PY
