#!/usr/bin/env bash
#
# panther_generalize baseline demo (2PC Cheetah, SIFT-100K).
#
# Runs the random_panther_doram SPU binary (rank 0 + rank 1) to
# measure end-to-end latency / communication, then runs a plaintext
# IVF oracle on real SIFT-100K with PANTHER's IVF parameters
# (total_cluster_size=9061, u=123 probes) to verify the algorithmic
# top-10 / recall — recall is independent of the PIR primitive used
# (FHE-PIR or our DORAM bridge).
#
# No flags needed if defaults are in place:
#   $HOME/hnsecw_build/spu/src/bazel-bin/experimental/panther/random_panther_doram
#   $HOME/hnsecw_build/duoram/cpir-read/cxx/spir_test{0,1}
#   $HOME/hnsecw_build/datasets/sift_base.npy
#   $HOME/hnsecw_build/datasets/sift_query.npy
#
# Override any of:
#   --spu-bin PATH      random_panther_doram binary
#   --duoram-dir PATH   directory containing spir_test0 / spir_test1
#   --sift-base PATH    SIFT base .npy (uses first 100000 rows)
#   --sift-query PATH   SIFT query .npy
#   --query-idx N       which query row to use (default 0)
#   --parties HOST:P,HOST:P
#   --python PATH       python with numpy + faiss

set -euo pipefail

HNSECW_BUILD=${HNSECW_BUILD:-$HOME/hnsecw_build}
SPU_BIN=${SPU_BIN:-$HNSECW_BUILD/spu/src/bazel-bin/experimental/panther/random_panther_doram}
DUORAM_BIN_DIR=${DUORAM_BIN_DIR:-$HNSECW_BUILD/duoram/cpir-read/cxx}
SIFT_BASE=${SIFT_BASE:-$HNSECW_BUILD/datasets/sift_base.npy}
SIFT_QUERY=${SIFT_QUERY:-$HNSECW_BUILD/datasets/sift_query.npy}
QUERY_IDX=${QUERY_IDX:-0}
PARTIES=${PARTIES:-127.0.0.1:9530,127.0.0.1:9531}
PYTHON=${PYTHON:-$HOME/miniforge3/envs/hnsecw/bin/python}
[[ -x "$PYTHON" ]] || PYTHON=python3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spu-bin)     SPU_BIN="$2"; shift 2;;
    --duoram-dir)  DUORAM_BIN_DIR="$2"; shift 2;;
    --sift-base)   SIFT_BASE="$2"; shift 2;;
    --sift-query)  SIFT_QUERY="$2"; shift 2;;
    --query-idx)   QUERY_IDX="$2"; shift 2;;
    --parties)     PARTIES="$2"; shift 2;;
    --python)      PYTHON="$2"; shift 2;;
    -h|--help)     sed -n '1,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

[[ -x "$SPU_BIN" ]]                       || { echo "SPU binary not found: $SPU_BIN" >&2; exit 1; }
[[ -x "$DUORAM_BIN_DIR/spir_test0" ]]     || { echo "Duoram spir_test0 not found in: $DUORAM_BIN_DIR" >&2; exit 1; }
[[ -f "$SIFT_BASE" ]]                     || { echo "SIFT base not found: $SIFT_BASE" >&2; exit 1; }
[[ -f "$SIFT_QUERY" ]]                    || { echo "SIFT query not found: $SIFT_QUERY" >&2; exit 1; }

LOG_DIR=/tmp/panther_demo
mkdir -p "$LOG_DIR"

export DUORAM_BIN_DIR

echo "[demo] Running panther_generalize (2PC Cheetah + Duoram bridge)"
( "$SPU_BIN" --rank=0 --parties="$PARTIES" > "$LOG_DIR/p0.log" 2>&1 & )
sleep 0.5
"$SPU_BIN" --rank=1 --parties="$PARTIES" > "$LOG_DIR/p1.log" 2>&1
wait

echo
echo "============================================================"
echo "[demo] Cost (measured, 2PC Cheetah, SIFT-100K, u=123 probes)"
echo "============================================================"
grep -oE "Step [0-9].*ms.*MB|Step 4.*wall.*MB\)|Total Latency: .*ms|Total Communication: .*MB" "$LOG_DIR/p0.log" || true

echo
echo "============================================================"
echo "[demo] Top-10 + recall (plaintext IVF oracle, real SIFT-100K)"
echo "============================================================"
"$PYTHON" - 2> >(grep -v "^WARNING" >&2) <<PYEOF
import numpy as np, faiss, sys, os
base = np.load("$SIFT_BASE").astype("float32")[:100000]
q = np.load("$SIFT_QUERY").astype("float32")[$QUERY_IDX:$QUERY_IDX+1]
N, D = base.shape
nlist = 9061
nprobe = 123
quantizer = faiss.IndexFlatL2(D)
index = faiss.IndexIVFFlat(quantizer, D, nlist)
index.train(base)
index.add(base)
index.nprobe = nprobe
_, ivf_top = index.search(q, 10)
gt = faiss.IndexFlatL2(D); gt.add(base)
_, gt_top = gt.search(q, 10)
ivf_top = ivf_top[0].tolist(); gt_top = gt_top[0].tolist()
overlap = set(ivf_top) & set(gt_top)
print(f"  Dataset: SIFT-100K (sift_base.npy[:100000]), real query #{$QUERY_IDX}")
print(f"  IVF params (matching panther random_panther_doram.cc):")
print(f"    total_cluster_size = {nlist}, u = {nprobe} probes")
print()
print(f"  IVF top-10:           {ivf_top}")
print(f"  Ground-truth top-10:  {gt_top}")
print(f"  Overlap: {sorted(overlap)}")
print(f"  recall@10 = {len(overlap)/10.0:.2f}")
PYEOF
echo "============================================================"
