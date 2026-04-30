#!/usr/bin/env bash
#
# Build HNSecW's HNSW index for a dataset.  Equivalent role to
# Panther's k-means cluster trainer (gen_panther_ivf_inputs.py)
# but for HNSW: produces the per-layer graph + uint8-quantized
# vectors + per-layer dummy ids that the .mpc / cpp paths consume.
#
# Output is the NPZ schema run_mpc_layer_search.py expects.
#
# Usage:
#   bash build_hnsecw_index.sh \
#     --dataset /path/to/X.npy \
#     --out /path/to/index.npz \
#     [--M 128] [--ef-construction 200] [--ef-base 16] \
#     [--tau 4] [--space l2|cosine] [--n N_subset]
#
# Defaults match the deployed setting from paper Sec. 4.1
# (M = 128, ef^{(0)} = 16, tau = 4).  Lower M values produce a
# sparser graph that needs a larger ef budget to reach the same
# recall (the residual log_2(N*d/M) term in the search-depth bound
# stays the same, but the constant in front of it grows).
#
# Examples:
#   # SIFT 1M (already in [0, 218] uint8 range)
#   bash build_hnsecw_index.sh \
#     --dataset $HOME/hnsecw_build/datasets/sift_base.npy \
#     --out    $HOME/hnsecw_build/results/sift_1m.npz \
#     --M 128 --space l2 --n 1000000
#
#   # DEEP 1M (unit-normalized float32, range [-0.55, 0.58])
#   bash build_hnsecw_index.sh \
#     --dataset $HOME/hnsecw_build/datasets/deep_base.npy \
#     --out    $HOME/hnsecw_build/results/deep_1m.npz \
#     --M 128 --space l2 --n 1000000
#   # The auto-detect in 2pc/bench/build_npz.py rescales (x+1)*127
#   # so DEEP doesn't get truncated to all-zero.
#
#   # LAION cosine (unit-normalized float32 in [-0.25, 0.26])
#   bash build_hnsecw_index.sh \
#     --dataset $HOME/hnsecw_build/datasets/laion_base.npy \
#     --out    $HOME/hnsecw_build/results/laion_200k.npz \
#     --M 128 --space cosine --n 200000
#
# After build, drive the MPC search via:
#   bash 2pc/bench/run_single_query.sh /path/to/index.npz
#
# or with the per-layer data file flow:
#   for L in 0 1 2; do
#     python3 2pc/bench/build_hnsecw_data_file.py \
#       --npz /path/to/index.npz --layer $L --num_queries 1 \
#       --out /tmp/hdata/layer$L.bin
#   done
#   HNSECW_DATA_DIR=/tmp/hdata python3 2pc/bench/run_mpc_layer_search.py \
#     --npz /path/to/index.npz --aby_bin /path/to/hnsecw_cli \
#     --addr 127.0.0.1 --port 21800 --mode single --protocol b2y

set -euo pipefail

DATASET=""
OUT=""
M=128
EF_CONSTRUCTION=200
EF_BASE=16
TAU=4
SPACE=l2
N_SUBSET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)         DATASET="$2"; shift 2;;
    --out)             OUT="$2"; shift 2;;
    --M)               M="$2"; shift 2;;
    --ef-construction) EF_CONSTRUCTION="$2"; shift 2;;
    --ef-base)         EF_BASE="$2"; shift 2;;
    --tau)             TAU="$2"; shift 2;;
    --space)           SPACE="$2"; shift 2;;
    --n)               N_SUBSET="$2"; shift 2;;
    -h|--help)         sed -n '1,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$DATASET" || -z "$OUT" ]]; then
  echo "Usage: bash build_hnsecw_index.sh --dataset X.npy --out index.npz [...]" >&2
  echo "       (see -h for full options)" >&2
  exit 1
fi

PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)

EXTRA=""
if [[ -n "$N_SUBSET" ]]; then EXTRA="$EXTRA --n $N_SUBSET"; fi

echo "[index] dataset=$DATASET out=$OUT M=$M ef_c=$EF_CONSTRUCTION ef_b=$EF_BASE tau=$TAU space=$SPACE"
"$PY" "$HERE/2pc/bench/build_npz.py" \
  --dataset "$DATASET" --out "$OUT" \
  --M "$M" --ef_construction "$EF_CONSTRUCTION" \
  --ef_base "$EF_BASE" --tau_base "$TAU" --tau_upper "$TAU" \
  --space "$SPACE" $EXTRA

echo
echo "[index] done.  Inspect with:"
echo "  python3 -c 'import numpy as np, json; d = np.load(\"$OUT\", allow_pickle=True); m = json.loads(str(d[\"meta_json\"][0])); print(m); print(\"vecs_0.shape=\", d[\"vecs_0\"].shape, \"unique_vals=\", len(np.unique(d[\"vecs_0\"])))'"
