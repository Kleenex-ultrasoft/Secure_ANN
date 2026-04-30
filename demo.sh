#!/usr/bin/env bash
#
# One-command MPC ANN query demo for HNSecW.
#
#   bash demo.sh                # runs Panther IVF + DORAM (3PC) on a
#                                 400-vector / 32-dim toy dataset and
#                                 prints the top-5 (id, dist) result.
#   bash demo.sh --hnsw-doram   # additionally run HNSW + DORAM on the
#                                 same input for a side-by-side anchor.
#
# Prereqs: MP-SPDZ built under $MP_SPDZ (default ~/hnsecw_build/MP-SPDZ),
# instructions.
#
# This is the smallest end-to-end MPC ANN run we ship.  The protocol
# is real — secret-shared centroids, DORAM-fetched bins, secret L2
# distance — but the dataset (first 400 SIFT vectors truncated to 32
# dims) is small enough that the whole demo finishes under a minute
# on a desktop.

set -euo pipefail

MP_SPDZ=${MP_SPDZ:-$HOME/hnsecw_build/MP-SPDZ}
SOURCE_DIR=${SOURCE_DIR:-$HOME/hnsecw_build/HNSecW}
SIFT_NPY=${SIFT_NPY:-$HOME/hnsecw_build/datasets/sift_base.npy}
PYTHON=${PYTHON:-$HOME/miniforge3/envs/hnsecw/bin/python}
ALSO_HNSW_DORAM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hnsw-doram) ALSO_HNSW_DORAM=1; shift;;
    --mp-spdz) MP_SPDZ="$2"; shift 2;;
    --source)  SOURCE_DIR="$2"; shift 2;;
    --sift)    SIFT_NPY="$2"; shift 2;;
    --python)  PYTHON="$2"; shift 2;;
    -h|--help)
      sed -n '1,18p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ ! -x "$MP_SPDZ/Scripts/ring.sh" ]]; then
  echo "MP-SPDZ not found at $MP_SPDZ.  Set MP_SPDZ or pass --mp-spdz." >&2
  exit 2
fi

if [[ ! -f "$SIFT_NPY" ]]; then
  echo "SIFT not found at $SIFT_NPY.  Generating a 400-row random demo dataset instead." >&2
  SIFT_NPY=/tmp/demo_random.npy
  "$PYTHON" -c "import numpy as np; np.save('$SIFT_NPY', np.random.default_rng(0).integers(0, 256, (400, 32)).astype(np.float32))"
fi

DEMO_DIR=/tmp/hnsecw_demo
mkdir -p "$DEMO_DIR"

echo
echo "[demo] Preparing 400 vectors (32 dims) from $SIFT_NPY"
"$PYTHON" -c "
import numpy as np
X = np.load('$SIFT_NPY').astype(np.float32)
sub = X[:400, :32]
np.save('$DEMO_DIR/X.npy', sub)
print('  shape =', sub.shape)
"

cd "$MP_SPDZ"

echo
echo "[demo] Generating Panther IVF inputs (3-out-of-3 additive shares)"
"$PYTHON" "$SOURCE_DIR/3pc/mp_spdz/bench/gen_panther_ivf_inputs.py" \
  --dataset_npy "$DEMO_DIR/X.npy" \
  --out_dir "Player-Data/demo_panther" \
  --N 400 --bin_size 24 --u 20 --K 10 --T 20 \
  --id_bits 16 --dist_bits 24 --seed 0 | tail -3

# stage inputs into the path the runner expects
cp -f Player-Data/demo_panther/Input-Binary-P0-0 Player-Data/
cp -f Player-Data/demo_panther/Input-Binary-P1-0 Player-Data/
cp -f Player-Data/demo_panther/Input-Binary-P2-0 Player-Data/

echo
echo "[demo] Compiling panther_ivf_doram (this may take a few seconds)"
./compile.py -R 64 panther_ivf_doram 400 32 20 24 20 10 16 24 \
  > "$DEMO_DIR/compile_panther.log" 2>&1
tail -3 "$DEMO_DIR/compile_panther.log"

echo
echo "[demo] Running Panther IVF + DORAM + secret-distance (3PC, replicated-ring-64)"
START=$(date +%s.%N)
./Scripts/ring.sh -v panther_ivf_doram-400-32-20-24-20-10-16-24 2>&1 \
  | tee "$DEMO_DIR/run_panther.log" \
  | grep -E "rank=|^Time =|^Data sent|^Global data sent" \
  | head -25
END=$(date +%s.%N)
printf "\n[demo] panther_ivf_doram wall time: %.3f s\n" "$(echo "$END - $START" | bc)"

if [[ "$ALSO_HNSW_DORAM" -eq 1 ]]; then
  echo
  echo "[demo] Generating HNSW DORAM inputs"
  if ! "$PYTHON" -c "import hnswlib" 2>/dev/null; then
    echo "  hnswlib not installed; pip install hnswlib  (skipping HNSW DORAM run)"
  else
    "$PYTHON" "$SOURCE_DIR/3pc/mp_spdz/bench/gen_mpspdz_inputs.py" \
      --dataset_npy "$DEMO_DIR/X.npy" \
      --out_dir Player-Data/demo_hnsw \
      --N 400 --M 16 --seed 0 | tail -3
    cp -f Player-Data/demo_hnsw/Input-Binary-P0-0 Player-Data/
    cp -f Player-Data/demo_hnsw/Input-Binary-P1-0 Player-Data/
    cp -f Player-Data/demo_hnsw/Input-Binary-P2-0 Player-Data/

    echo
    echo "[demo] Compiling hnsw_layer_search"
    ./compile.py -R 64 hnsw_layer_search 400 16 32 20 4 20 20 32 \
      > "$DEMO_DIR/compile_hnsw.log" 2>&1
    tail -3 "$DEMO_DIR/compile_hnsw.log"

    echo
    echo "[demo] Running HNSW + DORAM (3PC, replicated-ring-64)"
    START=$(date +%s.%N)
    ORAM_IMPL=optimal ./Scripts/ring.sh -v hnsw_layer_search-400-16-32-20-4-20-20-32 2>&1 \
      | tee "$DEMO_DIR/run_hnsw.log" \
      | grep -E "best_id|^Time =|^Data sent|^Global data sent" \
      | head -25
    END=$(date +%s.%N)
    printf "\n[demo] hnsw_layer_search wall time: %.3f s\n" "$(echo "$END - $START" | bc)"
  fi
fi

echo

LAN_LAT=$(grep -oE '^Time = [0-9.]+' "$DEMO_DIR/run_panther.log" | head -1 | grep -oE '[0-9.]+')
COMM_MB=$(grep -oE '^Data sent = [0-9.]+ MB' "$DEMO_DIR/run_panther.log" | head -1 | grep -oE '[0-9.]+')
GLOBAL_COMM=$(grep -oE '^Global data sent = [0-9.]+ MB' "$DEMO_DIR/run_panther.log" | head -1 | grep -oE '[0-9.]+')
if [[ -n "$LAN_LAT" && -n "$COMM_MB" ]]; then
  echo
  echo "============================================================"
  echo "[demo] HNSecW vs Panther IVF+DORAM demo (N=400 SIFT-32d toy)"
  echo "============================================================"
  echo "  Dataset: first 400 vectors of SIFT, truncated to D=32"
  echo "  This is the BASELINE (Panther IVF+DORAM at u=20 = full probe);"
  echo "  HNSecW saves rounds via batched layer-by-layer search."
  echo
  echo "  Top-K result (revealed by protocol):"
  grep -E "^rank=" "$DEMO_DIR/run_panther.log" | head -10 | sed 's/^/    /'
  echo
  echo "  Recall vs plaintext brute-force:"
  "$PYTHON" - <<PYEOF
import numpy as np, re
X = np.load('$DEMO_DIR/X.npy').astype(np.int64)
# Reproduce the EXACT query gen_panther_ivf_inputs.py emitted (seed=0,
# query = clip(mean + uniform[-5, 5], 0, 255)).  See
# 3pc/mp_spdz/bench/gen_panther_ivf_inputs.py:185.
rng = np.random.default_rng(0)
mean_vec = X.mean(axis=0).astype(np.int64)
D = X.shape[1]
q = np.clip(mean_vec + rng.integers(-5, 5, D), 0, 255).astype(np.int64)
diff = X - q; dist = (diff*diff).sum(axis=1)
pt_top10 = np.argsort(dist)[:10]
mpc_top10_ids = []
with open('$DEMO_DIR/run_panther.log') as f:
    for line in f:
        m = re.search(r'rank=\d+ id=(\d+)', line)
        if m: mpc_top10_ids.append(int(m.group(1)))
mpc_set = set(mpc_top10_ids[:10])
pt_set  = set(pt_top10.tolist())
overlap = mpc_set & pt_set
recall_at_10 = len(overlap) / 10.0
print(f"    plaintext top-10  : {pt_top10.tolist()}")
print(f"    MPC top-10        : {mpc_top10_ids[:10]}")
print(f"    overlap           : {sorted(overlap)}")
print(f"    recall@10         : {recall_at_10:.2f}")
PYEOF
  echo
  echo "  Total Latency       : $LAN_LAT s"
  echo "  Total Communication : ${GLOBAL_COMM:-$COMM_MB} MB"
  echo "============================================================"
fi
