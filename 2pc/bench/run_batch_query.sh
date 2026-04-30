#!/usr/bin/env bash
set -euo pipefail

NPZ=${1:?usage: run_batch_query.sh /path/to/input.npz}
Q=${Q:-4}
ABY_BIN=${ABY_BIN:-$HOME/hnsecw_build/ABY/build/bin/hnsecw_cli}
LOG_DIR=${LOG_DIR:-mpc_logs_batch}

python3 "$(dirname "$0")/run_mpc_layer_search.py" \
  --npz "$NPZ" \
  --aby_bin "$ABY_BIN" \
  --mode batch \
  --protocol b2y \
  --threads 64 \
  --num_queries "$Q" \
  --auto_thresh \
  --rtt_ms "${RTT_MS:-50}" \
  --bw_mbps "${BW_MBPS:-320}" \
  --log_dir "$LOG_DIR"
