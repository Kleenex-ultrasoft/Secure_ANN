#!/usr/bin/env bash
set -euo pipefail

NPZ=${1:?usage: run_single_query.sh /path/to/input.npz}
ABY_BIN=${ABY_BIN:-$HOME/hnsecw_build/ABY/build/bin/hnsecw_cli}
LOG_DIR=${LOG_DIR:-mpc_logs_single}

python3 "$(dirname "$0")/run_mpc_layer_search.py" \
  --npz "$NPZ" \
  --aby_bin "$ABY_BIN" \
  --mode single \
  --protocol b2y \
  --threads 64 \
  --log_dir "$LOG_DIR"
