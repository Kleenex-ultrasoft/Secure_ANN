#!/bin/bash
if [[ -n "${SPU_ENV:-}" ]]; then
    source "$SPU_ENV"
elif [[ -f "$HOME/miniconda3/bin/activate" ]]; then
    # Optional: activate local conda env if available.
    source "$HOME/miniconda3/bin/activate" sf-spu
fi
export PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=0
# Force MT
export OMP_NUM_THREADS=64
export YACL_NUM_THREADS=64
set -eo pipefail

# Argument: lan or wan
MODE=${1:-lan}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="$SCRIPT_DIR"
THROTTLE_SCRIPT="$ROOT/throttle_ns.sh"

cd "$ROOT"

CFG=results/spu_lookup_test.json
LOG=logs/spu_lookup_test.log

if [[ ! -d "$ROOT/bench_spu" ]]; then
    echo "missing bench_spu (expected at $ROOT/bench_spu)" >&2
    exit 1
fi

# Cleanup function (Host side)
cleanup_host() {
    # Only kill SPU-related python processes
    pkill -f "spu.utils.distributed" || true
    pkill -f "ForkServerProcess" || true
    
    # Check ports
    for port in 9327 9328 9930 9931; do
        if lsof -i :$port >/dev/null 2>&1; then
            fuser -k -n tcp $port || true
        fi
    done
    sleep 2
}

cleanup_host
echo ">>> Starting Lookup Test ($MODE) inside Namespace..."

# We use unshare to create a NEW network stack.
# We map the current user to root inside the namespace (-r) to allow 'tc' commands.
unshare -r -n bash -c "
    # 1. Setup Loopback inside namespace
    ip link set lo up
    
    # 2. Apply Throttling (affects only this namespace's lo)
    $THROTTLE_SCRIPT del || true
    $THROTTLE_SCRIPT $MODE
    $THROTTLE_SCRIPT show | head -n 5

    # 3. Generate Config
    python bench_spu/gen_ppd_cfg.py --protocol CHEETAH --field FM64 --out \"$CFG\"

    # 4. Address Fix
    python3 -c \"
import json
with open('$CFG', 'r') as f:
    c = json.load(f)
for node in c['nodes']:
    if isinstance(node, dict) and 'address' in node:
        port = node['address'].split(':')[-1]
        node['address'] = '127.0.0.1:' + port
json.dump(c, open('$CFG','w'), indent=2)\"

    # 5. Start SPU Nodes
    echo '>>> Starting SPU...'
    nohup python -m spu.utils.distributed -c \"$CFG\" up >\"$LOG\" 2>&1 &
    UP_PID=\$!

    echo '>>> Waiting 8s...'
    sleep 8

    if ! ps -p \$UP_PID > /dev/null; then
        echo 'ERROR: SPU died.'
        cat \"$LOG\"
        exit 1
    fi

    # 6. Run Benchmark
    PYTHON_SCRIPT=bench_spu/microbench_detailed.py
    
    echo '>>> Running Benchmark...'
    python \"\$PYTHON_SCRIPT\" --config \"$CFG\"

    # 7. Cleanup inside namespace
    kill -9 \$UP_PID 2>/dev/null || true
"

echo ">>> Test Complete."
cleanup_host
