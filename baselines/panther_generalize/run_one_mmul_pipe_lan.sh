#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="$SCRIPT_DIR"
PANTHER="$ROOT/OpenPanther"

# kill leftovers (no sudo needed)
pkill -f mmul_topk_pipeline_benchmark || true

cd "$ROOT"

unshare -Urn bash -lc '
set -euo pipefail
ip link set lo up
./throttle_ns.sh del || true
./throttle_ns.sh lan

cd "'"$PANTHER"'"
P0=9530
P1=9531
PARTIES="127.0.0.1:${P0},127.0.0.1:${P1}"
EMP=7111

echo "[runner] start rank=1"
timeout 180s stdbuf -oL -eL ./bazel-bin/experimental/panther/mmul_topk_pipeline_benchmark \
  --rank=1 --parties="$PARTIES" --emp_port=$EMP \
  --m=256 --d=768 --k=10 --iters=100 --warmup=0 \
  --elem_bits=8 --item_bits=31 --id_bits=20 --t=8 --seed=0 \
  --link_recv_timeout_ms=600000 --link_connect_retry_times=60 --link_connect_retry_interval_ms=1000 \
  > /tmp/mmul_pipe_r1.log 2>&1 &
p1=$!

sleep 1

echo "[runner] start rank=0"
timeout 180s stdbuf -oL -eL ./bazel-bin/experimental/panther/mmul_topk_pipeline_benchmark \
  --rank=0 --parties="$PARTIES" --emp_port=$EMP \
  --m=256 --d=768 --k=10 --iters=100 --warmup=0 \
  --elem_bits=8 --item_bits=31 --id_bits=20 --t=8 --seed=0 \
  --link_recv_timeout_ms=600000 --link_connect_retry_times=60 --link_connect_retry_interval_ms=1000 \
  > /tmp/mmul_pipe_r0.log 2>&1 &
p0=$!

wait $p0 || true
wait $p1 || true

echo "===== rank0 tail ====="
tail -n 200 /tmp/mmul_pipe_r0.log || true
echo "===== rank1 tail ====="
tail -n 200 /tmp/mmul_pipe_r1.log || true
'

