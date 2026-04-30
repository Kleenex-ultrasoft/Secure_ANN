#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="$SCRIPT_DIR"
PANTHER="$ROOT/OpenPanther"
THROTTLE="$ROOT/throttle_ns.sh"

OUT_LAN="$ROOT/results/panther_visited_lan.jsonl"
OUT_WAN="$ROOT/results/panther_visited_wan.jsonl"

mkdir -p "$ROOT/results" "$PANTHER/logs"

# Grid to benchmark
NVS=(128 256 512)     # |V| visited list length
MS=(32 64 128)        # |U| number of ids checked
ID_BITS=20

# Iterations (WAN should be small)
ITERS_LAN=1
ITERS_WAN=1
SEED=0

DRAIN_ITERS=100
DRAIN_SLEEP=0.02

cd "$PANTHER"
bazel build -c opt //experimental/panther:visited_test
cd "$ROOT"

run_mode () {
  local MODE="$1"
  local OUT="$2"
  : > "$OUT"

  for NV in "${NVS[@]}"; do
    for M in "${MS[@]}"; do
      local ITERS="$ITERS_LAN"
      if [[ "$MODE" == "wan" ]]; then ITERS="$ITERS_WAN"; fi

      # unique port per case
      local PORT=$((12000 + NV + M))

      unshare -Urn bash -lc "
set -euo pipefail
ip link set lo up

cd '$ROOT'
$THROTTLE del || true
$THROTTLE $MODE

cd '$PANTHER'
mkdir -p logs

# server (party=1)
./bazel-bin/experimental/panther/visited_test 1 $PORT $NV $M $ID_BITS $ITERS $SEED \
  >logs/panther_vis_server_${MODE}_NV${NV}_M${M}.raw 2>&1 &
spid=\$!
sleep 0.5

# client (party=2)
./bazel-bin/experimental/panther/visited_test 2 $PORT $NV $M $ID_BITS $ITERS $SEED \
  >logs/panther_vis_client_${MODE}_NV${NV}_M${M}.raw 2>&1


# give server time to flush and exit normally
for i in \$(seq 1 200); do
  if ! kill -0 \$spid 2>/dev/null; then
    break
  fi
  sleep 0.01
done
# if still alive, then kill
if kill -0 \$spid 2>/dev/null; then
  kill \$spid 2>/dev/null || true
fi
wait \$spid 2>/dev/null || true


# drain
for i in \$(seq 1 $DRAIN_ITERS); do
  if tc -s qdisc show dev lo 2>/dev/null | grep -q 'backlog 0b 0p'; then
    break
  fi
  sleep $DRAIN_SLEEP
done

tc -s qdisc show dev lo > logs/panther_vis_${MODE}_NV${NV}_M${M}.qdisc 2>&1 || true

cd '$ROOT'
$THROTTLE del || true
"

      python3 - "$MODE" \
        "$PANTHER/logs/panther_vis_server_${MODE}_NV${NV}_M${M}.raw" \
        "$PANTHER/logs/panther_vis_client_${MODE}_NV${NV}_M${M}.raw" \
        "$PANTHER/logs/panther_vis_${MODE}_NV${NV}_M${M}.qdisc" \
        >> "$OUT" <<'PY'
import json, sys, pathlib, re

mode = sys.argv[1]
srv = pathlib.Path(sys.argv[2]).read_text(errors="replace")
cli = pathlib.Path(sys.argv[3]).read_text(errors="replace")
qdisc = pathlib.Path(sys.argv[4]).read_text(errors="replace")

def last_json(txt):
    for line in txt.splitlines()[::-1]:
        line=line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                return None
    return None

j1 = last_json(srv) or {}
j2 = last_json(cli) or {}

# Sum both parties' reported comm (garbler dominates; evaluator can be ~0 and that's OK)
comm_sum = (j1.get("comm_total_bytes_reported", 0) or 0) + (j2.get("comm_total_bytes_reported", 0) or 0)

# Use client-side latency as "observed"; server should be similar
lat = j2.get("lat_ms", j1.get("lat_ms"))

# tc-sent
m = re.search(r"^qdisc tbf .*?\n\s*Sent (\d+) bytes .*?\(dropped (\d+), overlimits (\d+)", qdisc, re.M)
tc_sent = int(m.group(1)) if m else None
tbf_dropped = int(m.group(2)) if m else None
tbf_over = int(m.group(3)) if m else None

row = {
  "framework": "PANTHER",
  "mode": mode,
  "op": "visited_scan",
  "proto": "emp_sh2pc_gc",

  "nv": j2.get("nv", j1.get("nv")),
  "m": j2.get("m", j1.get("m")),
  "id_bits": j2.get("id_bits", j1.get("id_bits")),
  "iters": j2.get("iters", j1.get("iters")),
  "seed": j2.get("seed", j1.get("seed")),

  "lat_ms": lat,
  "comm_total_bytes_reported_sum": comm_sum,

  "comm_total_bytes_tc_sent": tc_sent,
  "tbf_dropped": tbf_dropped,
  "tbf_overlimits": tbf_over,
}

iters = row["iters"] or 1
if row["comm_total_bytes_tc_sent"] is not None:
    row["comm_total_bytes_tc_sent_per_iter"] = row["comm_total_bytes_tc_sent"] / iters
else:
    row["comm_total_bytes_tc_sent_per_iter"] = None

row["comm_total_bytes_reported_sum_per_iter"] = row["comm_total_bytes_reported_sum"] / iters
# ===========================


print(json.dumps(row))
PY
    done
  done
}

run_mode lan "$OUT_LAN"
run_mode wan "$OUT_WAN"

echo "Wrote:"
echo "  $OUT_LAN"
echo "  $OUT_WAN"
