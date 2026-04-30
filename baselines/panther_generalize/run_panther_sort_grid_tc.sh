#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="$SCRIPT_DIR"
PANTHER="$ROOT/OpenPanther"
THROTTLE="$ROOT/throttle_ns.sh"

OUT_LAN="$ROOT/results/panther_sort_lan.jsonl"
OUT_WAN="$ROOT/results/panther_sort_wan.jsonl"

mkdir -p "$ROOT/results" "$PANTHER/logs"

# sort-grid params (passed to the binary; not hardcoded into JSON)
NS=(64 128 256 512)

ITEM_BITS=23
DISCARD_BITS=5
ID_BITS=20
ALGO=0        # 0=both, 1=naive only, 2=bitonic only
SEED=0

DRAIN_ITERS=100
DRAIN_SLEEP=0.02

cd "$PANTHER"
bazel build -c opt //experimental/panther:topk_test
cd "$ROOT"

run_mode () {
  local MODE="$1"
  local OUT="$2"
  : > "$OUT"

  for N in "${NS[@]}"; do
    unshare -Urn bash -lc "
set -euo pipefail
ip link set lo up

cd '$ROOT'
$THROTTLE del || true
$THROTTLE $MODE

cd '$PANTHER'
mkdir -p logs

# server (party=1)
./bazel-bin/experimental/panther/topk_test 1 1111 $N $N $ITEM_BITS $DISCARD_BITS $ID_BITS $ALGO $SEED \
  >logs/panther_sort_server_${MODE}_N${N}.raw 2>&1 &
spid=\$!
sleep 1

# client (party=2)
./bazel-bin/experimental/panther/topk_test 2 1111 $N $N $ITEM_BITS $DISCARD_BITS $ID_BITS $ALGO $SEED \
  >logs/panther_sort_client_${MODE}_N${N}.raw 2>&1

kill \$spid 2>/dev/null || true
wait \$spid 2>/dev/null || true

# drain
for i in \$(seq 1 $DRAIN_ITERS); do
  if tc -s qdisc show dev lo 2>/dev/null | grep -q 'backlog 0b 0p'; then
    break
  fi
  sleep $DRAIN_SLEEP
done

tc -s qdisc show dev lo > logs/panther_sort_${MODE}_N${N}.qdisc 2>&1 || true

cd '$ROOT'
$THROTTLE del || true
"

    python3 - "$MODE" \
      "$PANTHER/logs/panther_sort_server_${MODE}_N${N}.raw" \
      "$PANTHER/logs/panther_sort_client_${MODE}_N${N}.raw" \
      "$PANTHER/logs/panther_sort_${MODE}_N${N}.qdisc" \
      >> "$OUT" <<'PY'
import json, sys, pathlib, re

mode = sys.argv[1]
srv = pathlib.Path(sys.argv[2]).read_text(errors="replace")
cli = pathlib.Path(sys.argv[3]).read_text(errors="replace")
qdisc = pathlib.Path(sys.argv[4]).read_text(errors="replace")

def last_json(txt):
    for line in txt.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                return None
    return None

def grab_ms(txt, key):
    m = re.search(rf"{key}\s*topk time:\s*(\d+)\s*ms", txt)
    return int(m.group(1)) if m else None

def grab_kb(txt, which):
    m = re.search(rf"Communication cost test_{which}_topk:\s*(\d+)\s*KBs", txt)
    return int(m.group(1)) if m else None

def parse_header(txt):
    # "From 512 elements top-512 23 5"
    m = re.search(r"From\s+(\d+)\s+elements\s+top-(\d+)\s+(\d+)\s+(\d+)", txt)
    if not m:
        return None
    return {
        "n": int(m.group(1)),
        "k": int(m.group(2)),
        "item_bits": int(m.group(3)),
        "discard_bits": int(m.group(4)),
    }

hdr = parse_header(cli) or {}
pj = last_json(cli) or {}

# times: use client-side times (observed)
lat_naive = grab_ms(cli, "Naive")
lat_bitonic = grab_ms(cli, "Bitonic")

# comm: sum server + client
naive_kb_sum = (grab_kb(srv, "naive") or 0) + (grab_kb(cli, "naive") or 0)
bitonic_kb_sum = (grab_kb(srv, "bitonic") or 0) + (grab_kb(cli, "bitonic") or 0)

# tc-sent
m = re.search(r"^qdisc tbf .*?\n\s*Sent (\d+) bytes .*?\(dropped (\d+), overlimits (\d+)", qdisc, re.M)
tc_sent = int(m.group(1)) if m else None
tbf_dropped = int(m.group(2)) if m else None
tbf_over = int(m.group(3)) if m else None

# Treat k==n as sort in the JSON output
n_run = hdr.get("n", pj.get("n"))
k_run = hdr.get("k", pj.get("k"))
op = "sort" if (n_run is not None and k_run is not None and int(n_run) == int(k_run)) else pj.get("op", "topk")

row = {
    "framework": "PANTHER",
    "mode": mode,
    "op": op,
    "proto": pj.get("proto", "emp_sh2pc_gc"),
    "algo": pj.get("algo", 0),

    "n": n_run,
    "k": k_run,
    "item_bits": hdr.get("item_bits", pj.get("item_bits")),
    "discard_bits": hdr.get("discard_bits", pj.get("discard_bits")),
    "id_bits": pj.get("id_bits"),
    "seed": pj.get("seed"),

    "lat_ms_naive": lat_naive,
    "lat_ms_bitonic": lat_bitonic,

    "comm_bytes_naive_sum": naive_kb_sum * 1024,
    "comm_bytes_bitonic_sum": bitonic_kb_sum * 1024,
    "comm_total_bytes_reported_sum": (naive_kb_sum + bitonic_kb_sum) * 1024,

    "comm_total_bytes_tc_sent": tc_sent,
    "tbf_dropped": tbf_dropped,
    "tbf_overlimits": tbf_over,
}

print(json.dumps(row))
PY
  done
}

run_mode lan "$OUT_LAN"
run_mode wan "$OUT_WAN"
echo "Wrote:"
echo "  $OUT_LAN"
echo "  $OUT_WAN"
