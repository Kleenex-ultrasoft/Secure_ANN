#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="$SCRIPT_DIR"
PANTHER="$ROOT/OpenPanther"
THROTTLE="$ROOT/throttle_ns.sh"

OUT_LAN="$ROOT/results/panther_ssip_lan.jsonl"
OUT_WAN="$ROOT/results/panther_ssip_wan.jsonl"
mkdir -p "$ROOT/results" "$PANTHER/logs"

MS=(16 32 64 128)
DS=(96 128 256 768 3072 7168)

ELEM_BITS=8
DISCARD_BITS=5

ITERS_LAN=20
ITERS_WAN=5

T=1  # start stable; increase later

run_mode () {
  local MODE="$1"
  local OUT="$2"
  : > "$OUT"

  local ITERS="$ITERS_LAN"
  if [[ "$MODE" == "wan" ]]; then ITERS="$ITERS_WAN"; fi

  for m in "${MS[@]}"; do
    for d in "${DS[@]}"; do
      # pick fresh ports per run to avoid reuse issues
      local P0=$((9700 + m + (d % 1000)))
      local P1=$((P0 + 1))
      local PARTIES="127.0.0.1:${P0},127.0.0.1:${P1}"

      unshare -Urn bash -lc "
set -euo pipefail
ip link set lo up

cd '$ROOT'
$THROTTLE del || true
$THROTTLE $MODE

cd '$PANTHER'
mkdir -p logs

./bazel-bin/experimental/panther/ss_ip_benchmark \
  --rank=1 --parties='$PARTIES' --m=$m --d=$d --iters=$ITERS --elem_bits=$ELEM_BITS --discard_bits=$DISCARD_BITS --t=$T \
  > logs/ssip_r1_${MODE}_m${m}_d${d}.log 2>&1 &
p1=\$!

sleep 0.5

./bazel-bin/experimental/panther/ss_ip_benchmark \
  --rank=0 --parties='$PARTIES' --m=$m --d=$d --iters=$ITERS --elem_bits=$ELEM_BITS --discard_bits=$DISCARD_BITS --t=$T \
  > logs/ssip_r0_${MODE}_m${m}_d${d}.log 2>&1

wait \$p1 || true

$THROTTLE del || true
"

      # Extract JSON from rank0 log (it prints the JSON line)
      python3 - "$MODE" "$m" "$d" \
        "$PANTHER/logs/ssip_r0_${MODE}_m${m}_d${d}.log" \
        >> "$OUT" <<'PY'
import json, sys, pathlib
mode = sys.argv[1]
m = int(sys.argv[2]); d = int(sys.argv[3])
txt = pathlib.Path(sys.argv[4]).read_text(errors="replace")

jline = None
for line in txt.splitlines()[::-1]:
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        jline = line
        break

if jline is None:
    print(json.dumps({"framework":"PANTHER","op":"ss_ip","mode":mode,"m":m,"d":d,"error":"no_json_found"}))
else:
    row = json.loads(jline)
    row["mode"] = mode
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
