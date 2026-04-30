#!/usr/bin/env bash
set -euo pipefail

DEV=${DEV:-lo}

del() {
  tc qdisc del dev "$DEV" root 2>/dev/null || true
}

# netem delay is one-way; the paper reports the round-trip RTT, so each
# direction gets RTT/2.
lan() {
  # LAN: 4000 Mbps, 1 ms RTT (paper §7.1).
  del
  tc qdisc add dev "$DEV" root handle 1: tbf rate 4000mbit burst 2mb limit 10mb
  tc qdisc add dev "$DEV" parent 1:1 handle 10: netem delay 0.5ms
}

wan() {
  # WAN: 320 Mbps, 50 ms RTT (paper §7.1).
  del
  tc qdisc add dev "$DEV" root handle 1: tbf rate 320mbit burst 2mb limit 10mb
  tc qdisc add dev "$DEV" parent 1:1 handle 10: netem delay 25ms
}

show() {
  tc -s qdisc show dev "$DEV"
}

case "${1:-}" in
  del) del ;;
  lan) lan ;;
  wan) wan ;;
  show) show ;;
  *)
    echo "Usage: $0 {lan|wan|del|show}"
    exit 1
    ;;
esac
