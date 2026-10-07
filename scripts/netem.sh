#!/usr/bin/env bash
# Emulates the two networks of the paper on the loopback interface, for runs
# where all parties are processes on one machine.  Needs root and the
# sch_netem kernel module.
#
#   sudo bash scripts/netem.sh lan    # 4000 Mbit/s, 1 ms round-trip time
#   sudo bash scripts/netem.sh wan    #  320 Mbit/s, 50 ms round-trip time
#   sudo bash scripts/netem.sh off
#
# A packet crosses the loopback interface once per direction, so the delay
# set here is half the round-trip time.
set -euo pipefail
DEV=${DEV:-lo}

tc qdisc del dev "$DEV" root 2>/dev/null || true
case "${1:-}" in
  lan) tc qdisc add dev "$DEV" root netem delay 0.5ms rate 4000mbit ;;
  wan) tc qdisc add dev "$DEV" root netem delay 25ms rate 320mbit ;;
  off) ;;
  *) echo "usage: $0 lan|wan|off" >&2; exit 1 ;;
esac
tc qdisc show dev "$DEV"
