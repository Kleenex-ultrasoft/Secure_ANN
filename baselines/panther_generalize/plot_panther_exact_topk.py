#!/usr/bin/env python3
"""
Plot Panther exact top-k results from JSONL and compare to the paper figure.

- Reads your JSONL (produced by the summed-runner) for LAN/WAN.
- Plots:
  (d) Exact top-k (LAN): runtime vs k (seconds)
  (e) Exact top-k (WAN): runtime vs k (seconds)
  (f) Exact top-k (Comm.): communication vs k (MiB), log y-axis

Notes:
- "SANNS" corresponds to naive top-k.
- "Panther" corresponds to bitonic top-k network.
- Uses comm_bytes_*_sum fields (total comm = ALICE+BOB). You can switch to server-only
  by changing comm_field_* to comm_bytes_*_alice if you add those fields later.
"""

import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt


def load_jsonl(path: str):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            rows.append(json.loads(line))
    return rows


def index_by_k(rows, mode: str):
    out = {}
    for r in rows:
        if r.get("mode") != mode:
            continue
        k = int(r["k"])
        out[k] = r
    return out


def mib(x_bytes: float) -> float:
    return x_bytes / (1024.0 * 1024.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lan", required=True, help="results/panther_topk_lan.jsonl")
    ap.add_argument("--wan", required=True, help="results/panther_topk_wan.jsonl")
    ap.add_argument("--out", default="panther_exact_topk_repro.png")
    ap.add_argument("--title_prefix", default="", help="Optional prefix for subplot titles")
    args = ap.parse_args()

    lan_rows = load_jsonl(args.lan)
    wan_rows = load_jsonl(args.wan)

    lan = index_by_k(lan_rows, "lan")
    wan = index_by_k(wan_rows, "wan")

    ks = sorted(set(lan.keys()) & set(wan.keys()))
    if not ks:
        raise SystemExit("No overlapping k values between LAN and WAN JSONL (check mode fields).")

    # Runtime fields (ms)
    rt_naive_lan = [lan[k]["lat_ms_naive"] / 1000.0 for k in ks]
    rt_bit_lan   = [lan[k]["lat_ms_bitonic"] / 1000.0 for k in ks]
    rt_naive_wan = [wan[k]["lat_ms_naive"] / 1000.0 for k in ks]
    rt_bit_wan   = [wan[k]["lat_ms_bitonic"] / 1000.0 for k in ks]

    # Communication fields (bytes)
    # Prefer summed protocol payload comm (paper-style).
    # Fallback to comm_total_bytes_reported_sum if you used that key instead.
    def get_comm(row, key_primary, key_fallback):
        if key_primary in row and row[key_primary] is not None:
            return float(row[key_primary])
        if key_fallback in row and row[key_fallback] is not None:
            return float(row[key_fallback])
        raise KeyError(f"Missing comm keys {key_primary} / {key_fallback}")

    comm_naive = [mib(get_comm(lan[k], "comm_bytes_naive_sum", "comm_bytes_naive")) for k in ks]
    comm_bit   = [mib(get_comm(lan[k], "comm_bytes_bitonic_sum", "comm_bytes_bitonic")) for k in ks]

    # Plot
    fig = plt.figure(figsize=(12.5, 4.0))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # (d) LAN runtime
    ax1.plot(ks, rt_naive_lan, marker="x", linewidth=2, label="SANNS")
    ax1.plot(ks, rt_bit_lan, marker="*", linewidth=2, label="Panther")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title(f"{args.title_prefix}Exact top-k (LAN)".strip())
    ax1.legend()

    # (e) WAN runtime
    ax2.plot(ks, rt_naive_wan, marker="x", linewidth=2, label="SANNS")
    ax2.plot(ks, rt_bit_wan, marker="*", linewidth=2, label="Panther")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Runtime (s)")
    ax2.set_title(f"{args.title_prefix}Exact top-k (WAN)".strip())
    ax2.legend()

    # (f) Communication (log y)
    width = 0.35
    x = list(range(len(ks)))
    ax3.bar([i - width/2 for i in x], comm_naive, width=width, label="SANNS")
    ax3.bar([i + width/2 for i in x], comm_bit, width=width, label="Panther")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(k) for k in ks])
    ax3.set_xlabel("k")
    ax3.set_ylabel("Communication cost (MiB)")
    ax3.set_yscale("log")
    ax3.set_title(f"{args.title_prefix}Exact top-k (Comm.)".strip())
    ax3.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


# python3 plot_panther_exact_topk.py \
#   --lan /home2/fahong/Experiment_12_19/Microbenchmark/results/panther_topk_lan.jsonl \
#   --wan /home2/fahong/Experiment_12_19/Microbenchmark/results/panther_topk_wan.jsonl \
#   --out /home2/fahong/Experiment_12_19/Microbenchmark/Pictures/panther_exact_topk_repro.png
