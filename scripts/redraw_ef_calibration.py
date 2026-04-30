"""Redraw fig:ef-calibration-compact with the actual fixed-T HNSW
recall numbers measured on 6 datasets.

This is the MPC-EQUIVALENT recall (fixed-T HNSW in plaintext) at
each ef.  Plot recall@10 vs ef per dataset.

Output: pdf next to this script.  Original figure (in
Workspace/figures_py/) is NOT touched.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path(__file__).parent.parent / "docs/measurements/ef_calibration_compact_data.csv"
OUT = Path(__file__).parent / "ef_calibration_compact_v2.pdf"

# read
data = {}  # ds -> [(ef, recall)]
with open(CSV) as f:
    for row in csv.DictReader(f):
        ds = row["dataset"]
        data.setdefault(ds, []).append((int(row["ef"]), float(row["recall_at_10"])))

# canonical order
ORDER = ["fashion_60k", "deep_1m_v2", "sift_1m",
         "sift_100k_M32", "laion_200k", "msmarco_200k"]
COLORS = {
    "fashion_60k":  "#888888",
    "deep_1m_v2":   "#009E73",
    "sift_1m":      "#0072B2",
    "sift_100k_M32":"#56B4E9",
    "laion_200k":   "#D55E00",
    "msmarco_200k": "#CC79A7",
}
LABELS = {
    "fashion_60k":  "Fashion 60K (D=784, sparse)",
    "deep_1m_v2":   "DEEP 1M (D=96)",
    "sift_1m":      "SIFT 1M (D=128)",
    "sift_100k_M32":"SIFT 100K (M=32)",
    "laion_200k":   "LAION 200K (cosine, D=512)",
    "msmarco_200k": "MSMARCO 200K (cosine, D=768)",
}

fig, ax = plt.subplots(figsize=(5.5, 3.6))
for ds in ORDER:
    if ds not in data:
        continue
    xs, ys = zip(*sorted(data[ds]))
    ax.plot(xs, ys, marker="o", color=COLORS[ds], label=LABELS[ds],
            linewidth=1.5, markersize=4)

ax.axhline(0.9, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(135, 0.92, "recall=0.9", fontsize=8, ha="right")

ax.set_xlabel("ef (fixed-T budget)")
ax.set_ylabel("recall@10  (single seed)")
ax.set_xticks([16, 32, 64, 96, 128])
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(10, 140)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
ax.set_title("ef calibration for fixed-T HNSW (= MPC equivalent)",
             fontsize=10)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")

# also a CSV summary for paper text
print()
print("ef-for-recall>=0.9 per dataset:")
for ds in ORDER:
    if ds not in data:
        continue
    rows = sorted(data[ds])
    hit = next((ef for ef, r in rows if r >= 0.9), None)
    print(f"  {ds:25s}: ef={hit if hit else '>128'}")
