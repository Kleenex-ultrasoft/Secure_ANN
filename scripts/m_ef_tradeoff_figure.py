"""Generate the M-ef trade-off figure for paper §4.

Shows for SIFT 1M (the hardest dataset in our benchmark):
- At M=96, ef=k+6=16 only gives recall 0.36 (FAIL paper claim)
- At M=128, ef=k+6=16 gives recall 0.954 (PASS paper claim)
- The recall curves vs ef for both M values

Establishes the M-ef trade-off and motivates the paper's M=128 default.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "m_ef_tradeoff.pdf"

# Empirical data from docs/measurements/multi_query_ef_sweep.csv
# (averaged over 50 sift_query.npy / deep_query.npy queries each)
ef_vals = [16, 32, 64, 96, 128]
sift_1m_M96  = [0.362, 0.778, 0.968, 0.982, 0.990]
sift_1m_M128 = [0.954, 0.990, 0.998, 0.998, 0.998]
deep_1m_M96  = [0.852, 0.920, 0.972, 0.982, 0.988]
deep_1m_M128 = [0.940, 0.986, 0.990, 0.994, 0.996]

fig, ax = plt.subplots(figsize=(5.0, 3.4))

ax.plot(ef_vals, sift_1m_M96, marker="o", color="#0072B2",
        linestyle="--", linewidth=1.2, markersize=4,
        label="SIFT 1M, M=96")
ax.plot(ef_vals, sift_1m_M128, marker="o", color="#0072B2",
        linewidth=1.6, markersize=5, label="SIFT 1M, M=128")
ax.plot(ef_vals, deep_1m_M96, marker="s", color="#009E73",
        linestyle="--", linewidth=1.2, markersize=4,
        label="DEEP 1M, M=96")
ax.plot(ef_vals, deep_1m_M128, marker="s", color="#009E73",
        linewidth=1.6, markersize=5, label="DEEP 1M, M=128")

ax.axhline(0.9, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
ax.text(132, 0.91, "0.9", fontsize=8, ha="right", va="bottom")

ax.axvline(16, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
ax.text(16.5, 0.05, "ef=k+6=16", fontsize=8, color="grey")

ax.set_xlabel(r"$ef$ (queue size)")
ax.set_ylabel(r"avg recall@10 (50 queries)")
ax.set_xticks(ef_vals)
ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9, ncol=2)
ax.set_title(r"$M$\textendash $ef$ trade-off at fixed-T HNSW",
             fontsize=10)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
