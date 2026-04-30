"""Redraw fig:ef-calibration-compact for paper §4 with the actual
multi-query data across 6 datasets at M=128, ef=k+6, tau=4 (T=ef+4).

Output: ef_calibration_v3.pdf (drop-in replacement candidate for
Pictures/efmin_offset_py.pdf in the Overleaf repo).

Layout: 2x3 grid, one panel per dataset. Each panel shows recall@k
across k ∈ {1, 5, 10, 20, 50, 100} with the "0.9 target" reference.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "ef_calibration_v3.pdf"

# Empirical multi-query data (50 queries from each dataset's standard
# query set, fixed-T HNSW at M=128 ef=k+6 tau=4).
# Source: ef_calibration_v3_data.csv (committed alongside).
DATA = {
    "Fashion 60K\n(L2, D=784)":    {1:1.000, 5:0.996, 10:0.998, 20:0.996, 50:0.998, 100:0.999},
    "SIFT 100K\n(L2, D=128)":      {1:1.000, 5:0.984, 10:0.982, 20:0.993, 50:0.996, 100:0.999},
    "DEEP 1M\n(L2, D=96)":         {1:0.880, 5:0.936, 10:0.940, 20:0.981, 50:0.989, 100:0.995},
    "SIFT 1M\n(L2, D=128)":        {1:0.980, 5:0.956, 10:0.954, 20:0.981, 50:0.986, 100:0.994},
    "LAION 200K\n(cosine, D=512)":  {1:0.200, 5:0.112, 10:0.108, 20:0.166, 50:0.273, 100:0.430},
    "MSMARCO 200K\n(cosine, D=768)":{1:0.080, 5:0.120, 10:0.100, 20:0.134, 50:0.230, 100:0.357},
}

ks = [1, 5, 10, 20, 50, 100]

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), sharey=True)
axes = axes.flatten()

for ax, (name, recalls) in zip(axes, DATA.items()):
    ys = [recalls[k] for k in ks]
    color = "#0072B2" if "L2" in name else "#D55E00"
    ax.plot(ks, ys, marker="o", color=color, linewidth=1.5, markersize=4)
    ax.axhline(0.9, color="black", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=7)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0, 0.5, 0.9, 1.0])
    ax.grid(True, alpha=0.25)
    ax.set_title(name, fontsize=8)
    if "L2" in name and min(ys) >= 0.9:
        ax.text(2, 0.05, "✓ paper claim holds", fontsize=7,
                color="#0072B2", style="italic")
    elif "L2" in name:
        ax.text(2, 0.05, f"(min: {min(ys):.2f})", fontsize=7,
                color="#0072B2")
    else:
        ax.text(2, 0.05, "✗ uint8 cosine quant\n  caps recall", fontsize=7,
                color="#D55E00", style="italic")

# Shared labels
for ax in axes[3:]:
    ax.set_xlabel(r"top-$k$", fontsize=8)
for ax in [axes[0], axes[3]]:
    ax.set_ylabel("avg recall@$k$", fontsize=8)

fig.suptitle(r"Fixed-$T$ HNSW recall@$k$ at $M{=}128$, $ef{=}k{+}6$, $\tau{=}4$ "
             "(50-query average)",
             fontsize=9, y=1.00)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
