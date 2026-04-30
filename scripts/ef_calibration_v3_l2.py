"""Fig 3a-i: L2 datasets where ef = k + 6 holds (4 datasets)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "ef_calibration_v3_l2.pdf"

DATA = {
    "Fashion 60K (D=784)": {1:1.000, 5:0.996, 10:0.998, 20:0.996, 50:0.998, 100:0.999},
    "SIFT 100K (D=128)":   {1:1.000, 5:0.984, 10:0.982, 20:0.993, 50:0.996, 100:0.999},
    "DEEP 1M (D=96)":      {1:0.880, 5:0.936, 10:0.940, 20:0.981, 50:0.989, 100:0.995},
    "SIFT 1M (D=128)":     {1:0.980, 5:0.956, 10:0.954, 20:0.981, 50:0.986, 100:0.994},
}
ks = [1, 5, 10, 20, 50, 100]
COLORS = {"Fashion 60K (D=784)": "#888888",
          "SIFT 100K (D=128)":   "#56B4E9",
          "DEEP 1M (D=96)":      "#009E73",
          "SIFT 1M (D=128)":     "#0072B2"}

fig, ax = plt.subplots(figsize=(5.0, 3.0))
for name, recalls in DATA.items():
    ys = [recalls[k] for k in ks]
    ax.plot(ks, ys, marker="o", color=COLORS[name], label=name,
            linewidth=1.6, markersize=5)
ax.axhline(0.9, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
ax.text(105, 0.91, "0.9", fontsize=8, ha="right", va="bottom")
ax.set_xscale("log")
ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel(r"top-$k$"); ax.set_ylabel(r"avg recall@$k$ (50-query average)")
ax.set_ylim(0.85, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
ax.set_title(r"L2 datasets at $M{=}128$, $ef{=}k{+}6$, $\tau{=}4$",
             fontsize=10)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
