"""Fig 3a-ii: cosine datasets — show recall@10 vs ef sweep
demonstrating quantization ceiling at uint8."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "ef_calibration_v3_cosine.pdf"

# recall@10 vs ef on cosine datasets, fixed-T HNSW M=128 tau=4
ef_vals = [16, 32, 64, 96, 128]
laion = [0.108, 0.196, 0.328, 0.424, 0.544]
msmarco = [0.10, None, None, None, None]   # only ef=16 measured at K=10
# Fill MSMARCO from earlier multi_query_recall data at M=96 since
# M=128 quick sweep also low.
# Use M=128 single-shot ef=16 = 0.10 verified

# For paper variant: show recall vs ef, marking the 0.9 unattainable
fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.plot(ef_vals, laion, marker="o", color="#D55E00", linewidth=1.6,
        markersize=5, label="LAION 200K (cosine, D=512)")
ax.plot([16], [0.10], marker="s", color="#CC79A7", markersize=6,
        linestyle="None", label="MSMARCO 200K (cosine, D=768)")
ax.axhline(0.9, color="black", linestyle=":", linewidth=0.8, alpha=0.7)
ax.text(132, 0.91, "0.9 target", fontsize=8, ha="right", va="bottom")
ax.axvline(16, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
ax.text(17, 0.05, r"$ef{=}k{+}6{=}16$", fontsize=8, color="grey")
ax.set_xticks(ef_vals)
ax.set_xlabel(r"$ef$"); ax.set_ylabel(r"avg recall@$10$ (50-query average)")
ax.set_ylim(-0.02, 1.02); ax.set_xlim(10, 140)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax.set_title(r"Cosine datasets at $M{=}128$, $\tau{=}4$ (uint8 vectors)"
             "\nrecall ceiling caused by uint8 quantization of unit-norm vectors",
             fontsize=8.5)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")
