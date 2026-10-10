"""
CVIP 2026 — Figure 3: Per-class mAP50 grouped bar chart.
Data from Table 4 (per-class mAP50, mean over 3 seeds).
Run: python draw_barchart_cvip.py   ->  perclass_map.png
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "font.family": "DejaVu Sans",
    "axes.linewidth": 1.1, "pdf.fonttype": 42,
})

# ── data (Table 4) ────────────────────────────────────────────────────────────
classes = ["Benign\nMass", "Malignant\nMass", "Benign\nCalc", "Malignant\nCalc"]
data = {
    "Universal": [0.0664, 0.1236, 0.0144, 0.0237],
    "DS-MoE":    [0.0843, 0.1421, 0.0099, 0.0234],
    "Gated MoE": [0.0802, 0.1133, 0.0082, 0.0338],
}
COLORS = {"Universal": "#2B6CB0", "DS-MoE": "#C05621", "Gated MoE": "#276749"}

x = np.arange(len(classes))
w = 0.26
fig, ax = plt.subplots(figsize=(7, 4.6))

for i, (name, vals) in enumerate(data.items()):
    off = (i - 1) * w
    bars = ax.bar(x + off, vals, w, label=name, color=COLORS[name],
                  edgecolor="white", linewidth=0.8, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.0015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.2, color="#2D3748")

# shaded region behind the two mass groups to highlight where DS-MoE wins
ax.axvspan(-0.5, 1.5, color="#2B6CB0", alpha=0.05, zorder=0)
ax.text(0.5, 0.153, "DS-MoE wins (mass)", ha="center", fontsize=9,
        style="italic", color="#2C5282")

ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10.5)
ax.set_ylabel("mAP@50", fontsize=12.5)
ax.set_ylim(0, 0.165)
ax.set_yticks(np.arange(0, 0.161, 0.04))
ax.grid(True, axis="y", alpha=0.25, linewidth=0.8, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10.5, loc="upper right", framealpha=0.96,
          edgecolor="#CBD5E0", ncol=1)

plt.tight_layout()
fig.savefig("perclass_map.pdf", bbox_inches="tight")
fig.savefig("perclass_map.png", dpi=300, bbox_inches="tight")
print("Saved: perclass_map.png / .pdf")
