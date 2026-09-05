"""Schematic for Section 8 (paper figure): the language-model battery from axis file to
depth profile. The profile panel is a cartoon shaped like the measured Pythia profiles
(content positive at the embedding and diluting; construction negative at the embedding
and rising past zero near a quarter depth); no data are read. Written S76 (2026-09-04).
Output: figures_canonical/fig_llm_pipeline.png (drawn at the 6.0 in text width).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

ORANGE, BLUE, GRAY, DGRAY, INK = "#e07b39", "#2166ac", "#9a9a9a", "#555555", "#222222"
CLASS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#9c755f"]
plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(6.0, 2.5), dpi=300)
P = {"p1": [0.010, 0.10, 0.185, 0.72], "p2": [0.225, 0.10, 0.155, 0.72], "p3": [0.410, 0.10, 0.205, 0.72], "p4": [0.700, 0.21, 0.285, 0.61]}
ax1 = fig.add_axes(P["p1"]); ax2 = fig.add_axes(P["p2"]); ax3 = fig.add_axes(P["p3"]); ax4 = fig.add_axes(P["p4"])
for ax in (ax1, ax2, ax3):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
def title(key, n, text):
    x0 = P[key][0]
    fig.text(x0, 0.905, f"{n}", fontsize=7.5, fontweight="bold", color=INK, va="bottom")
    fig.text(x0 + 0.022, 0.905, text, fontsize=6.3, color=INK, va="bottom")
title("p1", "1", "an axis file"); title("p2", "2", "last-token states"); title("p3", "3", "per layer, the same instrument"); title("p4", "4", "the depth profile")

ax1.add_patch(FancyBboxPatch((0.03, 0.06), 0.94, 0.90, boxstyle="round,pad=0.01", fc="#f7f7f7", ec=DGRAY, lw=0.6))
lines = ['{"geography": [', '  "The capital of ...",', '  "The river that ...",', '  ...16 prompts],', ' "law": [', '  "A contract is ...",', '  ...],', ' ... 8 classes}']
for i, t in enumerate(lines):
    ax1.text(0.07, 0.90 - i * 0.092, t, fontsize=4.9, family="DejaVu Sans Mono", color=INK, va="top")
ax1.text(0.50, 0.17, "+ strata.json: one nuisance\nlabel per prompt (optional)", fontsize=4.8, color=DGRAY, ha="center", va="top")
ax1.text(0.50, -0.02, "rung axis: one from any dataset", fontsize=4.6, color=DGRAY, ha="center", va="top")

nL = 7
for l in range(nL):
    y = 0.16 + l * 0.105
    ax2.add_patch(Rectangle((0.06, y), 0.50, 0.08, fc="#dddddd", ec=DGRAY, lw=0.5))
    ax2.text(0.31, y + 0.04, "embedding" if l == 0 else (f"layer {l}" if l < nL - 1 else "layer L"), fontsize=4.8, ha="center", va="center", color=INK)
    ax2.annotate("", xy=(0.80, y + 0.04), xytext=(0.58, y + 0.04), arrowprops=dict(arrowstyle="-|>", color=DGRAY, lw=0.6, mutation_scale=5))
    ax2.add_patch(Rectangle((0.82, y + 0.005), 0.15, 0.07, fc=CLASS[l % 8], ec="none", alpha=0.85))
ax2.annotate("", xy=(0.31, 0.15), xytext=(0.31, 0.03), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=6))
ax2.text(0.31, 0.01, "one prompt in", fontsize=5.0, ha="center", va="top", color=INK)
ax2.text(0.895, 0.01, "last-token\nstate", fontsize=4.8, ha="center", va="top", color=DGRAY)

rng = np.random.default_rng(1)
for i in range(16):
    for j in range(7):
        ax3.add_patch(Rectangle((0.30 + j * 0.052, 0.84 - i * 0.038), 0.047, 0.033, color=str(0.5 + 0.45 * rng.random()), lw=0))
    ax3.add_patch(Rectangle((0.19, 0.84 - i * 0.038), 0.08, 0.033, color=CLASS[i // 2], lw=0))
ax3.text(0.49, 0.90, "prompts × hidden units, layer ℓ", fontsize=5.0, ha="center", va="bottom", color=INK)
ax3.text(0.12, 0.55, "class", rotation=90, fontsize=5.0, ha="center", va="center", color=DGRAY)
ax3.annotate("", xy=(0.49, 0.14), xytext=(0.49, 0.22), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=6))
ax3.text(0.49, 0.12, "zoom(X, labels, strata)", fontsize=5.0, ha="center", va="top", color=INK, family="DejaVu Sans Mono")
ax3.text(0.49, 0.01, "δ(ℓ), p  ·  δ̄(ℓ), p  ·  strata p", fontsize=5.0, ha="center", va="top", color=INK)

ax4.set_xlim(0, 1); ax4.set_ylim(-0.16, 0.17)
ax4.spines[["top", "right"]].set_visible(False); ax4.tick_params(labelsize=5.2, length=2, pad=1)
ax4.set_xticks([0, 0.5, 1.0]); ax4.set_xticklabels(["0", "0.5", "1"]); ax4.set_yticks([-0.1, 0, 0.1])
ax4.set_xlabel("depth ℓ/L", fontsize=5.5, labelpad=1); ax4.set_ylabel("δ", fontsize=6.5, labelpad=0)
x = np.linspace(0, 1, 200)
ax4.fill_between(x, -0.02, 0.02, color="#e5e5e5", lw=0)
content = 0.12 * np.exp(-2.2 * x) + 0.03 * (1 - np.exp(-2.2 * x))
structure = -0.08 + 0.15 * (1 - np.exp(-4 * x)) - 0.05 * np.exp(-((x - 0.06) / 0.05) ** 2)
ax4.plot(x, content, color=ORANGE, lw=1.4); ax4.plot(x, structure, color=BLUE, lw=1.4)
ax4.plot(x, 0.047 + 0 * x, color=ORANGE, lw=0.8, ls="--"); ax4.plot(x, 0.053 + 0 * x, color=BLUE, lw=0.8, ls="--")
ax4.text(0.97, 0.135, "content: inherited, dilutes", color=ORANGE, fontsize=5.0, ha="right", va="center")
ax4.text(0.97, -0.05, "construction: built with depth", color=BLUE, fontsize=5.0, ha="right", va="center")
ax4.text(0.97, 0.078, "dashed: order-averaged, flat", color=DGRAY, fontsize=4.6, ha="right", va="center")
ax4.text(0.97, 0.0, "permutation band", color=DGRAY, fontsize=4.6, ha="right", va="center")
ax4.text(0.03, -0.135, "embedding = lexical control", color=INK, fontsize=4.6, ha="left", va="center")

for key in ("p1", "p2", "p3"):
    x0, _, w, _ = P[key]
    fig.add_artist(FancyArrowPatch((x0 + w + 0.004, 0.46), (x0 + w + 0.026, 0.46), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=6, color=DGRAY, lw=0.8))
out = Path(__file__).resolve().parent.parent / "figures_canonical" / "fig_llm_pipeline.png"
fig.savefig(out, dpi=300); print("wrote", out)
