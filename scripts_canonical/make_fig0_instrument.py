"""Schematic of the instrument (paper Figure 1): declare an axis -> two ladders at
matched sizes -> a slope per ladder, delta = theta_obs - theta_floor -> two nulls
(label permutation; permutation within nuisance strata) -> the reading.

Draws no data: every element is a drawn shape. Written for the arXiv v1 entry-point
pass (S76, 2026-09-04). Output: figures_canonical/fig_instrument.png (also copied to
the paper's figures/ directory by hand).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

CLASS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#9c755f"]
RED, GRAY, DGRAY, INK = "#b2182b", "#9a9a9a", "#555555", "#222222"
plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(6.0, 2.35), dpi=300)   # width = the jmlr text width, fonts print 1:1
W = [0.135, 0.215, 0.165, 0.205, 0.215]        # panel widths (fraction of figure)
G = 0.014                                      # gap for arrows
x0 = 0.008
axes = []
for w in W:
    ax = fig.add_axes([x0, 0.08, w, 0.74]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); axes.append(ax)
    x0 += w + G

def title(ax, n, text):
    ax.text(0.0, 1.08, f"{n}", fontsize=7.5, fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0.13, 1.08, text, fontsize=6.3, color=INK, ha="left", va="bottom")

# ---------- 1  declare an axis ----------
ax = axes[0]; title(ax, "1", "declare an axis")
rng = np.random.default_rng(0)
n_rows, n_cols = 24, 8
for i in range(n_rows):
    for j in range(n_cols):
        ax.add_patch(Rectangle((0.32 + j * 0.075, 0.93 - i * 0.034), 0.068, 0.030, color=str(0.45 + 0.5 * rng.random()), lw=0))
for i in range(n_rows):
    ax.add_patch(Rectangle((0.19, 0.93 - i * 0.034), 0.09, 0.030, color=CLASS[i // 3], lw=0))
ax.text(0.235, 0.985, "label", ha="center", va="bottom", fontsize=5.5, color=DGRAY)
ax.text(0.62, 0.985, "features", ha="center", va="bottom", fontsize=5.5, color=DGRAY)
ax.text(0.07, 0.53, "samples: trials, prompts, configurations", rotation=90, ha="center", va="center", fontsize=5.0, color=DGRAY)
ax.text(0.50, 0.06, "one class per sample", ha="center", va="top", fontsize=5.5, color=INK)

# ---------- 2  two ladders ----------
ax = axes[1]; title(ax, "2", "two ladders, same sizes")
rungs = [1, 2, 3, 4, 6, 8]
bx, bw, bh = 0.05, 0.13, 0.075
for k, r in enumerate(rungs):
    x = bx + k * 0.155
    for c in range(r):
        ax.add_patch(Rectangle((x, 0.60 + c * bh * 0.42), bw, bh * 0.40, color=CLASS[c], lw=0))
    cols = rng.choice(8, size=r, replace=True)
    for c in range(r):
        ax.add_patch(Rectangle((x, 0.12 + c * bh * 0.42), bw, bh * 0.40, color=CLASS[cols[c]], lw=0, alpha=0.45))
    ax.text(x + bw / 2, 0.075, f"{r}", ha="center", va="top", fontsize=5.5, color=DGRAY)
ax.text(0.02, 0.955, "declared: classes added in order", ha="left", va="bottom", fontsize=5.4, color=RED)
ax.text(0.02, 0.485, "floor: random same-size subsets", ha="left", va="bottom", fontsize=5.1, color=DGRAY)
ax.text(0.50, -0.02, "classes per rung", ha="center", va="top", fontsize=5.5, color=DGRAY)

# ---------- 3  slope per ladder ----------
ax = axes[2]; title(ax, "3", "one slope per ladder")
ax.add_patch(Rectangle((0.17, 0.10), 0.80, 0.80, fill=False, ec=DGRAY, lw=0.5))
xs = np.linspace(0.22, 0.92, 6)
floor = 0.52 + 0.30 * (xs - 0.22) / 0.70
obs = 0.22 + 0.64 * (xs - 0.22) / 0.70
ax.plot(xs, floor, "-o", color=GRAY, ms=2.2, lw=1.1)
ax.plot(xs, obs, "-o", color=RED, ms=2.2, lw=1.2)
ax.text(0.57, 0.03, "log n", ha="center", va="top", fontsize=5.8, color=DGRAY)
ax.text(0.09, 0.50, "log PR", ha="center", va="center", rotation=90, fontsize=5.8, color=DGRAY)
ax.text(0.24, 0.62, r"$\theta_{\rm floor}$", color=DGRAY, fontsize=7, ha="left", va="bottom")
ax.text(0.62, 0.36, r"$\theta_{\rm obs}$", color=RED, fontsize=7, ha="left", va="top")
ax.text(0.57, 0.93, r"$\delta=\theta_{\rm obs}-\theta_{\rm floor}$", color=INK, fontsize=6.8, ha="center", va="bottom")

# ---------- 4  two nulls ----------
ax = axes[3]; title(ax, "4", "two nulls")
def stripes(y, order, x=0.05, w=0.098, h=0.11, alpha=1.0):
    for k, c in enumerate(order):
        ax.add_patch(Rectangle((x + k * (w + 0.008), y), w, h, color=CLASS[c], lw=0, alpha=alpha))
stripes(0.80, list(range(8)))
ax.text(0.05, 0.925, "declared labels", fontsize=5.6, color=INK, va="bottom")
perm = rng.permutation(8); stripes(0.53, perm)
ax.text(0.05, 0.655, "null 1: permute all labels", fontsize=5.6, color=INK, va="bottom")
strata_perm = []
for a in range(0, 8, 2):
    pair = [a, a + 1]; rng.shuffle(pair); strata_perm += pair
stripes(0.20, strata_perm)
for a in range(4):
    ax.add_patch(Rectangle((0.05 + a * 2 * 0.106 - 0.006, 0.18), 2 * 0.106 - 0.004, 0.15, fill=False, ec=DGRAY, lw=0.6, ls=(0, (2, 1.5))))
ax.text(0.05, 0.36, "null 2: permute within strata", fontsize=5.6, color=INK, va="bottom")
ax.text(0.05, 0.10, "strata: carriers, topics, sessions", fontsize=5.0, color=DGRAY, va="top")

# ---------- 5  the reading ----------
ax = axes[4]; title(ax, "5", r"read $\delta$ against both")
ax.plot([0.08, 0.92], [0.22, 0.22], color=DGRAY, lw=0.6)
ax.text(0.50, 0.17, "0", ha="center", va="top", fontsize=5.5, color=DGRAY)
ax.text(0.92, 0.17, r"$\delta$", ha="center", va="top", fontsize=6.5, color=INK)
ax.add_patch(Rectangle((0.38, 0.28), 0.24, 0.09, color="#d9d9d9", lw=0))   # null 1 band (narrow)
ax.add_patch(Rectangle((0.24, 0.40), 0.52, 0.09, color="#b3b3b3", lw=0))   # null 2 band (wider)
ax.text(0.50, 0.325, "null 1 band", fontsize=5.0, color=INK, va="center", ha="center")
ax.text(0.50, 0.445, "null 2 band", fontsize=5.0, color="white", va="center", ha="center")
ax.plot([0.88], [0.56], marker="v", color=RED, ms=4.5, ls="none")
ax.plot([0.70], [0.56], marker="v", color=DGRAY, ms=4.5, ls="none")
ax.plot([0.50], [0.56], marker="v", color=GRAY, ms=4.5, ls="none")
ax.plot([0.88, 0.88], [0.25, 0.52], color=RED, lw=0.6, ls=":")
ax.plot([0.70, 0.70], [0.25, 0.52], color=DGRAY, lw=0.6, ls=":")
ax.plot([0.50, 0.50], [0.25, 0.52], color=GRAY, lw=0.6, ls=":")
ax.plot([0.06], [0.955], marker="v", color=RED, ms=4, ls="none")
ax.text(0.12, 0.955, "outside both: the label organizes\ncovariance accumulation", fontsize=5.0, color=INK, va="center")
ax.plot([0.06], [0.80], marker="v", color=DGRAY, ms=4, ls="none")
ax.text(0.12, 0.80, "inside null 2 only: composition\nthe labeling preserves", fontsize=5.0, color=INK, va="center")
ax.plot([0.06], [0.655], marker="v", color=GRAY, ms=4, ls="none")
ax.text(0.12, 0.655, "at zero: floor; the probe is blind\nto this axis", fontsize=5.0, color=INK, va="center")

# arrows between panels (figure coordinates)
x0 = 0.008
for w in W[:-1]:
    xa = x0 + w + 0.001; xb = x0 + w + G - 0.001
    fig.add_artist(FancyArrowPatch((xa, 0.45), (xb, 0.45), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=6, color=DGRAY, lw=0.8))
    x0 += w + G

fig.savefig("figures_canonical/fig_instrument.png", dpi=300)
print("wrote figures_canonical/fig_instrument.png")
