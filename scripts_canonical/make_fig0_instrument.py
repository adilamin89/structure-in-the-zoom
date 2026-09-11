"""Schematic of the instrument (paper Figure 1): declare an axis -> two ladders at
matched sizes -> a slope per ladder, delta = theta_obs - theta_floor -> two nulls
(label permutation; permutation within nuisance strata) -> the reading.

Draws no data: every element is a drawn shape. Written for the arXiv v1 entry-point
pass (S76, 2026-09-04); redrawn at 7 pt type in one row, 6.0 x 3.15 in (Session G
follow-up, 2026-09-09). Output: figures_canonical/fig_instrument.png and
../arxiv/figures/fig_instrument.png (shared by every version of the paper).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
CLASS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#9c755f"]
RED, GRAY, DGRAY, INK = "#b2182b", "#9a9a9a", "#555555", "#222222"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
FS = 7.0          # every label
FT = 7.4          # panel titles

fig = plt.figure(figsize=(6.0, 3.15), dpi=300)   # width = the jmlr text width, fonts print 1:1
W = [0.155, 0.195, 0.150, 0.210, 0.235]          # panel widths (fraction of figure)
G = 0.011                                        # gap for arrows
x0 = 0.008
axes = []
for w in W:
    ax = fig.add_axes([x0, 0.07, w, 0.76]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); axes.append(ax)
    x0 += w + G

def title(ax, n, text):
    ax.text(0.0, 1.07, f"{n}", fontsize=8, fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0.11, 1.07, text, fontsize=FT, color=INK, ha="left", va="bottom")

# ---------- 1  declare an axis ----------
ax = axes[0]; title(ax, "1", "declare an axis")
rng = np.random.default_rng(0)
n_rows, n_cols = 24, 8
for i in range(n_rows):
    for j in range(n_cols):
        ax.add_patch(Rectangle((0.34 + j * 0.078, 0.94 - i * 0.034), 0.070, 0.030, color=str(0.45 + 0.5 * rng.random()), lw=0))
for i in range(n_rows):
    ax.add_patch(Rectangle((0.20, 0.94 - i * 0.034), 0.10, 0.030, color=CLASS[i // 3], lw=0))
ax.text(0.25, 0.985, "label", ha="center", va="bottom", fontsize=FS, color=DGRAY)
ax.text(0.65, 0.985, "features", ha="center", va="bottom", fontsize=FS, color=DGRAY)
ax.text(0.09, 0.55, "samples: trials, prompts,\nconfigurations", rotation=90, ha="center", va="center", fontsize=FS, color=DGRAY, linespacing=1.0)
ax.text(0.58, 0.075, "one class\nper sample", ha="center", va="top", fontsize=FS, color=INK, linespacing=1.0)

# ---------- 2  two ladders ----------
ax = axes[1]; title(ax, "2", "matched ladders")
rungs = [1, 2, 3, 4, 6, 8]
bx, bw, bh = 0.05, 0.13, 0.075
for k, r in enumerate(rungs):
    x = bx + k * 0.155
    for c in range(r):
        ax.add_patch(Rectangle((x, 0.60 + c * bh * 0.42), bw, bh * 0.40, color=CLASS[c], lw=0))
    cols = rng.choice(8, size=r, replace=True)
    for c in range(r):
        ax.add_patch(Rectangle((x, 0.12 + c * bh * 0.42), bw, bh * 0.40, color=CLASS[cols[c]], lw=0, alpha=0.45))
    ax.text(x + bw / 2, 0.085, f"{r}", ha="center", va="top", fontsize=FS, color=DGRAY)
ax.text(0.02, 0.875, "declared: classes\nadded in order", ha="left", va="bottom", fontsize=FS, color=RED, linespacing=1.0)
ax.text(0.02, 0.40, "floor: random\nsame-size subsets", ha="left", va="bottom", fontsize=FS, color=DGRAY, linespacing=1.0)
ax.text(0.50, -0.01, "classes per rung", ha="center", va="top", fontsize=FS, color=DGRAY)

# ---------- 3  a slope per ladder ----------
ax = axes[2]; title(ax, "3", "a slope each")
ax.add_patch(Rectangle((0.19, 0.10), 0.78, 0.78, fill=False, ec=DGRAY, lw=0.5))
xs = np.linspace(0.24, 0.92, 6)
floor = 0.50 + 0.30 * (xs - 0.24) / 0.68
obs = 0.20 + 0.64 * (xs - 0.24) / 0.68
ax.plot(xs, floor, "-o", color=GRAY, ms=2.4, lw=1.1)
ax.plot(xs, obs, "-o", color=RED, ms=2.4, lw=1.2)
ax.text(0.58, 0.04, "log n", ha="center", va="top", fontsize=FS, color=DGRAY)
ax.text(0.10, 0.49, "log PR", ha="center", va="center", rotation=90, fontsize=FS, color=DGRAY)
ax.text(0.22, 0.70, r"$\theta_{\rm floor}$", color=DGRAY, fontsize=8, ha="left", va="bottom")
ax.text(0.63, 0.34, r"$\theta_{\rm obs}$", color=RED, fontsize=8, ha="left", va="top")
ax.text(0.56, 0.905, r"$\delta=\theta_{\rm obs}-\theta_{\rm floor}$", color=INK, fontsize=7.2, ha="center", va="bottom")

# ---------- 4  two nulls ----------
ax = axes[3]; title(ax, "4", "two nulls")
def stripes(y, order, x=0.05, w=0.100, h=0.10):
    for k, c in enumerate(order):
        ax.add_patch(Rectangle((x + k * (w + 0.008), y), w, h, color=CLASS[c], lw=0))
stripes(0.80, list(range(8)))
ax.text(0.05, 0.915, "declared labels", fontsize=FS, color=INK, va="bottom")
perm = rng.permutation(8); stripes(0.52, perm)
ax.text(0.05, 0.635, "null 1\npermute all labels", fontsize=FS, color=INK, va="bottom", linespacing=1.0)
strata_perm = []
for a in range(0, 8, 2):
    pair = [a, a + 1]; rng.shuffle(pair); strata_perm += pair
stripes(0.20, strata_perm)
for a in range(4):
    ax.add_patch(Rectangle((0.05 + a * 2 * 0.108 - 0.006, 0.18), 2 * 0.108 - 0.004, 0.14, fill=False, ec=DGRAY, lw=0.6, ls=(0, (2, 1.5))))
ax.text(0.05, 0.345, "null 2\npermute within strata", fontsize=FS, color=INK, va="bottom", linespacing=1.0)
ax.text(0.05, 0.135, "strata: carriers,\ntopics, sessions", fontsize=FS, color=DGRAY, va="top", linespacing=1.0)

# ---------- 5  the reading ----------
ax = axes[4]; title(ax, "5", r"read $\delta$ against both")
ax.plot([0.08, 0.92], [0.16, 0.16], color=DGRAY, lw=0.6)
ax.text(0.50, 0.11, "0", ha="center", va="top", fontsize=FS, color=DGRAY)
ax.text(0.92, 0.11, r"$\delta$", ha="center", va="top", fontsize=7.5, color=INK)
ax.add_patch(Rectangle((0.38, 0.22), 0.24, 0.09, color="#d9d9d9", lw=0))   # null 1 band (narrow)
ax.add_patch(Rectangle((0.24, 0.34), 0.52, 0.09, color="#b3b3b3", lw=0))   # null 2 band (wider)
ax.text(0.50, 0.265, "null 1 band", fontsize=6.5, color=INK, va="center", ha="center")
ax.text(0.50, 0.385, "null 2 band", fontsize=6.5, color="white", va="center", ha="center")
for xm, col in ((0.88, RED), (0.70, DGRAY), (0.50, GRAY)):
    ax.plot([xm], [0.50], marker="v", color=col, ms=4.5, ls="none")
    ax.plot([xm, xm], [0.19, 0.46], color=col, lw=0.6, ls=":")
readings = [(RED, 0.965, "outside both: the label\norganizes covariance\naccumulation"),
            (DGRAY, 0.805, "inside null 2 only:\ncomposition the\nlabeling preserves"),
            (GRAY, 0.645, "at zero: the floor;\nthe probe is blind\nto this axis")]
for col, y, txt in readings:
    ax.plot([0.05], [y], marker="v", color=col, ms=4, ls="none")
    ax.text(0.10, y, txt, fontsize=FS, color=INK, va="center", linespacing=1.0)

# arrows between panels (figure coordinates)
x0 = 0.008
for w in W[:-1]:
    xa = x0 + w + 0.001; xb = x0 + w + G - 0.001
    fig.add_artist(FancyArrowPatch((xa, 0.45), (xb, 0.45), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=6, color=DGRAY, lw=0.8))
    x0 += w + G

for out in (HERE.parent / "figures_canonical" / "fig_instrument.png", HERE.parent.parent / "arxiv" / "figures" / "fig_instrument.png"):
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
