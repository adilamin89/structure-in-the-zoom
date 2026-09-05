"""Figure: one population, three axes (S77). (A) a drawn schematic of the same
trials-by-neurons matrix subsampled three ways (random stimuli; stimulus classes
accumulated around the direction circle; spatially contiguous neurons) with the
three GT3 exponents and their readings; (B) the GT3 ladders (orientation_zoom.json,
as in make_fig1_2_ladder.py); (C) per-recording direction-aligned shifts with
bootstrap intervals (bootstrap_all_10_orient_fullneuron.json).
Out: ../figures_canonical/fig_three_axes.png + ../../arxiv/figures/
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import ScalarFormatter, NullFormatter
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.3, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_three_axes.png", HERE.parent.parent / "arxiv" / "figures" / "fig_three_axes.png"]
RED, GRAY, INK = "#b40426", "#8a8a8a", "#222222"
CLASS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#9c755f"]

fig = plt.figure(figsize=(6.0, 2.75))
axA = fig.add_axes([0.005, 0.02, 0.31, 0.90]); axB = fig.add_axes([0.405, 0.20, 0.245, 0.68]); axC = fig.add_axes([0.725, 0.20, 0.27, 0.68])
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
fig.text(0.005, 0.95, "A", fontsize=8.5, fontweight="bold", va="center"); fig.text(0.035, 0.95, "three ways to zoom one recording", fontsize=7.5, va="center")

rng = np.random.default_rng(3)
def matrix(ax, x0, y0, w, h, nrow=12, ncol=9, row_colors=None, row_edge=None, col_hi=None):
    cw, ch = w / ncol, h / nrow
    for i in range(nrow):
        for j in range(ncol):
            g = 0.55 + 0.4 * rng.random()
            ax.add_patch(Rectangle((x0 + j * cw, y0 + (nrow - 1 - i) * ch), cw * 0.9, ch * 0.9, color=(g, g, g), lw=0))
    if row_colors is not None:
        for i, c in enumerate(row_colors):
            if c is not None:
                ax.add_patch(Rectangle((x0 - 0.22, y0 + (nrow - 1 - i) * ch), 0.16, ch * 0.9, color=c, lw=0))
    if row_edge is not None:
        for i in row_edge:
            ax.add_patch(Rectangle((x0 - 0.04, y0 + (nrow - 1 - i) * ch - 0.02), w, ch * 0.9 + 0.04, fill=False, ec=INK, lw=0.8))
    if col_hi is not None:
        j0, j1 = col_hi
        ax.add_patch(Rectangle((x0 + j0 * cw - 0.03, y0 - 0.05), (j1 - j0) * cw - 0.1 * cw + 0.06, h + 0.1, fill=False, ec=RED, lw=1.0))

rows = [("random stimulus\nsubsets", "0.25", "floor", "gray"),
        ("stimulus classes,\nadded in order", "0.31", "structure,\n$\\delta = +0.24$", "classes"),
        ("neighboring\nneurons", "0.35", "floor again", "cols")]
y_tops = [9.3, 6.2, 3.1]
for (lab, expo, reading, kind), yt in zip(rows, y_tops):
    y0 = yt - 2.3
    if kind == "gray":
        matrix(axA, 0.7, y0, 2.6, 2.1, row_edge=[1, 4, 6, 9])
    elif kind == "classes":
        cols = [CLASS[k] for k in [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 7]]
        matrix(axA, 0.7, y0, 2.6, 2.1, row_colors=cols)
        # small direction circle with the accumulation order
        cx, cy, r = 4.6, y0 + 1.05, 0.75
        axA.add_patch(Circle((cx, cy), r, fill=False, ec=GRAY, lw=0.7))
        for k in range(8):
            a = k * np.pi / 4
            axA.plot([cx + r * np.cos(a)], [cy + r * np.sin(a)], "o", color=CLASS[k], ms=3.2)
        for k in range(3):
            a0, a1 = k * np.pi / 4, (k + 1) * np.pi / 4
            axA.annotate("", xy=(cx + 0.55 * r * np.cos(a1), cy + 0.55 * r * np.sin(a1)),
                         xytext=(cx + 0.55 * r * np.cos(a0), cy + 0.55 * r * np.sin(a0)),
                         arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6, mutation_scale=5, connectionstyle="arc3,rad=-0.3"))
    else:
        matrix(axA, 0.7, y0, 2.6, 2.1, col_hi=(2, 6))
        # small map of neuron positions with the contiguous patch
        cx, cy = 4.6, y0 + 1.05
        pts = rng.random((40, 2)) * 1.6 - 0.8
        near = np.linalg.norm(pts - np.array([0.15, -0.1]), axis=1) < 0.42
        axA.plot(cx + pts[~near, 0], cy + pts[~near, 1], ".", color=GRAY, ms=2.2)
        axA.plot(cx + pts[near, 0], cy + pts[near, 1], ".", color=RED, ms=2.6)
    axA.text(5.55, y0 + 1.75, lab, fontsize=6.2, va="center", color=INK, linespacing=0.95)
    axA.text(5.55, y0 + 1.0, "$\\theta_{\\rm obs} = " + expo + "$", fontsize=7.2, va="center", color=INK)
    axA.text(5.55, y0 + 0.30, "reads: " + reading, fontsize=6.2, va="center", color=RED if "structure" in reading else GRAY, linespacing=0.95)
axA.text(0.7, 9.65, "trials", fontsize=5.6, color=GRAY, rotation=90, va="top", ha="center")
axA.text(2.0, 9.62, "neurons", fontsize=5.6, color=GRAY, ha="center", va="bottom")

# ---- (B) GT3 ladders ----
oz = json.load(open(DATA / "orientation_zoom.json"))
gt3 = next(r for r in oz["results"] if "drifting_GT3" in r["name"])
s_sizes = [r[1] for r in gt3["struct_ladder"]]; s_pr = [r[2] for r in gt3["struct_ladder"]]
r_sizes = [r[1] for r in gt3["rand_ladder"]]; r_pr = [r[2] for r in gt3["rand_ladder"]]
d_eff = r_pr[-1]; n_grid = np.logspace(np.log10(min(s_sizes)), np.log10(max(s_sizes)), 200)
mp = (n_grid - 1) * d_eff / ((n_grid - 1) + d_eff)
ax = axB
ax.loglog(n_grid, mp, ls="--", color="0.55", lw=1.0, label="analytic crossover")
ax.loglog(r_sizes, r_pr, "o-", color=GRAY, lw=1.5, ms=4, label="random subsets (floor)")
ax.loglog(s_sizes, s_pr, "o-", color=RED, lw=1.7, ms=4, label="direction-aligned ladder")
ax.set_ylim(24, 62); ax.set_xticks([500, 1000, 2000, 4000]); ax.set_yticks([25, 30, 40, 50, 60])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
ax.text(560, 25.6, r"$\delta_{\rm dir} = +0.24$", fontsize=8.5, color=RED)
ax.set_xlabel("stimuli in rung"); ax.set_ylabel("participation ratio")
ax.set_title("B  GT3 along two axes", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left", handlelength=1.5)
ax.spines[["top", "right"]].set_visible(False)

# ---- (C) per-recording shifts ----
bs = json.load(open(DATA / "bootstrap_all_10_orient_fullneuron.json")); recs = bs["recordings"]
order = ["gratings_drifting_GT1", "gratings_drifting_GT2", "gratings_drifting_GT3", "gratings_local_GT1", "gratings_local_GT2",
         "gratings_local_GT3", "gratings_low_contrast_GT1", "gratings_low_contrast_GT2", "static_biased_TX40", "static_biased_TX42"]
labels = ["D1", "D2", "D3", "L1", "L2", "L3", "C1", "C2", "S1", "S2"]
ax = axC
for i, o in enumerate(order):
    r = next(x for x in recs if x["name"].startswith(o)); col = RED if i < 8 else GRAY
    ax.errorbar(i, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], fmt="o", color=col, ms=4.5, capsize=2.5, lw=1.2)
ax.axhline(0, color="k", lw=0.6, alpha=0.6); ax.axvspan(7.5, 9.5, color="0.93", zorder=0)
ax.text(8.5, 0.50, "static,\ndegenerate\nprobe", ha="center", fontsize=6, color="0.35")
ax.set_xticks(range(10)); ax.set_xticklabels(labels); ax.set_ylim(-0.06, 0.60)
ax.set_ylabel(r"$\delta_{\rm dir}$ (bootstrap 95% CI)"); ax.set_xlabel("recording")
ax.set_title("C  every grating recording", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
