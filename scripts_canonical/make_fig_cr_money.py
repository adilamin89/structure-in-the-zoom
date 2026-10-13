"""Camera-ready Figure 1: one population, many probes.
(A) drawn schematic: the same trials-by-neurons matrix subsampled three ways (random stimuli; direction classes
    accumulated around the circle; spatially contiguous neurons), with the three GT3 exponents and what each reads;
(B) GT3 ladders: random subsets (the empirical floor) and the direction-aligned ladder, Eq. 2 dashed as the iid
    Gaussian reference (orientation_zoom.json, as in make_fig1_2_ladder.py);
(C) Eq. 1 read off: delta against theta_floor for every ladder whose two slopes are stored; on the zero line an
    exponent is all floor. V1 random axis, 18 recordings
    (stringer_theta_decomposition.json); V1 spatial neighbours, 10 recordings (neuron_zoom_spatial.json); V1 direction
    axis, 8 grating recordings (orientation_zoom.json); a language model's topic axis, 13 layers of Pythia-160m
    (topic_class_ladder.json).
Width 6.0 in = the PMLR text width, so fonts print 1:1.
Out: ../figures_canonical/fig_cr_money.png (+ ../../neurreps_camera_ready/figures/ when present).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import ScalarFormatter, NullFormatter

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "savefig.dpi": 300,
                     "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent / "data"
OUTS = [HERE.parent / "figures_canonical" / "fig_cr_money.png",
        HERE.parent.parent / "neurreps_camera_ready" / "figures" / "fig_cr_money.png"]
RED, GRAY, INK, PURPLE, BLUEG = "#b40426", "#8a8a8a", "#222222", "#6a3d9a", "#4d6b8a"
CLASS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#9c755f"]


def load(name):
    p = DATA / name
    return json.load(open(p if p.exists() else RAW / name))


fig = plt.figure(figsize=(6.0, 2.3))
axA = fig.add_axes([0.0, 0.0, 0.335, 0.93])
axB = fig.add_axes([0.425, 0.17, 0.235, 0.70])
axC = fig.add_axes([0.755, 0.17, 0.235, 0.70])

# ---------------- (A) schematic ----------------
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
fig.text(0.004, 0.955, "A", fontsize=8, fontweight="bold", va="center")
fig.text(0.030, 0.955, "one recording, three probes", fontsize=7.5, va="center")
rng = np.random.default_rng(3)


def matrix(x0, y0, w, h, nrow=10, ncol=9, row_colors=None, row_edge=None, col_hi=None):
    cw, ch = w / ncol, h / nrow
    for i in range(nrow):
        for j in range(ncol):
            g = 0.55 + 0.4 * rng.random()
            axA.add_patch(Rectangle((x0 + j * cw, y0 + (nrow - 1 - i) * ch), cw * 0.9, ch * 0.9, color=(g, g, g), lw=0))
    if row_colors is not None:
        for i, c in enumerate(row_colors):
            axA.add_patch(Rectangle((x0 - 0.24, y0 + (nrow - 1 - i) * ch), 0.17, ch * 0.9, color=c, lw=0))
    if row_edge is not None:
        for i in row_edge:
            axA.add_patch(Rectangle((x0 - 0.04, y0 + (nrow - 1 - i) * ch - 0.02), w, ch * 0.9 + 0.04,
                                    fill=False, ec=INK, lw=0.7))
    if col_hi is not None:
        j0, j1 = col_hi
        axA.add_patch(Rectangle((x0 + j0 * cw - 0.03, y0 - 0.05), (j1 - j0) * cw - 0.1 * cw + 0.06, h + 0.1,
                                fill=False, ec=BLUEG, lw=1.0))


rows = [("random stimulus subsets", "0.25", "floor", GRAY),
        ("direction classes in order", "0.30", r"floor $+\ \delta = +0.24$", RED),
        ("neighbouring neurons", "0.35", "floor again", BLUEG)]
for k, ((lab, expo, reading, col), yt) in enumerate(zip(rows, [8.95, 5.95, 2.95])):
    y0 = yt - 2.25
    if k == 0:
        matrix(0.55, y0, 2.3, 2.05, row_edge=[1, 4, 6, 8])
    elif k == 1:
        matrix(0.55, y0, 2.3, 2.05, row_colors=[CLASS[c] for c in [0, 0, 1, 1, 2, 3, 4, 5, 6, 7]])
        cx, cy, r = 3.85, y0 + 1.02, 0.62
        axA.add_patch(Circle((cx, cy), r, fill=False, ec=GRAY, lw=0.6))
        for c in range(8):
            a = c * np.pi / 4
            axA.plot([cx + r * np.cos(a)], [cy + r * np.sin(a)], "o", color=CLASS[c], ms=2.8)
        for c in range(3):
            a0, a1 = c * np.pi / 4, (c + 1) * np.pi / 4
            axA.annotate("", xy=(cx + 0.55 * r * np.cos(a1), cy + 0.55 * r * np.sin(a1)),
                         xytext=(cx + 0.55 * r * np.cos(a0), cy + 0.55 * r * np.sin(a0)),
                         arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.5, mutation_scale=4.5,
                                         connectionstyle="arc3,rad=-0.3"))
    else:
        matrix(0.55, y0, 2.3, 2.05, col_hi=(2, 6))
        cx, cy = 3.85, y0 + 1.02
        pts = rng.random((38, 2)) * 1.3 - 0.65
        near = np.linalg.norm(pts - np.array([0.1, -0.05]), axis=1) < 0.36
        axA.plot(cx + pts[~near, 0], cy + pts[~near, 1], ".", color=GRAY, ms=2.0)
        axA.plot(cx + pts[near, 0], cy + pts[near, 1], ".", color=BLUEG, ms=2.4)
    axA.text(4.75, y0 + 1.72, lab, fontsize=6.0, va="center", color=INK)
    axA.text(4.75, y0 + 1.02, r"$\theta_{\rm obs} = " + expo + "$", fontsize=7.2, va="center", color=INK)
    axA.text(4.75, y0 + 0.36, "reads: " + reading, fontsize=6.4, va="center", color=col)
axA.text(0.38, 8.95, "trials", fontsize=6.0, color=GRAY, rotation=90, va="top", ha="center")
axA.text(1.70, 9.0, "neurons", fontsize=6.0, color=GRAY, ha="center", va="bottom")

# ---------------- (B) GT3 ladders ----------------
oz = load("orientation_zoom.json")
gt3 = next(r for r in oz["results"] if "drifting_GT3" in r["name"])
s_sizes = [r[1] for r in gt3["struct_ladder"]]; s_pr = [r[2] for r in gt3["struct_ladder"]]
r_sizes = [r[1] for r in gt3["rand_ladder"]]; r_pr = [r[2] for r in gt3["rand_ladder"]]
n_top, pr_top = r_sizes[-1], r_pr[-1]
d_eff = pr_top * n_top / (n_top - 1 - pr_top)                    # Eq. 2 through the top rung
n_grid = np.logspace(np.log10(min(s_sizes)), np.log10(max(s_sizes)), 200)
ax = axB
ax.loglog(n_grid, d_eff * (n_grid - 1) / (d_eff + n_grid), ls="--", color="0.6", lw=0.9, label="Eq. 2, iid reference")
ax.loglog(r_sizes, r_pr, "o-", color=GRAY, lw=1.3, ms=3.4, label="random subsets: floor")
ax.loglog(s_sizes, s_pr, "o-", color=RED, lw=1.5, ms=3.4, label="direction ladder")
ax.set_ylim(24, 78); ax.set_xticks([500, 1000, 2000, 4000]); ax.set_yticks([25, 30, 40, 50, 60])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
ax.annotate("", xy=(s_sizes[1], s_pr[1]), xytext=(r_sizes[1] if r_sizes[1] == s_sizes[1] else s_sizes[1],
            np.interp(np.log(s_sizes[1]), np.log(r_sizes), r_pr)),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6, mutation_scale=5))
ax.text(s_sizes[1] * 0.93, 37, "early\ndeficit", fontsize=6.3, color=INK, va="center", ha="right", linespacing=0.95)
ax.text(560, 25.6, r"$\delta_{\rm dir} = +0.24$", fontsize=7.5, color=RED)
ax.set_xlabel("stimuli in rung"); ax.set_ylabel("participation ratio")
ax.set_title("B  GT3, 11,311 neurons", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1)
ax.spines[["top", "right"]].set_visible(False)

# ---------------- (C) Eq. 1 read off: delta against theta_floor for every ladder with both slopes stored ----------------
ax = axC
rand = [r for r in load("stringer_theta_decomposition.json")["rows"] if r.get("status") == "ok"]
spat = load("neuron_zoom_spatial.json")["rows"]
dirx = [r for r in oz["results"] if r["name"].startswith("gratings_")]
llm = load("topic_class_ladder.json")["per_layer"]
ax.axhline(0, color=INK, lw=0.8, zorder=1)
ax.text(0.98, 0.012, "floor only", fontsize=6.0, color=INK, ha="right", va="bottom")
ax.plot([r["theta_floor_empirical"] for r in rand], [r["shift_obs"] for r in rand], "o", ms=3.0, mfc="white",
        mec=GRAY, mew=0.9, zorder=3)
ax.plot([r["theta_floor"] for r in spat], [r["delta_spatial"] for r in spat], "s", ms=2.8, mfc="white",
        mec=BLUEG, mew=0.9, zorder=3)
ax.plot([r["theta_random"] for r in dirx], [r["delta"] for r in dirx], "o", ms=3.4, color=RED, zorder=4)
ax.plot([r["theta_floor"] for r in llm], [r["delta_topic"] for r in llm], "D", ms=2.8, mfc="white", mec=PURPLE, mew=0.9, zorder=4)
ax.text(0.17, 0.33, "V1, stimulus\ndirection (8)", fontsize=6.3, color=RED, va="center", linespacing=0.95)
ax.text(0.55, -0.125, "V1, random stimuli (18)", fontsize=6.3, color=GRAY, va="center", ha="center")
ax.text(0.55, -0.18, "and neighbouring neurons (10)", fontsize=6.3, color=BLUEG, va="center", ha="center")
ax.text(0.63, 0.63, "language\nmodel, topic\n(13 layers)", fontsize=6.3, color=PURPLE, va="center", ha="right",
        linespacing=0.95)
ax.set_xlim(0, 1.0); ax.set_ylim(-0.21, 0.70)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax.set_xlabel(r"floor exponent $\theta_{\rm floor}$"); ax.set_ylabel(r"shift $\delta = \theta_{\rm obs} - \theta_{\rm floor}$")
ax.set_title("C  shift against floor", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

for out in OUTS:
    if out.parent.exists():
        fig.savefig(out, dpi=300); print("wrote", out)
print("V1 random: floor range", min(r["theta_floor_empirical"] for r in rand), max(r["theta_floor_empirical"] for r in rand))
print("V1 direction: delta range", min(r["delta"] for r in dirx), max(r["delta"] for r in dirx))
print("LLM delta range", min(r["delta_topic"] for r in llm), max(r["delta_topic"] for r in llm))
