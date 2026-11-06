"""Camera-ready Figure 1: one population, many probes.
(A) the measurement, drawn (steps 1-2 of fig_instrument.png): (1) declare an axis, one class per sample, the classes
    stimulus directions on a circle; (2) matched ladders, the declared one adding whole classes in order, the floor
    drawing random subsets of the same sizes, labels ignored;
(B) step 3 on one recording, GT3: random subsets (the empirical floor) and the direction-aligned ladder with their
    slopes, Eq. 2 dashed as the iid Gaussian reference (orientation_zoom.json, as in make_fig1_2_ladder.py);
(C) Eq. 1 read off: delta against theta_floor for every ladder whose two slopes are stored; on the zero line an
    exponent is all floor. V1 random axis, 18 recordings
    (stringer_theta_decomposition.json); V1 spatial neighbours, 10 recordings (neuron_zoom_spatial.json); V1 direction
    axis, 8 grating recordings (orientation_zoom.json); language models: the topic partition of
    the 32-prompt-per-class battery (world_knowledge32), every layer of Pythia-160m and Pythia-2.8B, the shift averaged
    over 50 random class orders (dbar50_run37) against that layer's floor (llm_battery_<model>.json).
    GT3's three readings (random stimuli, direction, neighbouring neurons) are ringed: the same neurons at three floors.
Width 6.0 in = the PMLR text width, so fonts print 1:1.
Out: ../figures_canonical/fig_probes.png (+ ../../neurreps_camera_ready/figures/ when present);
`--draft PATH` writes only PATH (a design mock-up that leaves the paper's figure untouched).
"""
import json
import sys
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
OUTS = [HERE.parent / "figures_canonical" / "fig_probes.png",
        HERE.parent.parent / "neurreps_camera_ready" / "figures" / "fig_probes.png"]
RED, GRAY, INK, PURPLE, BLUEG = "#b40426", "#8a8a8a", "#222222", "#6a3d9a", "#4d6b8a"
CLASS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#9c755f"]


def load(name):
    p = DATA / name
    return json.load(open(p if p.exists() else RAW / name))


DRAFT = Path(sys.argv[sys.argv.index("--draft") + 1]) if "--draft" in sys.argv else None
fig = plt.figure(figsize=(6.0, 2.3))
axA = fig.add_axes([0.0, 0.0, 0.345, 0.93])
axB = fig.add_axes([0.435, 0.17, 0.225, 0.70])
axC = fig.add_axes([0.755, 0.17, 0.235, 0.70])

# ---------------- (A) the measurement, drawn: declare an axis, match a floor ----------------
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
fig.text(0.004, 0.955, "A", fontsize=8, fontweight="bold", va="center")
fig.text(0.030, 0.955, "the measurement", fontsize=7.5, va="center")
rng = np.random.default_rng(3)
DG = "#555555"

# step 1: one class per sample; in V1 the classes are stimulus directions around the circle, in a language model
# the samples are prompts, the features a layer's units and the classes topics (Appendix I)
axA.text(0.15, 9.05, "declare an axis", fontsize=6.6, color=INK, va="center")
nrow, ncol, x0, y0, w, h = 16, 6, 1.15, 2.10, 2.0, 5.65
cw, ch = w / ncol, h / nrow
for i in range(nrow):
    axA.add_patch(Rectangle((x0 - 0.42, y0 + (nrow - 1 - i) * ch), 0.30, ch * 0.88, color=CLASS[i // 2], lw=0))
    for j in range(ncol):
        g = 0.50 + 0.45 * rng.random()
        axA.add_patch(Rectangle((x0 + j * cw, y0 + (nrow - 1 - i) * ch), cw * 0.9, ch * 0.88, color=(g, g, g), lw=0))
axA.text(x0 - 0.27, y0 + h + 0.12, "class", fontsize=5.8, color=DG, ha="center", va="bottom")
axA.text(x0 + w / 2 + 0.2, y0 + h + 0.12, "neurons\nor units", fontsize=5.8, color=DG, ha="center", va="bottom",
         linespacing=0.95)
axA.text(0.30, y0 + h / 2, "trials or prompts", fontsize=5.8, color=DG, rotation=90, ha="center", va="center")
cx, cy, r = 1.00, 0.92, 0.58
axA.add_patch(Circle((cx, cy), r, fill=False, ec=GRAY, lw=0.6))
for c in range(8):
    a = np.pi / 2 - c * np.pi / 4
    axA.plot([cx + r * np.cos(a)], [cy + r * np.sin(a)], "o", color=CLASS[c], ms=2.6, zorder=3)
axA.annotate("", xy=(cx + 0.5 * r * np.cos(np.pi / 2 - np.pi / 2), cy + 0.5 * r * np.sin(0)),
             xytext=(cx + 0.5 * r * np.cos(np.pi / 2), cy + 0.5 * r * np.sin(np.pi / 2)),
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.5, mutation_scale=4.5, connectionstyle="arc3,rad=-0.35"))
axA.text(cx + r + 0.16, cy, "8 directions\nor 8 topics", fontsize=5.8, color=DG, va="center", linespacing=0.95)

# step 2: the declared ladder adds whole classes in order; its floor draws random subsets of the same sizes
axA.text(4.30, 9.05, "match a floor", fontsize=6.6, color=INK, va="center")
rungs, bw, bh, gap, xs0 = [1, 2, 3, 4, 6, 8], 0.74, 0.29, 0.18, 4.50
for k, rr in enumerate(rungs):
    x = xs0 + k * (bw + gap)
    for c in range(rr):
        axA.add_patch(Rectangle((x, 5.15 + c * bh), bw, bh * 0.86, color=CLASS[c], lw=0))
    cols = rng.choice(8, size=rr, replace=True)
    for c in range(rr):
        axA.add_patch(Rectangle((x, 1.40 + c * bh), bw, bh * 0.86, color=CLASS[cols[c]], lw=0, alpha=0.40))
    axA.text(x + bw / 2, 1.18, f"{rr}", fontsize=6.0, color=DG, ha="center", va="top")
axA.text(4.30, 8.25, "declared: whole classes\nadded in order", fontsize=6.0, color=RED, va="center", linespacing=0.95)
axA.text(4.30, 4.45, "floor: random subsets\nof the same sizes", fontsize=6.0, color=DG, va="center", linespacing=0.95)
axA.text(xs0, 0.30, "rung size, in classes", fontsize=5.8, color=DG, ha="left", va="center")

# GT3 on each probe: the exponent, its own ladder's floor and the shift (theta_obs = theta_floor + delta)
_oz = load("orientation_zoom.json")
_r = next(r for r in load("stringer_theta_decomposition.json")["rows"] if r["name"].startswith("gratings_drifting_GT3"))
_d = next(r for r in _oz["results"] if r["name"].startswith("gratings_drifting_GT3"))
_s = next(r for r in load("neuron_zoom_spatial.json")["rows"] if r["name"].startswith("gratings_drifting_GT3"))
GT3 = [(_r["theta_obs"], _r["theta_floor_empirical"], _r["shift_obs"]),
       (_d["theta_structured"], _d["theta_random"], _d["delta"]),
       (_s["theta_spatial"], _s["theta_floor"], _s["delta_spatial"])]
print("GT3 (theta_obs, floor, delta):", [tuple(round(v, 3) for v in g) for g in GT3])

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
# the two slopes of step 3 (Eq. 1 on GT3: theta_obs = theta_floor + delta)
ax.text(2350, 33.0, r"$\theta_{\rm obs} = %.3f$" % _d["theta_structured"], fontsize=6.3, color=RED, ha="left", va="center")
ax.text(1260, 47.6, r"$\theta_{\rm floor} = %.3f$" % _d["theta_random"], fontsize=6.3, color=GRAY, ha="left", va="center")
ax.set_xlabel("stimuli in rung"); ax.set_ylabel("participation ratio")
ax.set_title("B  a slope each: GT3", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1)
ax.spines[["top", "right"]].set_visible(False)

# ---------------- (C) Eq. 1 read off: delta against theta_floor for every ladder with both slopes stored ----------------
ax = axC
rand = [r for r in load("stringer_theta_decomposition.json")["rows"] if r.get("status") == "ok"]
spat = load("neuron_zoom_spatial.json")["rows"]
dirx = [r for r in oz["results"] if r["name"].startswith("gratings_")]
LLM = {m: load(f"llm_battery_{m}.json")["batteries"]["world_knowledge32"]["layers"]
       for m in ("pythia-160m", "pythia-2.8b-deduped")}
ax.axhline(0, color=INK, lw=0.8, zorder=1)
ax.text(0.90, -0.03, "floor only", fontsize=6.0, color=INK, ha="center", va="top")
ax.plot([r["theta_floor_empirical"] for r in rand], [r["shift_obs"] for r in rand], "o", ms=3.0, mfc="white",
        mec=GRAY, mew=0.9, zorder=3)
ax.plot([r["theta_floor"] for r in spat], [r["delta_spatial"] for r in spat], "s", ms=2.8, mfc="white",
        mec=BLUEG, mew=0.9, zorder=3)
al = [v for r in load("allen_floors_all_sessions.json")["results"] if r.get("status") == "ok" for v in r["areas"].values()]
ax.plot([v["theta_floor_dir"] for v in al], [v["delta_dir8"] for v in al], "o", ms=1.6, color="#e8a0a8", zorder=2)
XLIM, YLIM = (-0.08, 1.0), (-0.21, 0.52)   # x from -0.08: four Allen direction-ladder floors are negative (to -0.064)
inside = sum(XLIM[0] < v["theta_floor_dir"] < XLIM[1] and YLIM[0] < v["delta_dir8"] < YLIM[1] for v in al)
print(f"CHECK Allen points {len(al)}, inside the axes {inside}, positive {sum(v['delta_dir8'] > 0 for v in al)}; "
      f"floor_dir {min(v['theta_floor_dir'] for v in al):+.3f}..{max(v['theta_floor_dir'] for v in al):+.3f}")
assert len(al) == 167 and inside == 167
ax.plot([r["theta_random"] for r in dirx], [r["delta"] for r in dirx], "o", ms=3.4, color=RED, zorder=4)
llm_x = [r["floor"] for m in LLM for r in LLM[m]]; llm_y = [r["dbar50_run37"] for m in LLM for r in LLM[m]]
print(f"CHECK LM points {len(llm_y)} (13 + 33), all positive {sum(y > 0 for y in llm_y)}; "
      f"range {min(llm_y):+.3f}..{max(llm_y):+.3f} (App I: +0.03 at the embedding to +0.08..+0.10 late); floors {min(llm_x):.2f}..{max(llm_x):.2f}")
assert len(llm_y) == 46 and all(y > 0 for y in llm_y)
ax.plot(llm_x, llm_y, "D", ms=2.4, color=PURPLE, mew=0, zorder=4)
# GT3's three readings: the same 11,311 neurons at three floors (Fig. 1 image of the abstract)
ax.plot([g[1] for g in GT3], [g[2] for g in GT3], "o", ms=7.0, mfc="none", mec=INK, mew=0.7, zorder=5)
ax.annotate("GT3", xy=(GT3[0][1], GT3[0][2]), xytext=(0.215, -0.085), fontsize=6.0, color=INK, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.4, shrinkA=0.5, shrinkB=3.2))
for g in GT3[1:]:
    ax.annotate("", xy=(g[1], g[2]), xytext=(0.215, -0.085), arrowprops=dict(arrowstyle="-", color=INK, lw=0.4,
                shrinkA=4.0, shrinkB=3.2))
ax.text(0.17, 0.36, "V1, stimulus\ndirection (8)", fontsize=6.3, color=RED, va="center", linespacing=0.95)
ax.text(0.17, 0.13, "Neuropixels,\n167 populations", fontsize=6.3, color="#c9606e", va="center", linespacing=0.95)
ax.text(0.46, -0.125, "V1, random stimuli (18)", fontsize=6.3, color=GRAY, va="center", ha="center")
ax.text(0.46, -0.18, "and neighbouring neurons (10)", fontsize=6.3, color=BLUEG, va="center", ha="center")
ax.text(0.99, 0.255, "language models, topics,\nevery layer (2 sizes),\naveraged over orders", fontsize=6.0, color=PURPLE,
        va="center", ha="right", linespacing=0.95)
ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax.set_xlabel(r"floor exponent $\theta_{\rm floor}$"); ax.set_ylabel(r"shift $\delta = \theta_{\rm obs} - \theta_{\rm floor}$")
ax.set_title("C  every ladder at its floor", loc="left", fontweight="bold", x=-0.17)
ax.spines[["top", "right"]].set_visible(False)

for out in ([DRAFT] if DRAFT else OUTS):
    if out.parent.exists():
        fig.savefig(out, dpi=300); print("wrote", out)
print("V1 random: floor range", min(r["theta_floor_empirical"] for r in rand), max(r["theta_floor_empirical"] for r in rand))
print("V1 direction: delta range", min(r["delta"] for r in dirx), max(r["delta"] for r in dirx))
print("LLM order-averaged range", min(llm_y), max(llm_y))
