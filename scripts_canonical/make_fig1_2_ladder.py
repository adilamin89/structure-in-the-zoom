"""Figures 1 and 2 (canonical): GT3 two-axis ladder and per-recording shifts.
Inputs: data_canonical/orientation_zoom.json, bootstrap_all_10_orient_fullneuron.json.
Out: ../figures_canonical/ (+ ../../arxiv/figures/ when present).

Fig 1 (fig_zoom_ladder_dir.png): GT3 ladders from stringer_orientation_zoom.json,
labeled as the direction-aligned construction, with the analytic MP crossover
marked as motivation (the reported floors are empirical).
Fig 2 (fig_delta_per_recording_dir.png): per-recording direction-aligned shifts
with bootstrap CIs; eight gratings as the primary endpoint, the two
orientation-biased static sessions shown but flagged as degenerate probes.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 7, "savefig.dpi": 300})
# figsize equals the printed width (jmlr textwidth 6.0 in) so fonts print 1:1

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUT = HERE.parent / "figures_canonical"
OUT.mkdir(parents=True, exist_ok=True)
ARXIV = HERE.parent.parent / "arxiv" / "figures"  # secondary copy only when the paper tree is present


def save(fig, name):
    fig.savefig(OUT / name, dpi=300)
    if ARXIV.exists():
        fig.savefig(ARXIV / name, dpi=300)
    print("wrote", OUT / name)
RED, GRAY = "#b40426", "#8a8a8a"

# ---------------- Fig 1: GT3 ladder ----------------
oz = json.load(open(DATA / "orientation_zoom.json"))
gt3 = next(r for r in oz["results"] if "drifting_GT3" in r["name"])
s_sizes = [r[1] for r in gt3["struct_ladder"]]
s_pr = [r[2] for r in gt3["struct_ladder"]]
r_sizes = [r[1] for r in gt3["rand_ladder"]]
r_pr = [r[2] for r in gt3["rand_ladder"]]

d_eff = r_pr[-1]
n_grid = np.logspace(np.log10(min(s_sizes)), np.log10(max(s_sizes)), 200)
mp = (n_grid - 1) * d_eff / ((n_grid - 1) + d_eff)

from matplotlib.ticker import ScalarFormatter, NullFormatter
fig, ax = plt.subplots(figsize=(3.9, 2.7))
ax.loglog(n_grid, mp, ls="--", color="0.55", lw=1.2,
          label="analytic crossover (motivation)")
ax.loglog(r_sizes, r_pr, "o-", color=GRAY, lw=1.6, ms=5,
          label="random stimulus subsets (empirical floor)")
ax.loglog(s_sizes, s_pr, "o-", color=RED, lw=1.8, ms=5,
          label="direction-aligned ladder")
ax.set_ylim(24, 62)
# readable log ticks: plain numbers at the rung sizes, no 10^k mantissas
ax.set_xticks([500, 1000, 2000, 4000])
ax.set_yticks([25, 30, 40, 50, 60])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())
ax.text(580, 25.5, r"$\delta_{\rm dir} = +0.24$", fontsize=10, color=RED,
        ha="left")
ax.set_xlabel("number of stimuli in rung")
ax.set_ylabel("participation ratio")
ax.set_title("One recording, two axes (GT3, 11,311 neurons)")
ax.legend(frameon=True, facecolor="white", edgecolor="0.85",
          fontsize=6.5, loc="upper left")  # clear of the ladders and the delta label
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig_zoom_ladder_dir.png")
plt.close(fig)
print("fig_zoom_ladder_dir.png written")

# ---------------- Fig 2: per-recording with CIs ----------------
bs = json.load(open(DATA / "bootstrap_all_10_orient_fullneuron.json"))
recs = bs["recordings"]
order = ["gratings_drifting_GT1", "gratings_drifting_GT2", "gratings_drifting_GT3",
         "gratings_local_GT1", "gratings_local_GT2", "gratings_local_GT3",
         "gratings_low_contrast_GT1", "gratings_low_contrast_GT2",
         "static_biased_TX40", "static_biased_TX42"]
labels = ["Drift\nGT1", "Drift\nGT2", "Drift\nGT3", "Local\nGT1", "Local\nGT2",
          "Local\nGT3", "LowC\nGT1", "LowC\nGT2", "Static\nTX40", "Static\nTX42"]
sel = []
for o in order:
    sel.append(next(r for r in recs if r["name"].startswith(o)))

fig, ax = plt.subplots(figsize=(5.4, 2.7))
for i, r in enumerate(sel):
    grating = i < 8
    col = RED if grating else GRAY
    ax.errorbar(i, r["mean"], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                fmt="o", color=col, ms=6, capsize=3, lw=1.4)
ax.axhline(0, color="k", lw=0.7, alpha=0.5)
ax.axvspan(7.5, 9.5, color="0.93", zorder=0)
ax.text(8.5, 0.50, "orientation-biased\nensembles\n(degenerate probe)",
        ha="center", fontsize=7.5, color="0.35")
ax.set_xticks(range(10))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel(r"$\delta_{\rm dir}$ (bootstrap, 95% CI)")
ax.set_title("Direction-aligned shift per recording: primary endpoint = eight gratings")
ax.set_ylim(-0.07, 0.60)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig_delta_per_recording_dir.png")
plt.close(fig)
print("fig_delta_per_recording_dir.png written")
