"""Figures 1 and 2 (canonical): GT3 two-axis ladder and per-recording shifts.
Inputs: data_canonical/orientation_zoom.json; Fig 2 (camera-ready): basin_memory/data/stringer_mismatch_ablation.json
(the paper's point estimates, Table C) + data_canonical/run73_m_out_of_n_all.json (m-out-of-n 95% intervals).
Out: ../figures_canonical/ (+ ../../arxiv/figures/ and ../../neurreps_camera_ready/figures/ when present).

Fig 1 (fig_zoom_ladder_dir.png): GT3 ladders from stringer_orientation_zoom.json,
labeled as the direction-aligned construction, with the analytic MP crossover
marked as motivation (the reported floors are empirical).
Fig 2 (fig_delta_per_recording_moon.png, camera-ready; supersedes fig_delta_per_recording_dir.png): per-recording
point estimates with m-out-of-n intervals (80% of each class's trials without replacement, 100 replicates, floor
redrawn); eight gratings as the primary endpoint, the two orientation-biased static sessions shown as points only
(degenerate probe, not resampled).
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
CR = HERE.parent.parent / "neurreps_camera_ready" / "figures"
BM = HERE.parent.parent.parent / "basin_memory" / "data"

def bm(name, local):
    """Raw-analysis file from basin_memory/data in the working tree, else its copy in data_canonical (public release)."""
    p = BM / name
    return p if p.exists() else DATA / local



def save(fig, name):
    fig.savefig(OUT / name, dpi=300)
    if ARXIV.exists() and name != "fig_delta_per_recording_moon.png":
        fig.savefig(ARXIV / name, dpi=300)
    if CR.exists():
        fig.savefig(CR / name, dpi=300)
    print("wrote", OUT / name)
RED, GRAY = "#b40426", "#8a8a8a"

# ---------------- Fig 1: GT3 ladder ----------------
oz = json.load(open(DATA / "orientation_zoom.json"))
gt3 = next(r for r in oz["results"] if "drifting_GT3" in r["name"])
s_sizes = [r[1] for r in gt3["struct_ladder"]]
s_pr = [r[2] for r in gt3["struct_ladder"]]
r_sizes = [r[1] for r in gt3["rand_ladder"]]
r_pr = [r[2] for r in gt3["rand_ladder"]]

# d_eff chosen so Eq. 2 passes through the top rung: PR = d(n-1)/(d+n)  =>  d = PR n / (n - 1 - PR)
n_top, pr_top = r_sizes[-1], r_pr[-1]
d_eff = pr_top * n_top / (n_top - 1 - pr_top)
n_grid = np.logspace(np.log10(min(s_sizes)), np.log10(max(s_sizes)), 200)
mp = d_eff * (n_grid - 1) / (d_eff + n_grid)  # Eq. 2 (ratio of expectations); was the (n-1)d/((n-1)+d) interpolant

from matplotlib.ticker import ScalarFormatter, NullFormatter
fig, ax = plt.subplots(figsize=(3.9, 2.7))
ax.loglog(n_grid, mp, ls="--", color="0.55", lw=1.2,
          label="Eq. 2 crossover (motivation)")
ax.loglog(r_sizes, r_pr, "o-", color=GRAY, lw=1.6, ms=5,
          label="random subsets (empirical floor)")
ax.loglog(s_sizes, s_pr, "o-", color=RED, lw=1.8, ms=5,
          label="direction-aligned ladder")
ax.set_ylim(24, 82)
# readable log ticks: plain numbers at the rung sizes, no 10^k mantissas
ax.set_xticks([500, 1000, 2000, 4000])
ax.set_yticks([25, 30, 40, 50, 60, 80])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())
ax.text(580, 25.5, r"$\delta_{\rm dir} = +0.24$", fontsize=10, color=RED,
        ha="left")
ax.set_xlabel("number of stimuli in rung")
ax.set_ylabel("participation ratio")
ax.set_title("One recording, two axes (GT3, 11,311 neurons)")
ax.legend(frameon=True, facecolor="white", edgecolor="0.85",
          fontsize=6.5, loc="upper left", ncol=1)  # above both curves (y range extended to 82)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig_zoom_ladder_dir.png")
plt.close(fig)
print("fig_zoom_ladder_dir.png written")

# ---------------- Fig 2: per-recording point estimates with m-out-of-n intervals ----------------
pts = {r["name"]: r["dir8"]["delta"] for r in json.load(open(bm("stringer_mismatch_ablation.json", "mismatch_ablation.json")))["rows"]}
moon = json.load(open(DATA / "run73_m_out_of_n_all.json"))["rows"]
order = ["gratings_drifting_GT1", "gratings_drifting_GT2", "gratings_drifting_GT3",
         "gratings_local_GT1", "gratings_local_GT2", "gratings_local_GT3",
         "gratings_low_contrast_GT1", "gratings_low_contrast_GT2",
         "static_biased_TX40", "static_biased_TX42"]
labels = ["Drift\nGT1", "Drift\nGT2", "Drift\nGT3", "Local\nGT1", "Local\nGT2",
          "Local\nGT3", "LowC\nGT1", "LowC\nGT2", "Static\nTX40", "Static\nTX42"]
fig, ax = plt.subplots(figsize=(5.4, 2.5))
inside = 0
for i, o in enumerate(order):
    name = next(k for k in pts if k.startswith(o))
    d = pts[name]
    if name in moon:
        lo, hi = moon[name]["ci95"]
        inside += lo <= d <= hi
        ax.vlines(i, lo, hi, color=RED, lw=2.2, alpha=0.55)
        ax.plot(i, d, "o", color=RED, ms=5.5)
    else:
        ax.plot(i, d, "o", color=GRAY, ms=5.5)
print(f"Fig 2: paper point inside its m-out-of-n interval in {inside}/8 grating recordings")
ax.axhline(0, color="k", lw=0.7, alpha=0.5)
ax.axvspan(7.5, 9.5, color="0.93", zorder=0)
ax.text(8.5, 0.30, "orientation-biased\nensembles\n(degenerate probe)",
        ha="center", fontsize=7.5, color="0.35")
ax.set_xticks(range(10))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel(r"$\delta_{\rm dir}$ (point, 95% interval)")
ax.set_ylim(-0.05, 0.55)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig_delta_per_recording_moon.png")
plt.close(fig)
