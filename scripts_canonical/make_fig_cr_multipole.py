"""Camera-ready (NeurReps 2026) Figures 2 and 3: the two-panel multipole/order figure and the Allen panel.
The arXiv-era make_fig3_4_allen_multipole.py draws the ordering contrast alone; this one keeps both panels.
Inputs: data_canonical/multipole_harmonics_8dir.json, run72_paper_designs_split.json (order-effect split),
run59_shift_by_direction_selectivity.json (DSI thirds),
allen_expansion_all_sessions.json. Out: ../figures_canonical/ (+ arxiv/figures when present).

Fig 3 (fig_allen_expansion.png): per-population direction-aligned shifts across
32 Allen Neuropixels sessions, by visual area, aligned vs shuffled in separate
sub-columns with annotated means and counts.
Fig 2 (fig_multipole_two_panel.png, three panels): GT3's class-mean correlation and its two sectors;
the order effect split into its class-mean and within-class parts; the shift of the most and least
direction-selective thirds per recording.

Outputs to figures/theta_eft/.
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
    CR = HERE.parent.parent / "neurreps_camera_ready" / "figures"
    if CR.exists():
        fig.savefig(CR / name, dpi=300)
    print("wrote", OUT / name)

RED, GRAY, GREEN, MAGENTA, BLUE = "#b40426", "#8a8a8a", "#3d7d1f", "#c51b7d", "#2166ac"

# ---------------- Fig 3: Allen expansion ----------------
rows = []
allen = json.load(open(DATA / "allen_expansion_all_sessions.json"))
for r in allen["results"]:
    if r.get("status") != "ok":
        continue
    for area, a in r["areas"].items():
        if a["delta_dir8"] is not None:
            rows.append((area, a["delta_dir8"], a.get("shuffled_mean")))

areas = ["VISp", "VISpm", "VISam", "VISal", "VISrl", "VISl"]
fig, ax = plt.subplots(figsize=(5.7, 2.9))
rng = np.random.default_rng(0)
OFF = 0.19
for i, area in enumerate(areas):
    d = np.array([x[1] for x in rows if x[0] == area])
    s = np.array([x[2] for x in rows if x[0] == area and x[2] is not None])
    ax.scatter(i - OFF + rng.uniform(-0.09, 0.09, len(d)), d, s=17, color=RED,
               alpha=0.8, zorder=3, edgecolors="none",
               label="direction-aligned" if i == 0 else None)
    ax.scatter(i + OFF + rng.uniform(-0.09, 0.09, len(s)), s, s=13, color=GRAY,
               alpha=0.65, zorder=2, edgecolors="none",
               label="shuffled labels" if i == 0 else None)
    ax.plot([i - OFF - 0.12, i - OFF + 0.12], [d.mean()] * 2, color="k",
            lw=2.2, zorder=4)
    ax.plot([i + OFF - 0.11, i + OFF + 0.11], [s.mean()] * 2, color="0.5",
            lw=1.2, zorder=4)
    ax.text(i - OFF - 0.16, d.mean(), f"{d.mean():+.2f}", ha="right",
            va="center", fontsize=7, color="k", fontweight="bold")
    npos = int((d > 0).sum())
    ax.text(i, -0.165, f"{npos}/{len(d)} > 0", ha="center", fontsize=7.5, color="0.25")
ax.axhline(0, color="k", lw=0.7, alpha=0.45, zorder=1)
ax.set_xlim(-0.85, len(areas) - 0.35)
ax.set_xticks(range(len(areas)))
ax.set_xticklabels(areas)
ax.set_ylim(-0.19, 0.47)
ax.set_ylabel(r"shift $\delta_{\rm dir}$")
ax.legend(frameon=False, loc="upper right", fontsize=7.5, handletextpad=0.2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig_allen_expansion.png")
plt.close(fig)
print("fig_allen_expansion.png written")

# ---------------- Fig 4: multipoles and order effects ----------------
# five harmonic coefficients of the GT3 class-mean correlation profile: the
# exact 8-direction decomposition (multipole_harmonics_8dir.json, Sec. 4);
# a, b2, c1 equal cos2theta_fit.json; (b2^2 + b4^2)/(c1^2 + c3^2) = 4.80
mh = json.load(open(DATA / "multipole_harmonics_8dir.json"))
gt3 = next(r for r in mh["rows"] if "drifting_GT3" in r["name"])["coef"]
a0, b, c, b4, c3 = gt3["a"], gt3["b2"], gt3["c1"], gt3["b4"], gt3["c3"]

split = json.load(open(DATA / "run72_paper_designs_split.json"))["rows"]      # order-effect split (Table 8)
tags = ["D1", "D2", "D3", "L1", "L2", "L3", "C1", "C2"]
o_tot = [split[t]["all"]["order_effect"] for t in tags]
o_mean = [split[t]["all"]["order_probe"] for t in tags]
o_with = [split[t]["all"]["order_noise"] for t in tags]
ds = json.load(open(DATA / "run59_shift_by_direction_selectivity.json"))["verdict"]   # registered DSI thirds
assert ds["tags"] == tags

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.2})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.0, 2.05), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]})

phi = np.linspace(0, np.pi, 400)
deg = np.degrees(phi)
Cfull = (a0 + b * np.cos(2 * phi) + c * np.cos(phi) + b4 * np.cos(4 * phi)
         + c3 * np.cos(3 * phi))
Ceven = a0 + b * np.cos(2 * phi) + b4 * np.cos(4 * phi)
Codd = c * np.cos(phi) + c3 * np.cos(3 * phi)
ax1.plot(deg, Cfull, color="k", lw=1.6, label="five-term fit")
ax1.plot(deg, Ceven, color=BLUE, lw=1.2, ls="--", label="even (orientation)")
ax1.plot(deg, Codd, color=RED, lw=1.2, ls=":", label="odd (direction)")
ax1.legend(frameon=False, loc="lower left", fontsize=5.8, handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)


def C_at(x_deg):
    x = np.radians(x_deg)
    return (a0 + b * np.cos(2 * x) + c * np.cos(x) + b4 * np.cos(4 * x)
            + c3 * np.cos(3 * x))


ax1.scatter([22.5], [C_at(22.5)], color="k", zorder=5, s=18)
ax1.text(30, 0.86, f"adjacent {C_at(22.5):.2f}", fontsize=6.3, color="k", ha="left", va="center")
ax1.scatter([180.0], [C_at(180.0)], color="k", zorder=5, s=18)
ax1.annotate(f"antipodal {C_at(180.0):.2f}", xy=(180, C_at(180.0)), xytext=(100, 0.62), fontsize=6.3, ha="center",
             va="center", arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5))
ax1.set_xlim(-4, 186); ax1.set_ylim(-0.42, 1.04); ax1.set_xticks([0, 45, 90, 135, 180])
ax1.set_xlabel(r"direction difference $\Delta\phi$ (deg)")
ax1.set_ylabel(r"class-mean correlation $C(\Delta\phi)$")
ax1.set_title(r"$\mathbf{A}$  GT3: two sectors", loc="left")
ax1.spines[["top", "right"]].set_visible(False)

x = np.arange(len(tags))
ax2.bar(x - 0.19, o_mean, width=0.36, color=GRAY, label="class means")
ax2.bar(x + 0.19, o_with, width=0.36, color=RED, label="within-class")
ax2.plot(x, o_tot, "o", color="k", ms=3.2, label="total", zorder=4)
ax2.axhline(0, color="k", lw=0.6)
ax2.text(-0.5, 0.285, "favours pairing", fontsize=6.0, color="0.35", va="center")
ax2.text(-0.5, -0.235, "favours adjacent", fontsize=6.0, color="0.35", va="center")
ax2.set_xticks(x); ax2.set_xticklabels(tags); ax2.set_ylim(-0.25, 0.31)
ax2.set_ylabel("order effect (paired $-$ adjacent)")
ax2.set_title(r"$\mathbf{B}$  order effect, split", loc="left")
ax2.legend(frameon=False, loc="upper right", fontsize=5.8, handlelength=1.0, borderaxespad=0.1, bbox_to_anchor=(1.0, 0.9))
ax2.spines[["top", "right"]].set_visible(False)

dsv, nds = ds["delta_DS"], ds["delta_nonDS"]
for k in x:
    ax3.plot([k, k], [nds[k], dsv[k]], color="0.75", lw=1.0, zorder=1)
ax3.plot(x, dsv, "o", color=RED, ms=3.6, label="most direction-selective third", zorder=3)
ax3.plot(x, nds, "o", mfc="white", mec=BLUE, mew=1.0, ms=3.4, label="least selective third", zorder=3)
ax3.set_xticks(x); ax3.set_xticklabels(tags); ax3.set_ylim(0, 0.74)
ax3.set_ylabel(r"shift $\delta_{\rm dir}$")
ax3.set_title(r"$\mathbf{C}$  where the shift lives", loc="left")
ax3.legend(frameon=False, loc="upper left", fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
ax3.spines[["top", "right"]].set_visible(False)
fig.tight_layout(w_pad=0.6)
save(fig, "fig_multipole_two_panel.png")
plt.close(fig)
print("fig_multipole_two_panel.png written; class-mean order part > 0:", sum(v > 0.01 for v in o_mean),
      "within-class < 0:", sum(v < 0 for v in o_with), "DS > nonDS:", sum(a > b for a, b in zip(dsv, nds)))
