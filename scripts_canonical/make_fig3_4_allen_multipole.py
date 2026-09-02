"""Figures 3 and 4 (canonical): multipole/order panel and Allen expansion panel.
Inputs: data_canonical/cos2theta_fit.json, antipodal_order.json,
allen_expansion_all_sessions.json. Out: ../figures_canonical/ (+ arxiv/figures when present).

Fig 3 (fig_allen_expansion.png): per-population direction-aligned shifts across
32 Allen Neuropixels sessions, by visual area, aligned vs shuffled in separate
sub-columns with annotated means and counts.
Fig 4 (fig_multipole_order.png): the measured C(dphi) multipole decomposition
and the accumulation-order effects it predicts.

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
    if ARXIV.exists():
        fig.savefig(ARXIV / name, dpi=300)
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
ax.set_ylabel(r"structural shift $\delta_{\rm dir}$")
ax.set_title("Direction-aligned shift across 32 Neuropixels sessions")
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

ao = json.load(open(DATA / "antipodal_order.json"))
seq = [r["delta_sequential"] for r in ao["rows"]]
par = [r["delta_paired"] for r in ao["rows"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.6),
                               gridspec_kw={"width_ratios": [1.1, 1.0]})

phi = np.linspace(0, np.pi, 400)
deg = np.degrees(phi)
Cfull = (a0 + b * np.cos(2 * phi) + c * np.cos(phi) + b4 * np.cos(4 * phi)
         + c3 * np.cos(3 * phi))
Ceven = a0 + b * np.cos(2 * phi) + b4 * np.cos(4 * phi)
Codd = c * np.cos(phi) + c3 * np.cos(3 * phi)
ax1.plot(deg, Cfull, color="k", lw=2.0)
ax1.plot(deg, Ceven, color=BLUE, lw=1.4, ls="--")
ax1.plot(deg, Codd, color=RED, lw=1.4, ls=":")
# label each curve where the full fit and the even sector actually
# separate (they differ only by the odd term, so mid-plot labels are
# ambiguous): full at the left edge, even at the right edge.
ax1.annotate(r"full fit $C(\Delta\phi)$", (10, 0.89), xytext=(38, 0.52),
             fontsize=7.5, color="k",
             arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
ax1.annotate("even (orientation)", (155, 0.59), xytext=(90, 0.44),
             fontsize=7.5, color=BLUE,
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
ax1.text(120, -0.20, "odd (direction)", fontsize=7.5, color=RED, ha="center")

def C_at(x_deg):
    x = np.radians(x_deg)
    return (a0 + b * np.cos(2 * x) + c * np.cos(x) + b4 * np.cos(4 * x)
            + c3 * np.cos(3 * x))

ax1.scatter([22.5], [C_at(22.5)], color=GREEN, zorder=5, s=55, marker="o")
ax1.annotate(f"adjacent\n$C = {C_at(22.5):.2f}$", (22.5, C_at(22.5)),
             xytext=(60, 0.90), fontsize=7.5, color=GREEN,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))
ax1.scatter([180.0], [C_at(180.0)], color=MAGENTA, zorder=5, s=55, marker="o")
ax1.annotate(f"antipodal\n$C = {C_at(180.0):.2f}$", (180.0, C_at(180.0)),
             xytext=(132, 0.90), fontsize=7.5, color=MAGENTA,
             arrowprops=dict(arrowstyle="->", color=MAGENTA, lw=0.8))
ax1.set_xlim(-4, 186)
ax1.set_ylim(-0.22, 1.02)
ax1.set_xlabel(r"orientation difference $\Delta\phi$ (deg)")
ax1.set_ylabel("signal correlation")
ax1.set_title(r"$\mathbf{A}$  Multipole content of the V1 code", loc="left")
ax1.spines[["top", "right"]].set_visible(False)

x = np.arange(len(seq))
ax2.bar(x - 0.19, seq, width=0.38, color=GREEN, label="sequential (adjacent)")
ax2.bar(x + 0.19, par, width=0.38, color=MAGENTA, label="antipodal-paired")
ax2.set_xticks(x)
ax2.set_xticklabels(["D1", "D2", "D3", "L1", "L2", "L3", "C1", "C2"])
ax2.set_xlabel("grating recording")
ax2.set_ylabel(r"structural shift $\delta$")
ax2.set_title(r"$\mathbf{B}$  Order effect (8/8 recordings)", loc="left")
ax2.set_ylim(0, 0.66)
ax2.legend(frameon=True, facecolor="white", edgecolor="0.85",
           fontsize=7, loc="upper right")
ax2.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig_multipole_order.png")
plt.close(fig)
print("fig_multipole_order.png written")
