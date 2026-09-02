"""Figure (App I): the sector balance follows single-neuron tuning across stimulus
types. From local_vs_fullfield_tuning.json (S74). Drawn at the printed width.
  (a) quadrupole-to-dipole ratio b2/c1 against the median direction-selectivity
      index of tuned neurons, one point per grating recording; the line b2/c1 = 1
      separates the quadrupole-dominant (director on RP^1, nematic) regime from
      the dipole-dominant (vector on S^1, polar) regime.
  (b) median orientation-selectivity index against median direction-selectivity
      index for the same recordings.
Out: ../figures_canonical/fig_stimulus_sector.png + ../../arxiv/figures/
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_stimulus_sector.png",
        HERE.parent.parent / "arxiv" / "figures" / "fig_stimulus_sector.png"]
TYPECOL = {"D": ("#a63603", "drifting, full field"), "L": ("#e6550d", "localized"),
           "C": ("#756bb1", "low contrast, full field")}


def short(name):
    kind = "D" if "drifting" in name else "L" if "local" in name else "C"
    return kind + name.split("GT")[1][0]


rows = [r for r in json.load(open(DATA / "local_vs_fullfield_tuning.json"))["rows"] if r["status"] == "ok"]
fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4))
plt.subplots_adjust(wspace=0.38, left=0.09, right=0.98, top=0.88, bottom=0.21)
ax = axes[0]
for r in rows:
    t = short(r["name"]); c, _ = TYPECOL[t[0]]
    ax.plot(r["median_dsi_tuned"], r["b2_over_c1"]["all"], "o", color=c, ms=5)
    ax.annotate(t, (r["median_dsi_tuned"], r["b2_over_c1"]["all"]), xytext=(4, 2), textcoords="offset points", fontsize=6)
ax.axhline(1.0, color="k", lw=0.6)
ax.text(0.02, 0.97, "quadrupole-dominant: director on $\\mathbb{RP}^1$", transform=ax.transAxes, va="top", fontsize=6.3, color="0.3")
ax.text(0.02, 0.05, "dipole-dominant: vector on $S^1$", transform=ax.transAxes, va="bottom", fontsize=6.3, color="0.3")
ax.set_xlabel("median DSI, tuned neurons"); ax.set_ylabel("$b_2/c_1$ (quadrupole / dipole)")
ax.set_ylim(0.3, 2.7); ax.set_xlim(0.2, 0.5)
ax.set_title("(a) sector balance follows direction tuning")
ax = axes[1]
for key, (c, lab) in TYPECOL.items():
    xs = [r["median_dsi_tuned"] for r in rows if short(r["name"])[0] == key]
    ys = [r["median_osi_tuned"] for r in rows if short(r["name"])[0] == key]
    ax.plot(xs, ys, "o", color=c, ms=5, label=lab)
for r in rows:
    ax.annotate(short(r["name"]), (r["median_dsi_tuned"], r["median_osi_tuned"]), xytext=(4, 2), textcoords="offset points", fontsize=6)
ax.set_xlabel("median DSI, tuned neurons"); ax.set_ylabel("median OSI, tuned neurons")
ax.set_xlim(0.2, 0.5); ax.set_ylim(0.34, 0.54)
ax.legend(frameon=False, loc="lower left")
ax.set_title("(b) single-neuron tuning by stimulus")
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists():
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300); print("wrote", out)
