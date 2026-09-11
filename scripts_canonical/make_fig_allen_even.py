"""Figure (Session F): the Allen shift against the even-sector amplitude over the 167 session-area populations
(the association of Section 5: r = +0.41, p = 3e-8; within areas +0.33; A_odd -0.04). Assembly as in run39:
delta = delta_dir8 per population, A_even = sqrt(b_quadrupole^2 + b4^2) from the harmonic fit.
Sources: allen_expansion_all_sessions.json, allen_multipoles_all_sessions.json.
Out: ../figures_canonical/fig_allen_even.png + ../../arxiv/figures/fig_allen_even.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.2, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_allen_even.png", HERE.parent.parent / "arxiv" / "figures" / "fig_allen_even.png"]
exp = json.load(open(DATA / "allen_expansion_all_sessions.json"))["results"]
mul = {m["session"]: m for m in json.load(open(DATA / "allen_multipoles_all_sessions.json"))["results"]}
rows = []
for s in exp:
    ms = mul.get(s["session"])
    if ms is None: continue
    for area, a in s["areas"].items():
        am = ms["areas"].get(area)
        if am is None: continue
        rows.append((area, a["delta_dir8"], float(np.sqrt(am["b_quadrupole"] ** 2 + am["b4"] ** 2)), abs(am["c_dipole"])))
areas = np.array([r[0] for r in rows]); d = np.array([r[1] for r in rows]); ae = np.array([r[2] for r in rows]); ao = np.array([r[3] for r in rows])
r, p = pearsonr(ae, d); ro, _ = pearsonr(ao, d)
print(f"{len(rows)} populations; r(delta, A_even) = {r:+.3f} (p = {p:.1e}); r(delta, A_odd) = {ro:+.3f}")
COL = {"VISp": "#b40426", "VISpm": "#2166ac", "VISam": "#e08214", "VISal": "#3d7d1f", "VISrl": "#756bb1", "VISl": "#4d4d4d"}
fig, ax = plt.subplots(figsize=(4.3, 2.5)); plt.subplots_adjust(left=0.13, right=0.77, top=0.95, bottom=0.18)
for area in COL:
    m = areas == area
    ax.plot(ae[m], d[m], "o", ms=3.4, color=COL[area], mec="white", mew=0.4, alpha=0.9, label=f"{area} ($n = {m.sum()}$)")
other = ~np.isin(areas, list(COL))
ax.plot(ae[other], d[other], "o", ms=3.0, color="#bdbdbd", mec="white", mew=0.4, label=f"other ($n = {other.sum()}$)")
b, a0 = np.polyfit(ae, d, 1); xx = np.linspace(ae.min(), ae.max(), 50)
ax.plot(xx, a0 + b * xx, "-", color="0.3", lw=0.9)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("even-sector amplitude $A_{\\mathrm{even}} = \\sqrt{b_2^2 + b_4^2}$"); ax.set_ylabel("direction-aligned shift $\\delta_{\\mathrm{dir}}$")

ax.set_ylim(-0.12, 0.5)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=6.0, handlelength=1.0, labelspacing=0.35, borderaxespad=0.0)
ax.spines[["top", "right"]].set_visible(False)
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
