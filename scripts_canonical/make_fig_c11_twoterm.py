"""Camera-ready Fig 3 (C11): the two-term account of the shift, measured against its prediction in five systems.
Nothing is fitted. Within-class part: the mean-field overlap law ln[K/(1+(K-1) sigma)]/ln K with split-half sigma. Class-mean
part: the full-set term log[(1+beta)^2/(1+(b2+2x)/w2)] carried through the rung sizes by the class means' variance fraction
(cr_h1_b245.classmean_prediction). delta = their sum.
Sources (each a committed artifact; CHECK lines assert every number the paper quotes from this figure):
  V1 imaging, 8 grating recordings ............ data_canonical/cr_h1_b245.json rows[*].B2 (pred) vs t9_* (Table 9)
  Allen Neuropixels, 167 populations ........... h1_allen_b2_predictions.json (committed before scoring) vs h1_allen/session_*.json tfres
  macaque V1, 3 Utah arrays .................... cr_h1_m_pvc11_twoterm.json
  XY nematic lattice (L 16/32, T 1.0/1.1) ...... h2_xy_twoterm.json lattice
  spin waves (same order, rotating fluctuations) h2_xy_twoterm.json spinwave
  XY lattice, polar-ordered side (T 0.6-0.85) . h1_xy_polar_twoterm.json (H1-R41)
Lattice ensembles with fewer configurations per class than twice the class's fluctuation dimension (n/d_w < 2) are drawn open:
there the split's two parts carry a finite-sample transfer (App K); the total is unaffected.
Out: ../figures_canonical/fig_c11_twoterm.png (+ camera-ready figures/)."""
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUT = HERE.parent / "figures_canonical"
CR = HERE.parent.parent / "neurreps_camera_ready" / "figures"
V1C, ALC, MQC, LAT = "#b40426", "#e98ea0", "#2a78d6", "#555555"   # validated all-pairs (dataviz validator, light mode)
PARTS = ("within", "class_mean", "delta")

# ---- V1 imaging
v1 = {p: ([], []) for p in PARTS}
for r in json.load(open(DATA / "cr_h1_b245.json"))["rows"]:
    b = r["B2"]
    for p, kp, km in (("within", "within_pred", "t9_within"), ("class_mean", "classmean_pred", "t9_class_means"),
                      ("delta", "delta_pred", "t9_delta")):
        v1[p][0].append(b[kp]); v1[p][1].append(b[km])
v1 = {p: (np.array(a), np.array(b)) for p, (a, b) in v1.items()}

# ---- Allen (prediction committed before the measured values were compared)
pred = {(p["session"], p["area"]): p for p in json.load(open(DATA / "h1_allen_b2_predictions.json"))}
al = {p: ([], []) for p in PARTS}; al_area = []
for f in sorted(glob.glob(str(DATA / "h1_allen" / "session_*.json"))):
    d = json.load(open(f))
    for area, v in d["areas"].items():
        q = pred[(d["session"], area)]
        for p, kp, km in (("within", "within_pred", "within"), ("class_mean", "classmean_pred", "class_means"),
                          ("delta", "delta_pred", "delta")):
            al[p][0].append(q[kp]); al[p][1].append(v["tfres"][km])
        al_area.append(area)
al = {p: (np.array(a), np.array(b)) for p, (a, b) in al.items()}; al_area = np.array(al_area)

# ---- macaque
mq_rows = json.load(open(DATA / "cr_h1_m_pvc11_twoterm.json"))["rows"]
mq = {p: (np.array([r[f"{'classmean' if p == 'class_mean' else p}_pred"] for r in mq_rows]),
          np.array([r[f"measured_{'class_means' if p == 'class_mean' else p}"] for r in mq_rows])) for p in PARTS}

# ---- lattice + spin waves
xy = json.load(open(DATA / "h2_xy_twoterm.json"))
def _lat(rows):
    return ({p: (np.array([r[f"pred_{p}"] for r in rows]), np.array([r[p] for r in rows])) for p in PARTS},
            np.array([r["per_class"] / r["PR_class"] for r in rows]))
lat, lat_nd = _lat(xy["lattice"])
sw = {p: (np.array([r[f"pred_{p}"] for r in xy["spinwave"]]), np.array([r[p] for r in xy["spinwave"]])) for p in PARTS}
pol_f = DATA / "h1_xy_polar_twoterm.json"
pol = pol_nd = None
if pol_f.exists():   # H1-R41: keys measured_*; measured delta = within + class-mean of the same split run
    prow = json.load(open(pol_f))["rows"]
    pol = {p: (np.array([r[f"pred_{p}"] for r in prow]), np.array([r[f"measured_{p}"] for r in prow])) for p in PARTS}
    pol_nd = np.array([r["n_over_d"] for r in prow])

# ---- CHECK: every number the text quotes from this figure
sgn = lambda a, b: int((np.sign(a) == np.sign(b)).sum())
r = lambda a, b: stats.pearsonr(a, b)[0]
visp = al_area == "VISp"
chk = {
    "V1 delta sign": (sgn(*v1["delta"]), 8), "V1 delta r": (round(r(*v1["delta"]), 2), 0.88),
    "V1 delta mean|err|": (round(float(np.abs(v1["delta"][0] - v1["delta"][1]).mean()), 3), 0.038),
    "V1 class-mean r": (round(r(*v1["class_mean"]), 2), 0.99),
    "Allen delta sign": (sgn(*al["delta"]), 153), "Allen n": (len(al_area), 167),
    "Allen VISp delta sign": (sgn(al["delta"][0][visp], al["delta"][1][visp]), 31),
    "Allen delta r": (round(r(*al["delta"]), 2), 0.68), "Allen VISp delta r": (round(r(al["delta"][0][visp], al["delta"][1][visp]), 2), 0.83),
    "Allen class-mean r": (round(r(*al["class_mean"]), 2), 0.95), "Allen class-mean sign": (sgn(*al["class_mean"]), 161),
    "macaque delta sign": (sgn(*mq["delta"]), 3),
    "lattice+spin-wave delta sign": (sgn(*lat["delta"]) + sgn(*sw["delta"]), 40), "lattice+spin-wave n": (len(lat_nd) + len(sw["delta"][0]), 40),
    "lattice Spearman": (round(stats.spearmanr(*lat["delta"])[0], 2), 0.87),
}
for k, (got, want) in chk.items():
    print(f"CHECK {k}: {got} (paper {want})")
    assert got == want, k
if pol is not None:
    for k, (got, want) in {"polar XY delta sign": (sgn(*pol["delta"]), 32), "polar XY n": (len(pol_nd), 32),
                           "all lattice + spin-wave delta sign": (sgn(*lat["delta"]) + sgn(*sw["delta"]) + sgn(*pol["delta"]), 72)}.items():
        print(f"CHECK {k}: {got} (paper {want})")
        assert got == want, k

# ---- draw
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.2))
TITLES = {"within": r"$\mathbf{A}$  fluctuations: within-class part",
          "class_mean": r"$\mathbf{B}$  class means: class-mean part",
          "delta": r"$\mathbf{C}$  the shift $\delta$"}
XL = {"within": r"overlap law (mean field, $\sigma$)", "class_mean": r"class-mean term ($\beta$, dimensions, alignment)",
      "delta": "two-term prediction"}
LIM = {"within": (-0.25, 0.75), "class_mean": (-1.35, 0.45), "delta": (-1.15, 0.6)}


def lattice(ax, P, nd, part, marker, label):
    x, y = P[part]
    big = nd >= 2
    ax.plot(x[big], y[big], marker, color=LAT, ms=3.0, mew=0.6, zorder=3, label=label)
    ax.plot(x[~big], y[~big], marker, color=LAT, mfc="none", ms=3.0, mew=0.6, zorder=3)


for ax, part in zip(axs, PARTS):
    lo, hi = LIM[part]
    if part == "delta":   # quadrants where prediction and measurement share a sign
        ax.fill_between([0, hi], 0, hi, color="0.93", lw=0, zorder=0)
        ax.fill_between([lo, 0], lo, 0, color="0.93", lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.7, ls="--", zorder=1)
    ax.axhline(0, color="0.75", lw=0.5, zorder=1); ax.axvline(0, color="0.75", lw=0.5, zorder=1)
    ax.plot(*al[part], "o", color=ALC, ms=1.7, mew=0, zorder=2, label="mouse V1, spikes (Allen, 167)")
    lattice(ax, lat, lat_nd, part, "s", "XY nematic lattice")
    if pol is not None:
        lattice(ax, pol, pol_nd, part, "s", None)
    ax.plot(*sw[part], "^", color=LAT, ms=3.2, mew=0, zorder=3, label="spin waves")
    ax.plot(*v1[part], "o", color=V1C, ms=3.6, mew=0, zorder=4, label="mouse V1, imaging (8)")
    ax.plot(*mq[part], "D", color=MQC, ms=3.4, mew=0, zorder=5, label="macaque V1 (3)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(XL[part]); ax.set_title(TITLES[part], loc="left")
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("measured")
h, l = axs[0].get_legend_handles_labels()
fig.tight_layout(w_pad=0.8, rect=(0, 0.07, 1, 1))
fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.0, frameon=False, handletextpad=0.2, columnspacing=1.2,
           bbox_to_anchor=(0.5, -0.01))
for dd in (OUT, CR):
    if dd.exists():
        fig.savefig(dd / "fig_c11_twoterm.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote fig_c11_twoterm.png")
