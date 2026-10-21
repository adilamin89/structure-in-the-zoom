"""Camera-ready Figure 3: the two measured numbers behind the shift, and the order effect's fluctuation part.
(A) within-class part of the direction shift (Table 9) against the mean-field growth ln[8/(1+7 sigma)]/ln 8, sigma = the
    split-half overlap of two classes' fluctuations relative to a class's own (cr_overlap_mining.json, a.raw.sigma);
(B) class-mean part (Table 9) against Tr B / Tr W, the class means' variance relative to the fluctuations' (cr_overlap_mining.json);
(C) within-class order effect averaged over sixteen origins (cr_order_effect_origins.json) against
    dK = Kn(22.5) - Kn(180), adjacent-minus-antipodal fluctuation overlap (cr_failure_interrogation.json).
Table 9 values: cr_order_and_withinbin_controls.json "orig" (the run Table 9 prints). CHECK lines assert printed numbers.
Out: ../figures_canonical/fig_c6_laws.png (+ camera-ready figures/)."""
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
TAGS = ["D1", "D2", "D3", "L1", "L2", "L3", "C1", "C2"]
COL = {"D": "k", "L": "#c51b7d", "C": "#8a8a8a"}

t9 = [r["orig"] for r in sorted(json.load(open(DATA / "cr_order_and_withinbin_controls.json"))["rows"], key=lambda r: r["name"])]
om = sorted(json.load(open(DATA / "cr_overlap_mining.json"))["rows"], key=lambda r: r["name"])
fi = sorted(json.load(open(DATA / "cr_failure_interrogation.json"))["rows"], key=lambda r: r["name"])
oo = sorted(json.load(open(DATA / "cr_order_effect_origins.json"))["rows"], key=lambda r: r["name"])
within = np.array([r["within"] for r in t9]); means = np.array([r["class_means"] for r in t9])
sigma = np.array([r["a"]["raw"]["sigma"] for r in om]); bw = np.array([r["a"]["trB_over_trW"] for r in om])
pred = np.log(8 / (1 + 7 * sigma)) / np.log(8)
dK = np.array([r["dK"] for r in fi]); oew = np.array([np.mean([o["within"] for o in r["origins"]]) for r in oo])
r_a, r_b, r_c = stats.pearsonr(pred, within)[0], stats.pearsonr(bw, means)[0], stats.pearsonr(dK, oew)[0]
print(f"CHECK within vs mean field r = {r_a:.2f} (paper 0.92), MAE {np.abs(pred - within).mean():.3f} (paper 0.054)")
print(f"CHECK class-mean part vs Tr B/Tr W r = {r_b:.2f} (paper -0.91); B/W range {bw.min():.2f}-{bw.max():.2f} (paper 0.07-0.28)")
print(f"CHECK order fluctuation part vs dK r = {r_c:.3f} (paper -0.997)")
assert round(r_a, 2) == 0.92 and round(r_b, 2) == -0.91 and round(r_c, 3) == -0.997

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.0))
panels = [(pred, within, "mean field $\\ln[8/(1+7\\sigma)]/\\ln 8$", "within-class part", f"$\\mathbf{{A}}$  one overlap, $r = {r_a:.2f}$"),
          (bw, means, "Tr$\\,B$/Tr$\\,W$ (class means / fluctuations)", "class-mean part", f"$\\mathbf{{B}}$  class means, $r = {r_b:.2f}$"),
          (dK, oew, "adjacent $-$ antipodal overlap", "order effect (fluctuations)", f"$\\mathbf{{C}}$  order, $r = {r_c:.3f}$")]
OFF = {(0, "L1"): (-11, -6), (0, "L3"): (-11, 2), (1, "L2"): (-11, -3), (2, "L2"): (-11, -6)}   # labels clear of neighbours
for k, (ax, (x, y, xl, yl, ti)) in enumerate(zip(axs, panels)):
    for t, xi, yi in zip(TAGS, x, y):
        ax.plot(xi, yi, "o", color=COL[t[0]], ms=3.8)
        ax.annotate(t, (xi, yi), xytext=OFF.get((k, t), (3, 2)), textcoords="offset points", fontsize=5.8, color=COL[t[0]])
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ti, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
lo, hi = 0.2, 0.72
axs[0].plot([lo, hi], [lo, hi], color="0.6", lw=0.7, ls="--"); axs[0].text(0.62, 0.66, "equality", fontsize=5.8, color="0.45", rotation=34)
axs[2].axhline(0, color="0.6", lw=0.6)
fig.tight_layout(w_pad=1.0)
for d in (OUT, CR):
    if d.exists():
        fig.savefig(d / "fig_c6_laws.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote fig_c6_laws.png")
