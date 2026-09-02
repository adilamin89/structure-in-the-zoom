"""Figure 5 (mechanism) for the arXiv version. Three panels, all from
committed JSONs (zero new computation):
  (a) principal-angle alignment vs class separation at 22.5 deg resolution
      (run9, 3 drifting recordings) with the covariance-preserving
      permutation-null band (run11) and the residualized-control points
      (run3b, GT3).
  (b) calibrated-model delta bars: shared modes / class-dependent modes /
      alignment-calibrated prediction (run2b, run11b), with GT3 observed
      marked. The prediction's overshoot is shown, not hidden.
  (c) CNN double dissociation by layer (run5b): delta_rot for plain,
      equivariant pre-pool, and invariant features; delta_digit on invariant
      conv3. Inset: rotation-tuned variance fraction by depth (run5c) with
      the stimulus baseline (run14).
Out: ../figures_canonical/fig_mechanism.png + ../../arxiv/figures/
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 6.8, "savefig.dpi": 300})
# figsize equals the printed width (jmlr textwidth 6.0 in) so fonts print 1:1
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_mechanism.png",
        HERE.parent.parent / "arxiv" / "figures" / "fig_mechanism.png"]

j = lambda f: json.load(open(DATA / f))
run9 = j("run9_alignment_225.json")
run3b = j("run3b_principal_angles_residualized.json")
run2b = j("run2b_corotating_seeds.json")
run11 = j("run11_bootstrap_prediction.json")
run11b = j("run11b_fresh_draw_prediction.json")
run5b = j("run5b_cnn_seeds.json")
run5c = j("run5c_cnn_multipole_fixed.json")
run14 = j("run14_stimulus_baseline.json")

fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.45),
                         gridspec_kw={"width_ratios": [0.95, 0.85, 1.20]})
plt.subplots_adjust(wspace=0.45, left=0.085, right=0.99, top=0.90,
                    bottom=0.20)

# ---------------- (a) alignment profile ----------------
ax = axes[0]
colors = {"GT1": "#c23b3b", "GT2": "#e08214", "GT3": "#7b3294"}
for rec in ["GT1", "GT2", "GT3"]:
    prof = run9[rec]["profile"]
    xs = [float(k) for k in prof]
    ys = [prof[k] for k in prof]
    ax.plot(xs, ys, "o-", ms=4, lw=1.4, color=colors[rec], label=rec)
# residualized control, GT3 (45-deg resolution)
res = run3b["GT3"]["residualized"]
ax.plot([float(k) for k in res], [res[k] for k in res], "s--", ms=5,
        lw=1.0, color="#7b3294", alpha=0.55, label="GT3 residualized")
# permutation-null band (range of angular modulation under
# covariance-preserving label permutation; centered at the profile floor)
null_mu, null_sd = run11["perm_null_range_mean"], run11["perm_null_range_sd"]
floor = min(min(run9[r]["profile"].values()) for r in ["GT1", "GT2", "GT3"])
ax.axhspan(floor, floor + null_mu + 2 * null_sd, color="0.85", zorder=0)  # described in the caption
ax.set_xlabel(r"class separation $\Delta\phi$ (deg)")
ax.set_ylabel("within-class subspace alignment")
ax.set_xlim(15, 185)
ax.set_ylim(0.02, 0.28)  # headroom for the legend above the 22.5-deg points
ax.set_xticks([45, 90, 135, 180])  # 22.5 is the first data point; labeling it collides with 45
ax.legend(fontsize=6.0, frameon=False, loc="upper center", ncol=2, columnspacing=0.9)
ax.set_title("(a) $\\pi$-periodic alignment")

# ---------------- (b) model bracket ----------------
ax = axes[1]
bars = [
    ("shared\nmodes", run2b["shared"]["mean"], run2b["shared"]["sd"],
     "#9ecae1"),
    ("class-dep.\nmodes", run2b["corotating"]["mean"],
     run2b["corotating"]["sd"], "#4292c6"),
    ("align.\ncalib.", run11b["boot_mean"],
     (run11b["boot_ci95"][1] - run11b["boot_ci95"][0]) / 2, "#08519c"),
]
xs = np.arange(len(bars))
for i, (lab, mu, err, c) in enumerate(bars):
    ax.bar(i, mu, 0.62, color=c, yerr=err, capsize=3,
           error_kw={"lw": 1.0})
ax.axhline(0.237, color="#c23b3b", lw=1.6, ls="-")
ax.text(-0.42, 0.226, "GT3 $+0.237$", color="#c23b3b", fontsize=6.8,
        ha="left", va="top")  # below the line, over the short shared-modes bar
ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in bars], fontsize=6.2)
ax.set_ylabel(r"model $\delta$")
ax.set_ylim(0, 0.56)
ax.set_title("(b) two knobs bracket GT3")
ax.annotate("overshoot\n(known defect)", xy=(2, run11b["boot_mean"]),
            xytext=(1.15, 0.44), fontsize=6.8,
            arrowprops={"arrowstyle": "->", "lw": 0.8})

# ---------------- (c) CNN dissociation ----------------
ax = axes[2]
layers = ["conv1", "conv2", "conv3"]
groups = [("plain", ["plain/conv1", "plain/conv2", "plain/conv3"],
           "#bdbdbd"),
          ("equiv. pre-pool",
           ["equivariant/conv1_equi", "equivariant/conv2_equi",
            "equivariant/conv3_equi"], "#74a9cf"),
          ("invariant",
           ["equivariant/conv1_inv", "equivariant/conv2_inv",
            "equivariant/conv3_inv"], "#2b8cbe")]
w = 0.26
for gi, (glab, keys, c) in enumerate(groups):
    mus = [run5b[k]["delta_rot_mean"] for k in keys]
    sds = [run5b[k]["delta_rot_sd"] for k in keys]
    ax.bar(np.arange(3) + (gi - 1) * w, mus, w, yerr=sds, capsize=2,
           color=c, label=glab, error_kw={"lw": 0.8})
# digit-axis marker on invariant conv3
dig = run5b["equivariant/conv3_inv"]
ax.errorbar([2 + w + 0.16], [dig["delta_digit_mean"]],
            yerr=[dig["delta_digit_sd"]], fmt="D", ms=5, color="#c23b3b",
            label=r"$\delta_{\mathrm{digit}}$ (invar.)")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(layers)
ax.set_ylabel(r"$\delta_{\mathrm{rot}}$ (5 seeds)")
ax.set_ylim(-0.035, 0.30)  # headroom so the inset sits above the bars
ax.legend(fontsize=5.8, frameon=False, loc="upper left", handlelength=1.2)
ax.set_title("(c) architectural null")

# inset: tuned fraction by depth + stimulus baseline
ins = ax.inset_axes([0.60, 0.56, 0.38, 0.32])  # above the bars, right of the legend
tf_plain = [run5c["plain"][l]["tuned_fraction"] for l in layers]
tf_equi = [run5c["equivariant"][f"{l}_equi"]["tuned_fraction"]
           for l in layers]
tf_inv = [run5c["equivariant"][f"{l}_inv"]["tuned_fraction"] for l in layers]
ins.plot([0, 1, 2], tf_plain, "o-", ms=3, lw=1, color="#bdbdbd")
ins.plot([0, 1, 2], tf_equi, "o-", ms=3, lw=1, color="#74a9cf")
ins.plot([0, 1, 2], tf_inv, "o-", ms=3, lw=1, color="#2b8cbe")
ins.axhline(run14["raw_pixels"]["tuned_fraction"], color="#c23b3b", lw=0.8,
            ls=":")
ins.text(0.02, run14["raw_pixels"]["tuned_fraction"] * 1.25, "pixels",
         fontsize=5.5, color="#c23b3b")
ins.set_yscale("log")
ins.set_xticks([0, 1, 2])
ins.set_xticklabels(["c1", "c2", "c3"], fontsize=5.5)
ins.tick_params(labelsize=5.5)
ins.set_title("tuned var. fraction", fontsize=5.8)

for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists():
        continue  # secondary copy only when the paper tree is present
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("wrote", out)
