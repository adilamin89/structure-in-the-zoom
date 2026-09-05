"""Figure: what sets the sign (S77). (A) a drawn picture of class means on a circle
with their within-class clouds in three regimes: low-rank clouds along an orbit
(delta > 0), isotropic clouds (delta < 0), an invariant representation (delta = 0);
(B) the measured within-class subspace alignment against class separation (run9,
run3b, run11), panel (a) of the S70 fig_mechanism.png.
Out: ../figures_canonical/fig_sign_rule.png + ../../arxiv/figures/
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.2, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_sign_rule.png", HERE.parent.parent / "arxiv" / "figures" / "fig_sign_rule.png"]
j = lambda f: json.load(open(DATA / f)); run9 = j("run9_alignment_225.json"); run3b = j("run3b_principal_angles_residualized.json"); run11 = j("run11_bootstrap_prediction.json")
CLASS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#9c755f"]
INK, GRAY = "#222222", "#8a8a8a"
fig = plt.figure(figsize=(6.0, 2.45))
axA = fig.add_axes([0.005, 0.02, 0.60, 0.86]); axB = fig.add_axes([0.715, 0.20, 0.275, 0.66])
fig.text(0.005, 0.95, "A", fontsize=8.5, fontweight="bold", va="center"); fig.text(0.035, 0.95, "the within-class geometry sets the sign", fontsize=7.5, va="center")
ax = axA; ax.set_xlim(0, 18); ax.set_ylim(0, 7.2); ax.set_aspect("equal"); ax.axis("off")
def regime(cx, cy, r, mode, title, verdict, colv):
    ax.add_patch(Circle((cx, cy), r, fill=False, ec=GRAY, lw=0.6, ls=(0, (2, 2))))
    for k in range(8):
        a = k * np.pi / 4
        if mode == "invariant":
            x, y = cx, cy
        else:
            x, y = cx + r * np.cos(a), cy + r * np.sin(a)
        if mode == "lowrank":
            ax.add_patch(Ellipse((x, y), 1.25, 0.28, angle=np.degrees(a) + 90, fc=CLASS[k], ec="none", alpha=0.45))
        elif mode == "isotropic":
            ax.add_patch(Circle((x, y), 0.55, fc=CLASS[k], ec="none", alpha=0.35))
        elif mode == "invariant":
            ax.add_patch(Circle((x, y), 0.8, fc="0.75", ec="none", alpha=0.35))
        ax.plot([x], [y], "o", color=CLASS[k] if mode != "invariant" else "0.3", ms=2.8, zorder=5)
    ax.text(cx, cy + r + 1.05, title, ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.text(cx, cy - r - 0.5, verdict, ha="center", va="top", fontsize=6.3, color=colv, fontweight="bold", linespacing=1.05)
regime(3.0, 3.6, 1.7, "lowrank", "orbit of class means,\nlow-rank clouds", "$\\delta > 0$: each class adds\na direction the floor\naverages away", "#b40426")
regime(9.0, 3.6, 1.7, "isotropic", "same means,\nisotropic clouds", "$\\delta < 0$: the structured\nrung is the more\ndiffuse one", "#2166ac")
regime(15.0, 3.6, 1.7, "invariant", "representation invariant\nto the axis", "$\\delta = 0$: nothing\nfor the ladder\nto read", "0.35")
# ---- (B) alignment profile ----
ax = axB; colors = {"GT1": "#c23b3b", "GT2": "#e08214", "GT3": "#7b3294"}
for rec in ["GT1", "GT2", "GT3"]:
    prof = run9[rec]["profile"]; ax.plot([float(k) for k in prof], [prof[k] for k in prof], "o-", ms=3.5, lw=1.3, color=colors[rec], label=rec)
res = run3b["GT3"]["residualized"]; ax.plot([float(k) for k in res], [res[k] for k in res], "s--", ms=4.5, lw=1.0, color="#7b3294", alpha=0.55, label="GT3 residualized")
null_mu, null_sd = run11["perm_null_range_mean"], run11["perm_null_range_sd"]
floor = min(min(run9[r]["profile"].values()) for r in ["GT1", "GT2", "GT3"])
ax.axhspan(floor, floor + null_mu + 2 * null_sd, color="0.85", zorder=0)
ax.set_xlabel(r"class separation $\Delta\phi$ (deg)"); ax.set_ylabel("within-class subspace alignment")
ax.set_xlim(15, 185); ax.set_ylim(0.02, 0.29); ax.set_xticks([45, 90, 135, 180])
ax.legend(fontsize=5.8, frameon=False, loc="upper center", ncol=2, columnspacing=0.8)
ax.set_title("B  measured alignment", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
