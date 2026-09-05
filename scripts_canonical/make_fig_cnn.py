"""Figure: architectural ground truth (S77). (A) a drawn architecture panel: the
shared trunk applied to the four rotations of an input and averaged over the
orbit, with the read-out points; (B) delta_rot by layer for plain, pre-pool and
invariant features with the digit axis on the invariant features (run5b); (C) the
rotation-tuned variance fraction by depth against the raw-pixel baseline
(run5c, run14). Panels B and C are the S70 fig_mechanism.png panel (c) and its inset.
Out: ../figures_canonical/fig_cnn.png + ../../arxiv/figures/
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.2, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_cnn.png", HERE.parent.parent / "arxiv" / "figures" / "fig_cnn.png"]
j = lambda f: json.load(open(DATA / f)); run5b = j("run5b_cnn_seeds.json"); run5c = j("run5c_cnn_multipole_fixed.json"); run14 = j("run14_stimulus_baseline.json")
INK, GRAY, RED = "#222222", "#8a8a8a", "#c23b3b"; C_PLAIN, C_PRE, C_INV = "#bdbdbd", "#74a9cf", "#2b8cbe"
fig = plt.figure(figsize=(6.0, 2.55))
axA = fig.add_axes([0.005, 0.03, 0.34, 0.88]); axB = fig.add_axes([0.455, 0.19, 0.28, 0.68]); axC = fig.add_axes([0.83, 0.19, 0.16, 0.68])
fig.text(0.005, 0.95, "A", fontsize=8.5, fontweight="bold", va="center"); fig.text(0.035, 0.95, "the equivariant network", fontsize=7.5, va="center")
ax = axA; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
def box(x, y, w, h, text, fc="white", ec=GRAY, fs=6.2, col=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=col)
# input and its four rotations
box(0.3, 4.2, 1.3, 1.6, "digit", fc="#f7f7f7")
ys = [8.3, 6.2, 4.1, 2.0]; labels = ["0°", "90°", "180°", "270°"]
for y, lab in zip(ys, labels):
    ax.annotate("", xy=(2.4, y + 0.5), xytext=(1.6, 5.0), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.6, mutation_scale=6))
    box(2.4, y, 1.0, 1.0, lab, fc="#f7f7f7")
    for i, name in enumerate(["conv1", "conv2", "conv3"]):
        box(3.75 + i * 1.35, y, 1.15, 1.0, name, fc=C_PRE if True else "white", ec=C_PRE, col="white", fs=6.0)
    ax.annotate("", xy=(8.2, 5.35), xytext=(7.85 + 0.05, y + 0.5), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.6, mutation_scale=6))
ax.text(5.8, 9.6, "one shared trunk, weights tied across the four copies", fontsize=5.8, ha="center", color=INK)
box(8.2, 4.5, 1.6, 1.7, "mean over\nthe orbit", fc=C_INV, ec=C_INV, col="white", fs=5.8)
ax.text(9.0, 4.05, "invariant by\nconstruction", fontsize=5.6, ha="center", va="top", color=C_INV)
ax.text(5.8, 0.75, "read here: pre-pool features (equivariant)", fontsize=5.8, ha="center", color=C_PRE)
ax.text(9.0, 6.75, "read here:\ninvariant", fontsize=5.6, ha="center", va="bottom", color=C_INV)
ax.text(0.3, 3.75, "plain CNN: one\ncopy, no pooling", fontsize=5.6, ha="left", va="top", color=GRAY)
# ---- (B) delta_rot by layer ----
ax = axB; layers = ["conv1", "conv2", "conv3"]
groups = [("plain", ["plain/conv1", "plain/conv2", "plain/conv3"], C_PLAIN),
          ("equivariant, pre-pool", ["equivariant/conv1_equi", "equivariant/conv2_equi", "equivariant/conv3_equi"], C_PRE),
          ("invariant", ["equivariant/conv1_inv", "equivariant/conv2_inv", "equivariant/conv3_inv"], C_INV)]
w = 0.26
for gi, (glab, keys, c) in enumerate(groups):
    ax.bar(np.arange(3) + (gi - 1) * w, [run5b[k]["delta_rot_mean"] for k in keys], w, yerr=[run5b[k]["delta_rot_sd"] for k in keys], capsize=2, color=c, label=glab, error_kw={"lw": 0.8})
dig = run5b["equivariant/conv3_inv"]
ax.errorbar([2 + w + 0.16], [dig["delta_digit_mean"]], yerr=[dig["delta_digit_sd"]], fmt="D", ms=5, color=RED, label=r"$\delta_{\mathrm{digit}}$ on invariant")
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(np.arange(3)); ax.set_xticklabels(layers)
ax.set_ylabel(r"$\delta_{\mathrm{rot}}$ (5 seeds)"); ax.set_ylim(-0.035, 0.16)
ax.legend(fontsize=5.8, frameon=False, loc="upper left", handlelength=1.2)
ax.set_title("B  zero where invariance is built in", loc="left", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
# ---- (C) tuned fraction ----
ax = axC
tf = lambda key: [run5c[key[0]][key[1].format(l)]["tuned_fraction"] for l in layers]
ax.plot([0, 1, 2], [run5c["plain"][l]["tuned_fraction"] for l in layers], "o-", ms=3, lw=1.1, color=C_PLAIN, label="plain")
ax.plot([0, 1, 2], [run5c["equivariant"][f"{l}_equi"]["tuned_fraction"] for l in layers], "o-", ms=3, lw=1.1, color=C_PRE, label="pre-pool")
ax.plot([0, 1, 2], [run5c["equivariant"][f"{l}_inv"]["tuned_fraction"] for l in layers], "o-", ms=3, lw=1.1, color=C_INV, label="invariant")
ax.axhline(run14["raw_pixels"]["tuned_fraction"], color=RED, lw=0.8, ls=":"); ax.text(0.02, run14["raw_pixels"]["tuned_fraction"] * 1.25, "raw pixels", fontsize=5.8, color=RED)
ax.set_yscale("log"); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(layers); ax.set_ylabel("tuned variance fraction", labelpad=1)
ax.set_title("C  inherited", loc="left", fontweight="bold"); ax.spines[["top", "right"]].set_visible(False)
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
