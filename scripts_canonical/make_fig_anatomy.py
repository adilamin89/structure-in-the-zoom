"""Figure (S78b): the anatomy of the deficit (runs 63 and 64a).
(A) The four exact terms of the per-rung deficit on drifting GT1 (all tuned neurons): the
trace term, the fluctuation-pooling term, the between-class term and the mean-fluctuation
coupling; (B) the pooling term of the direction-selective third against log(k/8), the value
for fluctuations fully private to each class, on all eight recordings; (C) where the
within-class modes live in neuron space: the share of the tuned population carrying the top
within-class mode, per recording, in the recording's own units (where its amplitude sits) and at
equal weight per neuron (its extent; run 68).
Out: ../figures_canonical/fig_anatomy.png + ../../arxiv/figures/fig_anatomy.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.3, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_anatomy.png", HERE.parent.parent / "arxiv" / "figures" / "fig_anatomy.png"]
dec = json.load(open(DATA / "run63_deficit_decomposition.json"))["rows"]
loc = json.load(open(DATA / "run64a_within_class_localization.json"))["rows"]
tags = list(dec)
RED, BLUE, GRAY, INK, GREEN, ORANGE = "#b40426", "#2166ac", "#8a8a8a", "#222222", "#3d7d1f", "#e08214"
CLASSES = [1, 2, 3, 4, 6, 8]

fig = plt.figure(figsize=(6.0, 2.35))
axA = fig.add_axes([0.095, 0.19, 0.285, 0.66]); axB = fig.add_axes([0.455, 0.19, 0.235, 0.66]); axC = fig.add_axes([0.775, 0.19, 0.215, 0.66])
for ax, letter, title in ((axA, "A", "the deficit, term by term (GT1)"), (axB, "B", "private fluctuations: $\\log(k/8)$"), (axC, "C", "where the modes live")):
    ax.set_title(f"$\\mathbf{{{letter}}}$  {title}", loc="left", fontsize=7.6); ax.spines[["top", "right"]].set_visible(False)

# (A) the four terms on GT1, all tuned neurons
rungs = dec["D1"]["data"]["full"]["rungs"]
axA.axhline(0, color="k", lw=0.6, alpha=0.5)
axA.plot(CLASSES, [r["deficit"] for r in rungs], "o-", color=INK, lw=1.6, ms=3.4, label="deficit")
axA.plot(CLASSES, [r["P"] for r in rungs], "s-", color=RED, lw=1.3, ms=3.0, label="fluctuation pooling $P$")
axA.plot(CLASSES, [r["Cb"] for r in rungs], "^-", color=BLUE, lw=1.1, ms=3.0, label="between-class $C_b$")
axA.plot(CLASSES, [r["Cx"] for r in rungs], "v-", color=GREEN, lw=1.1, ms=3.0, label="coupling $C_x$")
axA.plot(CLASSES, [r["A"] for r in rungs], "D-", color=GRAY, lw=1.0, ms=2.6, label="trace $A$")
axA.set_xticks(CLASSES); axA.set_xlim(0.6, 8.4); axA.set_xlabel("classes accumulated"); axA.set_ylabel("log PR below the floor")
axA.legend(loc="lower right", frameon=False, handlelength=1.5, labelspacing=0.25, borderaxespad=0.2)

# (B) P of the direction-selective third against log(k/8)
ref = np.log(np.array(CLASSES) / 8.0)
axB.plot([-2.2, 0.05], [-2.2, 0.05], "-", color="0.6", lw=0.8)
cols = {"D": INK, "L": RED, "C": ORANGE}; mk = {"D": "o", "L": "^", "C": "s"}
for t in tags:
    P = [r["P"] for r in dec[t]["data"]["DS"]["rungs"]]
    axB.plot(ref, P, mk[t[0]], ms=3.4, color=cols[t[0]], mfc=cols[t[0]] if t[0] != "C" else "white", mew=0.8, alpha=0.9, label=t)
axB.text(-2.15, -0.02, "each class brings\nits own subspace:\n$P = \\log(k/8)$", fontsize=5.6, va="top", color="0.35")
axB.set_xlabel("$\\log(k/8)$, $k$ classes accumulated"); axB.set_ylabel("pooling term $P$, DS third")
axB.set_xlim(-2.2, 0.1); axB.set_ylim(-2.2, 0.1)
axB.text(0.98, 0.03, "D drifting  L localized\nC low contrast", transform=axB.transAxes, fontsize=5.8, ha="right", va="bottom", color="0.35")

# (C) the share of the population carrying the top within-class mode: the recording's own units (filled; where
# the amplitude sits) and equal weight per neuron (open; the extent), from runs 64a and 68
z = json.load(open(DATA / "run68_v1_scale_and_zscore.json"))["rows"]
x = np.arange(len(tags))
own = [100 * loc[t]["data"]["summary"]["pr_neuron_top1_mean"] / loc[t]["data"]["summary"]["N"] for t in tags]
eq = [100 * z[t]["localization_zscored"]["pr_neuron_top1_mean"] / z[t]["N"] for t in tags]
for i, t in enumerate(tags):
    axC.plot([i, i], [own[i], eq[i]], "-", color="0.75", lw=0.8, zorder=1)
    axC.plot(i, own[i], mk[t[0]], ms=4.0, color=cols[t[0]], mfc=cols[t[0]], mew=0.8, zorder=3)
    axC.plot(i, eq[i], mk[t[0]], ms=4.0, color=cols[t[0]], mfc="white", mew=0.8, zorder=3)
axC.set_yscale("log"); axC.set_ylim(0.1, 100)
axC.set_yticks([0.1, 1, 10, 100]); axC.set_yticklabels(["0.1%", "1%", "10%", "100%"])
axC.set_xticks(x); axC.set_xticklabels(tags); axC.set_ylabel("neurons carrying the top mode", fontsize=7)
axC.set_xlabel("grating recording")
axC.text(0.03, 0.97, "open: equal weight per neuron\nfilled: the recording's units", transform=axC.transAxes, fontsize=5.8, va="top", color="0.35")
axC.text(0.97, 0.03, "of 10,000$-$21,000\ntuned neurons", transform=axC.transAxes, fontsize=5.6, ha="right", va="bottom", color="0.35")

for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists():
        continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
