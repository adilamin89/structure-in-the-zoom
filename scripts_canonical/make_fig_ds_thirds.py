"""Figure (S78): the direction-selective third and the rung-four stall (runs 59 and 59b).
(A) the per-rung log-PR deficit below the floor on drifting GT1 for the three DSI-sorted
thirds and the full tuned population; (B) the share of the climb that remains after four
classes, per third, on all eight grating recordings; (C) the full-population shift against
the direction-selective fraction across the eight recordings.
Out: ../figures_canonical/fig_ds_thirds.png + ../../arxiv/figures/fig_ds_thirds.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.3, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_ds_thirds.png",
        HERE.parent.parent / "arxiv" / "figures" / "fig_ds_thirds.png"]
r59b = json.load(open(DATA / "run59b_per_rung_thirds.json"))
r59 = json.load(open(DATA / "run59_shift_by_direction_selectivity.json"))
rows = r59b["rows"]; tags = list(rows)
RED, BLUE, GRAY, INK = "#b40426", "#2166ac", "#8a8a8a", "#222222"
COL = {"DS": RED, "random": GRAY, "nonDS": BLUE}
LAB = {"DS": "direction-selective third", "random": "random third", "nonDS": "orientation-only third"}
MK = {"DS": "o", "random": "D", "nonDS": "s"}
CLASSES = [1, 2, 3, 4, 6, 8]

fig = plt.figure(figsize=(6.0, 2.4))
axA = fig.add_axes([0.075, 0.19, 0.31, 0.66])
axB = fig.add_axes([0.455, 0.19, 0.235, 0.66])
axC = fig.add_axes([0.775, 0.19, 0.215, 0.66])
for ax, letter, title in ((axA, "A", "the rung-four stall, drifting GT1"),
                          (axB, "B", "share of the climb after four classes"),
                          (axC, "C", "the shift tracks the DS fraction")):
    ax.set_title(f"$\\mathbf{{{letter}}}$  {title}", loc="left", fontsize=7.6)
    ax.spines[["top", "right"]].set_visible(False)

# ---- (A) per-rung deficit on GT1 ----
row = rows["D1"]
axA.axvspan(4, 8.4, color="0.94", zorder=0)
axA.axvline(4, color="0.55", lw=0.7, ls=(0, (3, 2)), zorder=1)
axA.axhline(0, color="k", lw=0.6, alpha=0.5, zorder=1)
axA.plot(CLASSES, row["full"]["deficit"], "-", color=INK, lw=0.9, alpha=0.55, label="all tuned neurons", zorder=2)
for s in ("DS", "random", "nonDS"):
    axA.plot(CLASSES, row["subsets"][s]["deficit"], MK[s] + "-", ms=3.3, lw=1.4, color=COL[s], label=LAB[s], zorder=3)
y4 = row["subsets"]["nonDS"]["deficit"][3]
axA.annotate("an even code has every class\nmean it will have by four classes:\nthe orientation-only ladder stalls",
             xy=(4, y4), xytext=(1.0, 0.09), fontsize=6.2, color=BLUE, va="top",
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7, shrinkB=2))
axA.text(6.0, -0.985, "late rungs", ha="center", va="bottom", fontsize=6.4, color="0.35")
axA.set_xticks(CLASSES); axA.set_xlim(0.6, 8.4); axA.set_ylim(-1.0, 0.36)
axA.set_xlabel("classes accumulated"); axA.set_ylabel("log PR below the floor")
axA.legend(loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.2)

# ---- (B) late fraction per third, eight recordings ----
x = np.arange(len(tags))
for i, t in enumerate(tags):
    a, b = rows[t]["subsets"]["DS"]["late_fraction"], rows[t]["subsets"]["nonDS"]["late_fraction"]
    axB.plot([i, i], [b, a], "-", color="0.75", lw=0.8, zorder=1)
for s in ("DS", "random", "nonDS"):
    axB.plot(x, [rows[t]["subsets"][s]["late_fraction"] for t in tags], MK[s], ms=3.6, color=COL[s],
             mec="white", mew=0.4, label=LAB[s], zorder=3, ls="none")
axB.axhline(0, color="k", lw=0.6, alpha=0.5)
axB.set_xticks(x); axB.set_xticklabels(tags); axB.set_ylim(-0.12, 1.12)
axB.set_xlabel("grating recording"); axB.set_ylabel("deficit at four classes / at one class")
axB.text(0.02, 0.97, "D drifting  L localized  C low contrast", transform=axB.transAxes, fontsize=5.9, va="top", color="0.35")

# ---- (C) full shift vs DS fraction ----
v = r59["verdict"]; fr = np.array(v["ds_fraction"]); df = np.array(v["delta_full"])
rho = spearmanr(fr, df)[0]
mark = {"D": "o", "L": "^", "C": "s"}
for i, t in enumerate(v["tags"]):
    axC.plot(fr[i], df[i], mark[t[0]], ms=4.2, color=INK, mfc=RED if t[0] == "L" else ("white" if t[0] == "C" else INK), mew=0.8)
    dx, dy = (0.012, 0.004) if t not in ("C1",) else (0.012, -0.014)
    axC.text(fr[i] + dx, df[i] + dy, t, fontsize=6.2, color="0.25", va="center")
axC.text(0.03, 0.96, f"Spearman $\\rho$ = {rho:.2f}", transform=axC.transAxes, fontsize=6.6, va="top")
axC.set_xlabel("direction-selective fraction of tuned neurons")
axC.set_ylabel("direction-aligned shift $\\delta$")
axC.set_xlim(0.32, 0.86); axC.set_ylim(0.15, 0.50)

for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists():
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("wrote", out)
