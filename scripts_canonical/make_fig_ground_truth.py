"""Figure (Session F): the ground truth of Section 6.1 in one panel each.
(A) 2D Ising, the uncentered signed-sort shift against T/T_c for L = 32, 64, 128: the critical value varies slowly
(0.40, 0.35, 0.33) while the off-critical values collapse (0.15, 0.08, 0.04 at 0.95 T_c); the centered estimator at
T_c, -0.47 (L = 32), marked. (B) XY nematic with polar coupling J1/J2 = 0.42: the centered shift against T at L = 32
and 64, positive near the nematic transition and zero in the disordered phase; the ordered side is sampling-limited
(director frozen within a chain; open markers). The nematic order parameter S2 (gray, right axis) locates the
transition. (C) the two-species XY lattice: the share of the ladder's climb left after four classes against the
polar fraction f at T = 1.2 and 1.4; filled markers have five or more valid chains, open fewer (the polar phase
freezes the director), the count printed.
Sources: ising_wolff_matrix.json, ising_L128_tc.json, nematic_polar_delta.json, run62_two_species_xy.json.
Out: ../figures_canonical/fig_ground_truth.png + ../../arxiv/figures/fig_ground_truth.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.2, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_ground_truth.png", HERE.parent.parent / "arxiv" / "figures" / "fig_ground_truth.png"]
INK, RED, BLUE, GRAY, ORANGE, PURPLE = "#222222", "#b40426", "#2166ac", "#8a8a8a", "#e08214", "#756bb1"

fig = plt.figure(figsize=(6.0, 2.3))
axA = fig.add_axes([0.075, 0.20, 0.255, 0.64]); axB = fig.add_axes([0.415, 0.20, 0.235, 0.64]); axC = fig.add_axes([0.79, 0.20, 0.20, 0.64])
for ax, letter, title in ((axA, "A", "Ising: conditioning on $m$"), (axB, "B", "XY nematic: the orbit"), (axC, "C", "two species: the stall")):
    ax.set_title(f"$\\mathbf{{{letter}}}$  {title}", loc="left", fontsize=7.6); ax.spines[["top", "right"]].set_visible(False)

# ---- (A) Ising ----
w = json.load(open(DATA / "ising_wolff_matrix.json")); l128 = json.load(open(DATA / "ising_L128_tc.json"))
rows = w["results"] + l128["results"]
cols = {32: "#9ecae1", 64: BLUE, 128: INK}
for L in (32, 64, 128):
    r = sorted([x for x in rows if x["L"] == L], key=lambda x: x["T_over_Tc"])
    t = [x["T_over_Tc"] for x in r]; m = [x["uncentered_signed"]["mean"] for x in r]; s = [x["uncentered_signed"]["sd"] for x in r]
    axA.errorbar(t, m, yerr=s, fmt="o-", ms=3.0, lw=1.2, color=cols[L], capsize=1.5, label=f"$L = {L}$")
cen = [x for x in w["results"] if x["L"] == 32 and abs(x["T_over_Tc"] - 1) < 1e-6][0]["centered_absm"]["mean"]
axA.plot([1.0], [cen], "D", ms=4.2, color=RED, mfc="white", mew=1.2, zorder=5)
axA.text(1.06, cen, "centered estimator\nat $T_c$: $-0.47$", fontsize=5.8, color=RED, va="center", ha="left")
axA.axvline(1.0, color="0.6", lw=0.7, ls=(0, (3, 2))); axA.axhline(0, color="k", lw=0.5)
axA.set_xlabel("$T / T_c$"); axA.set_ylabel("shift $\\delta$ (uncentered, signed sort)", labelpad=1)
axA.set_xlim(0.55, 1.85); axA.set_ylim(-0.6, 0.62)
axA.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.2, labelspacing=0.3)

# ---- (B) XY nematic ----
n = json.load(open(DATA / "nematic_polar_delta.json"))["results"]
for L, col in ((32, "#9ecae1"), (64, BLUE)):
    r = sorted([x for x in n if x["L"] == L], key=lambda x: x["T"])
    for x in r:
        frozen = x["T"] < 1.0
        if frozen:
            axB.plot([x["T"]], [x["delta_mean"]], "o", ms=3.2, color=col, mfc="white", mew=1.0, alpha=0.6, clip_on=True)
        else:
            axB.errorbar([x["T"]], [x["delta_mean"]], yerr=[x["delta_sd"]], fmt="o", ms=3.2, color=col, mfc=col, mew=1.0, capsize=1.5, lw=0.9)
    keep = [x for x in r if x["T"] >= 1.0]
    axB.plot([x["T"] for x in keep], [x["delta_mean"] for x in keep], "-", lw=1.2, color=col, label=f"$L = {L}$")
axB.axhline(0, color="k", lw=0.5)
axB.set_xlabel("temperature $T$ ($J_2 = 1$)"); axB.set_ylabel("shift $\\delta$ (centered)", labelpad=1)
axB.set_xlim(0.1, 2.1); axB.set_ylim(-0.32, 0.30)
axB2 = axB.twinx(); axB2.spines[["top", "left"]].set_visible(False)
r32 = sorted([x for x in n if x["L"] == 32], key=lambda x: x["T"])
axB2.plot([x["T"] for x in r32], [x["S2"] for x in r32], "-", color=GRAY, lw=0.9, alpha=0.8)
axB2.set_ylim(0, 1.0); axB2.tick_params(axis="y", colors=GRAY, labelsize=6.5)
axB2.set_ylabel("nematic order $S_2$ ($L = 32$)", color=GRAY, fontsize=6.5, labelpad=2)
axB.text(0.13, -0.30, "open: director frozen\n(sampling-limited)", fontsize=5.4, color="0.35", va="bottom")
axB.legend(frameon=False, loc="upper right", handlelength=1.4, borderaxespad=0.2, labelspacing=0.3)

# ---- (C) two-species stall ----
r62 = json.load(open(DATA / "run62_two_species_xy.json"))["cells"]
for T, col, mk in ((1.2, INK, "o"), (1.4, ORANGE, "s")):
    cells = sorted([c for c in r62 if abs(c["T"] - T) < 1e-9 and c["late_fraction"] is not None], key=lambda c: c["f"])
    dy = -0.075 if T == 1.2 else 0.03
    for c in cells:
        solid = c["n_valid"] >= 5
        axC.plot(c["f"], c["late_fraction"], mk, ms=4.2, color=col, mfc=col if solid else "white", mew=1.0)
        axC.text(c["f"] + 0.035, c["late_fraction"] + dy, str(c["n_valid"]), fontsize=5.4, color=col, va="bottom")
    axC.plot([], [], mk, color=col, label=f"$T = {T}$")
axC.axhline(0, color="k", lw=0.5)
axC.set_xlabel("polar fraction $f$"); axC.set_ylabel("climb left after four classes", labelpad=1)
axC.set_xlim(-0.08, 1.12); axC.set_ylim(-0.12, 1.05); axC.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); axC.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
axC.legend(frameon=False, loc="upper left", handlelength=1.2, borderaxespad=0.2, labelspacing=0.3)
axC.text(0.05, 0.36, "filled: $\\geq 5$ valid chains,\ncount printed in the\nseries' color", transform=axC.transAxes, fontsize=5.6, color="0.35", ha="left", va="center")

for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
