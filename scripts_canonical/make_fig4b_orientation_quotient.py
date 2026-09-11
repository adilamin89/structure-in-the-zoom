"""Schematic for Section 4 (paper Figure 4): the direction circle and its orientation
quotient; one tuning curve split into its even (orientation) and odd (direction)
sectors; the two accumulation orders as paths on the circle. The tuning-curve panel
uses the GT3 sector ratio b2/c1 = 2.4 (cos2theta_fit.json) for its amplitudes; the
rest is drawn. Written S76 (2026-09-04). Output: figures_canonical/fig_orientation_quotient.png.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
fit = json.load(open(DATA / "cos2theta_fit.json"))
b, c = fit["b_cos2theta"], fit["c_costheta"]          # GT3: 0.33, 0.14

CLASS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948", "#76b7b2", "#9c755f"]
RED, BLUE, GREEN, MAGENTA, GRAY, INK = "#b40426", "#2166ac", "#3d7d1f", "#c51b7d", "#8a8a8a", "#222222"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(6.0, 2.35), dpi=300)
axA = fig.add_axes([0.01, 0.04, 0.30, 0.82]); axB = fig.add_axes([0.385, 0.25, 0.265, 0.57]); axC = fig.add_axes([0.68, 0.04, 0.31, 0.82])
for ax in (axA, axC):
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal"); ax.axis("off")

def title(ax, letter, text, y=None):
    x0 = ax.get_position().x0
    fig.text(x0, 0.90, letter, fontsize=8, fontweight="bold", va="bottom", ha="left")
    fig.text(x0 + 0.03, 0.90, text, fontsize=7, va="bottom", ha="left")

# ---------- A: the direction circle and its orientation quotient ----------
title(axA, "A", "eight drift directions, four orientations")
th = np.linspace(0, 2 * np.pi, 400)
axA.plot(np.cos(th), np.sin(th), color=GRAY, lw=0.8)
angles = np.arange(8) * np.pi / 4
for k, a in enumerate(angles):
    axA.plot([np.cos(a)], [np.sin(a)], "o", color=CLASS[k], ms=6, zorder=4)
    axA.annotate("", xy=(0.78 * np.cos(a), 0.78 * np.sin(a)), xytext=(0.55 * np.cos(a), 0.55 * np.sin(a)),
                 arrowprops=dict(arrowstyle="-|>", color=CLASS[k], lw=0.9, mutation_scale=6))
    axA.text(1.17 * np.cos(a), 1.17 * np.sin(a), f"{int(np.degrees(a))}°", ha="center", va="center", fontsize=5.5, color=INK)
for k in range(4):   # antipodal pairs share an orientation
    a = angles[k]
    axA.plot([np.cos(a), -np.cos(a)], [np.sin(a), -np.sin(a)], color=GRAY, lw=0.6, ls=(0, (2, 2)), zorder=1)
axA.text(0, -1.33, "antipodal pairs (dashed) share an orientation\nand differ only in drift direction",
         ha="center", va="top", fontsize=5.9, color=INK)

# ---------- B: the measured GT3 kernel and its two sectors (the five-term fit of Sec 4) ----------
title(axB, "B", "the GT3 kernel, two sectors")
a0 = fit["a"]; b4, c3 = 0.0716, 0.0638          # the ell = 4 and ell = 3 terms of the five-term fit (App I)
dphi = np.linspace(0, np.pi, 400); ddeg = np.degrees(dphi)
Cfull = a0 + b * np.cos(2 * dphi) + c * np.cos(dphi) + b4 * np.cos(4 * dphi) + c3 * np.cos(3 * dphi)
Ceven = a0 + b * np.cos(2 * dphi) + b4 * np.cos(4 * dphi)
Codd = c * np.cos(dphi) + c3 * np.cos(3 * dphi)
C_at = lambda x: a0 + b * np.cos(2 * np.radians(x)) + c * np.cos(np.radians(x)) + b4 * np.cos(4 * np.radians(x)) + c3 * np.cos(3 * np.radians(x))
axB.plot(ddeg, Cfull, color="k", lw=1.5)
axB.plot(ddeg, Ceven, color=BLUE, lw=1.0, ls="--")
axB.plot(ddeg, Codd, color=RED, lw=1.0, ls=":")
axB.plot([22.5], [C_at(22.5)], "o", color=GREEN, ms=4.5, zorder=5)
axB.plot([180.0], [C_at(180.0)], "o", color=MAGENTA, ms=4.5, zorder=5)
axB.annotate(f"adjacent {C_at(22.5):.2f}", (22.5, C_at(22.5)), xytext=(38, 0.80), fontsize=5.6, color=GREEN, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6))
axB.annotate(f"antipodal {C_at(180.0):.2f}", (180.0, C_at(180.0)), xytext=(176, 0.93), fontsize=5.6, color=MAGENTA, ha="right", va="center",
             arrowprops=dict(arrowstyle="-", color=MAGENTA, lw=0.6))
axB.text(92, 0.60, "even (orientation)", color=BLUE, fontsize=5.6, ha="center", va="center")
axB.text(92, -0.16, "odd (direction)", color=RED, fontsize=5.6, ha="center", va="center")
axB.set_xlim(-3, 183); axB.set_ylim(-0.24, 1.02)
axB.set_xticks([0, 45, 90, 135, 180]); axB.set_xticklabels(["0°", "45°", "90°", "135°", "180°"], fontsize=6)
axB.set_yticks([0, 0.5, 1.0]); axB.tick_params(axis="y", labelsize=6)
axB.set_xlabel("class separation $\\Delta\\phi$", fontsize=6.5, labelpad=1); axB.set_ylabel("signal correlation", fontsize=6.5, labelpad=1)
axB.spines[["top", "right"]].set_visible(False)
axB.text(0.5, -0.36, "black: the five-term fit on GT3", transform=axB.transAxes, fontsize=5.3, color=GRAY, ha="center", va="top")

# ---------- C: two ways to climb the ladder ----------
title(axC, "C", "two ways to climb the ladder")
for cx, order, col, lab in [(-0.64, list(range(8)), GREEN, "sequential\n(adjacent)"),
                            (0.64, [0, 4, 1, 5, 2, 6, 3, 7], MAGENTA, "paired\n(antipodal)")]:
    r = 0.52
    axC.plot(cx + r * np.cos(th), r * np.sin(th), color=GRAY, lw=0.7)
    pts = [(cx + r * np.cos(angles[k]), r * np.sin(angles[k])) for k in range(8)]
    for k in range(8):
        axC.plot([pts[k][0]], [pts[k][1]], "o", color=CLASS[k], ms=4.2, zorder=4)
    for i in range(3):   # first three steps of the order
        p, q = pts[order[i]], pts[order[i + 1]]
        axC.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=7,
                     connectionstyle="arc3,rad=-0.25" if col == GREEN else "arc3,rad=0.0"), zorder=5)
    axC.text(cx, -0.66, lab, ha="center", va="top", fontsize=5.6, color=col)
axC.text(0, -1.02, "adjacent pairs cohere more (0.78)\nthan antipodal pairs (0.59), so the\nsequential ladder climbs faster, 8 of 8",
         ha="center", va="top", fontsize=5.9, color=INK)
axC.set_ylim(-1.45, 0.85)

for out in (HERE.parent / "figures_canonical" / "fig_orientation_quotient.png", HERE.parent.parent / "arxiv" / "figures" / "fig_orientation_quotient.png"):
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
