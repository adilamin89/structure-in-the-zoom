"""Figure: the sector balance across scales (S77). (A) a drawn block of neurons
sorted by orientation (antipodal preferences: the odd harmonic cancels, the even
adds), a random block, and a direction-sorted block; (B) the measured blocking
factor (b2/c1)(K)/(b2/c1)(1) on the eight grating recordings, median and range per
block type, with Eq. B(K) at the measured coherences; (C) retention of the shift
at K = 32 per recording (run50b, run51); (D) Eq. B(K) for three map types.
Sources: sector_balance_scale.json, run50b_graining_sectors.json, run51_spatial_blocking.json.
Out: ../figures_canonical/fig_blocking.png + ../../arxiv/figures/
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.2, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_blocking.png", HERE.parent.parent / "arxiv" / "figures" / "fig_blocking.png"]
INK, GRAY, BLUE, ORANGE = "#222222", "#8a8a8a", "#3182bd", "#e6550d"
def short(name):
    kind = "D" if "drifting" in name else "L" if "local" in name else "C"; return kind + name.split("GT")[1][0]
sb = json.load(open(DATA / "sector_balance_scale.json")); rows = [r for r in sb["rows"] if r["status"] == "ok"]
r50 = {r["name"]: r for r in json.load(open(DATA / "run50b_graining_sectors.json"))["rows"] if r["status"] == "ok"}
r51 = {r["name"]: r for r in json.load(open(DATA / "run51_spatial_blocking.json"))["rows"] if r["status"] == "ok"}
def B(K, r1, r2): K = np.asarray(K, float); return (1 / K + (1 - 1 / K) * r2) / (1 / K + (1 - 1 / K) * r1)

fig = plt.figure(figsize=(6.0, 2.95))
axA = fig.add_axes([0.005, 0.05, 0.25, 0.86]); axB = fig.add_axes([0.335, 0.22, 0.175, 0.62])
axC = fig.add_axes([0.60, 0.22, 0.165, 0.62]); axD = fig.add_axes([0.835, 0.22, 0.155, 0.62])
for ax, letter, text in [(axA, "A", "averaging neurons into blocks")]:
    fig.text(0.005, 0.95, letter, fontsize=8.5, fontweight="bold", va="center"); fig.text(0.035, 0.95, text, fontsize=7.5, va="center")
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
def block(ax, y, angles, title, odd, even, col):
    ax.add_patch(FancyBboxPatch((0.5, y - 0.55), 4.2, 1.1, boxstyle="round,pad=0.05,rounding_size=0.25", fc="white", ec=GRAY, lw=0.7))
    for i, a in enumerate(angles):
        cx, cy = 1.15 + i * 1.0, y
        ax.add_patch(Circle((cx, cy), 0.36, fc="#f2f2f2", ec=GRAY, lw=0.6))
        ax.annotate("", xy=(cx + 0.30 * np.cos(a), cy + 0.30 * np.sin(a)), xytext=(cx - 0.30 * np.cos(a), cy - 0.30 * np.sin(a)),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, mutation_scale=6))
    ax.text(0.5, y + 0.75, title, fontsize=6.4, va="bottom", color=INK)
    ax.text(0.5, y - 0.75, "odd (direction): " + odd + "\neven (orientation): " + even, fontsize=5.9, va="top", color=INK, linespacing=1.15)
block(axA, 8.55, [0, np.pi, 0, np.pi], "sorted by orientation (antipodal preferences)", "cancels, $\\rho_1 \\approx 0$", "adds, $\\rho_2 \\approx 0.2$", BLUE)
block(axA, 5.15, [0.4, 2.3, 4.0, 5.5], "random block", "cancels, $\\rho_1 \\approx 0$", "cancels, $\\rho_2 \\approx 0$", GRAY)
block(axA, 1.75, [0, 0.1, -0.1, 0.05], "sorted by direction", "adds, $\\rho_1 \\approx 0.2$", "adds, $\\rho_2 \\approx 0.1$", ORANGE)

# ---- (B) measured blocking factor ----
ax = axB
arms = [("ori_sorted", "orientation-sorted", BLUE), ("dir_sorted", "direction-sorted", ORANGE), ("random", "random", GRAY)]
K = np.array(rows[0]["graining_flow"]["K"], float)
for arm, lab, col in arms:
    M = np.array([np.array(r["graining_flow"][arm], float) / float(r["graining_flow"][arm][0]) for r in rows])
    ax.fill_between(K, M.min(0), M.max(0), color=col, alpha=0.18, lw=0)
    ax.plot(K, np.median(M, 0), "o-", ms=2.5, lw=1.4, color=col, label=lab)
ax.plot(K, B(K, 0.0, 0.2), "--", color=BLUE, lw=0.9); ax.plot(K, B(K, 0.2, 0.1), "--", color=ORANGE, lw=0.9)
ax.axhline(1, color="k", lw=0.5)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(["1", "4", "16", "64"])
ax.set_xlabel("neurons per block $K$"); ax.set_ylabel("$(b_2/c_1)(K)\\,/\\,(b_2/c_1)(1)$", labelpad=1)
ax.set_title("B  blocking factor", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left", handlelength=1.3); ax.text(0.98, 0.03, "dashed: Eq. 3 at\nthe measured $\\rho$", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color="0.35")
ax.spines[["top", "right"]].set_visible(False)

# ---- (C) retention at K = 32 ----
ax = axC
tags = [short(r["name"]) for r in rows]
armsC = [("ori_sorted", "ori.-sorted", BLUE, r50), ("dir_sorted", "dir.-sorted", ORANGE, r50), ("random", "random", "0.6", r50), ("spatial", "anatomical", "#31a354", r51)]
w = 0.8 / len(armsC)
for i, (arm, lab, col, src) in enumerate(armsC):
    vals = [src[r["name"]]["retention_K32"][arm] for r in rows]
    ax.bar(np.arange(len(rows)) + (i - 1.5) * w, vals, width=w, color=col, label=lab, edgecolor="none")
ax.axhline(1.0, color="k", lw=0.5); ax.axhline(0.0, color="k", lw=0.5)
ax.set_xticks(np.arange(len(rows))); ax.set_xticklabels(tags, fontsize=6.2); ax.set_ylabel("shift retained", labelpad=1); ax.set_xlabel("recording")
ax.set_title("C  shift at $K = 32$", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.6, columnspacing=0.6, handlelength=1.0, borderaxespad=0.1)
ax.set_ylim(-0.05, 1.95); ax.spines[["top", "right"]].set_visible(False)

# ---- (D) B(K) for three map types ----
ax = axD
Kd = np.logspace(0, np.log2(256) * np.log10(2), 100); Kd = 2 ** np.linspace(0, 8, 100)
ax.plot(Kd, B(Kd, 0.0, 0.2), "-", color=BLUE, lw=1.5, label="orientation columns\n($\\rho_2 = 0.2,\\ \\rho_1 = 0$)")
ax.plot(Kd, B(Kd, 0.1, 0.2), "-", color="#756bb1", lw=1.5, label="both maps\n($\\rho_2 = 0.2,\\ \\rho_1 = 0.1$)")
ax.plot(Kd, B(Kd, 0.0, 0.0), "-", color=GRAY, lw=1.5, label="salt-and-pepper\n($\\rho_1 = \\rho_2 = 0$)")
ax.plot([32], [1.0], "o", color=INK, ms=4.5, zorder=5); ax.text(32, 1.35, "mouse,\nmeasured", fontsize=5.6, ha="center", va="bottom", color=INK)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([1, 4, 16, 64, 256]); ax.set_xticklabels(["1", "4", "16", "64", "256"])
ax.set_ylim(0.7, 60); ax.set_xlabel("units pooled $K$"); ax.set_ylabel("$B(K)$", labelpad=1)
ax.set_title("D  by map type", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left", fontsize=5.4, handlelength=1.2, labelspacing=0.5)
ax.spines[["top", "right"]].set_visible(False)
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
