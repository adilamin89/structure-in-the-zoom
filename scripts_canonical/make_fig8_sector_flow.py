"""Figure: the sector balance across scales (S75). Three panels from
committed artifacts, drawn at the printed width (jmlr 6.0 in) so fonts print
1:1.
  (A) graining flow of the quadrupole-to-dipole ratio b2/c1 with block size K
      on all eight grating recordings: orientation-sorted blocks amplify the
      quadrupole, direction-sorted blocks the dipole, random blocks leave the
      balance unchanged (sector_balance_scale.json).
  (B) Allen Neuropixels per-area median b2/|c1| (bars, sector_balance_scale
      allen_per_area) against the per-area mean direction-aligned shift
      (points; the Table 3 rule, areas with >= 20 populations), same order.
  (C) retention of the direction-aligned shift at K = 32 per recording under
      orientation-sorted, direction-sorted, random (run50b) and, once run51
      has finished, anatomical (spatial k-means) blocks.
Out: ../figures_canonical/fig_sector_flow.png + ../../arxiv/figures/
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.3, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_sector_flow.png",
        HERE.parent.parent / "arxiv" / "figures" / "fig_sector_flow.png"]

def short(name):
    """D1/L1/C1 style tag from the recording file stem."""
    kind = "D" if "drifting" in name else "L" if "local" in name else "C"
    return kind + name.split("GT")[1][0]


TYPECOL = {"D": "#a63603", "L": "#e6550d", "C": "#756bb1"}
ARMSTYLE = {"ori_sorted": ("-", "orientation-sorted"), "dir_sorted": ("--", "direction-sorted"),
            "random": (":", "random")}
AREAS = ["VISp", "VISpm", "VISam", "VISal", "VISrl", "VISl"]

sb = json.load(open(DATA / "sector_balance_scale.json"))
rows = [r for r in sb["rows"] if r["status"] == "ok"]
r50 = {r["name"]: r for r in json.load(open(DATA / "run50b_graining_sectors.json"))["rows"] if r["status"] == "ok"}
r51_path = DATA / "run51_spatial_blocking.json"
r51 = {r["name"]: r for r in json.load(open(r51_path))["rows"] if r["status"] == "ok"} if r51_path.exists() else {}
allen = json.load(open(DATA / "allen_expansion_all_sessions.json"))
dmean = []
for a in AREAS:
    vals = [s["areas"][a]["delta_dir8"] for s in allen["results"] if s.get("status") == "ok" and a in s.get("areas", {})]
    dmean.append(float(np.mean(vals)))

fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.3), gridspec_kw={"width_ratios": [1.15, 0.85, 1.2]})
plt.subplots_adjust(wspace=0.80, left=0.08, right=0.975, top=0.86, bottom=0.25)

# ---------------- (A) graining flow ----------------
ax = axes[0]
for r in rows:
    tag = short(r["name"]); col = TYPECOL[tag[0]]
    K = np.array(r["graining_flow"]["K"], float)
    for arm, (ls, _) in ARMSTYLE.items():
        ax.plot(K, np.array(r["graining_flow"][arm], float), ls, color=col, lw=0.9, alpha=0.9)
ax.axhline(1.0, color="k", lw=0.5)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(["1", "4", "16", "64"])
ax.set_ylim(0.2, 60)
ax.set_xlabel("neurons per block $K$"); ax.set_ylabel("$b_2/c_1$ (quadrupole / dipole)")
ax.set_title("(A) blocking flow of $b_2/c_1$")
h1 = [Line2D([], [], color="0.3", ls=ls, lw=1.0, label=lab) for ls, lab in ARMSTYLE.values()]
h2 = [Line2D([], [], color=c, lw=2.0, label=lab) for c, lab in
      [(TYPECOL["D"], "drifting"), (TYPECOL["L"], "localized"), (TYPECOL["C"], "low contrast")]]
leg1 = ax.legend(handles=h1, loc="upper left", frameon=False, handlelength=1.7, borderaxespad=0.2)
ax.add_artist(leg1)
ax.legend(handles=h2, loc="upper left", bbox_to_anchor=(0.0, 0.71), frameon=False, handlelength=1.2, borderaxespad=0.2, fontsize=6.0, labelspacing=0.25)

# ---------------- (B) Allen areas ----------------
ax = axes[1]
med = [sb["allen_per_area"][a]["median_b2_over_c1"] for a in AREAS]
x = np.arange(len(AREAS))
ax.bar(x, med, color="#9ecae1", edgecolor="#3182bd", lw=0.6, width=0.6)
ax.set_ylabel("median $b_2/|c_1|$", color="#3182bd"); ax.set_ylim(0, 15)
ax.set_xticks(x); ax.set_xticklabels(AREAS, rotation=45, ha="right")
ax2 = ax.twinx()
ax2.plot(x, dmean, "o", color="#a50f15", ms=3.5)
ax2.set_ylabel(r"mean $\delta_{\mathrm{dir}}$", color="#a50f15", labelpad=3); ax2.set_ylim(0, 0.25)
ax2.tick_params(axis="y", labelsize=6.5)
ax.set_title("(B) Allen areas")

# ---------------- (C) retention at K = 32 ----------------
ax = axes[2]
tags = [short(r["name"]) for r in rows]
arms = [("ori_sorted", "orientation-sorted", "#3182bd"), ("dir_sorted", "direction-sorted", "#e6550d"),
        ("random", "random", "0.6")]  # short legend labels below keep the key inside the panel
if r51:
    arms.append(("spatial", "anatomical", "#31a354"))
SHORTLAB = {"orientation-sorted": "ori.-sorted", "direction-sorted": "dir.-sorted", "random": "random", "anatomical": "anatomical"}
w = 0.8 / len(arms)
for i, (arm, lab, col) in enumerate(arms):
    src = r51 if arm == "spatial" else r50
    vals = [src[r["name"]]["retention_K32"][arm] for r in rows]
    ax.bar(np.arange(len(rows)) + (i - (len(arms) - 1) / 2) * w, vals, width=w, color=col, label=SHORTLAB[lab], edgecolor="none")
ax.axhline(1.0, color="k", lw=0.5); ax.axhline(0.0, color="k", lw=0.5)
ax.set_xticks(np.arange(len(rows))); ax.set_xticklabels(tags)
ax.set_ylabel("shift retained", labelpad=2)
ax.set_xlabel("grating recording")
ax.set_title("(C) shift retained at $K = 32$")
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.8, columnspacing=0.6, handlelength=1.0, borderaxespad=0.2, handletextpad=0.4)
ax.set_ylim(-0.05, 1.95)

for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists():
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300); print("wrote", out)
print("allen area means", dict(zip(AREAS, [round(v, 3) for v in dmean])), "| spatial arm:", bool(r51))
