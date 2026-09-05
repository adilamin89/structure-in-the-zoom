"""Figure (appendix): the blocking flow of b2/c1 per recording (S77; panel (A)
of the S75 fig_sector_flow.png with the legend outside the data).
Source: sector_balance_scale.json. Out: ../figures_canonical/fig_sector_flow_full.png + ../../arxiv/figures/
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_sector_flow_full.png", HERE.parent.parent / "arxiv" / "figures" / "fig_sector_flow_full.png"]
def short(name):
    kind = "D" if "drifting" in name else "L" if "local" in name else "C"; return kind + name.split("GT")[1][0]
TYPECOL = {"D": "#a63603", "L": "#e6550d", "C": "#756bb1"}
ARMSTYLE = {"ori_sorted": ("-", "orientation-sorted"), "dir_sorted": ("--", "direction-sorted"), "random": (":", "random")}
rows = [r for r in json.load(open(DATA / "sector_balance_scale.json"))["rows"] if r["status"] == "ok"]
fig, ax = plt.subplots(figsize=(4.6, 2.7))
for r in rows:
    col = TYPECOL[short(r["name"])[0]]; K = np.array(r["graining_flow"]["K"], float)
    for arm, (ls, _) in ARMSTYLE.items():
        ax.plot(K, np.array(r["graining_flow"][arm], float), ls, color=col, lw=0.9, alpha=0.9)
ax.axhline(1.0, color="k", lw=0.5); ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(["1", "4", "16", "64"]); ax.set_ylim(0.2, 60)
ax.set_xlabel("neurons per block $K$"); ax.set_ylabel("$b_2/c_1$ (quadrupole / dipole)")
ax.set_title("Blocking flow of the sector balance, eight recordings")
h1 = [Line2D([], [], color="0.3", ls=ls, lw=1.0, label=lab) for ls, lab in ARMSTYLE.values()]
h2 = [Line2D([], [], color=c, lw=2.0, label=lab) for c, lab in [(TYPECOL["D"], "drifting"), (TYPECOL["L"], "localized"), (TYPECOL["C"], "low contrast")]]
ax.legend(handles=h1 + h2, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=1.7)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
