"""Figure: declared order vs order average on Pythia-2.8B (S77; panel (a) of the
S73 fig_nulls.png as its own figure). Source: run37_inferential_nulls.json.
Out: ../figures_canonical/fig_path_partition.png + ../../arxiv/figures/
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DC = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_path_partition.png", HERE.parent.parent / "arxiv" / "figures" / "fig_path_partition.png"]
r37 = json.load(open(DC / "run37_inferential_nulls.json"))
fig, ax = plt.subplots(figsize=(3.8, 2.7))
colors = {"world_knowledge": "#d95f0e", "language_type": "#2171b5"}
names = {"world_knowledge": "content", "language_type": "construction"}
for axis_name, col in colors.items():
    L = r37["models"]["pythia-2.8b-deduped"]["axes"][axis_name]["layers"]
    x = np.array([l["layer"] for l in L]) / (len(L) - 1)
    d = np.array([l["delta"] for l in L]); nm = np.array([l["null_mean"] for l in L]); ns = np.array([l["null_sd"] for l in L])
    da = np.array([l["delta_orderavg"] for l in L]); nma = np.array([l["null_mean_orderavg"] for l in L]); nsa = np.array([l["null_sd_orderavg"] for l in L])
    ax.fill_between(x, nm - 2 * ns, nm + 2 * ns, color=col, alpha=0.10, lw=0)
    ax.fill_between(x, nma - 2 * nsa, nma + 2 * nsa, color=col, alpha=0.22, lw=0)
    ax.plot(x, d, "-", color=col, lw=1.8, label=f"{names[axis_name]}: declared order")
    ax.plot(x, da, "--", color=col, lw=1.6, label=f"{names[axis_name]}: order-averaged")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("normalized depth $\\ell/L$"); ax.set_ylabel("$\\delta$")
ax.set_title("Pythia-2.8B: the order and the partition")
ax.set_ylim(-0.17, 0.21)
ax.legend(fontsize=6.2, loc="upper right", frameon=False)
ax.text(0.98, 0.03, "bands: $\\pm 2$ SD of the 500-permutation null\n(light: declared; dark: order-averaged)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color="0.35")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
