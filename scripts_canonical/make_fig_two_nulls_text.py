"""Figure: the two nulls on natural text (S77; panels (b),(c) of the S73
fig_nulls.png). Sources: run42_blimp_battery.json, run43b_baroni_64pairs.json.
Out: ../figures_canonical/fig_two_nulls_text.png + ../../arxiv/figures/
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
OUTS = [HERE.parent / "figures_canonical" / "fig_two_nulls_text.png", HERE.parent.parent / "arxiv" / "figures" / "fig_two_nulls_text.png"]
r42 = json.load(open(DC / "run42_blimp_battery.json")); r43 = json.load(open(DC / "run43b_baroni_64pairs.json"))
fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5))
def panel(ax, layers, title):
    x = np.array([l["layer"] for l in layers]) / (len(layers) - 1)
    d = np.array([l["delta_plain"] for l in layers])
    pm = np.array([l["perm_null_mean"] for l in layers]); ps = np.array([l["perm_null_sd"] for l in layers])
    sm = np.array([l["strat_null_mean"] for l in layers]); ss = np.array([l["strat_null_sd"] for l in layers])
    ax.fill_between(x, pm - 2 * ps, pm + 2 * ps, color="0.6", alpha=0.35, lw=0, label="label permutation ($\\pm 2$ SD)")
    ax.fill_between(x, sm - 2 * ss, sm + 2 * ss, color="#c51b7d", alpha=0.25, lw=0, label="within-pair swap ($\\pm 2$ SD)")
    ax.plot(x, d, "-", color="k", lw=1.8, label="observed $\\delta$")
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("normalized depth $\\ell/L$"); ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
panel(axes[0], r42["models"]["pythia-2.8b-deduped"]["B_grammaticality"]["layers"], "(a) BLiMP, 64 pairs: composition only")
panel(axes[1], r43["models"]["pythia-2.8b-deduped"]["layers"], "(b) Baroni, 64 pairs: signal beyond carriers")
axes[0].set_ylabel("$\\delta$"); axes[0].legend(fontsize=6.0, loc="lower left", frameon=False)
fig.tight_layout(w_pad=1.5)
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
