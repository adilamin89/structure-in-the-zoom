"""Figure: LLM depth profiles (S77 restyle of make_fig6_llm.py). Two panels from
committed JSONs (runs 17/18/19/26): (a) content axis, (b) construction axis, at
four Pythia scales, on normalized depth; pooled shuffle band. The embedding-excess
panel and the OLMo/GPT-Neo inset of the S73 figure are superseded by
fig_architectures.png. Drawn at the printed width so fonts print 1:1.
Out: ../figures_canonical/fig_llm_depth.png + ../../arxiv/figures/
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 6.8, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_llm_depth.png",
        HERE.parent.parent / "arxiv" / "figures" / "fig_llm_depth.png"]
MODELS = [("run17_multiclass_battery.json", "160m"), ("run18_pythia410m_battery.json", "410m"),
          ("run19_pythia1b_battery.json", "1B"), ("run26_pythia28b_battery.json", "2.8B")]
WARM = {"160m": "#fdbe85", "410m": "#fd8d3c", "1B": "#e6550d", "2.8B": "#a63603"}
COOL = {"160m": "#bdd7e7", "410m": "#6baed6", "1B": "#3182bd", "2.8B": "#08519c"}

def profile(data, axis):
    ls = data["axes"][axis]["layers"]
    x = np.array([lr["layer"] for lr in ls], float); x /= x.max()
    return x, np.array([lr["delta"] for lr in ls]), np.array([lr["shuffle_mean"] for lr in ls])

fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.55))
plt.subplots_adjust(wspace=0.32, left=0.09, right=0.985, top=0.89, bottom=0.19)
all_sh = []
for ax, axis, title, cmap in [(axes[0], "world_knowledge", "(a) content axis: inherited, dilutes", WARM),
                              (axes[1], "language_type", "(b) construction axis: built with depth", COOL)]:
    for fname, tag in MODELS:
        x, y, sh = profile(json.load(open(DATA / fname)), axis)
        all_sh.extend(sh)
        ax.plot(x, y, "o-", ms=3, lw=1.3, color=cmap[tag], label=f"Pythia-{tag}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("normalized depth $\\ell/L$"); ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"$\delta$")
sh = np.array(all_sh)
for ax in axes:
    ax.axhspan(sh.mean() - 2 * sh.std(), sh.mean() + 2 * sh.std(), color="0.9", zorder=0)
    ax.set_ylim(-0.22, 0.25)
axes[0].legend(frameon=False, loc="lower left", ncol=2, columnspacing=0.8)
axes[1].legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8)
axes[1].text(0.98, 0.04, "band: pooled shuffle range", transform=axes[1].transAxes, ha="right", fontsize=6, color="0.4")
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
