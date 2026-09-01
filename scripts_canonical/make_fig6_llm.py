"""Figure 6 (LLM depth profiles) for the arXiv version. From committed
JSONs (runs 17/18/19/26 + run25 OLMo):
  (a) world_knowledge (content axis): delta vs normalized depth at four
      Pythia scales — positive at the embedding, declining.
  (b) language_type (structural axis): negative at the embedding, rising —
      the content/structure crossover at all four scales.
  (c) embedding-excess delta(L) - delta(0) for all six axes on Pythia-2.8B
      (32 layers): network-built organization is visible as positive excess.
      Inset: OLMo-1B (16/class, run25) — content replicates cross-family,
      the structural profile does not.
Out: ../figures_canonical/fig_llm_depth.png + ../../arxiv/figures/
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_llm_depth.png",
        HERE.parent.parent / "arxiv" / "figures" / "fig_llm_depth.png"]

MODELS = [("run17_multiclass_battery.json", "160m", "#fdbe85"),
          ("run18_pythia410m_battery.json", "410m", "#fd8d3c"),
          ("run19_pythia1b_battery.json", "1B", "#e6550d"),
          ("run26_pythia28b_battery.json", "2.8B", "#a63603")]
COOL = {"160m": "#bdd7e7", "410m": "#6baed6", "1B": "#3182bd",
        "2.8B": "#08519c"}


def profile(data, axis):
    ls = data["axes"][axis]["layers"]
    x = np.array([lr["layer"] for lr in ls], float)
    x /= x.max()
    y = np.array([lr["delta"] for lr in ls])
    sh = np.array([lr["shuffle_mean"] for lr in ls])
    return x, y, sh


fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
plt.subplots_adjust(wspace=0.30, left=0.055, right=0.985, top=0.89,
                    bottom=0.17)

all_sh = []
for panel, axis, title, cmap in [
        (0, "world_knowledge", "(a) content axis: inherited, diluted", None),
        (1, "language_type", "(b) structural axis: built along the declared path", COOL)]:
    ax = axes[panel]
    for fname, tag, warm in MODELS:
        d = json.load(open(DATA / fname))
        x, y, sh = profile(d, axis)
        all_sh.extend(sh)
        c = cmap[tag] if cmap else warm
        ax.plot(x, y, "o-", ms=3, lw=1.3, color=c, label=f"Pythia-{tag}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("normalized depth $\\ell/L$")
    if panel == 0:
        ax.set_ylabel(r"$\delta$")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
sh_arr = np.array(all_sh)
for panel in (0, 1):
    axes[panel].axhspan(sh_arr.mean() - 2 * sh_arr.std(),
                        sh_arr.mean() + 2 * sh_arr.std(), color="0.9",
                        zorder=0)
    axes[panel].set_ylim(-0.30, 0.30)

# ---------------- (c) embedding excess, all axes, 2.8B ----------------
ax = axes[2]
d28 = json.load(open(DATA / "run26_pythia28b_battery.json"))
# random is excluded here: its embedding-layer floor is unstable at n=128
# (delta(0) = -0.31 on 2.8B), which would inflate the excess baseline;
# it appears as the shaded control band of panels (a,b) instead.
AXCOL = {"world_knowledge": ("world knowledge", "#e6550d"),
         "language_type": ("construction type", "#3182bd"),
         "ethical": ("ethical concept", "#31a354"),
         "tqa_category": ("TruthfulQA category", "#969696"),
         "hs_activity": ("HellaSwag activity", "#bcbddc"),
         "arc_topic": ("ARC science topic", "#c994c7")}
for axis, (lab, c) in AXCOL.items():
    ls = d28["axes"][axis]["layers"]
    x = np.array([lr["layer"] for lr in ls], float)
    x /= x.max()
    y = np.array([lr["delta"] - ls[0]["delta"] for lr in ls])
    lw = 1.6 if axis in ("world_knowledge", "language_type", "ethical") \
        else 0.8
    ax.plot(x, y, "-", lw=lw, color=c,
            label=lab, alpha=0.95 if lw > 1 else 0.55)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("normalized depth $\\ell/L$")
ax.set_ylabel(r"embedding excess $\delta(\ell)-\delta(0)$")
ax.set_title("(c) Pythia-2.8B: excess over the embedding", fontsize=10)
ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower right")
ax.set_ylim(-0.22, 0.42)

# inset: cross-family (OLMo, solid) vs same-corpus different-architecture
# (GPT-Neo, dashed) — the rise tracks the corpus, the shape the architecture
d25 = json.load(open(DATA / "run25_olmo1b_16pc_battery.json"))
d29 = json.load(open(DATA / "run29_gptneo_battery.json"))
ins = ax.inset_axes([0.06, 0.60, 0.40, 0.34])
for d, style in [(d25, "-"), (d29, "--")]:
    for axis, c in [("world_knowledge", "#e6550d"),
                    ("language_type", "#3182bd")]:
        ls = d["axes"][axis]["layers"]
        x = np.array([lr["layer"] for lr in ls], float)
        x /= x.max()
        y = np.array([lr["delta"] for lr in ls])
        ins.plot(x, y, style, lw=1.0, color=c)
ins.axhline(0, color="k", lw=0.5)
ins.tick_params(labelsize=6)
ins.set_title("OLMo-1B (solid) / GPT-Neo (dashed)", fontsize=6)
ins.set_ylim(-0.35, 0.30)

for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists():
        continue  # secondary copy only when the paper tree is present
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print("wrote", out)
