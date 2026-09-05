"""Figure: the construction axis across architectures, and the blind probe (S77).
(a) declared-order construction profiles on normalized depth for Pythia-2.8B,
    GPT-Neo-1.3B, RedPajama-INCITE-3B, Mamba-2.8B, OLMo-2-1B and OLMo-1B, with
    the last zero crossing marked (the depth after which the profile stays
    non-negative; the rule of Table 5).
(b) leading-eigenvalue fraction of the full prompt set by layer (run55
    diagnostics) for OLMo-1B against Pythia-2.8B and OLMo-2-1B: the rogue
    dimension that blinds the linear probe.
Sources: run37_inferential_nulls.json, run47_fourth_cell_redpajama.json,
run53_mamba_fifth_cell.json, run54_olmo2_1b_construction.json,
run55_blind_probe_physics.json.
Out: ../figures_canonical/fig_architectures.png + ../../arxiv/figures/
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.6, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DC = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_architectures.png", HERE.parent.parent / "arxiv" / "figures" / "fig_architectures.png"]
SRC = [("Pythia-2.8B (Pile)", "run37_inferential_nulls.json", "pythia-2.8b-deduped", "#08519c"),
       ("RedPajama-INCITE-3B", "run47_fourth_cell_redpajama.json", "RedPajama-INCITE-Base-3B-v1", "#3182bd"),
       ("GPT-Neo-1.3B (Pile)", "run37_inferential_nulls.json", "gpt-neo-1.3B", "#6baed6"),
       ("Mamba-2.8B (Pile)", "run53_mamba_fifth_cell.json", "mamba-2.8b-hf", "#d95f0e"),
       ("OLMo-2-1B (Dolma)", "run54_olmo2_1b_construction.json", "OLMo-2-0425-1B", "#31a354"),
       ("OLMo-1B (Dolma)", "run37_inferential_nulls.json", "OLMo-1B-hf", "0.55")]
cache = {}
def last_cross(y):
    L = len(y) - 1; i = L
    while i > 0 and y[i - 1] >= 0: i -= 1
    if i == 0: return 0.0
    return (i - 1 + (0 - y[i - 1]) / (y[i] - y[i - 1])) / L
fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), gridspec_kw={"width_ratios": [1.45, 1.0]})
plt.subplots_adjust(wspace=0.35, left=0.08, right=0.985, top=0.88, bottom=0.19)
ax = axes[0]
rows = []
for lab, f, k, col in SRC:
    d = cache.setdefault(f, json.load(open(DC / f)))
    L = d["models"][k]["axes"]["language_type"]["layers"]
    y = np.array([l["delta"] for l in L]); x = np.arange(len(y)) / (len(y) - 1)
    ls = "-" if "OLMo-1B" not in lab else ":"
    ax.plot(x, y, ls, color=col, lw=1.5, label=lab)
    if "OLMo-1B" not in lab:
        c = last_cross(y); ax.plot([c], [0], "v", color=col, ms=5, zorder=5); rows.append((lab, c))
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("normalized depth $\\ell/L$"); ax.set_ylabel("construction $\\delta$ (declared order)")
ax.set_title("(a) one shape, one crossing depth per architecture")
ax.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.4)
ax.set_ylim(-0.36, 0.14)
ax.text(0.02, 0.96, "$\\blacktriangledown$ last zero crossing", transform=ax.transAxes, va="top", fontsize=6.5, color="0.3")
ax.spines[["top", "right"]].set_visible(False)
# (b) blind probe: top-eigenvalue fraction by layer
ax = axes[1]
r55 = json.load(open(DC / "run55_blind_probe_physics.json"))
for k, lab, col in [("OLMo-1B-hf", "OLMo-1B", "0.2"), ("pythia-2.8b-deduped", "Pythia-2.8B", "#08519c"), ("OLMo-2-0425-1B", "OLMo-2-1B", "#31a354")]:
    L = r55["models"][k]["axes"]["language_type"]["layers"]
    x = np.arange(len(L)) / (len(L) - 1); f = np.array([l["diag"]["top1_eig_frac"] for l in L])
    ax.plot(x, f, "o-", ms=2.5, lw=1.3, color=col, label=lab)
ax.set_ylim(0, 1.0); ax.set_xlabel("normalized depth $\\ell/L$"); ax.set_ylabel("leading-eigenvalue fraction")
ax.set_title("(b) the blind probe")
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.03, 0.40))
ax.spines[["top", "right"]].set_visible(False)
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
print("last zero crossings:", [(l, round(c, 2)) for l, c in rows])
