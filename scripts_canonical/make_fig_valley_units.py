"""Figure (queue 27n): the construction profiles of the readable models with depth measured in units of each
model's own landmark, the valley (the layer of the construction minimum), instead of the total depth. If the six
profiles collapse in valley units, the shape is one curve with one scale per architecture; if not, the valley and
the rise are two scales. Sources as in make_fig_architectures.py. Out: figures_canonical/fig_valley_units.png (not
a paper figure; Paper B material)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.6, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DC = HERE.parent / "data_canonical"
OUT = HERE.parent / "figures_canonical" / "fig_valley_units.png"
SRC = [("Pythia-160m", "run37_inferential_nulls.json", "pythia-160m", "#c6dbef"),
       ("Pythia-410m", "run37_inferential_nulls.json", "pythia-410m-deduped", "#9ecae1"),
       ("Pythia-1B", "run37_inferential_nulls.json", "pythia-1b-deduped", "#4292c6"),
       ("Pythia-2.8B", "run37_inferential_nulls.json", "pythia-2.8b-deduped", "#08519c"),
       ("RedPajama-INCITE-3B", "run47_fourth_cell_redpajama.json", "RedPajama-INCITE-Base-3B-v1", "#3182bd"),
       ("GPT-Neo-1.3B", "run37_inferential_nulls.json", "gpt-neo-1.3B", "#6baed6"),
       ("Mamba-2.8B", "run53_mamba_fifth_cell.json", "mamba-2.8b-hf", "#d95f0e"),
       ("OLMo-2-1B", "run54_olmo2_1b_construction.json", "OLMo-2-0425-1B", "#31a354")]
cache = {}
def last_cross(y):
    L = len(y) - 1; i = L
    while i > 0 and y[i - 1] >= 0: i -= 1
    return 0.0 if i == 0 else (i - 1 + (0 - y[i - 1]) / (y[i] - y[i - 1]))
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
plt.subplots_adjust(wspace=0.32, left=0.07, right=0.99, top=0.86, bottom=0.2)
rows = []
for lab, f, k, col in SRC:
    d = cache.setdefault(f, json.load(open(DC / f)))
    try: L = d["models"][k]["axes"]["language_type"]["layers"]
    except KeyError: print("missing", lab); continue
    y = np.array([l["delta"] for l in L]); n = len(y); ell = np.arange(n)
    v = int(np.argmin(y[: max(2, n // 2)])); v = max(v, 1)   # the valley: the construction minimum in the first half
    c = last_cross(y)
    rows.append((lab, n - 1, v, c, y[v], y[-1]))
    axes[0].plot(ell / (n - 1), y, "-", color=col, lw=1.3, label=lab)
    axes[1].plot(ell / v, y, "-", color=col, lw=1.3)
    axes[2].plot(ell / c if c > 0 else ell, y, "-", color=col, lw=1.3)
for ax, t, xl in zip(axes, ["(a) depth over total depth", "(b) depth in valley units", "(c) depth in crossing units"],
                     ["$\\ell/L$", "$\\ell/\\ell_{\\mathrm{valley}}$", "$\\ell/\\ell_{\\mathrm{cross}}$"]):
    ax.axhline(0, color="k", lw=0.6); ax.set_title(t); ax.set_xlabel(xl); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("construction $\\delta$, declared order"); axes[0].legend(frameon=False, ncol=2, fontsize=5.8, loc="lower right")
axes[1].set_xlim(0, 12); axes[2].set_xlim(0, 4)
OUT.parent.mkdir(exist_ok=True); fig.savefig(OUT); print("wrote", OUT)
print("model, L, valley layer, crossing layer, delta at valley, delta at end")
for r in rows: print("%-20s L=%2d valley=%2d cross=%5.1f  d_valley=%+.3f d_end=%+.3f  valley/L=%.2f cross/L=%.2f" % (r[0], r[1], r[2], r[3], r[4], r[5], r[2]/r[1], r[3]/r[1]))
