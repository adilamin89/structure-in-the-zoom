"""Figure 7 — the S73 inferential results in one figure.

(a) Pythia-2.8B: declared-path delta (solid) against its 500-permutation null
    band, and the order-averaged deltabar (dashed) against its own null band,
    for the content and structural axes. The crossover shape belongs to the
    declared path; the order-averaged statistic is flat and label-linked at
    nearly every depth.
(b) Two null levels on BLiMP (design B, 64 pairs): plain delta against the
    label-free permutation band and the within-pair swap band. The swap null
    absorbs the whole signal.
(c) Two null levels on the Baroni contrasts at 64 pairs: a label-linked
    residual emerges beyond the swap band at mid and late depth.
(d) Spontaneous sessions: circular-shift z against frame-permutation z per
    session for the three state axes; time blocks pass the permutation but
    not the shift.

Sources: run37_inferential_nulls.json, run42_blimp_battery.json,
run43b_baroni_64pairs.json, run40_spont_state_axis.json.
Out: ../figures_canonical/fig_nulls.png (+ ../../arxiv/figures/ if present)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DC = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_nulls.png",
        HERE.parent.parent / "arxiv" / "figures" / "fig_nulls.png"]

r37 = json.load(open(DC / "run37_inferential_nulls.json"))
r42 = json.load(open(DC / "run42_blimp_battery.json"))
r43 = json.load(open(DC / "run43b_baroni_64pairs.json"))
r40 = json.load(open(DC / "run40_spont_state_axis.json"))

fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6),
                         gridspec_kw={"width_ratios": [1.25, 1, 1, 0.95]})

# (a) declared path vs order-averaged, Pythia-2.8B
ax = axes[0]
colors = {"world_knowledge": "#d95f0e", "language_type": "#2171b5"}
names = {"world_knowledge": "content", "language_type": "structural"}
for axis_name, col in colors.items():
    L = r37["models"]["pythia-2.8b-deduped"]["axes"][axis_name]["layers"]
    x = np.array([l["layer"] for l in L]) / (len(L) - 1)
    d = np.array([l["delta"] for l in L])
    nm = np.array([l["null_mean"] for l in L]); ns = np.array([l["null_sd"] for l in L])
    da = np.array([l["delta_orderavg"] for l in L])
    nma = np.array([l["null_mean_orderavg"] for l in L]); nsa = np.array([l["null_sd_orderavg"] for l in L])
    ax.fill_between(x, nm - 2 * ns, nm + 2 * ns, color=col, alpha=0.10, lw=0)
    ax.fill_between(x, nma - 2 * nsa, nma + 2 * nsa, color=col, alpha=0.22, lw=0)
    ax.plot(x, d, "-", color=col, lw=1.8, label=f"{names[axis_name]}: declared path")
    ax.plot(x, da, "--", color=col, lw=1.6, label=f"{names[axis_name]}: order-averaged")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("normalized depth $\\ell/L$")
ax.set_ylabel("$\\delta$")
ax.set_title("(a) Pythia-2.8B: path vs partition", fontsize=10.5)
ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=1)
ax.text(0.98, 0.04, "bands: $\\pm 2$ SD of the 500-permutation null\n(light: declared; dark: order-averaged)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="0.35")

# (b) BLiMP two nulls
def two_null_panel(ax, layers, title, ylim=None):
    x = np.array([l["layer"] for l in layers]) / (len(layers) - 1)
    d = np.array([l["delta_plain"] for l in layers])
    pm = np.array([l["perm_null_mean"] for l in layers]); ps = np.array([l["perm_null_sd"] for l in layers])
    sm = np.array([l["strat_null_mean"] for l in layers]); ss = np.array([l["strat_null_sd"] for l in layers])
    ax.fill_between(x, pm - 2 * ps, pm + 2 * ps, color="0.6", alpha=0.35, lw=0,
                    label="label-free permutation ($\\pm 2$ SD)")
    ax.fill_between(x, sm - 2 * ss, sm + 2 * ss, color="#c51b7d", alpha=0.25, lw=0,
                    label="within-pair swap ($\\pm 2$ SD)")
    ax.plot(x, d, "-", color="k", lw=1.8, label="observed $\\delta$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("normalized depth $\\ell/L$")
    ax.set_title(title, fontsize=10.5)
    if ylim:
        ax.set_ylim(*ylim)

two_null_panel(axes[1], r42["models"]["pythia-2.8b-deduped"]["B_grammaticality"]["layers"],
               "(b) BLiMP, 64 pairs: composition only")
axes[1].legend(fontsize=7, loc="lower left", frameon=False)
two_null_panel(axes[2], r43["models"]["pythia-2.8b-deduped"]["layers"],
               "(c) Baroni, 64 pairs: signal beyond carriers")

# (d) spont two nulls
ax = axes[3]
mk = {"runspeed8": ("o", "#1b9e77", "running speed"),
      "pupil8": ("s", "#7570b3", "pupil area"),
      "time_block8": ("^", "#d95f02", "time blocks")}
for tag, (m, c, lab) in mk.items():
    zp = [r[tag]["perm_z"] for r in r40["rows"]]
    zs = [r[tag]["shift_z"] for r in r40["rows"]]
    ax.scatter(zp, zs, marker=m, color=c, s=34, label=lab, zorder=3, edgecolor="k", lw=0.4)
for v in (-2, 2):
    ax.axhline(v, color="0.6", lw=0.7, ls=":")
    ax.axvline(v, color="0.6", lw=0.7, ls=":")
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
lim = 18
ax.plot([-lim, lim], [-lim, lim], color="0.8", lw=0.7)
ax.set_xlim(-lim, lim); ax.set_ylim(-9.5, 6)
ax.set_xlabel("$z$ vs frame permutation")
ax.set_ylabel("$z$ vs circular shift")
ax.set_title("(d) spontaneous sessions: two nulls", fontsize=10.5)
ax.legend(fontsize=7, loc="upper left", frameon=False)

for a in axes:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.tight_layout(w_pad=1.2)
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists():
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(f"wrote {out}")
