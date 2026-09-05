"""Figure (appendix): the two nulls in vivo on the spontaneous sessions (S77;
panel (d) of the S73 fig_nulls.png). Source: run40_spont_state_axis.json.
Out: ../figures_canonical/fig_state_axes.png + ../../arxiv/figures/
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DC = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_state_axes.png", HERE.parent.parent / "arxiv" / "figures" / "fig_state_axes.png"]
r40 = json.load(open(DC / "run40_spont_state_axis.json"))
fig, ax = plt.subplots(figsize=(3.4, 2.7))
mk = {"runspeed8": ("o", "#1b9e77", "running speed"), "pupil8": ("s", "#7570b3", "pupil area"), "time_block8": ("^", "#d95f02", "time blocks")}
for tag, (m, c, lab) in mk.items():
    ax.scatter([r[tag]["perm_z"] for r in r40["rows"]], [r[tag]["shift_z"] for r in r40["rows"]],
               marker=m, color=c, s=34, label=lab, zorder=3, edgecolor="k", lw=0.4)
for v in (-2, 2):
    ax.axhline(v, color="0.6", lw=0.7, ls=":"); ax.axvline(v, color="0.6", lw=0.7, ls=":")
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
ax.plot([-18, 18], [-18, 18], color="0.8", lw=0.7); ax.set_xlim(-18, 18); ax.set_ylim(-9.5, 6)
ax.set_xlabel("$z$ vs frame permutation"); ax.set_ylabel("$z$ vs circular shift")
ax.set_title("Spontaneous sessions: two nulls"); ax.legend(fontsize=6.4, loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
