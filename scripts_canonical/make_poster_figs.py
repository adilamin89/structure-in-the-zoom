"""Poster panels (NeurReps 2026), drawn at print size from the released artifacts; no number typed by hand.
(1) poster_three_axes.png: recording GT3, three exponents split into floor (gray) and shift (red):
    random stimuli (stringer_theta_decomposition), stimulus direction (stringer_mismatch_ablation, dir8),
    contiguous neurons (stringer_neuron_zoom_spatial).
(2) poster_split.png: per grating recording, the direction shift and its two exact parts, within-class covariance and
    class means (run72_paper_designs_split, design A, all neurons).
Out: ../../neurreps_camera_ready/poster/ and ../figures_canonical/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
BM = HERE.parent.parent.parent / "basin_memory" / "data"
OUTS = [p for p in (HERE.parent.parent / "neurreps_camera_ready" / "poster",) if p.parent.exists()] + [HERE.parent / "figures_canonical"]
for o in OUTS:
    o.mkdir(parents=True, exist_ok=True)

def bm(name, local):
    """Raw-analysis file from basin_memory/data in the working tree, else its copy in data_canonical (public release)."""
    p = BM / name
    return p if p.exists() else DATA / local

plt.rcParams.update({"font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
                     "legend.fontsize": 17, "savefig.dpi": 300})
RED, GRAY, BLUE = "#b40426", "#9a9a9a", "#2166ac"


def save(fig, name):
    for o in OUTS:
        fig.savefig(o / name, dpi=300, bbox_inches="tight")
    print("wrote", name)


def gt3(rows, key="name"):
    return next(r for r in rows if "drifting_GT3" in r[key])


# ---- (1) three axes on GT3 ----
rnd = gt3(json.load(open(bm("stringer_theta_decomposition.json", "theta_decomposition.json")))["rows"])
dirn = gt3(json.load(open(bm("stringer_mismatch_ablation.json", "mismatch_ablation.json")))["rows"])["dir8"]
spat_all = json.load(open(bm("stringer_neuron_zoom_spatial.json", "neuron_zoom_spatial.json")))
spat_rows = spat_all.get("rows") or spat_all.get("results") or spat_all
spat = gt3(spat_rows if isinstance(spat_rows, list) else list(spat_rows.values()))
fl_s = spat.get("theta_floor", spat.get("theta_floor_random"))
d_s = spat.get("delta", spat.get("delta_spatial"))
axes = [("random\nstimuli", rnd["theta_floor_empirical"], rnd["shift_obs"]),
        ("stimulus\ndirection", dirn["theta_floor"], dirn["delta"]),
        ("contiguous\nneurons", fl_s, d_s)]
fig, ax = plt.subplots(figsize=(8.0, 5.2))
for i, (lab, fl, d) in enumerate(axes):
    ax.bar(i, fl, color=GRAY, width=0.62, label="floor" if i == 0 else None)
    ax.bar(i, d, bottom=fl, color=RED, width=0.62, label=r"shift $\delta$" if i == 0 else None)
    ax.text(i, fl + max(d, 0) + 0.012, f"{fl + d:.2f}", ha="center", fontsize=22, weight="bold")
    ax.text(i, fl / 2, f"{fl:.2f}", ha="center", va="center", fontsize=17, color="white")
    if d > 0.05:
        ax.text(i, fl + d / 2, f"+{d:.2f}", ha="center", va="center", fontsize=17, color="white")
ax.set_xticks(range(3)); ax.set_xticklabels([a[0] for a in axes])
ax.set_ylabel("scaling exponent")
ax.set_ylim(0, 0.42)
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "poster_three_axes.png"); plt.close(fig)

# ---- (2) the split of the shift ----
m = json.load(open(DATA / "run72_paper_designs_split.json"))["rows"]
keys = ["D1", "D2", "D3", "L1", "L2", "L3", "C1", "C2"]
d = [m[k]["all"]["A"]["delta"] for k in keys]
w = [m[k]["all"]["A"]["slope_noise"] for k in keys]
b = [m[k]["all"]["A"]["slope_probe"] for k in keys]
fig, ax = plt.subplots(figsize=(10.0, 5.2))
x = range(len(keys)); wd = 0.27
ax.bar([i - wd for i in x], w, wd, color=BLUE, label="within-class fluctuations")
ax.bar(list(x), d, wd, color=RED, label=r"shift $\delta$")
ax.bar([i + wd for i in x], b, wd, color=GRAY, label="class means")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(list(x)); ax.set_xticklabels(keys)
ax.set_xlabel("grating recording (D drifting, L localized, C low contrast)")
ax.set_ylabel("slope relative to floor")
ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16))
ax.spines[["top", "right"]].set_visible(False)
save(fig, "poster_split.png"); plt.close(fig)
print("three axes:", [(a[0].replace(chr(10), " "), round(a[1], 3), round(a[2], 3)) for a in axes])
