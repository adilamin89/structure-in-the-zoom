"""C3 (Oct 16): what the floor measures. For every random-stimulus ladder (18 recordings), the local floor slope between
consecutive rungs against n / d_eff (d_eff = PR at the full stimulus set). The iid reference (Eq. 2) gives local slope
~ d/(d+n) = 1/(1+n/d). Also: the random ladder's slope restricted to the direction ladder's window (>= 500 stimuli)
against the direction ladder's floor. Out: data_canonical/c3_floor_crossover.json, figures_canonical/fig_floor_crossover.png"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
def load(n):
    p = HERE.parent / "data_canonical" / n
    return json.load(open(p if p.exists() else HERE.parent / "data" / n))
rows = [r for r in load("stringer_theta_decomposition.json")["rows"] if r.get("status") == "ok"]
oz = {r["name"]: r for r in load("orientation_zoom.json")["results"]}
pts, win = [], []
for r in rows:
    z = sorted((a[1], a[2]) for a in r["zoom_ladder"]); n = np.array([a for a, _ in z], float); pr = np.array([b for _, b in z])
    d = r["d_eff_pop"]; kind = r["name"].split("_")[0] + ("_" + r["name"].split("_")[1] if r["name"].startswith("gratings") else "")
    ln, lp = np.log(n), np.log(pr)
    for k in range(len(n) - 1):
        pts.append(dict(name=r["name"], kind=kind, x=float(np.sqrt(n[k] * n[k + 1]) / d), s=float((lp[k + 1] - lp[k]) / (ln[k + 1] - ln[k]))))
    if r["name"] in oz and r["name"].startswith("gratings"):
        m = n >= 500
        win.append(dict(name=r["name"], high_window=float(np.polyfit(ln[m], lp[m], 1)[0]), direction_floor=oz[r["name"]]["theta_random"],
                        full_window=float(np.polyfit(ln, lp, 1)[0]), d_eff=d))
x = np.array([p["x"] for p in pts]); s = np.array([p["s"] for p in pts])
iid = 1 / (1 + x)
resid = s - iid
out = dict(n_points=len(pts), corr_local_slope_vs_iid=float(np.corrcoef(s, iid)[0, 1]),
           median_ratio_measured_over_iid=float(np.median(s / iid)), window=win, points=pts)
json.dump(out, open(HERE.parent / "data_canonical" / "c3_floor_crossover.json", "w"), indent=1)
print("points", len(pts), "corr(measured, iid)", round(out["corr_local_slope_vs_iid"], 3), "median measured/iid", round(out["median_ratio_measured_over_iid"], 2))
for w in win: print(w["name"][:34], "high-window", round(w["high_window"], 3), "direction floor", round(w["direction_floor"], 3))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
col = {"gratings_drifting": "#b40426", "gratings_local": "#e8743b", "gratings_low": "#7a4fa3", "static_biased": "#4d6b8a", "spont": "#59a14f"}
for p in pts:
    k = next(c for c in col if p["kind"].startswith(c) or p["name"].startswith(c))
    ax[0].plot(p["x"], p["s"], "o", ms=5, color=col[k], alpha=0.8)
g = np.logspace(-2.2, 2, 200); ax[0].plot(g, 1 / (1 + g), "--", color="0.4", label="iid reference (Eq. 2): 1/(1+n/d)")
ax[0].set_xscale("log"); ax[0].set_xlabel("n / d_eff  (how far up the crossover the rung sits)"); ax[0].set_ylabel("local floor slope")
for k, c in col.items(): ax[0].plot([], [], "o", color=c, label=k.replace("_", " "))
ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("A  every random ladder, rung by rung", loc="left")
hw = [w["high_window"] for w in win]; dfl = [w["direction_floor"] for w in win]
ax[1].plot(hw, dfl, "o", color="#b40426", ms=8); ax[1].plot([0, 0.16], [0, 0.16], "--", color="0.5")
for w in win: ax[1].annotate(w["name"].split("_")[1][:5] + " " + w["name"].split("_")[-3] if False else "", (w["high_window"], w["direction_floor"]))
ax[1].set_xlabel("random ladder's own slope, rungs >= 500 stimuli"); ax[1].set_ylabel("direction ladder's floor")
ax[1].set_title("B  same window, same floor (8 grating recordings)", loc="left")
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(HERE.parent / "figures_canonical" / "fig_floor_crossover.png", dpi=150)
print("wrote figure")
