"""Why localized gratings are dipole-dominant: per-neuron tuning under the three
grating conditions (drifting full-field, localized, low-contrast full-field).

CONTEXT (S74, 2026-09-02): multipole_harmonics_8dir.py showed the class-mean
correlation profile is quadrupole-dominant (|b2| > |c1|) in the five full-field
recordings and dipole-dominant in the three localized-grating recordings. This
script FORMALIZES the same-day exploratory pass that asked why. It is not a
prediction: the expectations below are the exploratory findings, written down so
the released artifact carries them with the code that computes them.

MEASURES (per recording, 8 direction classes, class means over stimuli):
  - tuned fraction: one-way ANOVA across the 8 classes, p < 0.01 per neuron
  - OSI = |sum_k R_k e^{2i theta_k}| / sum_k R_k, DSI = |sum_k R_k e^{i theta_k}|
    / sum_k R_k, on min-subtracted class means (tuned neurons)
  - direction-preference cardinal fraction (within 22.5 deg of 0/90/180/270)
    among DSI > 0.3 neurons; chance 0.50 (a retinal DSGC signature would raise it)
  - spatial clustering in the imaging plane: r between a neuron's OSI (DSI) and
    the mean OSI (DSI) of its 10 nearest neighbours, vs a label shuffle
  - coverage terciles: neurons ranked by the mean OSI of their 30 nearest
    neighbours (a retinotopic proxy for how much of the aperture their receptive
    field sees); OSI, DSI and response range per tercile
  - b2/c1 of the class-mean correlation profile for all neurons, tuned-only,
    top-10% tuned, and z-scored neurons (subpopulation-artifact controls)

EXPECTATIONS (from the exploratory pass; recorded, not predicted):
  X1 localized: median DSI higher (~0.41-0.46 vs 0.33-0.36 drifting) and median
     OSI lower (~0.37-0.40 vs 0.46-0.50); fraction DSI > 0.5 up to ~0.43.
  X2 b2/c1 < 1 for localized under all four subpopulation controls (not a
     mixing or baseline artifact).
  X3 cardinal fraction at chance in all conditions (no retinal-DS signature).
  X4 OSI spatially clustered under localized gratings (r ~0.2-0.3 vs ~0.05
     full-field): the aperture's retinotopic footprint; DSI not clustered (~0.05).
  X5 DSI flat across coverage terciles under localized gratings while OSI and
     response range rise toward the footprint centre: the direction-selectivity
     gain is global, not an aperture-edge effect.

Out: ../data_canonical/local_vs_fullfield_tuning.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = HERE.parent / "data_canonical" / "local_vs_fullfield_tuning.json"
NB = 8


def profile_b2_c1(M):
    Cm = np.corrcoef(M)
    prof = np.array([np.mean([Cm[i, (i + k) % NB] for i in range(NB)]) for k in range(NB)])
    ang = 2 * np.pi * np.arange(NB) / NB
    c1 = 2 / NB * np.sum(prof * np.cos(ang))
    b2 = 2 / NB * np.sum(prof * np.cos(2 * ang))
    return float(b2 / c1)


def analyze(f, rng):
    dat = np.load(f, allow_pickle=True).item()
    X = dat["sresp"].T.astype(np.float64)
    phi = np.asarray(dat["istim"]).ravel() % (2 * np.pi)
    b = np.floor(phi / (2 * np.pi) * NB).astype(int) % NB
    counts = np.bincount(b, minlength=NB)
    if counts.min() < 10:
        return {"name": f.stem, "status": "degenerate_bins", "bin_counts": counts.tolist()}
    M = np.stack([X[b == k].mean(0) for k in range(NB)])
    n = X.shape[0]
    grand = X.mean(0)
    ssb = sum(counts[k] * (M[k] - grand) ** 2 for k in range(NB))
    ssw = sum(((X[b == k] - M[k]) ** 2).sum(0) for k in range(NB))
    F = (ssb / (NB - 1)) / (ssw / (n - NB) + 1e-12)
    tuned = (1 - stats.f.cdf(F, NB - 1, n - NB)) < 0.01
    R = M - M.min(0, keepdims=True) + 1e-9
    ang = 2 * np.pi * np.arange(NB) / NB
    z1 = (R * np.exp(1j * ang[:, None])).sum(0) / R.sum(0)
    z2 = (R * np.exp(2j * ang[:, None])).sum(0) / R.sum(0)
    dsi, osi = np.abs(z1), np.abs(z2)
    pdir = np.degrees(np.angle(z1)) % 360
    ds = tuned & (dsi > 0.3)
    dd = pdir[ds] % 90
    cardinal = float(((dd < 22.5) | (dd > 67.5)).mean())
    pos = np.array([[s["med"][0], s["med"][1]] for s in dat["stat"]], float)
    tree = cKDTree(pos)
    _, nb10 = tree.query(pos, k=11)
    _, nb30 = tree.query(pos, k=31)
    r_dsi = float(np.corrcoef(dsi[tuned], dsi[nb10[:, 1:]].mean(1)[tuned])[0, 1])
    r_osi = float(np.corrcoef(osi[tuned], osi[nb10[:, 1:]].mean(1)[tuned])[0, 1])
    shuf = []
    for _ in range(20):
        d2 = rng.permutation(dsi)
        shuf.append(np.corrcoef(d2[tuned], d2[nb10[:, 1:]].mean(1)[tuned])[0, 1])
    nbosi = osi[nb30[:, 1:]].mean(1)
    q = np.quantile(nbosi[tuned], [1 / 3, 2 / 3])
    terc = [tuned & (nbosi < q[0]), tuned & (nbosi >= q[0]) & (nbosi < q[1]), tuned & (nbosi >= q[1])]
    rng_resp = M.max(0) - M.min(0)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Mz = np.stack([Z[b == k].mean(0) for k in range(NB)])
    top = F >= np.quantile(F, 0.9)
    return {"name": f.stem, "status": "ok", "n_stimuli": int(n), "n_neurons": int(X.shape[1]),
            "bin_counts": counts.tolist(), "tuned_fraction": float(tuned.mean()),
            "median_osi_tuned": float(np.median(osi[tuned])), "median_dsi_tuned": float(np.median(dsi[tuned])),
            "frac_dsi_gt_0p5_tuned": float((dsi[tuned] > 0.5).mean()), "frac_ds_dsi_gt_0p3": float(ds.mean()),
            "cardinal_direction_fraction": cardinal, "corr_osi_dsi_tuned": float(np.corrcoef(osi[tuned], dsi[tuned])[0, 1]),
            "spatial_r_dsi": r_dsi, "spatial_r_dsi_shuffle": float(np.mean(shuf)), "spatial_r_osi": r_osi,
            "coverage_terciles": {"dsi": [float(dsi[t].mean()) for t in terc], "osi": [float(osi[t].mean()) for t in terc],
                                  "response_range": [float(rng_resp[t].mean()) for t in terc]},
            "b2_over_c1": {"all": profile_b2_c1(M), "tuned_only": profile_b2_c1(M[:, tuned]),
                           "top10pct_tuned": profile_b2_c1(M[:, top]), "zscored": profile_b2_c1(Mz)}}


def main():
    rng = np.random.default_rng(0)
    rows = [analyze(f, rng) for f in sorted(RAW.glob("gratings_*.npy"))]
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['name'][:32]:32s} degenerate"); continue
        t = r["coverage_terciles"]; bb = r["b2_over_c1"]
        print(f"  {r['name'][:32]:32s} tuned {r['tuned_fraction']:.2f} OSI {r['median_osi_tuned']:.2f} DSI {r['median_dsi_tuned']:.2f} "
              f"DSI>.5 {r['frac_dsi_gt_0p5_tuned']:.2f} cardinal {r['cardinal_direction_fraction']:.2f} | spatial r OSI {r['spatial_r_osi']:+.2f} "
              f"DSI {r['spatial_r_dsi']:+.2f} (shuf {r['spatial_r_dsi_shuffle']:+.2f}) | DSI by coverage {np.round(t['dsi'],3).tolist()} "
              f"OSI {np.round(t['osi'],2).tolist()} | b2/c1 all {bb['all']:.2f} tuned {bb['tuned_only']:.2f} top10 {bb['top10pct_tuned']:.2f} z {bb['zscored']:.2f}")
    ok = [r for r in rows if r["status"] == "ok"]
    loc = [r for r in ok if "local" in r["name"]]; ff = [r for r in ok if "local" not in r["name"]]
    verdict = {
        "X1_dsi_up_osi_down_localized": bool(min(r["median_dsi_tuned"] for r in loc) > max(r["median_dsi_tuned"] for r in ff)
                                             and max(r["median_osi_tuned"] for r in loc) < min(r["median_osi_tuned"] for r in ff)),
        "X2_b2_over_c1_below_1_all_controls_localized": bool(all(v < 1 for r in loc for v in r["b2_over_c1"].values())),
        "X3_cardinal_at_chance_all": bool(all(abs(r["cardinal_direction_fraction"] - 0.5) < 0.05 for r in ok)),
        "X4_osi_clustered_localized_not_fullfield": bool(min(r["spatial_r_osi"] for r in loc) > 0.15 and max(r["spatial_r_osi"] for r in ff) < 0.1
                                                       and max(r["spatial_r_dsi"] for r in ok) < 0.1),
        "X5_dsi_flat_across_coverage_localized": bool(all(max(r["coverage_terciles"]["dsi"]) - min(r["coverage_terciles"]["dsi"]) < 0.02 for r in loc)
                                                    and all(r["coverage_terciles"]["osi"][2] - r["coverage_terciles"]["osi"][0] > 0.05 for r in loc)),
    }
    print("VERDICT", json.dumps(verdict, indent=1))
    json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
