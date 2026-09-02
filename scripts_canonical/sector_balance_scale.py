"""Sector balance across scales (S74, 2026-09-02): is the dipole/quadrupole balance
of the class-mean correlation a single-neuron property, and does it flow under
coarse-graining? Plus the balance per Allen visual area.

Follows multipole_harmonics_8dir.py (localized gratings dipole-dominant) and
local_vs_fullfield_tuning.py (single-neuron DSI up, OSI down). Exploratory pass
on 2026-09-02 found the identities below; this script formalizes them.

A. ADDITIVITY (identity, checked numerically): the eight-class circular profile
   of the class-mean correlation is the trace-normalized class-mean covariance
   across neurons, so its harmonic coefficients are sums over neurons of each
   neuron's tuning-curve harmonic power. Prediction b2/c1 from per-neuron
   harmonics (sum_n |z2_n|^2 / sum_n |z1_n|^2 on class-standardized tuning
   curves) must equal the profile's b2/c1 to numerical precision.
B. COARSE-GRAINING FLOW: average K neurons per block and recompute b2/c1 for
   K = 1..64 with blocks formed by (i) preferred-orientation sorting, (ii)
   preferred-direction sorting, (iii) random assignment. Expectation from the
   exploratory pass: random blocks leave b2/c1 invariant (label-blind graining
   preserves the balance); orientation-sorted blocks amplify the quadrupole
   (antipodal preferences share a block and their odd harmonics cancel) and
   cross the localized recordings to quadrupole-dominant by K ~ 8; direction-
   sorted blocks amplify the dipole. Mesoscale readouts therefore inherit the
   balance set by the anatomy's blocking rule: orientation columns act as
   orientation-sorted blocks (cross-species prediction, Paper C).
C. ALLEN PER-AREA BALANCE: b_quadrupole / |c_dipole| per session-area population
   from allen_multipoles_all_sessions.json (full-field drifting gratings); the
   fraction quadrupole-dominant per area and the median J1/J2.

Out: ../data_canonical/sector_balance_scale.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
DATA = HERE.parent / "data_canonical"
OUT = DATA / "sector_balance_scale.json"
NB = 8
KS = [1, 2, 4, 8, 16, 32, 64]


def profile_ratio(M):
    Cm = np.corrcoef(M)
    prof = np.array([np.mean([Cm[i, (i + k) % NB] for i in range(NB)]) for k in range(NB)])
    ang = 2 * np.pi * np.arange(NB) / NB
    c1 = 2 / NB * np.sum(prof * np.cos(ang))
    b2 = 2 / NB * np.sum(prof * np.cos(2 * ang))
    return float(b2 / c1), float(b2), float(c1)


def main():
    rows = []
    for f in sorted(RAW.glob("gratings_*.npy")):
        dat = np.load(f, allow_pickle=True).item()
        X = dat["sresp"].T.astype(np.float64)
        phi = np.asarray(dat["istim"]).ravel() % (2 * np.pi)
        b = np.floor(phi / (2 * np.pi) * NB).astype(int) % NB
        counts = np.bincount(b, minlength=NB)
        if counts.min() < 10:
            rows.append({"name": f.stem, "status": "degenerate_bins"}); continue
        M = np.stack([X[b == k].mean(0) for k in range(NB)])
        ang = 2 * np.pi * np.arange(NB) / NB
        meas, b2, c1 = profile_ratio(M)
        Ms = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
        Msc = Ms - Ms.mean(0)
        P1 = float(np.sum(np.abs((Msc * np.exp(1j * ang[:, None])).sum(0)) ** 2))
        P2 = float(np.sum(np.abs((Msc * np.exp(2j * ang[:, None])).sum(0)) ** 2))
        pred = P2 / P1
        Mc = M - M.mean(0)
        pref_ori = (np.angle((Mc * np.exp(2j * ang[:, None])).sum(0)) / 2) % np.pi
        pref_dir = np.angle((Mc * np.exp(1j * ang[:, None])).sum(0)) % (2 * np.pi)
        rng = np.random.default_rng(0)
        flow = {"K": KS, "ori_sorted": [], "dir_sorted": [], "random": []}
        for K in KS:
            nblk = M.shape[1] // K
            for key, order in (("ori_sorted", np.argsort(pref_ori)), ("dir_sorted", np.argsort(pref_dir)),
                               ("random", rng.permutation(M.shape[1]))):
                Mb = np.stack([M[:, order[i * K:(i + 1) * K]].mean(1) for i in range(nblk)], 1)
                flow[key].append(profile_ratio(Mb)[0])
        rows.append({"name": f.stem, "status": "ok", "b2_over_c1_measured": meas, "b2": b2, "c1": c1,
                     "b2_over_c1_from_neuron_harmonics": pred, "additivity_ratio": meas / pred, "graining_flow": flow})
        print(f"  {f.stem[:32]:32s} meas {meas:.3f} pred {pred:.3f} ratio {meas/pred:.4f} | K=64 ori {flow['ori_sorted'][-1]:.2f} "
              f"dir {flow['dir_sorted'][-1]:.2f} rand {flow['random'][-1]:.2f} | first K with ori-sorted > 1: "
              f"{next((K for K, v in zip(KS, flow['ori_sorted']) if v > 1), None)}")
    # Allen per-area
    al = json.load(open(DATA / "allen_multipoles_all_sessions.json"))
    per_area = {}
    for s in al["results"]:
        if s.get("status") not in (None, "ok"):
            continue
        for area, a in (s.get("areas") or {}).items():
            if not isinstance(a, dict) or "b_quadrupole" not in a or "c_dipole" not in a:
                continue
            per_area.setdefault(area, []).append(a)
    allen = {}
    for area, v in per_area.items():
        r = np.array([x["b_quadrupole"] / abs(x["c_dipole"]) if abs(x["c_dipole"]) > 1e-9 else np.nan for x in v])
        allen[area] = {"n": len(v), "median_b2_over_c1": float(np.nanmedian(r)),
                       "frac_quadrupole_dominant": float(np.nanmean(r > 1)),
                       "median_J1_over_J2": float(np.nanmedian([x.get("J1_over_J2", np.nan) for x in v])),
                       "median_cardinal_fraction": float(np.nanmedian([x.get("cardinal_fraction", np.nan) for x in v]))}
    for area in sorted(allen, key=lambda a: -allen[a]["n"]):
        a = allen[area]
        print(f"  Allen {area:6s} n={a['n']:3d} median b2/|c1| {a['median_b2_over_c1']:.2f} frac quad-dominant {a['frac_quadrupole_dominant']:.2f} "
              f"J1/J2 {a['median_J1_over_J2']:.2f} cardinal {a['median_cardinal_fraction']:.2f}")
    ok = [r for r in rows if r["status"] == "ok"]
    verdict = {"A_additivity_max_abs_dev": float(max(abs(r["additivity_ratio"] - 1) for r in ok)),
               "B_random_blocks_invariant_max_rel_change_K64": float(max(abs(r["graining_flow"]["random"][-1] / r["graining_flow"]["random"][0] - 1) for r in ok)),
               "B_ori_sorted_all_quadrupole_dominant_by_K": {r["name"]: next((K for K, v in zip(KS, r["graining_flow"]["ori_sorted"]) if v > 1), None) for r in ok},
               "B_dir_sorted_monotone_decrease": bool(all(all(np.diff(r["graining_flow"]["dir_sorted"]) <= 1e-9) for r in ok)),
               "C_allen_frac_quadrupole_dominant_VISp": allen.get("VISp", {}).get("frac_quadrupole_dominant")}
    print("VERDICT", json.dumps(verdict, indent=1))
    json.dump({"rows": rows, "allen_per_area": allen, "verdict": verdict}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
