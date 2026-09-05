"""Run 59 (S77, 2026-09-05) - Does the direction-aligned shift live in the
direction-selective neurons?

WHY: across the eight grating recordings the direction-aligned shift grows
with the dipole share of the code (Spearman +0.86 against c1/b2, -0.90
against the antipodal correlation C(180)). The calibrated model carries the
dipole in its class means and misses each recording by an amount monotone in
that share; runs 56-58 showed that the within-class rotation (matched with an
odd gain field), the gain field's baseline (removed) and the within-class
spectrum (grows 22-28% alike on all three) are not what orders the
recordings. A homogeneous Gaussian model with dense modes cannot represent a
subpopulation whose rate, and with it its variability, vanishes at the
antipode of its preferred direction. This run asks the data directly.

DESIGN: for each of the eight grating recordings, per-neuron DSI and OSI by
vector strength on the eight class means (tuned neurons by F-test p < 0.01,
as in local_vs_fullfield_tuning.py); three neuron subsets of equal size
n = floor(n_tuned / 3): the top third by DSI ("DS"), the bottom third by DSI
("non-DS"), and a random third; on each subset the eight-class direction
ladder (1, 2, 3, 4, 6, 8 classes, angular order) against ten-draw floors of
the same trial counts, the shift delta, three label shuffles, and the
subset's own sector balance b2/c1 from the class-mean correlation profile.
Also per recording: the fraction of tuned neurons with DSI > 0.3 and the
median DSI, correlated with the full-population shift across recordings.

REGISTERED EXPECTATIONS (before the run):
H1: delta(DS third) > delta(non-DS third) in 8 of 8 recordings.
H2: the sorting does what it should: the DS third is dipole-dominant
    (b2/c1 < 1) and the non-DS third quadrupole-dominant (b2/c1 > 1) in 8 of 8.
H3: the full-population shift lies between the two subset shifts in 8 of 8
    (DS above, non-DS below), and the random third lies within 0.05 of the
    full-population value.
H4 (the material constant): across the eight recordings the full-population
    shift tracks the DS fraction (DSI > 0.3 among tuned neurons) with
    Spearman rho >= 0.8, i.e. the recording-to-recording variation of the
    shift is the fraction of direction-selective neurons.
A miss is reported at full volume.

Out: ../data_canonical/run59_shift_by_direction_selectivity.json (+ .log)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = DATA / "run59_shift_by_direction_selectivity.json"
spec = importlib.util.spec_from_file_location("run2", HERE / "run2_calibrated_corotating.py")
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
NB = 8
N_SHUF = 3


def short(name):
    kind = "D" if "drifting" in name else "L" if "local" in name else "C"
    return kind + name.split("GT")[1][0]


def profile_b2_c1(M):
    """DFT of the circular profile of the class-mean correlation (as in
    multipole_harmonics_8dir): returns b2/c1 and the profile."""
    Mz = (M - M.mean(0, keepdims=True)) / (M.std(0, keepdims=True) + 1e-9)
    C = np.corrcoef(Mz)
    prof = np.array([np.mean([C[i, (i + k) % NB] for i in range(NB)]) for k in range(NB)])
    ang = 2 * np.pi * np.arange(NB) / NB
    c1 = 2 * np.mean(prof * np.cos(ang)); b2 = 2 * np.mean(prof * np.cos(2 * ang))
    return float(b2 / c1) if c1 != 0 else float("inf"), prof.tolist()


def main():
    t0 = time.time()
    oz = {r["name"]: r for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    names = [n for n in oz if "gratings_" in n and "static" not in n]
    out = {"design": {"n_null": r2.N_NULL, "n_shuf": N_SHUF}, "rows": {}}
    for name in names:
        tag = short(name)
        dat = np.load(RAW / f"{name}.npy", allow_pickle=True).item()
        X = np.asarray(dat["sresp"], np.float32); X /= X.std() + 1e-9
        phi = np.asarray(dat["istim"], float).ravel() % (2 * np.pi)
        bl = np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, NB + 1)) - 1, 0, NB - 1)
        Xt = np.ascontiguousarray(X.T); del dat, X
        n, N = Xt.shape
        counts = np.bincount(bl, minlength=NB)
        M = np.stack([Xt[bl == k].mean(0) for k in range(NB)])
        grand = Xt.mean(0)
        ssb = sum(counts[k] * (M[k] - grand) ** 2 for k in range(NB))
        ssw = sum(((Xt[bl == k] - M[k]) ** 2).sum(0) for k in range(NB))
        F = (ssb / (NB - 1)) / (ssw / (n - NB) + 1e-12)
        tuned = (1 - stats.f.cdf(F, NB - 1, n - NB)) < 0.01
        R = M - M.min(0, keepdims=True) + 1e-9
        ang = 2 * np.pi * np.arange(NB) / NB
        dsi = np.abs((R * np.exp(1j * ang[:, None])).sum(0) / R.sum(0))
        osi = np.abs((R * np.exp(2j * ang[:, None])).sum(0) / R.sum(0))
        tidx = np.where(tuned)[0]; nsub = len(tidx) // 3
        order = tidx[np.argsort(dsi[tidx])]
        rng = np.random.default_rng(0)
        subsets = {"DS": order[-nsub:], "nonDS": order[:nsub], "random": rng.choice(tidx, nsub, replace=False)}
        row = {"name": name, "n_trials": int(n), "n_neurons": int(N), "n_tuned": int(len(tidx)), "n_sub": int(nsub),
               "frac_ds_dsi_gt_0p3": float((dsi[tidx] > 0.3).mean()), "median_dsi_tuned": float(np.median(dsi[tidx])),
               "median_osi_tuned": float(np.median(osi[tidx])), "full_delta": float(oz[name]["delta"]), "subsets": {}}
        print(f"[{tag}] trials {n} neurons {N} tuned {len(tidx)} sub {nsub} | DS frac {row['frac_ds_dsi_gt_0p3']:.2f} median DSI {row['median_dsi_tuned']:.2f} | full delta {row['full_delta']:+.3f}", flush=True)
        for sname, idx in subsets.items():
            Xs = np.ascontiguousarray(Xt[:, idx])
            d, sh = r2.ladder_delta(Xs, bl, np.random.default_rng(42), n_shuf=N_SHUF)
            b2c1, prof = profile_b2_c1(M[:, idx])
            row["subsets"][sname] = {"delta": float(d), "shuffle_mean": float(np.mean(sh)), "shuffle_sd": float(np.std(sh)), "b2_over_c1": b2c1,
                                     "median_dsi": float(np.median(dsi[idx])), "median_osi": float(np.median(osi[idx])), "profile": prof}
            print(f"  {sname:6s}: delta {d:+.3f} (shuffle {np.mean(sh):+.3f}) b2/c1 {b2c1:.2f} median DSI {np.median(dsi[idx]):.2f} OSI {np.median(osi[idx]):.2f} | {time.time()-t0:.0f}s", flush=True)
            del Xs
        out["rows"][tag] = row
        json.dump(out, open(OUT, "w"), indent=1)
        del Xt
    rows = out["rows"]; tags = list(rows)
    ds = [rows[t]["subsets"]["DS"]["delta"] for t in tags]; nd = [rows[t]["subsets"]["nonDS"]["delta"] for t in tags]
    rd = [rows[t]["subsets"]["random"]["delta"] for t in tags]; fd = [rows[t]["full_delta"] for t in tags]
    b_ds = [rows[t]["subsets"]["DS"]["b2_over_c1"] for t in tags]; b_nd = [rows[t]["subsets"]["nonDS"]["b2_over_c1"] for t in tags]
    fr = [rows[t]["frac_ds_dsi_gt_0p3"] for t in tags]; md = [rows[t]["median_dsi_tuned"] for t in tags]
    verdict = {"tags": tags, "delta_DS": ds, "delta_nonDS": nd, "delta_random": rd, "delta_full": fd,
               "H1_DS_gt_nonDS_count": int(sum(a > b for a, b in zip(ds, nd))),
               "H2_sorting_count": int(sum((a < 1) and (b > 1) for a, b in zip(b_ds, b_nd))), "b2c1_DS": b_ds, "b2c1_nonDS": b_nd,
               "H3_bracket_count": int(sum((a >= f >= b) for a, f, b in zip(ds, fd, nd))),
               "H3_random_within_0.05_count": int(sum(abs(r - f) <= 0.05 for r, f in zip(rd, fd))),
               "H4_rho_full_vs_DSfrac": float(spearmanr(fr, fd)[0]), "H4_rho_full_vs_medianDSI": float(spearmanr(md, fd)[0]),
               "rho_DSdelta_vs_DSfrac": float(spearmanr(fr, ds)[0]), "ds_fraction": fr}
    verdict["H1"] = verdict["H1_DS_gt_nonDS_count"] == 8; verdict["H2"] = verdict["H2_sorting_count"] == 8
    verdict["H3"] = verdict["H3_bracket_count"] == 8 and verdict["H3_random_within_0.05_count"] == 8; verdict["H4"] = verdict["H4_rho_full_vs_DSfrac"] >= 0.8
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
