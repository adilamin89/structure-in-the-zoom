"""Run 50c (S74) - Two-key blocking: does sorting by retinotopic footprint AND
phase recover the localized recordings' shift under coarse-graining?

App E (run50b) states that localized recordings retain only 0.41-0.56 of
delta_dir at K = 32 under direction-sorted blocks because the aperture's
retinotopic footprint mixes strongly and weakly driven neurons within a phase.
Test: sort units first by a footprint proxy (mean OSI of the 30 nearest
neighbours in the imaging plane, the same proxy as local_vs_fullfield_tuning;
4 quantile bands) and within each band by preferred direction phase; average
K = 32 units per block; compare K = 32 retention with the single-key
direction-sorted blocks of run50b. Two floor seeds. All eight recordings.

REGISTERED PREDICTIONS (before the run):
T1: two-key retention exceeds single-key direction-sorted retention in all
    three localized recordings, reaching >= 0.6 in at least two of them.
T2: in the five full-field recordings the two keys make no material
    difference (|change| < 0.15), because their footprint is uniform.
A T1 miss means the footprint explanation in App E is wrong or incomplete and
must be softened to "no single-phase sorting preserves it".

Out: ../data_canonical/run50c_twokey_blocking.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
DATA = HERE.parent / "data_canonical"
OUT = DATA / "run50c_twokey_blocking.json"
NB = 8; BIN_COUNTS = [1, 2, 3, 4, 6, 8]; N_NULL = 10; K = 32; N_BANDS = 4


def pr_c(X):
    Xc = X - X.mean(0); G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum()); return tr * tr / tr2 if tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, float)); y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    return float(np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, y, rcond=None)[0][1])


def delta_dir(Xt, bl, rng):
    members = [np.where(bl == k)[0] for k in range(NB)]
    sizes, prs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c]); sizes.append(len(sel)); prs.append(pr_c(Xt[sel]))
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(pr_c(Xt[rng.choice(len(Xt), s, replace=False)]), 1e-9))
    return slope(sizes, prs) - slope(sizes, np.exp(nl.mean(0)))


def blocked(Xt, order):
    nblk = Xt.shape[1] // K
    return np.stack([Xt[:, order[i * K:(i + 1) * K]].mean(1) for i in range(nblk)], 1)


def main():
    r50b = {r["name"]: r for r in json.load(open(DATA / "run50b_graining_sectors.json"))["rows"] if r["status"] == "ok"}
    rows = []
    for f in sorted(RAW.glob("gratings_*.npy")):
        dat = np.load(f, allow_pickle=True).item()
        X = np.asarray(dat["sresp"], np.float32); X /= X.std() + 1e-9
        istim = np.asarray(dat["istim"], float)
        bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, NB + 1)) - 1, 0, NB - 1)
        if np.bincount(bl, minlength=NB).min() < 10:
            rows.append({"name": f.stem, "status": "degenerate_bins"}); continue
        Xt = np.ascontiguousarray(X.T)
        ang = 2 * np.pi * np.arange(NB) / NB
        M = np.stack([Xt[bl == k].mean(0) for k in range(NB)]); Mc = M - M.mean(0)
        ph_dir = np.angle((Mc * np.exp(1j * ang[:, None])).sum(0)) % (2 * np.pi)
        R = M - M.min(0, keepdims=True) + 1e-9
        osi = np.abs((R * np.exp(2j * ang[:, None])).sum(0) / R.sum(0))
        pos = np.array([[s["med"][0], s["med"][1]] for s in dat["stat"]], float)
        _, nb30 = cKDTree(pos).query(pos, k=31)
        footprint = osi[nb30[:, 1:]].mean(1)
        band = np.digitize(footprint, np.quantile(footprint, np.linspace(0, 1, N_BANDS + 1)[1:-1]))
        order_two = np.lexsort((ph_dir, band))          # primary key band, secondary key phase
        d1 = float(np.mean([delta_dir(Xt, bl, np.random.default_rng(42 + s)) for s in (0, 1)]))
        d_two = float(np.mean([delta_dir(blocked(Xt, order_two), bl, np.random.default_rng(42 + s)) for s in (0, 1)]))
        single = r50b[f.stem]["retention_K32"]["dir_sorted"]
        rows.append({"name": f.stem, "status": "ok", "delta_K1": d1, "delta_twokey_K32": d_two,
                     "retention_twokey": d_two / d1, "retention_single_dir_sorted": single})
        print(f"  {f.stem[:32]:32s} K=1 {d1:+.3f} | two-key K=32 {d_two:+.3f} retention {d_two/d1:.2f} | single-key dir-sorted {single:.2f}", flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    loc = [r for r in ok if "local" in r["name"]]; ff = [r for r in ok if "local" not in r["name"]]
    verdict = {"T1_twokey_beats_single_all_localized": bool(all(r["retention_twokey"] > r["retention_single_dir_sorted"] for r in loc)),
               "T1_localized_ge_0p6_count": sum(r["retention_twokey"] >= 0.6 for r in loc),
               "T2_fullfield_abs_change_lt_0p15_all": bool(all(abs(r["retention_twokey"] - r["retention_single_dir_sorted"]) < 0.15 for r in ff)),
               "localized_retention_twokey_vs_single": [(round(r["retention_twokey"], 2), round(r["retention_single_dir_sorted"], 2)) for r in loc],
               "fullfield_retention_twokey_vs_single": [(round(r["retention_twokey"], 2), round(r["retention_single_dir_sorted"], 2)) for r in ff]}
    print("VERDICT", json.dumps(verdict))
    json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1); print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
