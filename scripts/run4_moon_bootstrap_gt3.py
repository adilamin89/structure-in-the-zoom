"""Run 4 - m-out-of-n (without replacement) bootstrap pilot on GT3.

The paper's within-class bootstrap resamples WITH replacement, which reduces
effective class diversity and centers the bootstrap distribution above the
point estimate (Table 2: GT3 point +0.237, bootstrap center 0.261; low-contrast
GT1 point outside its own percentile CI). Subsampling without replacement
(m = 0.8 n per class) does not duplicate stimuli, so it should not suppress
early-rung diversity.

REGISTERED EXPECTATIONS (before run):
B1: the m-out-of-n distribution is centered near the point estimate
    (|center - 0.237| < half the with-replacement offset of 0.024).
B2: the 95% interval excludes zero (sign conclusion unchanged).

100 replicates, fork-parallel. Out: feedback_runs/run4_moon_bootstrap.json
"""
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
N_BINS = 8
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_NULL = 10
FRAC = 0.8
N_REP = 100

_X = None
_BL = None


def pr_c(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, float))
    y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def delta_on(idx_keep, rng):
    Xt, bl = _X[idx_keep], _BL[idx_keep]
    members = [np.where(bl == b)[0] for b in range(N_BINS)]
    sizes, prs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        sizes.append(len(sel))
        prs.append(pr_c(Xt[sel]))
    th_o = slope(sizes, prs)
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(pr_c(Xt[rng.choice(len(Xt), s, replace=False)]),
                                  1e-9))
    return th_o - slope(sizes, np.exp(nl.mean(axis=0)))


def one_rep(rep):
    rng = np.random.default_rng(10_000 + rep)
    keep = []
    for b in range(N_BINS):
        idx = np.where(_BL == b)[0]
        m = int(round(FRAC * len(idx)))
        keep.append(rng.choice(idx, m, replace=False))
    d = delta_on(np.concatenate(keep), rng)
    if rep % 5 == 0:
        print(f"rep {rep}: delta={d:+.4f}", flush=True)
    return d


def init():
    global _X, _BL
    dat = np.load(DATA / "gratings_drifting_GT3_2019_04_05_1.npy",
                  allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    _BL = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
                  0, N_BINS - 1)
    _X = np.ascontiguousarray(X.T)


if __name__ == "__main__":
    mp.set_start_method("fork")
    init()
    point = delta_on(np.arange(len(_BL)), np.random.default_rng(42))
    print(f"point estimate (full data, this pipeline): {point:+.4f} "
          f"(paper: +0.237)", flush=True)
    with mp.Pool(10) as pool:
        deltas = pool.map(one_rep, range(N_REP))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    out = {"point": point, "center": float(deltas.mean()),
           "median": float(np.median(deltas)),
           "ci95": [float(lo), float(hi)], "n_rep": N_REP, "frac": FRAC,
           "paper_with_replacement_center": 0.261, "paper_point": 0.237}
    print(f"m-out-of-n: center={deltas.mean():+.4f} median={np.median(deltas):+.4f} "
          f"CI95=[{lo:+.4f},{hi:+.4f}] | point={point:+.4f}", flush=True)
    json.dump(out, open(HERE / "run4_moon_bootstrap.json", "w"), indent=1)
    print("DONE run4", flush=True)
