"""Camera-ready App G: the direction shift along the sampling axis (finite-size sweep on GT3), one generator.

Superseded for the paper by cr_finite_size_split.py (same design + the Sec 7 split). Replaces the generator-less `finite_size_scaling` block of physics_full_suite.json (C5, Oct 20; PI asked for a recompute).
Same estimator as the paper's direction ladder and cr_graining_table.py: centred PR, eight direction bins accumulated
from 0 deg in rungs of 1, 2, 3, 4, 6, 8 bins, floor = ten label-blind random ladders of the same sizes; rungs with fewer
than ten samples are skipped (App K). For each stimulus budget n, DRAWS random subsets of n presentations (without
replacement, labels kept) are laddered; at n = 4442 (the full set) the draws differ only in their floor seeds.

EXPECTATION (written before the run, from the old artifact): delta ~ 0 at n <= 100 (-0.02, -0.00), small at 200 (+0.02),
rising through n = 500-2000 to ~ +0.24 at the full set.
Input: raw/gratings_drifting_GT3.npy. Out: data_canonical/cr_finite_size_sweep.json"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw" / "gratings_drifting_GT3.npy"   # Stringer et al. 2021 grating recording (Figshare 8279387), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_finite_size_sweep.json"
N_BINS, BIN_COUNTS, N_NULL, MIN_RUNG = 8, [1, 2, 3, 4, 6, 8], 10, 10
BUDGETS, DRAWS = [50, 100, 200, 500, 1000, 2000, 4442], 10


def pr(X):
    """Centred PR of a samples x units matrix, via the smaller Gram matrix."""
    X = X - X.mean(axis=0)
    G = X @ X.T if X.shape[0] <= X.shape[1] else X.T @ X
    G = G.astype(np.float64)
    return float(np.trace(G) ** 2 / (G * G).sum())


def slope(sizes, logs):
    return float(np.polyfit(np.log(sizes), logs, 1)[0])


def shift(Xt, bins, rng):
    members = [np.where(bins == b)[0] for b in range(N_BINS)]
    sel = [np.concatenate(members[:c]) for c in BIN_COUNTS]
    sel = [s for s in sel if len(s) >= MIN_RUNG]
    if len(sel) < 2:
        return None
    sizes = [len(s) for s in sel]
    th_obs = slope(sizes, [np.log(pr(Xt[s])) for s in sel])
    null = np.array([[np.log(pr(Xt[rng.choice(len(Xt), n, replace=False)])) for n in sizes] for _ in range(N_NULL)])
    th_floor = slope(sizes, null.mean(axis=0))
    return dict(delta=th_obs - th_floor, theta_obs=th_obs, theta_floor=th_floor, rungs=len(sizes))


def main():
    d = np.load(RAW, allow_pickle=True).item()
    X = np.asarray(d["sresp"], dtype=np.float32).T                     # trials x neurons
    X /= X.std() + 1e-9
    bins = np.clip(np.digitize(np.asarray(d["istim"], float), np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
    rows = []
    for n in BUDGETS:
        reps = []
        for k in range(DRAWS):
            rng = np.random.default_rng(10_000 * n + k)
            idx = np.arange(len(X)) if n >= len(X) else np.sort(rng.choice(len(X), n, replace=False))
            r = shift(X[idx], bins[idx], rng)
            if r is not None:
                reps.append(r)
        dl = np.array([r["delta"] for r in reps])
        row = dict(n=n, draws=len(reps), delta_mean=float(dl.mean()), delta_sd=float(dl.std(ddof=1)),
                   rungs_used=sorted({r["rungs"] for r in reps}), reps=reps)
        rows.append(row)
        print(f"n={n:5d} draws={len(reps):2d} delta {row['delta_mean']:+.3f} +- {row['delta_sd']:.3f} "
              f"rungs {row['rungs_used']}", flush=True)
    json.dump(dict(estimator="centred PR; rungs 1,2,3,4,6,8 of 8 direction bins (rungs < 10 samples skipped); "
                             "floor = 10 random ladders per draw", draws=DRAWS, rows=rows), open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
