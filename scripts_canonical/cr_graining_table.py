"""Camera-ready App G table: coarse-graining of GT3 with preference-sorted and random neuron blocks, one estimator.

Same estimator as the paper's direction ladder (and estimator_and_graining_controls.py): centred PR, eight direction bins
accumulated from 0 deg in rungs of 1, 2, 3, 4, 6, 8 bins, floor = ten label-blind random ladders of the same sizes.
Both arms share one floor seed at each K, so the ungrained population (K = 1) returns one value in both columns.
Blocks: K neurons averaged into one unit, neurons ordered by preferred direction (circular mean over the eight bins)
or by a fixed random permutation.
Input: raw/gratings_drifting_GT3.npy. Out: data_canonical/cr_graining_table.json"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw" / "gratings_drifting_GT3.npy"   # Stringer et al. 2019 grating recording (figshare), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_graining_table.json"
N_BINS, BIN_COUNTS, N_NULL, KS = 8, [1, 2, 3, 4, 6, 8], 10, [1, 2, 4, 8, 16, 32]


def pr(X):
    """Centred PR of a samples x units matrix, via the smaller Gram matrix."""
    X = X - X.mean(axis=0)
    G = X @ X.T if X.shape[0] <= X.shape[1] else X.T @ X
    G = G.astype(np.float64)
    return float(np.trace(G) ** 2 / (G * G).sum())


def slope(sizes, logs):
    return float(np.polyfit(np.log(sizes), logs, 1)[0])


def shift(Xt, bins, seed):
    members = [np.where(bins == b)[0] for b in range(N_BINS)]
    sel = [np.concatenate(members[:c]) for c in BIN_COUNTS]
    sizes = [len(s) for s in sel]
    th_obs = slope(sizes, [np.log(pr(Xt[s])) for s in sel])
    rng = np.random.default_rng(seed)
    null = np.array([[np.log(pr(Xt[rng.choice(len(Xt), n, replace=False)])) for n in sizes] for _ in range(N_NULL)])
    th_floor = slope(sizes, null.mean(axis=0))
    return th_obs - th_floor, th_obs, th_floor


def main():
    d = np.load(RAW, allow_pickle=True).item()
    X = np.asarray(d["sresp"], dtype=np.float32)                       # neurons x trials
    X /= X.std() + 1e-9
    bins = np.clip(np.digitize(np.asarray(d["istim"], float), np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
    means = np.stack([X[:, bins == b].mean(axis=1) for b in range(N_BINS)])
    pref = np.angle((means * np.exp(1j * np.linspace(0, 2 * np.pi, N_BINS, endpoint=False))[:, None]).sum(0)) % (2 * np.pi)
    orders = {"pref_sorted": np.argsort(pref), "random_blocks": np.random.default_rng(7).permutation(X.shape[0])}
    rows = []
    for K in KS:
        row = {"K": K, "units": X.shape[0] // K}
        for tag, order in orders.items():
            n = X.shape[0] // K
            V = X[order[: n * K]].reshape(n, K, X.shape[1]).mean(axis=1)
            dl, to, tf = shift(np.ascontiguousarray(V.T), bins, seed=1000 + K)
            row[tag] = dict(delta=dl, theta_obs=to, theta_floor=tf)
        rows.append(row)
        print(f"K={K:2d} units={row['units']:5d}  sorted {row['pref_sorted']['delta']:+.3f}  "
              f"random {row['random_blocks']['delta']:+.3f}", flush=True)
    json.dump(dict(estimator="centred PR; rungs 1,2,3,4,6,8 of 8 direction bins; floor = 10 random ladders, "
                             "seed shared by both arms at each K", rows=rows), open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
