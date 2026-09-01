"""Two estimator controls: centering choice and graining-block assignment.

REGISTERED EXPECTATIONS (before run):
A9.8 control - uncentered PR on the real GT3 direction ladder. Attack: subset-
  mean centering removes the class mean at one-class rungs, mechanically
  lowering structured-rung PR. If delta_dir remains positive UNCENTERED, the
  centering-artifact reading is excluded. Expectation: positive but different
  magnitude (the mean mode adds a large shared eigenvalue to both arms).
A9.1 control - coarse-graining with RANDOM block assignment (no preference
  sorting) on GT3, same K ladder. Attack: preference-sorted graining is
  aligned to the probe. Expectation: random blocks mix tuning, so delta
  should persist at small K (averaging few random cells preserves per-block
  tuning diversity) and degrade at large K; either outcome is reported.

Output: data/estimator_graining_controls.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stringer_v1" / "natimg"
OUT = HERE / "data" / "estimator_graining_controls.json"

N_BINS = 8
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_NULL = 10


def pr_gram(X, centered=True):
    if centered:
        X = X - X.mean(axis=0)
    G = (X @ X.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr = float(np.trace(G))
    tr2 = float((G * G).sum())
    if not np.isfinite(tr) or not np.isfinite(tr2) or tr2 <= 0:
        return 1.0
    return tr * tr / tr2


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.maximum(np.asarray(prs, dtype=float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def delta_dir(Xt, bin_idx, rng, centered=True):
    members = [np.where(bin_idx == b)[0] for b in range(N_BINS)]
    sizes, prs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        sizes.append(len(sel))
        prs.append(pr_gram(Xt[sel], centered))
    th_o = slope(sizes, prs)
    null_logs = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            null_logs[d, k] = np.log(max(pr_gram(Xt[rng.choice(len(Xt), s, replace=False)],
                                                 centered), 1e-9))
    return th_o - slope(sizes, np.exp(null_logs.mean(axis=0)))


def main():
    fp = DATA / "gratings_drifting_GT3_2019_04_05_1.npy"
    dat = np.load(fp, allow_pickle=True).item()
    X = np.asarray(dat["sresp"], dtype=np.float32)      # neurons x stimuli
    X /= (X.std() + 1e-9)
    istim = np.asarray(dat["istim"], dtype=float)
    bin_idx = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
                      0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)                      # stimuli x neurons

    out = {}
    # ---- A9.8: centered vs uncentered on the same ladder ----
    d_cent = delta_dir(Xt, bin_idx, np.random.default_rng(42), centered=True)
    d_unc = delta_dir(Xt, bin_idx, np.random.default_rng(42), centered=False)
    out["A9_8_estimator"] = {"delta_dir_centered": d_cent,
                             "delta_dir_uncentered": d_unc}
    print(f"A9.8  centered={d_cent:+.4f}  uncentered={d_unc:+.4f}", flush=True)

    # ---- A9.1: preference-sorted vs random-block coarse-graining ----
    # tuning preference per neuron: circular mean of response over direction bins
    means = np.stack([Xt[bin_idx == b].mean(axis=0) for b in range(N_BINS)])  # bins x neurons
    angles = np.linspace(0, 2 * np.pi, N_BINS, endpoint=False)
    pref = np.angle((means * np.exp(1j * angles)[:, None]).sum(axis=0)) % (2 * np.pi)
    order_pref = np.argsort(pref)
    rng = np.random.default_rng(7)
    order_rand = rng.permutation(X.shape[0])

    out["A9_1_coarse_graining"] = []
    for K in [1, 4, 16, 32]:
        row = {"K": K}
        for tag, order in [("pref_sorted", order_pref), ("random_blocks", order_rand)]:
            n_vox = X.shape[0] // K
            vox = X[order[: n_vox * K]].reshape(n_vox, K, X.shape[1]).mean(axis=1)
            row[tag] = delta_dir(np.ascontiguousarray(vox.T), bin_idx,
                                 np.random.default_rng(100 + K), centered=True)
        out["A9_1_coarse_graining"].append(row)
        print(f"A9.1  K={K}: pref={row['pref_sorted']:+.4f} "
              f"random={row['random_blocks']:+.4f}", flush=True)

    with OUT.open("w") as f:
        json.dump(out, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
