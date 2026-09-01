"""Synthetic von Mises population at kappa=8 (V1-like regime) - appendix replacement.

The committed kappa=2 synthetic sits in the wrong regime (broad tuning gives a
negative category-zoom delta, opposite the paper's positive V1 headline).
Real V1 orientation tuning is sharp; kappa=8 half-width ~19 deg matches
mouse V1. This runs the same orientation-accumulation ladder + coarse-graining
sweep at kappa=8 and, for reference, a kappa sweep.

Output: data/theta_synthetic_kappa8.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "theta_synthetic_kappa8.json"

N_NEUR = 2048
N_STIM = 512
N_BINS = 8
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_NULL = 10
NOISE_SD = 0.3
KAPPAS = [2.0, 4.0, 8.0, 16.0]
CG_LEVELS = [1, 4, 16, 64]


def pr_trace(X):
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    if n < 3:
        return 1.0
    G = Xc @ Xc.T / (n - 1)
    tr = np.trace(G)
    tr2 = float((G * G).sum())
    return float(tr * tr / tr2) if tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.maximum(np.asarray(prs, dtype=float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def make_population(kappa, seed=0):
    rng = np.random.default_rng(seed)
    prefs = rng.uniform(0, np.pi, N_NEUR)
    thetas = rng.uniform(0, np.pi, N_STIM)
    resp = np.exp(kappa * np.cos(2 * (thetas[None, :] - prefs[:, None]))) / np.exp(kappa)
    resp += NOISE_SD * rng.standard_normal(resp.shape)
    return resp, prefs, thetas


def ladder_delta(Xt, bin_idx, rng):
    n = Xt.shape[0]
    members = [np.where(bin_idx == b)[0] for b in range(N_BINS)]
    sizes = [len(np.concatenate(members[:c])) for c in BIN_COUNTS]
    obs = [pr_trace(Xt[np.concatenate(members[:c])]) for c in BIN_COUNTS]
    theta_obs = slope(sizes, obs)
    null_logs = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            null_logs[d, k] = np.log(max(pr_trace(Xt[rng.choice(n, s, replace=False)]), 1e-9))
    theta_floor = slope(sizes, np.exp(null_logs.mean(axis=0)))
    return theta_obs - theta_floor


def run_kappa(kappa):
    resp, prefs, thetas = make_population(kappa, seed=int(kappa * 10))
    bin_idx = np.minimum((thetas / (np.pi / N_BINS)).astype(int), N_BINS - 1)
    order = np.argsort(prefs)
    out = {"kappa": kappa, "coarse_graining": []}
    for K in CG_LEVELS:
        n_vox = N_NEUR // K
        vox = resp[order[: n_vox * K]].reshape(n_vox, K, N_STIM).mean(axis=1)
        d_cat = ladder_delta(vox.T, bin_idx, np.random.default_rng(7))
        out["coarse_graining"].append({"K": K, "n_voxels": n_vox, "delta_cat": d_cat})
        print(f"kappa={kappa} K={K}: delta_cat={d_cat:+.4f}", flush=True)
    return out


def main():
    results = [run_kappa(k) for k in KAPPAS]
    with OUT.open("w") as f:
        json.dump({"noise_sd": NOISE_SD, "n_neurons": N_NEUR, "n_stimuli": N_STIM,
                   "results": results}, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
