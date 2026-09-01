"""2D Ising Wolff: delta(T, L) for magnetization-aligned ladders - COMMITTED artifact.

Regenerates the rescue-session result (ledger §B) with a registered construction:
  - Wolff cluster sampling, 200 burn-in steps, 200 samples spaced 5 steps.
  - Configurations sorted by magnetization m; 8 equal-count bins.
  - Structured ladder: accumulate bins in sorted order at counts {1,2,3,4,6,8}
    (sizes 25..200), mirroring the V1 orientation-bin accumulation ladder.
  - Floor: slope of the mean null log-PR over 10 random same-size config subsets
    (identical to the paper's floor definition).
  - delta = theta_obs - theta_floor. Uncertainty: mean +/- SD over 5 independent
    Wolff chains per (T, L). Binder cumulant U = 1 - <m^4>/(3<m^2>^2) per (T, L).

Output: data/ising_wolff_delta.json (kill-safe incremental writes).
"""
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "ising_wolff_delta.json"

T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))
T_FRACS = [0.66, 0.88, 1.00, 1.06, 1.32, 1.76]
SIZES_L = [32, 64]
N_CHAINS = 5
N_BURN = 200
N_SAMPLES = 200
N_SPACING = 5
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_BINS = 8
N_NULL = 10


def wolff_step(spins, p_add, rng):
    L = spins.shape[0]
    i, j = rng.integers(L), rng.integers(L)
    target = spins[i, j]
    cluster = {(i, j)}
    queue = deque([(i, j)])
    while queue:
        ci, cj = queue.popleft()
        for ni, nj in ((ci - 1) % L, cj), ((ci + 1) % L, cj), (ci, (cj - 1) % L), (ci, (cj + 1) % L):
            if (ni, nj) not in cluster and spins[ni, nj] == target and rng.random() < p_add:
                cluster.add((ni, nj))
                queue.append((ni, nj))
    for ci, cj in cluster:
        spins[ci, cj] = -target


def sample_chain(L, T, seed):
    rng = np.random.default_rng(seed)
    p_add = 1.0 - np.exp(-2.0 / T)
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
    for _ in range(N_BURN):
        wolff_step(spins, p_add, rng)
    configs = np.empty((N_SAMPLES, L * L), dtype=np.float64)
    for s in range(N_SAMPLES):
        for _ in range(N_SPACING):
            wolff_step(spins, p_add, rng)
        configs[s] = spins.ravel()
    return configs


def pr_trace(X):
    """PR via traces of the Gram matrix: Tr(G)^2 / Tr(G^2). No eigenvalues."""
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


def delta_one_chain(configs, rng):
    n = configs.shape[0]
    m = configs.mean(axis=1)
    order = np.argsort(m)
    per_bin = n // N_BINS
    sizes = [c * per_bin for c in BIN_COUNTS]
    obs_prs = [pr_trace(configs[order[:s]]) for s in sizes]
    theta_obs = slope(sizes, obs_prs)
    null_logs = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            null_logs[d, k] = np.log(max(pr_trace(configs[rng.choice(n, s, replace=False)]), 1e-9))
    theta_floor = slope(sizes, np.exp(null_logs.mean(axis=0)))
    return theta_obs, theta_floor, m


def main():
    results = []
    for L in SIZES_L:
        for tf in T_FRACS:
            T = tf * T_C
            t0 = time.time()
            deltas, obs_l, floor_l, m_all = [], [], [], []
            for c in range(N_CHAINS):
                configs = sample_chain(L, T, seed=10_000 * L + 100 * c + int(tf * 100))
                rng = np.random.default_rng(77_000 + 100 * c + int(tf * 100))
                th_o, th_f, m = delta_one_chain(configs, rng)
                deltas.append(th_o - th_f)
                obs_l.append(th_o)
                floor_l.append(th_f)
                m_all.append(m)
            m_all = np.concatenate(m_all)
            binder = float(1.0 - np.mean(m_all**4) / (3.0 * np.mean(m_all**2) ** 2))
            row = {
                "L": L, "T_over_Tc": tf, "T": T,
                "delta_mean": float(np.mean(deltas)), "delta_sd": float(np.std(deltas)),
                "theta_obs_mean": float(np.mean(obs_l)), "theta_floor_mean": float(np.mean(floor_l)),
                "binder_U": binder, "abs_m_mean": float(np.abs(m_all).mean()),
                "n_chains": N_CHAINS, "n_samples": N_SAMPLES,
            }
            results.append(row)
            print(f"L={L} T/Tc={tf}: delta={row['delta_mean']:+.4f}±{row['delta_sd']:.4f} "
                  f"U={binder:.3f} |m|={row['abs_m_mean']:.3f} ({time.time()-t0:.0f}s)", flush=True)
            with OUT.open("w") as f:
                json.dump({"T_c": T_C, "construction": "m-sorted 8-bin accumulation, "
                           "counts 1/2/3/4/6/8, floor=slope of mean null logPR (10 draws)",
                           "results": results}, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
