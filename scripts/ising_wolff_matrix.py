"""Ising delta construction matrix - fresh verification of the susceptibility hinge.

The rescue session (2026-08-21, recovered from transcript) computed delta with:
  - UNCENTERED Gram PR: G = C C^T / N (no mean subtraction; magnetization mode kept)
  - m-sorted ladder at config fractions {0.1, 0.25, 0.5, 0.75, 1.0}
  - single-draw random floor
and reported delta(Tc) = +0.398 (L=32) / +0.437 (L=64).

The V1 pipeline uses CENTERED PR. This script runs the full construction matrix so
the paper's claim can be stated for the construction that actually matches V1:

  arms: (centering: uncentered | centered) x (sort: signed-m | abs-m)
  rungs: fractions {0.1, 0.25, 0.5, 0.75, 1.0} (verbatim rescue rungs)
  floor: 10 random same-size draws, slope of mean log PR
  chains: 3 independent Wolff chains -> mean +/- SD
  temps: T/Tc in {0.66, 0.88, 0.95, 1.00, 1.06, 1.15, 1.32, 1.76}; L in {32, 64}

Output: data/ising_wolff_matrix.json
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "ising_wolff_matrix.json"

T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))
TEMPS = [1.5, 2.0, 2.15, T_C, 2.4, 2.6, 3.0, 4.0]
SIZES_L = [32, 64]
N_CHAINS = 3
N_EQUIL = 300
N_CONFIGS = 200
N_SPACING = 3
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
N_NULL = 10


def wolff_chain(L, T, seed):
    rng = np.random.default_rng(seed)
    N = L * L
    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta)
    spins = rng.choice([-1, 1], size=N).astype(np.float64)

    def step():
        site = rng.integers(0, N)
        s0 = spins[site]
        cluster = {site}
        stack = [site]
        while stack:
            s = stack.pop()
            x, y = divmod(s, L)
            for nb in (((x + 1) % L) * L + y, ((x - 1) % L) * L + y,
                       x * L + (y + 1) % L, x * L + (y - 1) % L):
                if nb not in cluster and spins[nb] == s0 and rng.random() < p_add:
                    cluster.add(nb)
                    stack.append(nb)
        for s in cluster:
            spins[s] = -spins[s]

    for _ in range(N_EQUIL):
        step()
    configs = np.zeros((N_CONFIGS, N))
    for c in range(N_CONFIGS):
        for _ in range(N_SPACING):
            step()
        configs[c] = spins
    return configs, rng


def pr(M, centered):
    if centered:
        M = M - M.mean(axis=0)
    G = M @ M.T / M.shape[1]
    ev = np.linalg.eigvalsh(G)
    ev = ev[ev > 1e-10]
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if len(ev) else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.maximum(np.asarray(prs, dtype=float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def delta_arm(configs, rng, centered, sort_key):
    mags = configs.mean(axis=1)
    key = mags if sort_key == "signed" else np.abs(mags)
    cs = configs[np.argsort(key)]
    sizes = [max(10, int(f * N_CONFIGS)) for f in FRACS]
    obs = [pr(cs[:ns], centered) for ns in sizes]
    theta_obs = slope(sizes, obs)
    null_logs = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, ns in enumerate(sizes):
            null_logs[d, k] = np.log(max(pr(configs[rng.choice(N_CONFIGS, ns, replace=False)],
                                            centered), 1e-9))
    theta_floor = slope(sizes, np.exp(null_logs.mean(axis=0)))
    return theta_obs - theta_floor


ARMS = [("uncentered_signed", False, "signed"),
        ("uncentered_absm", False, "abs"),
        ("centered_signed", True, "signed"),
        ("centered_absm", True, "abs")]


def main():
    results = []
    for L in SIZES_L:
        for T in TEMPS:
            t0 = time.time()
            per_arm = {name: [] for name, _, _ in ARMS}
            binder_ms = []
            for c in range(N_CHAINS):
                configs, rng = wolff_chain(L, T, seed=50_000 + 1000 * L + 10 * c + int(T * 7))
                binder_ms.append(configs.mean(axis=1))
                for name, cent, sk in ARMS:
                    per_arm[name].append(delta_arm(configs, rng, cent, sk))
            m = np.concatenate(binder_ms)
            row = {"L": L, "T": T, "T_over_Tc": T / T_C,
                   "binder_U": float(1 - np.mean(m**4) / (3 * np.mean(m**2) ** 2)),
                   "abs_m": float(np.abs(m).mean()),
                   "chi": float(L * L * np.var(np.abs(m)))}
            for name in per_arm:
                row[name] = {"mean": float(np.mean(per_arm[name])),
                             "sd": float(np.std(per_arm[name]))}
            results.append(row)
            print(f"L={L} T/Tc={T/T_C:.3f}: " +
                  " ".join(f"{n}={row[n]['mean']:+.3f}±{row[n]['sd']:.3f}"
                           for n in per_arm) +
                  f" U={row['binder_U']:.3f} ({time.time()-t0:.0f}s)", flush=True)
            with OUT.open("w") as f:
                json.dump({"T_c": T_C, "results": results}, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
