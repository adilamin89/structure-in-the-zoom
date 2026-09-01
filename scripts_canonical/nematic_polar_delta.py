"""2D XY-nematic with weak polar coupling - the SYMMETRY-MATCHED ground truth.

Effective mapping (registered 2026-08-22 BEFORE running):
V1's signal correlation C(dphi) = a + c cos(dphi) + b cos(2 dphi) + d4 cos(4 dphi)
is a multipole expansion: quadrupole b (orientation, RP^1 nematic order) dominates
the dipole c (direction) by b/c = 2.35. The matched lattice model is therefore a
2D XY model with nematic + weak polar exchange:

    E = - sum_<ij> [ J2 cos 2(phi_i - phi_j) + J1 cos(phi_i - phi_j) ]
    J2 = 1,  J1/J2 = c/b = 0.42  (couplings SET BY the measured multipoles)

Aligned ladder: configurations binned by their global director angle Psi
(phase of the Q tensor, on [0, pi)) into 8 bins, accumulated - the analog of
the V1 orientation bins. Floor: 10 random same-size config subsets. PR is
CENTERED (V1-matched observable). Rungs with <10 configs skipped.

REGISTERED PREDICTIONS:
P1: delta > 0 near the nematic transition with the CENTERED observable -
    opposite to Z2 Ising (centered delta(Tc) = -0.47) - because the nematic
    order parameter is an ORBIT (RP^1 director classes = distinct subspaces,
    sector-accumulating branch), not a scalar (mode-conditioning branch).
    This is the mechanism claimed for V1's positive delta.
P2: |delta| is extremal near the transition (located by the S2(T) drop) and
    small deep in the ordered and disordered phases.
P3: The polar term J1 makes full-circle (S^1) binning of the mean polarization
    weaker than director (RP^1) binning, mirroring b > c in V1.

Output: data/nematic_polar_delta.json
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "nematic_polar_delta.json"

J2, J1 = 1.0, 0.42
TEMPS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
SIZES_L = [32, 64]
N_CHAINS = 6
N_BURN_SWEEPS = 600       # post-anneal equilibration at target T
N_ANNEAL_SWEEPS = 1200    # linear ramp T_hot -> T (v1 froze: S2 non-monotonic at low T)
T_HOT = 2.5
N_SAMPLES = 200
N_SPACING_SWEEPS = 5
N_BINS = 8
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_NULL = 10


def neighbor_sum_terms(phi):
    """cos/sin sums over the 4 neighbors for both harmonics."""
    n1c = n1s = n2c = n2s = 0.0
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        nb = np.roll(phi, sh, axis=ax)
        n1c = n1c + np.cos(nb)
        n1s = n1s + np.sin(nb)
        n2c = n2c + np.cos(2 * nb)
        n2s = n2s + np.sin(2 * nb)
    return n1c, n1s, n2c, n2s


def sweep(phi, T, rng, mask_a, mask_b):
    """Checkerboard Metropolis: propose new random angles per site."""
    for mask in (mask_a, mask_b):
        prop = rng.uniform(0, 2 * np.pi, phi.shape)
        n1c, n1s, n2c, n2s = neighbor_sum_terms(phi)
        e_old = -(J1 * (np.cos(phi) * n1c + np.sin(phi) * n1s)
                  + J2 * (np.cos(2 * phi) * n2c + np.sin(2 * phi) * n2s))
        e_new = -(J1 * (np.cos(prop) * n1c + np.sin(prop) * n1s)
                  + J2 * (np.cos(2 * prop) * n2c + np.sin(2 * prop) * n2s))
        acc = (rng.random(phi.shape) < np.exp(np.minimum((e_old - e_new) / T, 0))) & mask
        phi[acc] = prop[acc]


def sample_chain(L, T, seed):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, (L, L))
    ii, jj = np.indices((L, L))
    mask_a = (ii + jj) % 2 == 0
    mask_b = ~mask_a
    for k in range(N_ANNEAL_SWEEPS):
        t_cur = T_HOT + (T - T_HOT) * (k + 1) / N_ANNEAL_SWEEPS
        sweep(phi, t_cur, rng, mask_a, mask_b)
    for _ in range(N_BURN_SWEEPS):
        sweep(phi, T, rng, mask_a, mask_b)
    configs = np.empty((N_SAMPLES, L * L))
    psis = np.empty(N_SAMPLES)
    s2s = np.empty(N_SAMPLES)
    for s in range(N_SAMPLES):
        for _ in range(N_SPACING_SWEEPS):
            sweep(phi, T, rng, mask_a, mask_b)
        configs[s] = phi.ravel()
        z2 = np.exp(2j * phi).mean()
        psis[s] = (np.angle(z2) / 2) % np.pi
        s2s[s] = np.abs(z2)
    return configs, psis, s2s, rng


def pr_centered(A):
    Ac = A - A.mean(axis=0)
    n = Ac.shape[0]
    if n < 3:
        return 1.0
    G = (Ac @ Ac.T).astype(np.float64) / (n - 1)
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


def features(configs):
    """Represent each config by (cos, sin, cos2, sin2) per site - the natural
    observables of the polar+nematic fields (angle itself is not a vector)."""
    return np.concatenate([np.cos(configs), np.sin(configs),
                           np.cos(2 * configs), np.sin(2 * configs)], axis=1)


def delta_director(feat, psis, rng):
    bin_idx = np.minimum((psis / (np.pi / N_BINS)).astype(int), N_BINS - 1)
    members = [np.where(bin_idx == b)[0] for b in range(N_BINS)]
    sizes, prs = [], []
    acc = []
    for c in BIN_COUNTS:
        acc = [members[b] for b in range(c)]
        sel = np.concatenate(acc)
        if len(sel) < 10:
            continue
        sizes.append(len(sel))
        prs.append(pr_centered(feat[sel]))
    if len(sizes) < 3:
        return float("nan"), 0
    theta_obs = slope(sizes, prs)
    null_logs = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            null_logs[d, k] = np.log(max(pr_centered(feat[rng.choice(len(feat), s,
                                                                     replace=False)]), 1e-9))
    return theta_obs - slope(sizes, np.exp(null_logs.mean(axis=0))), len(sizes)


def main():
    results = []
    for L in SIZES_L:
        for T in TEMPS:
            t0 = time.time()
            deltas, s2_all = [], []
            n_rungs_min = 99
            for c in range(N_CHAINS):
                configs, psis, s2s, rng = sample_chain(L, T, seed=7000 + 100 * c + int(T * 13) + L)
                d, nr = delta_director(features(configs), psis, rng)
                if np.isfinite(d):
                    deltas.append(d)
                n_rungs_min = min(n_rungs_min, nr)
                s2_all.append(s2s)
            s2m = float(np.concatenate(s2_all).mean())
            row = {"L": L, "T": T, "S2": s2m,
                   "delta_mean": float(np.mean(deltas)) if deltas else None,
                   "delta_sd": float(np.std(deltas)) if deltas else None,
                   "n_valid_chains": len(deltas), "min_rungs": int(n_rungs_min)}
            results.append(row)
            print(f"L={L} T={T}: S2={s2m:.3f} delta="
                  f"{row['delta_mean'] if row['delta_mean'] is not None else 'nan'}"
                  f"±{row['delta_sd'] if row['delta_sd'] is not None else '-'} "
                  f"(rungs>={n_rungs_min}, {time.time()-t0:.0f}s)", flush=True)
            with OUT.open("w") as f:
                json.dump({"J2": J2, "J1": J1, "J1_over_J2_from_V1_multipoles": 0.42,
                           "results": results}, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
