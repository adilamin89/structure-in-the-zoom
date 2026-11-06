"""Lattice ground truth for App F / Sec 7.

Estimators are the paper's: ising_wolff_matrix.delta_arm (centered PR, |m|-sorted rungs, 10-draw floor) and
nematic_polar_delta's pr_centered / slope / features (director bins, cumulative-bin rungs, 10-draw floor).
New here: numba samplers (Wolff for Ising; Metropolis + reflection-Wolff hybrid for the XY nematic, fixed cluster counts), many independent chains per point,
and an XY design with no chain selection (each ensemble = 40 chains x 5 samples).

Usage:
  python lattice_ground_truth.py r31            # reproduce the paper's artifacts (canonical code, canonical seeds)
  python lattice_ground_truth.py ising L T_over_Tc n_chains   -> data_canonical/lattice_ground_truth_ising_L{L}_t{T}.json
  python lattice_ground_truth.py xy L T J1                    -> data_canonical/lattice_ground_truth_xy_L{L}_T{T}_J{J1}.json
  python lattice_ground_truth.py xytime L T J1                -> single-chain angular vs time-ordered ladder (R34 iii)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUTDIR = HERE.parent / "data_canonical"

import ising_wolff_matrix as iwm  # noqa: E402
import nematic_polar_delta as nem  # noqa: E402

T_C = iwm.T_C
FRACS = iwm.FRACS
N_NULL = 10


# ---------------------------------------------------------------- Ising (Wolff, numba)
@njit(cache=True)
def _wolff_step(spins, L, p_add, stack, inclust):
    N = L * L
    site = np.random.randint(N)
    s0 = spins[site]
    top = 0
    stack[top] = site
    top += 1
    inclust[site] = True
    size = 0
    while top > 0:
        top -= 1
        s = stack[top]
        size += 1
        x = s // L
        y = s % L
        for k in range(4):
            if k == 0:
                nb = ((x + 1) % L) * L + y
            elif k == 1:
                nb = ((x - 1 + L) % L) * L + y
            elif k == 2:
                nb = x * L + (y + 1) % L
            else:
                nb = x * L + (y - 1 + L) % L
            if (not inclust[nb]) and spins[nb] == s0 and np.random.random() < p_add:
                inclust[nb] = True
                stack[top] = nb
                top += 1
    # flip and clear
    for s in range(N):
        if inclust[s]:
            spins[s] = -spins[s]
            inclust[s] = False
    return size


@njit(cache=True)
def ising_chain(L, T, seed, n_equil_sweeps, n_configs):
    np.random.seed(seed)
    N = L * L
    p_add = 1.0 - np.exp(-2.0 / T)
    spins = np.where(np.random.random(N) < 0.5, -1.0, 1.0)
    stack = np.empty(N, dtype=np.int64)
    inclust = np.zeros(N, dtype=np.bool_)
    flipped = 0
    nsteps = 0
    while flipped < n_equil_sweeps * N:
        flipped += _wolff_step(spins, L, p_add, stack, inclust)
        nsteps += 1
    # a FIXED number of cluster steps between samples (about one sweep of flipped sites, from the equilibration's mean
    # cluster size); a state-dependent stopping rule ("until N sites flipped") biases the sampled distribution
    k = max(1, int(np.ceil(N / (flipped / nsteps))))
    out = np.empty((n_configs, N))
    for c in range(n_configs):
        for _ in range(k):
            _wolff_step(spins, L, p_add, stack, inclust)
        out[c] = spins
    return out


def ising_point(L, t_over_tc, n_chains, n_configs=400):
    """Each chain: 400 configs; the estimate uses the second 200 (paper's size), the first 200 = equilibration check."""
    T = t_over_tc * T_C
    rows = []
    for c in range(n_chains):
        t0 = time.time()
        seed = 910_000 + 1000 * L + 10 * c + int(round(100 * t_over_tc))
        cfg = ising_chain(L, T, seed, 200, n_configs)
        rng = np.random.default_rng(seed)
        half = n_configs // 2
        row = {"chain": c, "seed": seed}
        for tag, block in (("first", cfg[:half]), ("second", cfg[half:])):
            iwm.N_CONFIGS = len(block)
            row[tag] = {name: iwm.delta_arm(block, rng, cent, sk) for name, cent, sk in iwm.ARMS}
        m = cfg[half:].mean(axis=1)
        row["m2"] = float(np.mean(m ** 2))
        row["m4"] = float(np.mean(m ** 4))
        row["secs"] = time.time() - t0
        rows.append(row)
        print(f"ising L={L} t={t_over_tc} chain {c}: centered_absm={row['second']['centered_absm']:+.4f} "
              f"(first half {row['first']['centered_absm']:+.4f}) {row['secs']:.0f}s", flush=True)
    m2 = np.mean([r["m2"] for r in rows])
    m4 = np.mean([r["m4"] for r in rows])
    summ = {}
    for name, _, _ in iwm.ARMS:
        v = np.array([r["second"][name] for r in rows])
        v1 = np.array([r["first"][name] for r in rows])
        summ[name] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "se": float(v.std(ddof=1) / np.sqrt(len(v))),
                      "n_neg": int((v < 0).sum()), "first_half_mean": float(v1.mean()),
                      "first_half_se": float(v1.std(ddof=1) / np.sqrt(len(v1)))}
    return {"L": L, "t_over_tc": t_over_tc, "T": T, "n_chains": n_chains, "binder_U": float(1 - m4 / (3 * m2 ** 2)),
            "summary": summ, "chains": rows}


# ---------------------------------------------------------------- XY nematic + polar (Metropolis, numba)
@njit(cache=True)
def _local_e(phi, L, i, j, a, J1, J2):
    e = 0.0
    for k in range(4):
        if k == 0:
            ni, nj = (i + 1) % L, j
        elif k == 1:
            ni, nj = (i - 1 + L) % L, j
        elif k == 2:
            ni, nj = i, (j + 1) % L
        else:
            ni, nj = i, (j - 1 + L) % L
        d = a - phi[ni, nj]
        e -= J1 * np.cos(d) + J2 * np.cos(2.0 * d)
    return e


@njit(cache=True)
def _sweep(phi, L, T, J1, J2, width):
    acc = 0
    for i in range(L):
        for j in range(L):
            old = phi[i, j]
            new = old + width * (2.0 * np.random.random() - 1.0)
            de = _local_e(phi, L, i, j, new, J1, J2) - _local_e(phi, L, i, j, old, J1, J2)
            if de <= 0.0 or np.random.random() < np.exp(-de / T):
                phi[i, j] = new % (2.0 * np.pi)
                acc += 1
    return acc / (L * L)


@njit(cache=True)
def _pair_e(d, J1, J2):
    return -(J1 * np.cos(d) + J2 * np.cos(2.0 * d))


@njit(cache=True)
def _wolff_xy(phi, L, T, J1, J2, stack, inclust):
    """Wolff embedding for any pair energy invariant under a global reflection phi -> 2a - phi: a neighbour j joins
    with p = 1 - exp(min(0, -dE/T)), dE = e(R phi_i - phi_j) - e(phi_i - phi_j) (both at pre-flip values); detailed
    balance holds because the reverse move has -dE on every boundary bond. Reflections with the axis perpendicular to
    the director implement polar (phi -> phi + pi) domain flips, which local Metropolis cannot."""
    N = L * L
    a = np.random.random() * np.pi
    site = np.random.randint(N)
    top = 0
    stack[top] = site
    top += 1
    inclust[site] = True
    size = 0
    while top > 0:
        top -= 1
        s = stack[top]
        size += 1
        i = s // L
        j = s % L
        pi_ = phi[i, j]
        ri = 2.0 * a - pi_
        for k in range(4):
            if k == 0:
                ni, nj = (i + 1) % L, j
            elif k == 1:
                ni, nj = (i - 1 + L) % L, j
            elif k == 2:
                ni, nj = i, (j + 1) % L
            else:
                ni, nj = i, (j - 1 + L) % L
            nb = ni * L + nj
            if inclust[nb]:
                continue
            pj = phi[ni, nj]
            de = _pair_e(ri - pj, J1, J2) - _pair_e(pi_ - pj, J1, J2)
            if de > 0.0 and np.random.random() < 1.0 - np.exp(-de / T):
                inclust[nb] = True
                stack[top] = nb
                top += 1
    for s in range(N):
        if inclust[s]:
            i = s // L
            j = s % L
            phi[i, j] = (2.0 * a - phi[i, j]) % (2.0 * np.pi)
            inclust[s] = False
    return size


@njit(cache=True)
def _hybrid_sweep(phi, L, T, J1, J2, width, stack, inclust, k):
    """One Metropolis sweep plus a FIXED number k of Wolff clusters (fixed: a state-dependent count biases)."""
    _sweep(phi, L, T, J1, J2, width)
    for _ in range(k):
        _wolff_xy(phi, L, T, J1, J2, stack, inclust)


@njit(cache=True)
def xy_chain_w(L, T, J1, J2, seed, hot, n_equil, n_samples, spacing):
    """Hybrid Metropolis + Wolff chain; hot=True random start, hot=False ordered start at a random angle."""
    np.random.seed(seed)
    if hot:
        phi = np.random.random((L, L)) * 2.0 * np.pi
    else:
        phi = np.full((L, L), np.random.random() * 2.0 * np.pi)
    stack = np.empty(L * L, dtype=np.int64)
    inclust = np.zeros(L * L, dtype=np.bool_)
    width = np.pi
    tot_flip = 0
    tot_n = 0
    for k in range(n_equil):
        a = _sweep(phi, L, T, J1, J2, width)
        width = min(np.pi, max(0.05, width * (1.0 + 0.5 * (a - 0.4))))
        flipped = 0
        n = 0
        while flipped < L * L and n < 50 * L * L:
            flipped += _wolff_xy(phi, L, T, J1, J2, stack, inclust)
            n += 1
        tot_flip += flipped
        tot_n += n
    kw = max(1, min(50 * L * L, int(np.ceil(L * L / (tot_flip / max(tot_n, 1) + 1e-9)))))
    configs = np.empty((n_samples, L * L))
    energies = np.empty(n_samples)
    for s in range(n_samples):
        for _ in range(spacing):
            _hybrid_sweep(phi, L, T, J1, J2, width, stack, inclust, kw)
        configs[s] = phi.ravel()
        energies[s] = _energy(phi, L, J1, J2)
    return configs, energies


@njit(cache=True)
def _energy(phi, L, J1, J2):
    e = 0.0
    for i in range(L):
        for j in range(L):
            for (ni, nj) in (((i + 1) % L, j), (i, (j + 1) % L)):
                d = phi[i, j] - phi[ni, nj]
                e -= J1 * np.cos(d) + J2 * np.cos(2.0 * d)
    return e / (L * L)


@njit(cache=True)
def xy_chain(L, T, J1, J2, seed, hot, n_anneal, n_equil, n_samples, spacing):
    """hot=True: random start annealed from T=2.5; hot=False: ordered start at a random director (no anneal)."""
    np.random.seed(seed)
    if hot:
        phi = np.random.random((L, L)) * 2.0 * np.pi
    else:
        phi = np.full((L, L), np.random.random() * 2.0 * np.pi)
    width = np.pi
    for k in range(n_anneal):
        t_cur = 2.5 + (T - 2.5) * (k + 1) / n_anneal
        a = _sweep(phi, L, t_cur, J1, J2, width)
        width = min(np.pi, max(0.05, width * (1.0 + 0.5 * (a - 0.4))))
    for k in range(n_equil):
        a = _sweep(phi, L, T, J1, J2, width)
        width = min(np.pi, max(0.05, width * (1.0 + 0.5 * (a - 0.4))))
    # width frozen from here (detailed balance during sampling)
    configs = np.empty((n_samples, L * L))
    energies = np.empty(n_samples)
    for s in range(n_samples):
        for _ in range(spacing):
            _sweep(phi, L, T, J1, J2, width)
        configs[s] = phi.ravel()
        energies[s] = _energy(phi, L, J1, J2)
    return configs, energies


def order_params(configs):
    z1 = np.exp(1j * configs).mean(axis=1)
    z2 = np.exp(2j * configs).mean(axis=1)
    psi = (np.angle(z2) / 2) % np.pi
    return np.abs(z1), np.abs(z2), psi


def ladder_delta(feat, order, sizes, rng):
    """delta for a ladder that takes the first `size` configs of `order`, vs a same-size random floor (paper's)."""
    prs = [nem.pr_centered(feat[order[:s]]) for s in sizes]
    theta_obs = nem.slope(sizes, prs)
    null_logs = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            null_logs[d, k] = np.log(max(nem.pr_centered(feat[rng.choice(len(feat), s, replace=False)]), 1e-9))
    return theta_obs - nem.slope(sizes, np.exp(null_logs.mean(axis=0)))


def director_ladder(feat, psi, rng):
    bins = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
    order = np.argsort(bins, kind="stable")
    counts = np.bincount(bins, minlength=nem.N_BINS)
    sizes = [int(counts[:c].sum()) for c in nem.BIN_COUNTS]
    sizes = [s for s in sizes if s >= 10]
    sizes = sorted(set(sizes))
    if len(sizes) < 3:
        return float("nan"), sizes, counts
    return ladder_delta(feat, order, sizes, rng), sizes, counts


def class_mean_stats(feat, psi, N):
    """Odd-sector share of class-mean variance and the harmonic content of the class-mean kernel on the 8-cycle."""
    bins = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
    mu = feat.mean(axis=0)
    means = np.array([feat[bins == b].mean(axis=0) - mu if (bins == b).sum() else np.zeros(feat.shape[1])
                      for b in range(nem.N_BINS)])
    odd = (means[:, :2 * N] ** 2).sum()
    tot = (means ** 2).sum()
    K = means @ means.T
    # circulant average: k(d) = mean over b of K[b, b+d]
    kd = np.array([np.mean([K[b, (b + d) % nem.N_BINS] for b in range(nem.N_BINS)]) for d in range(nem.N_BINS)])
    spec = np.abs(np.fft.rfft(kd))  # harmonics 0..4 of the bin index
    return float(odd / tot) if tot > 0 else float("nan"), [float(x) for x in spec]


def xy_point(L, T, J1, n_ens=8, chains_per_ens=40, samples_per_chain=5, start="hot", n_equil=None, check_equil=None):
    J2 = 1.0
    n_anneal = 0                       # hybrid Metropolis + Wolff (validated: hot = cold = pure Wolff energies)
    n_equil = n_equil or (300 if L <= 32 else 500)
    check_equil = check_equil or n_equil
    hot_prod = start == "hot"          # 'cold' = ordered start at a uniformly random angle (same symmetric ensemble)
    spacing = 10
    N = L * L
    ens_rows = []
    for e in range(n_ens):
        t0 = time.time()
        cfgs, ens = [], []
        for c in range(chains_per_ens):
            seed = 77_000_000 + 100_000 * int(round(J1 * 100)) + 1000 * L + 40 * e + c + int(round(T * 1000)) * 7
            x, en = xy_chain_w(L, T, J1, J2, seed, hot_prod, n_equil, samples_per_chain, spacing)
            cfgs.append(x)
            ens.append(en)
        cfg = np.concatenate(cfgs)
        energies = np.concatenate(ens)
        m1, s2, psi = order_params(cfg)
        # director autocorrelation within chains (consecutive samples, spacing sweeps apart): mean Re z_t z*_{t+1} / |z|^2
        z = np.exp(2j * psi).reshape(chains_per_ens, samples_per_chain)
        dir_ac = float(np.mean(np.real(z[:, 1:] * np.conj(z[:, :-1]))))
        feat = nem.features(cfg)
        rng = np.random.default_rng(1_000 + e)
        d_dir, sizes, counts = director_ladder(feat, psi, rng)
        perm = rng.permutation(len(psi))
        d_perm, _, _ = director_ladder(feat, psi[perm], rng)
        d_scalar = ladder_delta(feat, np.argsort(s2), sizes, rng) if len(sizes) >= 3 else float("nan")
        odd, harm = class_mean_stats(feat, psi, N)
        odd_p, harm_p = class_mean_stats(feat, psi[perm], N)
        row = {"ens": e, "delta_director": d_dir, "delta_permuted": d_perm, "delta_scalar_S2": d_scalar,
               "rung_sizes": sizes, "bin_counts": [int(x) for x in counts],
               "S2": float(s2.mean()), "S2_var": float(s2.var()), "director_ac_lag1": dir_ac, "M1": float(m1.mean()), "E": float(energies.mean()),
               "odd_share": odd, "odd_share_perm": odd_p, "kernel_harmonics": harm, "kernel_harmonics_perm": harm_p,
               "secs": time.time() - t0}
        ens_rows.append(row)
        print(f"xy L={L} T={T} J1={J1} ens {e}: d_dir={d_dir:+.4f} perm={d_perm:+.4f} scalar={d_scalar:+.4f} "
              f"S2={row['S2']:.3f} M1={row['M1']:.3f} bins={row['bin_counts']} {row['secs']:.0f}s", flush=True)
    # equilibration check: chains from the OTHER start (ordered at a random angle if production is hot, and vice versa)
    cold_E, cold_S2 = [], []
    for c in range(chains_per_ens if check_equil == n_equil else 16):
        seed = 88_000_000 + 100_000 * int(round(J1 * 100)) + 1000 * L + c + int(round(T * 1000)) * 7
        x, en = xy_chain_w(L, T, J1, J2, seed, not hot_prod, check_equil, samples_per_chain, spacing)
        _, s2c, _ = order_params(x)
        cold_E.append(en.mean())
        cold_S2.append(s2c.mean())

    def ms(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))), "n": int(len(v)),
                "n_pos": int((v > 0).sum())}
    hot_E = [r["E"] for r in ens_rows]     # production chains (named 'hot' when start == 'hot')
    hot_S2 = [r["S2"] for r in ens_rows]
    summ = {k: ms([r[k] for r in ens_rows]) for k in
            ("delta_director", "delta_permuted", "delta_scalar_S2", "S2", "M1", "E", "odd_share", "odd_share_perm",
             "director_ac_lag1")}
    summ["chi2"] = float(N * np.mean([r["S2_var"] for r in ens_rows]) / T)
    summ["equil_check"] = {"hot_E": ms(hot_E), "cold_E": ms(cold_E), "hot_S2": ms(hot_S2), "cold_S2": ms(cold_S2)}
    summ["kernel_harmonics_mean"] = np.mean([r["kernel_harmonics"] for r in ens_rows], axis=0).tolist()
    summ["kernel_harmonics_perm_mean"] = np.mean([r["kernel_harmonics_perm"] for r in ens_rows], axis=0).tolist()
    return {"L": L, "T": T, "J1": J1, "J2": J2, "n_ens": n_ens, "chains_per_ens": chains_per_ens, "start": start,
            "check_equil": check_equil, "equil_check_keys": "hot_* = production chains, cold_* = the other start",
            "samples_per_chain": samples_per_chain, "n_anneal": n_anneal, "n_equil": n_equil, "spacing": spacing,
            "summary": summ, "ensembles": ens_rows}


def xy_time_point(L, T, J1, n_chains=6):
    """R34 (iii): the old single-chain design (200 samples, 5 sweeps apart): angular bin ladder vs time-ordered ladder."""
    rows = []
    for c in range(n_chains):
        seed = 99_000_000 + 1000 * L + c + int(round(T * 1000)) * 7 + 100_000 * int(round(J1 * 100))
        nem.J1 = J1                     # the paper's own sampler and schedule (nematic_polar_delta.sample_chain)
        x, psi, _, _ = nem.sample_chain(L, T, seed)
        feat = nem.features(x)
        rng = np.random.default_rng(seed)
        d_dir, sizes, counts = director_ladder(feat, psi, rng)
        d_time = ladder_delta(feat, np.arange(len(x)), sizes, rng) if len(sizes) >= 3 else float("nan")
        rows.append({"chain": c, "delta_director": d_dir, "delta_time": d_time, "rung_sizes": sizes,
                     "bins_nonempty": int((counts > 0).sum())})
        print(f"xytime L={L} T={T} chain {c}: dir={d_dir:+.4f} time={d_time:+.4f} bins={int((counts > 0).sum())}",
              flush=True)
    return {"L": L, "T": T, "J1": J1, "chains": rows}


def xy_time_paper(L, T):
    """R34 (iii), fixed: the paper's own chains (nematic_polar_delta seeds and sampler) and its rung rule (cumulative bins
    1,2,3,4,6,8, rungs with < 10 configurations skipped, repeated sizes kept), against a time-ordered ladder at the SAME
    rung sizes (the first s samples of the chain), both with the paper's 10-draw floor."""
    rows = []
    for c in range(nem.N_CHAINS):
        nem.J1 = 0.42
        configs, psis, _, rng = nem.sample_chain(L, T, seed=7000 + 100 * c + int(T * 13) + L)
        feat = nem.features(configs)
        bins = np.minimum((psis / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
        members = [np.where(bins == b)[0] for b in range(nem.N_BINS)]
        sizes, sets = [], []
        for k in nem.BIN_COUNTS:
            sel = np.concatenate([members[b] for b in range(k)])
            if len(sel) >= 10:
                sizes.append(len(sel))
                sets.append(sel)
        rec = {"chain": c, "rung_sizes": sizes, "bins_nonempty": int(sum(len(m) > 0 for m in members))}
        if len(sizes) >= 3:
            r1 = np.random.default_rng(1)
            d_paper, _ = nem.delta_director(feat, psis, np.random.default_rng(rng.integers(1 << 30)))
            prs_t = [nem.pr_centered(feat[:s]) for s in sizes]
            nl = np.zeros((N_NULL, len(sizes)))
            for d in range(N_NULL):
                for k, s in enumerate(sizes):
                    nl[d, k] = np.log(max(nem.pr_centered(feat[r1.choice(len(feat), s, replace=False)]), 1e-9))
            fl = nem.slope(sizes, np.exp(nl.mean(axis=0)))
            rec["delta_director_paper_rule"] = d_paper
            rec["delta_time_same_sizes"] = nem.slope(sizes, prs_t) - fl
            rec["n_distinct_sizes"] = len(set(sizes))
            # how time-ordered the angular ladder is: mean sample index of each rung's newest members
            rec["rung_mean_time"] = [float(np.mean(s)) for s in sets]
        rows.append(rec)
        print(f"xytime2 T={T} chain {c}: sizes {sizes} bins {rec['bins_nonempty']} "
              f"dir {rec.get('delta_director_paper_rule', float('nan')):+.4f} time {rec.get('delta_time_same_sizes', float('nan')):+.4f}",
              flush=True)
    return {"L": L, "T": T, "chains": rows}


# ---------------------------------------------------------------- R31 reproduction (canonical code + seeds)
def r31():
    out = {}
    L, T = 32, T_C
    per = []
    for c in range(iwm.N_CHAINS):
        configs, rng = iwm.wolff_chain(L, T, seed=50_000 + 1000 * L + 10 * c + int(T * 7))
        vals = {name: iwm.delta_arm(configs, rng, cent, sk) for name, cent, sk in iwm.ARMS}
        per.append(vals["centered_absm"])
    art = json.load(open(OUTDIR / "ising_wolff_matrix.json"))
    ref = [r for r in art["results"] if r["L"] == 32 and abs(r["T"] - T_C) < 1e-9][0]["centered_absm"]["mean"]
    out["ising"] = {"per_chain": per, "mean": float(np.mean(per)), "sd": float(np.std(per)), "artifact": ref,
                    "abs_diff": abs(float(np.mean(per)) - ref)}
    print("R31a ising", out["ising"], flush=True)
    art = json.load(open(OUTDIR / "nematic_polar_delta.json"))
    out["nematic"] = []
    for T in (1.0, 1.2):
        deltas, chains = [], []
        for c in range(nem.N_CHAINS):
            configs, psis, s2s, rng = nem.sample_chain(L, T, seed=7000 + 100 * c + int(T * 13) + L)
            d, nr = nem.delta_director(nem.features(configs), psis, rng)
            bins = np.minimum((psis / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
            cnt = np.bincount(bins, minlength=nem.N_BINS)
            chains.append({"chain": c, "delta": d, "n_rungs": nr, "bins_nonempty": int((cnt > 0).sum()),
                           "bin_counts": cnt.tolist()})
            if np.isfinite(d):
                deltas.append(d)
            print(f"R31b T={T} chain {c}: delta={d:+.4f} rungs={nr} bins={cnt.tolist()}", flush=True)
        ref = [r for r in art["results"] if r["L"] == 32 and r["T"] == T][0]
        out["nematic"].append({"T": T, "mean": float(np.mean(deltas)), "n_valid": len(deltas),
                               "artifact": ref["delta_mean"], "artifact_n_valid": ref["n_valid_chains"],
                               "abs_diff": abs(float(np.mean(deltas)) - ref["delta_mean"]), "chains": chains})
    json.dump(out, open(OUTDIR / "lattice_ground_truth_r31.json", "w"), indent=1)
    assert out["ising"]["abs_diff"] < 1e-9, out["ising"]
    for r in out["nematic"]:
        assert r["abs_diff"] < 1e-9, r
    print("R31 reproduced", flush=True)


def main():
    mode = sys.argv[1]
    if mode == "r31":
        r31()
    elif mode == "ising":
        L, t, n = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
        res = ising_point(L, t, n)
        json.dump(res, open(OUTDIR / f"lattice_ground_truth_ising_L{L}_t{t:.2f}.json", "w"), indent=1)
    elif mode == "xy":
        L, T, J1 = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        start = sys.argv[5] if len(sys.argv) > 5 else "hot"
        ne = int(sys.argv[6]) if len(sys.argv) > 6 else None
        ce = int(sys.argv[7]) if len(sys.argv) > 7 else None
        res = xy_point(L, T, J1, start=start, n_equil=ne, check_equil=ce)
        tag = "" if start == "hot" and ne is None else f"_{start}{ne}"
        json.dump(res, open(OUTDIR / f"lattice_ground_truth_xy_L{L}_T{T:.2f}_J{J1:.2f}{tag}.json", "w"), indent=1)
    elif mode == "xytime2":
        L, T = int(sys.argv[2]), float(sys.argv[3])
        json.dump(xy_time_paper(L, T), open(OUTDIR / f"lattice_ground_truth_xytime2_L{L}_T{T:.2f}.json", "w"), indent=1)
    elif mode == "xytime":
        L, T, J1 = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        res = xy_time_point(L, T, J1)
        json.dump(res, open(OUTDIR / f"lattice_ground_truth_xytime_L{L}_T{T:.2f}_J{J1:.2f}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
