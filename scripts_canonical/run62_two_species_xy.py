"""Run 62 (S78b, 2026-09-06) - The two-species XY lattice: the rung-four stall on
ground truth, with the polar fraction as the knob.

WHY: Section 4 says an even code has every class mean it will have by four
classes of a direction ladder, so the ladder stalls there unless an odd
(polar) component carries it on; Section 7 and Figure 8 show it in the cells
(the orientation-only third stalls, the direction-selective third climbs). The
matched lattice of Section 6.1 is an XY nematic with a uniform polar coupling.
Here the polar coupling lives on a fraction f of the sites, the lattice form of
a direction-selective subpopulation, and f is the declared constant.

MODEL: E = -sum_<ij> [ J2 cos 2(phi_i - phi_j) + J1_ij cos(phi_i - phi_j) ],
J2 = 1, J1_ij = J1 (p_i + p_j)/2 with p_i = 1 on a random fraction f of the
sites and 0 on the rest, f in {0, 0.25, 0.5, 0.75, 1}. J1 = 1.3, the value the
paper assigns to the localized (direction-dominant) recordings, so that the
polar sites order at the temperatures where the director still fluctuates
(the original J1 = 0.42 orders the polar component only where the director
is frozen and the bins are empty; nematic_polar_delta.py, App D). Checkerboard
Metropolis with the original anneal / burn / spacing schedule, L = 32,
T in {1.0, 1.2, 1.4} (the transition side of the original sweep), twelve chains
per cell, 200 configurations per chain (v2, same day: twelve chains, and the four-class
and one-class rungs found by class count; v1 found them by position, which
misassigned rungs whenever a sparse early rung was skipped).

THE STALL TEST: eight classes in which class c and class c + 4 are antipodes.
The director angle Psi (the phase of the nematic order parameter, on [0, pi))
is binned into four classes; the sign s of the polarization along the director,
s = sign Re(P e^{-i Psi}) with P = mean exp(i phi), splits each into two. Under
the global flip phi -> phi + pi the director is unchanged and P changes sign,
so class c maps to class c + 4. Declared order 0, 1, 2, 3 (the four directors,
s > 0) then 4, 5, 6, 7 (their antipodes): the first half holds one member of
each pair. The ladder [1, 2, 3, 4, 6, 8] on the site features (cos, sin, cos 2,
sin 2), the centered participation ratio, ten random-subset floors of the same
sizes; per rung the log-PR deficit below the floor; the late fraction is the
deficit at four classes over the deficit at one class (run 59b, rung 1.3).
Also the eight-class shift, and the original eight-bin director-only shift.
Chains with fewer than five rungs (sparse classes) are excluded and counted.

REGISTERED EXPECTATIONS (written before the run):
X1: at f = 0 the sign label is noise, so the ladder has every class mean by
    four classes: the late fraction is within 0.15 of zero (chain mean) at
    every T with at least three valid chains.
X2: the late fraction increases with f: Spearman over f >= 0.9 on the chain
    means at every such T, and at f = 1 it exceeds 0.3 at the T where the
    f = 1 eight-class shift is largest.
X3: the eight-class shift increases with f (Spearman over f >= 0.9 at every
    such T).
X4: the director-only shift is positive at every f at every such T (the
    nematic orbit accumulates whatever the polar fraction).
A miss is reported at full volume.

Out: ../data_canonical/run62_two_species_xy.json (+ .log)
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run62_two_species_xy.json"

J2, J1 = 1.0, 1.3
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
TEMPS = [1.0, 1.2, 1.4]
L = 32
N_CHAINS = 12
N_BURN_SWEEPS = 600
N_ANNEAL_SWEEPS = 1200
T_HOT = 2.5
N_SAMPLES = 200
N_SPACING_SWEEPS = 5
N_BINS = 8
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_NULL = 10
MIN_RUNGS = 5


def neighbor_sums(phi, p):
    n1c = n1s = n2c = n2s = n1cp = n1sp = 0.0
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        nb = np.roll(phi, sh, axis=ax); pb = np.roll(p, sh, axis=ax)
        n1c = n1c + np.cos(nb); n1s = n1s + np.sin(nb)
        n2c = n2c + np.cos(2 * nb); n2s = n2s + np.sin(2 * nb)
        n1cp = n1cp + pb * np.cos(nb); n1sp = n1sp + pb * np.sin(nb)
    return n1c, n1s, n2c, n2s, n1cp, n1sp


def site_energy(ang, p, n1c, n1s, n2c, n2s, n1cp, n1sp):
    """-[J2 sum_j cos 2(phi_i - phi_j) + (J1/2) sum_j (p_i + p_j) cos(phi_i - phi_j)]"""
    c, s = np.cos(ang), np.sin(ang)
    polar = 0.5 * J1 * (p * (c * n1c + s * n1s) + (c * n1cp + s * n1sp))
    return -(J2 * (np.cos(2 * ang) * n2c + np.sin(2 * ang) * n2s) + polar)


def sweep(phi, p, T, rng, mask_a, mask_b):
    for mask in (mask_a, mask_b):
        prop = rng.uniform(0, 2 * np.pi, phi.shape)
        terms = neighbor_sums(phi, p)
        e_old = site_energy(phi, p, *terms); e_new = site_energy(prop, p, *terms)
        acc = (rng.random(phi.shape) < np.exp(np.minimum((e_old - e_new) / T, 0))) & mask
        phi[acc] = prop[acc]


def sample_chain(L, T, f, seed):
    rng = np.random.default_rng(seed)
    p = (rng.random((L, L)) < f).astype(float)
    phi = rng.uniform(0, 2 * np.pi, (L, L))
    ii, jj = np.indices((L, L)); mask_a = (ii + jj) % 2 == 0; mask_b = ~mask_a
    for k in range(N_ANNEAL_SWEEPS):
        sweep(phi, p, T_HOT + (T - T_HOT) * (k + 1) / N_ANNEAL_SWEEPS, rng, mask_a, mask_b)
    for _ in range(N_BURN_SWEEPS):
        sweep(phi, p, T, rng, mask_a, mask_b)
    configs = np.empty((N_SAMPLES, L * L)); psis = np.empty(N_SAMPLES); signs = np.empty(N_SAMPLES); s2s = np.empty(N_SAMPLES); pol = np.empty(N_SAMPLES)
    for s in range(N_SAMPLES):
        for _ in range(N_SPACING_SWEEPS):
            sweep(phi, p, T, rng, mask_a, mask_b)
        configs[s] = phi.ravel()
        z2 = np.exp(2j * phi).mean(); z1 = np.exp(1j * phi).mean()
        psi = (np.angle(z2) / 2) % np.pi
        psis[s] = psi; s2s[s] = np.abs(z2); pol[s] = np.abs(z1)
        signs[s] = 1.0 if np.real(z1 * np.exp(-1j * psi)) >= 0 else -1.0
    return configs, psis, signs, s2s, pol, rng


def pr_centered(A):
    Ac = A - A.mean(axis=0); n = Ac.shape[0]
    if n < 3:
        return 1.0
    G = (Ac @ Ac.T).astype(np.float64) / (n - 1)
    tr = float(np.trace(G)); tr2 = float((G * G).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, float)); y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    return float(np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, y, rcond=None)[0][1])


def features(configs):
    return np.concatenate([np.cos(configs), np.sin(configs), np.cos(2 * configs), np.sin(2 * configs)], axis=1)


def ladder(feat, cls, rng, n_classes=N_BINS):
    """Deficit per rung in the declared class order 0..n-1; returns sizes,
    deficits, the shift, and the number of rungs."""
    members = [np.where(cls == b)[0] for b in range(n_classes)]
    sizes, prs, counts = [], [], []
    for c in BIN_COUNTS:
        if c > n_classes:
            continue
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        sizes.append(len(sel)); prs.append(pr_centered(feat[sel])); counts.append(c)
    if len(sizes) < 3:
        return None
    obs = np.log(np.maximum(prs, 1e-9))
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(pr_centered(feat[rng.choice(len(feat), s, replace=False)]), 1e-9))
    flo = nl.mean(axis=0)
    deficit = (obs - flo).tolist()
    return {"sizes": sizes, "classes": counts, "deficit": deficit, "delta": float(slope(sizes, np.exp(obs)) - slope(sizes, np.exp(flo))), "n_rungs": len(sizes)}


def main():
    t0 = time.time()
    out = {"design": {"J2": J2, "J1": J1, "fracs": FRACS, "temps": TEMPS, "L": L, "chains": N_CHAINS, "n_samples": N_SAMPLES,
                      "bin_counts": BIN_COUNTS, "n_null": N_NULL, "min_rungs": MIN_RUNGS}, "cells": []}
    for T in TEMPS:
        for f in FRACS:
            rows = []
            for c in range(N_CHAINS):
                configs, psis, signs, s2s, pol, rng = sample_chain(L, T, f, seed=62000 + 100 * c + int(T * 13) + int(f * 100))
                feat = features(configs)
                dbin = np.minimum((psis / (np.pi / 4)).astype(int), 3)
                cls8 = dbin + 4 * (signs < 0)
                stall = ladder(feat, cls8, rng)
                dir8 = np.minimum((psis / (np.pi / N_BINS)).astype(int), N_BINS - 1)
                director = ladder(feat, dir8, rng)
                row = {"chain": c, "S2": float(s2s.mean()), "P": float(pol.mean()), "class_counts": np.bincount(cls8, minlength=8).tolist(),
                       "stall": stall, "director": director}
                if stall is not None and stall["n_rungs"] >= MIN_RUNGS and 1 in stall["classes"] and 4 in stall["classes"]:
                    d1 = stall["deficit"][stall["classes"].index(1)]; d4 = stall["deficit"][stall["classes"].index(4)]
                    if d1 != 0:
                        row["late_fraction"] = float(d4 / d1)
                rows.append(row)
            valid = [r for r in rows if "late_fraction" in r]
            dvalid = [r for r in rows if r["director"] is not None and r["director"]["n_rungs"] >= MIN_RUNGS]
            cell = {"T": T, "f": f, "n_valid": len(valid), "n_valid_director": len(dvalid), "S2": float(np.mean([r["S2"] for r in rows])),
                    "P": float(np.mean([r["P"] for r in rows])),
                    "late_fraction": float(np.mean([r["late_fraction"] for r in valid])) if valid else None,
                    "late_fraction_sd": float(np.std([r["late_fraction"] for r in valid])) if valid else None,
                    "delta8": float(np.mean([r["stall"]["delta"] for r in valid])) if valid else None,
                    "delta_director": float(np.mean([r["director"]["delta"] for r in dvalid])) if dvalid else None,
                    "deficit_mean": [float(np.mean([r["stall"]["deficit"][k] for r in valid])) for k in range(6)] if valid and all(len(r["stall"]["deficit"]) == 6 for r in valid) else None,
                    "n_rungs_by_chain": [r["stall"]["n_rungs"] if r["stall"] is not None else 0 for r in rows],
                    "chains": rows}
            out["cells"].append(cell)
            lf = f"{cell['late_fraction']:.2f}±{cell['late_fraction_sd']:.2f}" if valid else "n/a"
            print(f"T={T} f={f:.2f}: S2 {cell['S2']:.2f} |P| {cell['P']:.2f} | valid {len(valid)}/{N_CHAINS} | late {lf} | delta8 "
                  f"{cell['delta8'] if valid else float('nan'):+.3f} | director {cell['delta_director'] if dvalid else float('nan'):+.3f} | {time.time()-t0:.0f}s", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
    verdict = {}
    for T in TEMPS:
        cells = [c for c in out["cells"] if c["T"] == T]
        ok = [c for c in cells if c["n_valid"] >= 3]
        v = {"n_valid_by_f": {str(c["f"]): c["n_valid"] for c in cells}}
        f0 = next((c for c in ok if c["f"] == 0.0), None)
        v["X1"] = bool(f0 is not None and abs(f0["late_fraction"]) <= 0.15) if f0 else None
        if len(ok) >= 4:
            fs = [c["f"] for c in ok]
            v["X2_rho"] = float(spearmanr(fs, [c["late_fraction"] for c in ok])[0])
            v["X3_rho"] = float(spearmanr(fs, [c["delta8"] for c in ok])[0])
            v["X2"] = bool(v["X2_rho"] >= 0.9); v["X3"] = bool(v["X3_rho"] >= 0.9)
        f1 = next((c for c in ok if c["f"] == 1.0), None)
        v["late_at_f1"] = f1["late_fraction"] if f1 else None
        dok = [c for c in cells if c["n_valid_director"] >= 3]
        v["X4"] = bool(dok and all(c["delta_director"] > 0 for c in dok)) if dok else None
        verdict[str(T)] = v
    # X2's second clause: late fraction > 0.3 at f = 1 at the T with the largest f = 1 eight-class shift
    f1cells = [c for c in out["cells"] if c["f"] == 1.0 and c["n_valid"] >= 3]
    if f1cells:
        best = max(f1cells, key=lambda c: c["delta8"])
        verdict["X2_late_gt_0.3_at_best_T"] = {"T": best["T"], "late": best["late_fraction"], "pass": bool(best["late_fraction"] > 0.3)}
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
