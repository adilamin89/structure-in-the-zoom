"""Run 13 - Repair the named defect and re-predict (item-4 upgrade).

run11b: the one-parameter calibration predicts delta = +0.32 [0.29, 0.34]
against the observed +0.237, and the overshoot was attributed to the model's
gain modulation over-recovering the 180-deg alignment (the modulation pattern
inherits the even-dominated class-mean kernel). If that attribution is right,
giving the GAIN pathway its own odd-harmonic admixture (a second field D with
pure cos(dphi) correlation, modulation pattern = normalize(M + gamma D))
should lower the model's 180-deg alignment toward the measured 0.19 and move
the alignment-calibrated delta prediction toward the observed value.

REGISTERED EXPECTATIONS (before run):
G1: increasing gamma lowers the model's 180-deg alignment at fixed s.
G2: the two-parameter fit (s, gamma) to the four measured alignment points
    (the observed delta never enters the fit) yields a predicted delta closer
    to +0.237 than the one-parameter +0.318; success = prediction within
    [0.19, 0.29].

Out: feedback_runs/run13_gain_dipole_repair.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "run2", HERE / "run2_calibrated_corotating.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)

MEASURED = {45: 0.139, 90: 0.048, 135: 0.092, 180: 0.192}
N_BINS, RANK, N_PER = r2.N_BINS, 10, 120
S_VALS = [0.5, 0.65, 0.8, 1.0]
G_VALS = [0.0, 0.5, 1.0, 2.0]


def make_two_param(n_stim, n_neur, bl, r_within, pr_within, s, gamma, seed=1):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, N_BINS, endpoint=False)
    dphi = np.abs(ang[:, None] - ang[None, :])
    Cm = r2.A0 + r2.B2 * np.cos(2 * dphi) + r2.C1 * np.cos(dphi) \
        + r2.B4 * np.cos(4 * dphi)
    np.fill_diagonal(Cm, r2.A0 + r2.B2 + r2.C1 + r2.B4)
    Cm = Cm / Cm[0, 0]
    w, V = np.linalg.eigh(Cm)
    L = V @ np.diag(np.sqrt(np.maximum(w, 0)))
    M = L @ rng.standard_normal((N_BINS, n_neur))
    # odd-harmonic field for the gain pathway (pure dipole kernel)
    Cd = 0.5 + 0.5 * np.cos(dphi)
    wd, Vd = np.linalg.eigh(Cd)
    Ld = Vd @ np.diag(np.sqrt(np.maximum(wd, 0)))
    D = Ld @ rng.standard_normal((N_BINS, n_neur))
    G = M + gamma * D
    Ghat = G / (np.sqrt((G ** 2).mean(axis=1, keepdims=True)) + 1e-9)
    s2 = float((M ** 2).mean())
    sig2_total = s2 * (1 - r_within) / max(r_within, 1e-3)
    K = max(int(round(pr_within)), 2)
    frac_iso = 0.1
    U = np.linalg.qr(rng.standard_normal((n_neur, K)))[0]
    g = rng.standard_normal((n_stim, K)) * np.sqrt(
        sig2_total * (1 - frac_iso) * n_neur / K)
    modulation = (1 - s) + s * Ghat[bl]
    Xs = (M[bl] + (g @ U.T) * modulation
          + rng.standard_normal((n_stim, n_neur)) * np.sqrt(
              sig2_total * frac_iso))
    return np.ascontiguousarray(Xs.astype(np.float32))


def subspace(V):
    Vc = V - V.mean(axis=0)
    Gm = Vc @ Vc.T
    w, U = np.linalg.eigh(Gm)
    order = np.argsort(w)[::-1][:RANK]
    B = Vc.T @ U[:, order]
    B /= np.linalg.norm(B, axis=0, keepdims=True) + 1e-12
    return B


def profile_of(Xs, bl, rng):
    subs = []
    for b in range(N_BINS):
        idx = rng.choice(np.where(bl == b)[0], N_PER, replace=False)
        subs.append(subspace(Xs[idx]))
    prof = {sep: [] for sep in (45, 90, 135, 180)}
    for i in range(N_BINS):
        for j in range(i + 1, N_BINS):
            sep = min((j - i) % N_BINS, (i - j) % N_BINS) * 45
            sv = np.linalg.svd(subs[i].T @ subs[j], compute_uv=False)
            prof[sep].append(float(np.mean(np.clip(sv, 0, 1) ** 2)))
    return {k: float(np.mean(v)) for k, v in prof.items()}


def main():
    (n_stim, n_neur), bl, rw, pw = r2.measure_gt3()
    cells = {}
    for s in S_VALS:
        for gam in G_VALS:
            ds, profs = [], []
            for seed in (1, 2, 3):
                Xs = make_two_param(n_stim, 4000, bl, rw, pw, s, gam,
                                    seed=seed)
                d, _ = r2.ladder_delta(Xs, bl,
                                       np.random.default_rng(600 + seed))
                ds.append(d)
                profs.append(profile_of(Xs, bl,
                                        np.random.default_rng(700 + seed)))
            prof = {k: float(np.mean([p[k] for p in profs]))
                    for k in profs[0]}
            err = float(np.mean([(prof[k] - MEASURED[k]) ** 2
                                 for k in MEASURED]))
            cells[(s, gam)] = {"delta": float(np.mean(ds)),
                               "profile": prof, "err": err}
            print(f"s={s:.2f} g={gam:.1f}: delta={np.mean(ds):+.3f} | "
                  f"45/90/135/180 = {prof[45]:.3f}/{prof[90]:.3f}/"
                  f"{prof[135]:.3f}/{prof[180]:.3f} | err={err:.5f}",
                  flush=True)
    best = min(cells, key=lambda k: cells[k]["err"])
    print(f"best fit (s={best[0]}, gamma={best[1]}): "
          f"delta_pred={cells[best]['delta']:+.3f} (observed +0.237; "
          f"one-parameter prediction was +0.318)", flush=True)
    json.dump({"cells": {f"{s}_{g}": v for (s, g), v in cells.items()},
               "best": {"s": best[0], "gamma": best[1],
                        "delta_pred": cells[best]["delta"]},
               "observed": 0.237, "one_param_pred": 0.318},
              open(HERE / "run13_gain_dipole_repair.json", "w"), indent=1)
    print("DONE run13", flush=True)


if __name__ == "__main__":
    main()
