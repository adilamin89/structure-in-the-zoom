"""Run 6 - Calibrate the co-rotation strength to the measured principal-angle
profile, then predict delta with NO free parameter (referee M2/Q2).

Partial co-rotation s in [0,1]: within-class mode patterns are
(1-s) * shared + s * class-modulated, so s=0 is the shared-mode model
(delta +0.080) and s=1 the fully co-rotating model (delta +0.491). For each s
we compute BOTH delta AND the model's own principal-angle alignment profile
(same estimator as run3: top-10 subspaces, 120 trials per class, mean cos^2
principal angles by angular separation). s* is chosen to match the measured
GT3 profile (raw, r=10: 45deg 0.139, 90deg 0.048, 135deg 0.092, 180deg 0.192)
by least squares. delta(s*) is then a zero-free-parameter prediction of the
measured GT3 shift (+0.237).

REGISTERED EXPECTATIONS (before run):
S1: delta(s) increases monotonically in s.
S2: the model's overall alignment level decreases with s (shared modes give
    near-total subspace overlap; class-modulated modes decorrelate it).
S3: SUCCESS CRITERION: delta(s*) lands in [0.15, 0.35] around GT3's +0.237.
    A miss is reported as a miss; it would mean partial co-rotation of this
    one-parameter form cannot jointly match alignment and delta.

Out: feedback_runs/run6_corotation_calibration.json
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

MEASURED = {45: 0.139, 90: 0.048, 135: 0.092, 180: 0.192}  # GT3 raw, r=10
N_BINS = r2.N_BINS
RANK = 10
N_PER = 120


def make_partial(n_stim, n_neur, bl, r_within, pr_within, s, seed=1):
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
    s2 = float((M ** 2).mean())
    sig2_total = s2 * (1 - r_within) / max(r_within, 1e-3)
    K = max(int(round(pr_within)), 2)
    frac_iso = 0.1
    U = np.linalg.qr(rng.standard_normal((n_neur, K)))[0]
    g = rng.standard_normal((n_stim, K)) * np.sqrt(
        sig2_total * (1 - frac_iso) * n_neur / K)
    Mhat = M / (np.sqrt((M ** 2).mean(axis=1, keepdims=True)) + 1e-9)
    modulation = (1 - s) + s * Mhat[bl]
    W = (g @ U.T) * modulation
    Xs = (M[bl] + W
          + rng.standard_normal((n_stim, n_neur)) * np.sqrt(
              sig2_total * frac_iso))
    return np.ascontiguousarray(Xs.astype(np.float32))


def class_subspace(V, r):
    Vc = V - V.mean(axis=0)
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    return Vt[:r].T


def alignment_profile(Xs, bl, rng):
    subs = []
    for b in range(N_BINS):
        idx = rng.choice(np.where(bl == b)[0], N_PER, replace=False)
        subs.append(class_subspace(Xs[idx], RANK))
    by_sep = {}
    for i in range(N_BINS):
        for j in range(i + 1, N_BINS):
            sep = min((j - i) % N_BINS, (i - j) % N_BINS) * 45
            s_ = np.linalg.svd(subs[i].T @ subs[j], compute_uv=False)
            by_sep.setdefault(sep, []).append(float(np.mean(s_ ** 2)))
    return {sep: float(np.mean(v)) for sep, v in by_sep.items()}


def main():
    (n_stim, n_neur), bl, rw, pw = r2.measure_gt3()
    print(f"calibration: corr={rw:.3f} PR={pw:.1f}", flush=True)
    out = {"measured_profile": MEASURED, "gt3_delta": 0.237, "sweep": {}}
    best = None
    for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        ds, profs = [], []
        for seed in (1, 2, 3):
            Xs = make_partial(n_stim, 4000, bl, rw, pw, s, seed=seed)
            d, _ = r2.ladder_delta(Xs, bl, np.random.default_rng(300 + seed))
            ds.append(d)
            profs.append(alignment_profile(Xs, bl,
                                           np.random.default_rng(400 + seed)))
        prof = {sep: float(np.mean([p[sep] for p in profs]))
                for sep in profs[0]}
        err = float(np.mean([(prof[sep] - MEASURED[sep]) ** 2
                             for sep in MEASURED]))
        out["sweep"][str(s)] = {"delta": float(np.mean(ds)),
                                "delta_sd": float(np.std(ds)),
                                "profile": {str(k): v for k, v in prof.items()},
                                "sq_err_vs_measured": err}
        print(f"s={s:.1f}: delta={np.mean(ds):+.3f}±{np.std(ds):.3f} | "
              f"align 45/90/135/180 = {prof[45]:.3f}/{prof[90]:.3f}/"
              f"{prof[135]:.3f}/{prof[180]:.3f} | err={err:.5f}", flush=True)
        if best is None or err < best[1]:
            best = (s, err, float(np.mean(ds)))
    out["s_star"] = best[0]
    out["delta_at_s_star"] = best[2]
    print(f"s* = {best[0]} -> delta = {best[2]:+.3f} (GT3 +0.237; "
          f"registered success band [0.15, 0.35])", flush=True)
    json.dump(out, open(HERE / "run6_corotation_calibration.json", "w"),
              indent=1)
    print("DONE run6", flush=True)


if __name__ == "__main__":
    main()
