"""Paired seed sweep for the calibrated generative model.

The original single-seed run uses different seeds per arm
(1/2/3), so arm differences are confounded with realization noise and no
across-seed uncertainty exists for +0.08 / -0.57 / doubled-b.

Design: N_SEEDS paired repetitions. Within one seed, all four arms share the
SAME rng draws (identical shapes consumed in identical order; only the harmonic
coefficients / noise split differ) = common random numbers. Arms:
  A calibrated v2 (low-rank within-class variability, frac_iso=0.1)
  B harmonics-off (b2=c1=b4=0), otherwise as A
  C isotropic-noise (frac_iso=1.0, same total variance), otherwise as A
  D doubled quadrupole (b2 -> 2*b2), otherwise as A

REGISTERED EXPECTATIONS (before run, 2026-08-23):
  R1-E1: delta(A) > 0 in >= 18/20 seeds; median in [0.02, 0.20].
  R1-E2: |delta(B)| median < 0.05 (no-harmonics null).
  R1-E3: delta(C) < 0 in >= 18/20 seeds (isotropic noise flips the sign).
  R1-E4: delta(D) < delta(A) in >= 15/20 seeds (entry coherence: doubling b
         lowers delta), paired within seed.

Output: data/calibrated_seed_sweep.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stringer_v1" / "natimg"
OUT = HERE / "data" / "calibrated_seed_sweep.json"

N_BINS = 8
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_NULL = 10
N_SEEDS = 20
A0, B2, C1, B4 = 0.3947, 0.3298, 0.1402, 0.072


def pr_c(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, float))
    y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def ladder_delta(Xt, bin_idx, rng):
    members = [np.where(bin_idx == b)[0] for b in range(N_BINS)]
    sizes, prs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        sizes.append(len(sel))
        prs.append(pr_c(Xt[sel]))
    th_o = slope(sizes, prs)
    logs = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            logs[d, k] = np.log(max(pr_c(Xt[rng.choice(len(Xt), s, replace=False)]), 1e-9))
    th_f = slope(sizes, np.exp(logs.mean(axis=0)))
    return th_o - th_f


def measure_gt3():
    dat = np.load(DATA / "gratings_drifting_GT3_2019_04_05_1.npy", allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)
    rng = np.random.default_rng(0)
    cors, wc_prs = [], []
    for b in range(N_BINS):
        idx = rng.choice(np.where(bl == b)[0], 120, replace=False)
        V = Xt[idx]
        wc_prs.append(pr_c(V))
        Vn = V - V.mean(axis=1, keepdims=True)
        Vn /= np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-9
        Cm = Vn @ Vn.T
        cors.append((Cm.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
    return Xt.shape, bl, float(np.mean(cors)), float(np.median(wc_prs))


def make_synthetic(n_stim, n_neur, bl, r_within, pr_within, b2, c1, b4,
                   frac_iso, seed):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, N_BINS, endpoint=False)
    dphi = np.abs(ang[:, None] - ang[None, :])
    Cm = A0 + b2 * np.cos(2 * dphi) + c1 * np.cos(dphi) + b4 * np.cos(4 * dphi)
    np.fill_diagonal(Cm, A0 + b2 + c1 + b4)
    Cm = Cm / Cm[0, 0]
    w, V = np.linalg.eigh(Cm)
    L = V @ np.diag(np.sqrt(np.maximum(w, 0)))
    M = (L @ rng.standard_normal((N_BINS, n_neur)))
    s2 = float((M ** 2).mean())
    sig2_total = s2 * (1 - r_within) / max(r_within, 1e-3)
    K = max(int(round(pr_within)), 2)
    U = np.linalg.qr(rng.standard_normal((n_neur, K)))[0]
    g = rng.standard_normal((n_stim, K)) * np.sqrt(sig2_total * (1 - frac_iso)
                                                   * n_neur / K)
    Xs = (M[bl] + g @ U.T
          + rng.standard_normal((n_stim, n_neur)) * np.sqrt(sig2_total * max(frac_iso, 1e-12)))
    return np.ascontiguousarray(Xs.astype(np.float32))


def main():
    (n_stim, n_neur), bl, r_w, pr_w = measure_gt3()
    n_syn = 4000
    print(f"GT3 calib: {n_stim} stim, r_within={r_w:.3f}, pr_within={pr_w:.1f}", flush=True)
    arms = {
        "A_calibrated": dict(b2=B2, c1=C1, b4=B4, frac_iso=0.1),
        "B_no_harmonics": dict(b2=0.0, c1=0.0, b4=0.0, frac_iso=0.1),
        "C_isotropic_noise": dict(b2=B2, c1=C1, b4=B4, frac_iso=1.0),
        "D_doubled_b": dict(b2=2 * B2, c1=C1, b4=B4, frac_iso=0.1),
    }
    rows = []
    t0 = time.time()
    for s in range(N_SEEDS):
        row = {"seed": s}
        for name, kw in arms.items():
            Xs = make_synthetic(n_stim, n_syn, bl, r_w, pr_w, seed=1000 + s, **kw)
            row[name] = ladder_delta(Xs, bl, np.random.default_rng(5000 + s))
        rows.append(row)
        print(f"seed {s:2d}  A={row['A_calibrated']:+.4f}  B={row['B_no_harmonics']:+.4f}"
              f"  C={row['C_isotropic_noise']:+.4f}  D={row['D_doubled_b']:+.4f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)
    a = np.array([r["A_calibrated"] for r in rows])
    b = np.array([r["B_no_harmonics"] for r in rows])
    c = np.array([r["C_isotropic_noise"] for r in rows])
    d = np.array([r["D_doubled_b"] for r in rows])
    summary = {
        "registration": "calibrated_seed_sweep.py docstring (pre-run)",
        "n_seeds": N_SEEDS,
        "A_median": float(np.median(a)), "A_n_positive": int((a > 0).sum()),
        "B_median_abs": float(np.median(np.abs(b))),
        "C_median": float(np.median(c)), "C_n_negative": int((c < 0).sum()),
        "D_minus_A_median": float(np.median(d - a)),
        "D_lt_A_count": int((d < a).sum()),
        "rows": rows,
    }
    OUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    print("DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
