"""Data-calibrated generative synthetic - ground truth matched to GT3.

Replaces the kappa-regime synthetics. The generative model is calibrated to the
measured GT3 quantities, then asked whether the observed delta EMERGES:
  - class-mean correlation structure = the four-term multipole fit
    C(dphi) = a + b cos2 + c cos1 + b4 cos4  (a=0.3947, b=0.3298, c=0.1402, b4=0.072)
  - within-class trial correlation matched to the value MEASURED from GT3
  - same population size, class sizes, ladder, floor, shuffle control

REGISTERED EXPECTATIONS (before run):
E1: delta_dir on the calibrated synthetic lands in [0.15, 0.35]
    (GT3 measured +0.237), with shuffle control near zero.
E2: setting b = c = b4 = 0 (isotropic class means) gives delta near zero.
E3: doubling b (stronger quadrupole) increases delta.

V1 RESULT (registered miss, kept): with rank-1 class means + ISOTROPIC trial
noise, E1 = -0.567 and E3 = -0.748 (E2 = 0, shuffle = 0): a single-class rung
is an isotropic maximal-PR ball after centering, forcing the
mode-conditioning sign. LESSON: the sign of delta encodes the dimensionality
of within-class variability relative to between-class structure.
V2 FIX (registered before the v2 run): trial variability = shared LOW-RANK
modes (gain-like), calibrated to BOTH the measured within-class trial
correlation AND the measured within-class centered PR of GT3.
V2 EXPECTATIONS: E1' in [0.1, 0.4] and positive; E2' near 0; E3' more
positive than E1'.

Output: data/synthetic_calibrated_gt3.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stringer_v1" / "natimg"
OUT = HERE / "data" / "synthetic_calibrated_gt3.json"

N_BINS = 8
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_NULL = 10
N_SHUF = 5
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


def ladder_delta(Xt, bin_idx, rng, n_shuf=0):
    members = [np.where(bin_idx == b)[0] for b in range(N_BINS)]
    sizes, prs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        sizes.append(len(sel))
        prs.append(pr_c(Xt[sel]))
    th_o = slope(sizes, prs)
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(pr_c(Xt[rng.choice(len(Xt), s, replace=False)]), 1e-9))
    th_f = slope(sizes, np.exp(nl.mean(axis=0)))
    shufs = []
    for s in range(n_shuf):
        srng = np.random.default_rng(900 + s)
        perm = bin_idx[srng.permutation(len(bin_idx))]
        m2 = [np.where(perm == b)[0] for b in range(N_BINS)]
        sz2, pr2 = [], []
        for c in BIN_COUNTS:
            sel = np.concatenate(m2[:c])
            if len(sel) < 10:
                continue
            sz2.append(len(sel))
            pr2.append(pr_c(Xt[sel]))
        shufs.append(slope(sz2, pr2) - th_f)
    return th_o - th_f, shufs


def measure_gt3():
    dat = np.load(DATA / "gratings_drifting_GT3_2019_04_05_1.npy", allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)
    # within-class trial correlation + within-class centered PR (the v2
    # calibration target: the dimensionality of within-class variability)
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


def make_synthetic(n_stim, n_neur, bl, r_within, pr_within, b2=B2, c1=C1, b4=B4,
                   seed=1):
    """v2: class means with the target multipole correlation; within-class
    variability = K shared low-rank modes (gain-like) with K set by the
    MEASURED within-class centered PR, plus a small isotropic term. Total
    noise variance tuned to the measured within-class trial correlation."""
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
    frac_iso = 0.1                                 # 90% of variability in K modes
    U = np.linalg.qr(rng.standard_normal((n_neur, K)))[0]   # shared modes
    g = rng.standard_normal((n_stim, K)) * np.sqrt(sig2_total * (1 - frac_iso)
                                                   * n_neur / K)
    Xs = (M[bl] + g @ U.T
          + rng.standard_normal((n_stim, n_neur)) * np.sqrt(sig2_total * frac_iso))
    return np.ascontiguousarray(Xs.astype(np.float32))


def main():
    (n_stim, n_neur), bl, r_within, pr_within = measure_gt3()
    print(f"GT3: {n_stim} stim x {n_neur} neur, within-class corr = {r_within:.3f}, "
          f"within-class PR = {pr_within:.1f}", flush=True)
    n_neur_syn = 4000  # compute-friendly; PR ladder regime preserved (n_stim << n_neur)

    out = {"measured_within_class_corr": r_within,
           "measured_within_class_pr": pr_within, "n_neur_syn": n_neur_syn}
    Xs = make_synthetic(n_stim, n_neur_syn, bl, r_within, pr_within, seed=1)
    # self-check: realized within-class PR of the synthetic
    idx0 = np.where(bl == 0)[0][:120]
    out["realized_within_class_pr"] = pr_c(Xs[idx0])
    print(f"synthetic realized within-class PR = {out['realized_within_class_pr']:.1f}",
          flush=True)
    d, sh = ladder_delta(Xs, bl, np.random.default_rng(42), n_shuf=N_SHUF)
    out["E1_calibrated"] = {"delta_dir": d, "shuffle_mean": float(np.mean(sh)),
                            "shuffle_sd": float(np.std(sh))}
    print(f"E1 calibrated: delta={d:+.4f} shuffle={np.mean(sh):+.4f}±{np.std(sh):.4f}",
          flush=True)

    Xi = make_synthetic(n_stim, n_neur_syn, bl, r_within, pr_within,
                        b2=0, c1=0, b4=0, seed=2)
    di, _ = ladder_delta(Xi, bl, np.random.default_rng(43))
    out["E2_isotropic"] = {"delta_dir": di}
    print(f"E2 isotropic means: delta={di:+.4f}", flush=True)

    Xb = make_synthetic(n_stim, n_neur_syn, bl, r_within, pr_within,
                        b2=2 * B2, seed=3)
    db, _ = ladder_delta(Xb, bl, np.random.default_rng(44))
    out["E3_double_quadrupole"] = {"delta_dir": db}
    print(f"E3 doubled b: delta={db:+.4f}", flush=True)

    with OUT.open("w") as f:
        json.dump(out, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
