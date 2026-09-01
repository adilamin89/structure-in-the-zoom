"""Run 2 - Calibrated GT3 generative model with CO-ROTATING within-class modes.

The v2 calibrated model (calibrated_synthetic.py) draws within-class
variability as low-rank modes SHARED across classes and under-predicts GT3
threefold (+0.08 vs +0.24). The feedback toy (toy_delta.py) shows shared modes
add no dimensions as classes accumulate; class-dependent (co-rotating) modes
raise delta 3-4x. This run tests that on the calibrated model itself.

REGISTERED EXPECTATIONS (before run):
R1: co-rotating within-class modes raise delta relative to the shared-mode
    baseline at identical calibration targets; success = delta_corot in
    [0.15, 0.35], i.e. the threefold gap to GT3 (+0.237) substantially closes.
R2: shuffle control stays near zero in all variants.
R3: doubling b2 (between-class correlation only, within-class geometry fixed)
    still does NOT raise delta under co-rotation (predict negative-to-flat):
    the two-knobs resolution says the b2-at-fixed-everything derivative is
    non-positive regardless of mode geometry.
R4: raising b2 WITH proportional within-class co-rotation strength (the
    biological covariation path: stronger tuning -> stronger tuned gain
    variability) raises delta - the empirical Allen direction.

Out: feedback_runs/run2_calibrated_corotating.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts_canonical"))
DATA = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"

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
    dat = np.load(DATA / "gratings_drifting_GT3_2019_04_05_1.npy",
                  allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
                 0, N_BINS - 1)
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


def make_synthetic(n_stim, n_neur, bl, r_within, pr_within, b2=B2, c1=C1, b4=B4,
                   within="shared", gain_scale=1.0, seed=1):
    """within='shared':     v2 baseline, modes identical for all classes.
    within='corotating':    mode k of class j has pattern u_k modulated
                            elementwise by the class-mean profile m_j
                            (trial-to-trial gain fluctuations of the tuned
                            response), so within-class subspaces rotate with
                            the class label.
    gain_scale multiplies the co-rotating mode amplitude (R4 covariation path).
    """
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
    frac_iso = 0.1
    U = np.linalg.qr(rng.standard_normal((n_neur, K)))[0]
    g = rng.standard_normal((n_stim, K)) * np.sqrt(sig2_total * (1 - frac_iso)
                                                   * n_neur / K)
    if within == "shared":
        W = g @ U.T
    else:  # corotating: modulate each mode by the class-mean profile
        Mhat = M / (np.sqrt((M ** 2).mean(axis=1, keepdims=True)) + 1e-9)
        W = (g @ U.T) * Mhat[bl] * gain_scale
    Xs = (M[bl] + W
          + rng.standard_normal((n_stim, n_neur)) * np.sqrt(sig2_total * frac_iso))
    return np.ascontiguousarray(Xs.astype(np.float32))


def realized_wc(Xs, bl):
    idx0 = np.where(bl == 0)[0][:120]
    return pr_c(Xs[idx0])


def main():
    (n_stim, n_neur), bl, r_within, pr_within = measure_gt3()
    print(f"GT3: {n_stim} stim, within-class corr = {r_within:.3f}, "
          f"within-class PR = {pr_within:.1f}", flush=True)
    n_syn = 4000
    out = {"gt3_delta_reference": 0.237}

    for name, kw in [
        ("shared_baseline", dict(within="shared", seed=1)),
        ("corotating", dict(within="corotating", seed=1)),
        ("corotating_seed2", dict(within="corotating", seed=7)),
    ]:
        Xs = make_synthetic(n_stim, n_syn, bl, r_within, pr_within, **kw)
        d, sh = ladder_delta(Xs, bl, np.random.default_rng(42), n_shuf=N_SHUF)
        out[name] = {"delta": d, "shuffle_mean": float(np.mean(sh)),
                     "realized_wc_pr": realized_wc(Xs, bl)}
        print(f"{name}: delta={d:+.4f} shuffle={np.mean(sh):+.4f} "
              f"wcPR={out[name]['realized_wc_pr']:.1f}", flush=True)

    # R3: b2 doubled, within-class geometry fixed (both mode types)
    for name, kw in [
        ("shared_b2x2", dict(within="shared", b2=2 * B2, seed=3)),
        ("corotating_b2x2", dict(within="corotating", b2=2 * B2, seed=3)),
    ]:
        Xs = make_synthetic(n_stim, n_syn, bl, r_within, pr_within, **kw)
        d, _ = ladder_delta(Xs, bl, np.random.default_rng(44))
        out[name] = {"delta": d}
        print(f"R3 {name}: delta={d:+.4f}", flush=True)

    # R4: biological covariation path - b2 up WITH gain amplitude up
    for name, kw in [
        ("corotating_b2x2_gainx2", dict(within="corotating", b2=2 * B2,
                                        gain_scale=2.0, seed=3)),
        ("corotating_gainx2", dict(within="corotating", gain_scale=2.0, seed=3)),
    ]:
        Xs = make_synthetic(n_stim, n_syn, bl, r_within, pr_within, **kw)
        d, _ = ladder_delta(Xs, bl, np.random.default_rng(44))
        out[name] = {"delta": d}
        print(f"R4 {name}: delta={d:+.4f}", flush=True)

    json.dump(out, open(HERE / "run2_calibrated_corotating.json", "w"), indent=1)
    print("DONE run2", flush=True)


if __name__ == "__main__":
    main()
