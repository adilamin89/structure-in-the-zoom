"""H2-R41: the XY director ladder's within / class-mean split as configurations per class grow past the class's own
fluctuation dimension. Feature-space split (same algebra as run72.logparts / run63.traces: w1 = Tr C_W, w2 = ||C_W||^2,
b1 = Tr C_B, b2 = ||C_B||^2, x = Tr C_W C_B with 1/n normalisation), validated against r63.traces before use.
Usage: python h2_xy_asymptote.py L T  -> data_canonical/h2_xy_asymptote_L{L}_T{T}.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h2_lattice as h  # noqa: E402
import h2_xy_split as xs  # noqa: E402
import nematic_polar_delta as nem  # noqa: E402

J1 = 0.42
N_FLOOR = 6


def traces_fs(F, lab):
    """Feature-space traces of the pooled within-class and class-mean covariances of rows F with labels lab."""
    n, d = F.shape
    mu = F.mean(0)
    CW = np.zeros((d, d))
    CB = np.zeros((d, d))
    for c in np.unique(lab):
        Fc = F[lab == c]
        mc = Fc.mean(0)
        R = Fc - mc
        CW += R.T @ R
        dm = (mc - mu)[:, None]
        CB += len(Fc) * (dm @ dm.T)
    CW /= n
    CB /= n
    return float(np.trace(CB)), float(np.trace(CW)), float((CB * CB).sum()), float((CW * CW).sum()), float((CW * CB).sum())


def logparts(F, lab):
    b1, w1, b2, w2, x = traces_fs(F, lab)
    return {"logpr": np.log((b1 + w1) ** 2 / (b2 + w2 + 2 * x)), "noise": np.log(w1 ** 2 / w2),
            "probe": np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2))}


def slope(sizes, ys):
    return float(np.polyfit(np.log(sizes), ys, 1)[0])


def split(F, lab, sets, rng):
    lad, flo, sizes = [], [], []
    for sel in sets:
        sizes.append(len(sel))
        lad.append(logparts(F[sel], lab[sel]))
        acc = {k: 0.0 for k in lad[-1]}
        for _ in range(N_FLOOR):
            idx = rng.choice(len(F), len(sel), replace=False)
            p = logparts(F[idx], lab[idx])
            for k in acc:
                acc[k] += p[k] / N_FLOOR
        flo.append(acc)
    return {k: slope(sizes, [l[k] - f[k] for l, f in zip(lad, flo)]) for k in ("logpr", "noise", "probe")}


def class_pr_and_overlap(F, lab, rng):
    """One class's within PR; split-half shape overlap Kn by director-class separation (feature space)."""
    S_half = []
    for c in range(nem.N_BINS):
        Fc = F[lab == c]
        p = rng.permutation(len(Fc))
        A, B = Fc[p[: len(Fc) // 2]], Fc[p[len(Fc) // 2:]]
        S_half.append([np.cov(A.T, bias=True), np.cov(B.T, bias=True)])
    Sc = np.cov(F[lab == 0].T, bias=True)
    pr_class = float(np.trace(Sc) ** 2 / (Sc * Sc).sum())
    K = np.zeros((8, 8))
    for a in range(8):
        for b in range(8):
            K[a, b] = 0.5 * ((S_half[a][0] * S_half[b][1]).sum() + (S_half[a][1] * S_half[b][0]).sum())
    kd = np.diag(K)
    Kn = K / np.sqrt(np.outer(kd, kd))
    sep = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
    prof = [float(Kn[sep == s].mean()) for s in range(5)]
    sigma = float(Kn[sep > 0].mean())
    return pr_class, prof, sigma


def validate():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(120, 50)) + np.repeat(rng.normal(size=(8, 50)), 15, axis=0)
    lab = np.repeat(np.arange(8), 15)
    a = traces_fs(F, lab)
    G = F - F.mean(0)
    b = xs.r72.r63.traces(G, lab)
    assert np.allclose(a, b, rtol=1e-5), (a, b)


def ensemble(L, T, n_chains, e):
    cfgs = []
    for c in range(n_chains):
        seed = 33_000_000 + 1_000_000 * e + 10_000 * L + c + int(round(T * 100)) * 100_000_000
        x, _ = h.xy_chain_w(L, T, J1, 1.0, seed, True, 300 if L <= 32 else 500, 5, 10)
        cfgs.append(x)
    return np.concatenate(cfgs)


def main():
    validate()
    L, T = int(sys.argv[1]), float(sys.argv[2])
    per_list = [25, 100, 400, 1600] if L <= 16 else [25, 100, 200, 400]
    out = {"L": L, "T": T, "J1": J1, "cells": {}}
    path = h.OUTDIR / f"h2_xy_asymptote_L{L}_T{T:.2f}.json"
    for per in per_list:
        rows = []
        for e in range(2):
            cfg = ensemble(L, T, per * 8 // 5, e)
            _, _, psi = h.order_params(cfg)
            lab = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
            F = nem.features(cfg)
            F = F - F.mean(0)
            F /= F.std()
            order = np.argsort(lab, kind="stable")
            cnt = np.bincount(lab, minlength=8)
            sizes = sorted({int(cnt[:k].sum()) for k in nem.BIN_COUNTS if cnt[:k].sum() >= 10})
            rng = np.random.default_rng(900 + e)
            sp = split(F, lab, [order[:s] for s in sizes], rng)
            b1, w1, b2, w2, x = traces_fs(F, lab)
            pr_class, prof, sigma = class_pr_and_overlap(F, lab, rng)
            mf = float(np.log(8 / (1 + 7 * sigma)) / np.log(8))
            r = {"ens": e, "per_class": per, "delta": sp["logpr"], "within": sp["noise"], "class_mean": sp["probe"],
                 "PR_W": w1 ** 2 / w2, "PR_B": b1 ** 2 / b2, "PR_class": pr_class, "n_over_dclass": per / pr_class,
                 "overlap_profile": prof, "sigma": sigma, "mean_field": mf}
            rows.append(r)
            print(f"L={L} T={T} {per}/class ens {e}: d {r['delta']:+.3f} w {r['within']:+.3f} m {r['class_mean']:+.3f} "
                  f"PR_W {r['PR_W']:.0f} PR_class {pr_class:.0f} n/d {r['n_over_dclass']:.2f} sigma {sigma:.2f} mf {mf:+.3f} "
                  f"Kn {np.round(prof, 2).tolist()}", flush=True)
        out["cells"][per] = rows
        json.dump(out, open(path, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
