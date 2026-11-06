"""Are the across-recording laws protocol contrasts? Eight grating recordings = 3 mice
(GT1, GT2, GT3) x 3 protocols (drifting, localized, low-contrast; low-contrast GT3 absent). Predictors and outcomes come from
opposite temporal halves (epochs of 48 consecutive presentations, A = even, B = odd; both assignments, averaged):
  predictors (selection half)   DS fraction (tuned units F p < 0.01 over 8 classes; first-harmonic vector strength > 0.3)
  outcomes (other half)         delta (Table 9 ladder), dK = Kn(22.5) - Kn(180) (Fig 2C overlap, 16 classes) and the
                                rank-10 subspace alignment at 180 deg and its adjacent-minus-antipodal margin (classmean_alignment_all8
                                estimator, 16 classes, up to 120 presentations per class).
Analysis: raw Pearson / Spearman across the 8; mouse-and-protocol demeaned (two-way additive fit, least squares, missing cell
kept missing) residual correlation with an exact restricted permutation (predictor permuted within protocol: 3! 3! 2! = 72);
within-protocol slopes; leave-one-mouse-out correlations.
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler protocol_contrasts.py [n_workers]   Out: data_canonical/protocol_contrasts.json"""
import importlib.util
import itertools
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
OUT = HERE.parent / "data_canonical" / "protocol_contrasts.json"
E, RANK, N_PER = 48, 10, 120
RUNGS = (1, 2, 3, 4, 6, 8)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def ds_fraction(X, lab):
    n = len(lab); counts = np.bincount(lab, minlength=8)
    M = np.stack([X[lab == k].mean(0) for k in range(8)]); g = X.mean(0)
    ssb = sum(counts[k] * (M[k] - g) ** 2 for k in range(8)); ssw = sum(((X[lab == k] - M[k]) ** 2).sum(0) for k in range(8))
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ssb / 7) / (ssw / (n - 8))
    tuned = np.nan_to_num(1 - stats.f.cdf(F, 7, n - 8), nan=1.0) < 0.01
    R = M - M.min(0, keepdims=True) + 1e-9
    dsi = np.abs((R * np.exp(1j * 2 * np.pi * np.arange(8)[:, None] / 8)).sum(0) / R.sum(0))
    return float((dsi[tuned] > 0.3).mean())


def alignment_profile(X, lab16, rng):
    n_per = min(int(np.bincount(lab16, minlength=16).min()), N_PER); subs = []
    for b in range(16):
        V = X[rng.choice(np.where(lab16 == b)[0], n_per, replace=False)].astype(np.float64)
        _, _, Vt = np.linalg.svd(V - V.mean(0), full_matrices=False); subs.append(Vt[:RANK].T)
    by = {}
    for i in range(16):
        for j in range(i + 1, 16):
            s = min((j - i) % 16, (i - j) % 16)
            by.setdefault(s, []).append(float(np.mean(np.clip(np.linalg.svd(subs[i].T @ subs[j], compute_uv=False), 0, 1) ** 2)))
    return {s: float(np.mean(v)) for s, v in by.items()}


def run_one(fp):
    ow = _load("ow", "cr_order_and_withinbin_controls.py"); om = _load("om", "overlap_identity_checks.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab, lab16 = ow.bins(phi, 8), ow.bins(phi, 16); ep = (np.arange(len(phi)) // E) % 2
    out = {"name": fp.stem, "ways": []}
    for s, m in ((0, 1), (1, 0)):
        S, M = ep == s, ep == m
        Xm, lm, l16 = X[M], lab[M], lab16[M]
        mm = [np.where(lm == b)[0] for b in range(8)]
        G = (Xm @ Xm.T).astype(np.float64)
        lad = ow.ladder(G, lm, [np.concatenate(mm[:c]) for c in RUNGS], np.random.default_rng(42))
        idx = [np.where(l16 == c)[0] for c in range(16)]
        tr, _, T = om.overlaps(G, idx); K = om.K_of(tr, tr, T); kd = np.diag(K); Kn = K / np.sqrt(np.outer(kd, kd))
        a = np.arange(16); sep = np.minimum(np.abs(np.subtract.outer(a, a)), 16 - np.abs(np.subtract.outer(a, a)))
        al = alignment_profile(Xm, l16, np.random.default_rng(0))
        out["ways"].append({"ds_fraction": ds_fraction(X[S], lab[S]), "delta": lad["delta"],
                            "dK": float(Kn[sep == 1].mean() - Kn[sep == 8].mean()),
                            "align_180": al[8], "align_margin": al[1] - al[8]})
    for k in out["ways"][0]:
        out[k] = float(np.mean([w[k] for w in out["ways"]]))
    print(f"{fp.stem[9:30]:22s} DS {out['ds_fraction']:.3f} delta {out['delta']:+.3f} dK {out['dK']:+.3f} align180 {out['align_180']:.3f} "
          f"margin {out['align_margin']:+.3f} ({time.time() - t0:.0f}s) DONE", flush=True)
    return out


def demean(y, mouse, prot):
    """Residual of a two-way additive (mouse + protocol) least-squares fit."""
    ms, ps = sorted(set(mouse)), sorted(set(prot))
    D = np.column_stack([np.ones(len(y))] + [[m == q for m in mouse] for q in ms[1:]] + [[p == q for p in prot] for q in ps[1:]]).astype(float)
    return y - D @ np.linalg.lstsq(D, y, rcond=None)[0]


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(run_one, sorted(RAW.glob("gratings*.npy"))))
    mouse = [r["name"].split("_GT")[1][0] for r in rows]
    prot = [next(k for k in ("low_contrast", "drifting", "local") if k in r["name"]) for r in rows]
    x = np.array([r["ds_fraction"] for r in rows]); res = {}
    groups = {p: [i for i in range(8) if prot[i] == p] for p in set(prot)}
    for out_k in ("delta", "dK", "align_180", "align_margin"):
        y = np.array([r[out_k] for r in rows]); xr, yr = demean(x, mouse, prot), demean(y, mouse, prot)
        r0 = float(np.corrcoef(xr, yr)[0, 1]) if xr.std() > 0 and yr.std() > 0 else None
        null = []
        for perms in itertools.product(*[list(itertools.permutations(g)) for g in groups.values()]):
            xp = x.copy()
            for g, pg in zip(groups.values(), perms):
                xp[list(g)] = x[list(pg)]
            xpr = demean(xp, mouse, prot)
            null.append(np.corrcoef(xpr, yr)[0, 1] if xpr.std() > 0 else 0)
        null = np.array(null)
        within = {p: (float(np.polyfit(x[g], y[g], 1)[0]) if len(g) > 1 else None) for p, g in groups.items()}
        lomo = {m: float(np.corrcoef(x[[i for i in range(8) if mouse[i] != m]], y[[i for i in range(8) if mouse[i] != m]])[0, 1]) for m in sorted(set(mouse))}
        res[out_k] = {"pearson_raw": float(np.corrcoef(x, y)[0, 1]), "spearman_raw": float(stats.spearmanr(x, y)[0]),
                      "pearson_demeaned": r0, "restricted_perm_p_two_sided": float(np.mean(np.abs(null) >= abs(r0) - 1e-12)) if r0 is not None else None,
                      "within_protocol_slopes": within, "leave_one_mouse_out_r": lomo}
        print(out_k, json.dumps(res[out_k]), flush=True)
    json.dump({"rows": rows, "mouse": mouse, "protocol": prot, "associations": res}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
