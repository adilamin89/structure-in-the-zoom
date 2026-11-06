"""H4-R1c (Oct 24; registration H4-R1c in data_canonical/ex0_REGISTERED.md, commit d4f222cd): follow-up to H4-R1, whose decoders
read the residual offsets a half-A subspace leaves in half B (its own estimation noise, linearly decodable in high dimension).

Same input, halves and neuron draws as h4_r1_meanfree.py. Per measured half (both halves):
  (a) 5-fold CV inside the half (StratifiedKFold, as h3_v1_decode); each fold's 7-dim class-mean subspace is estimated from the
      TRAINING folds (all neurons) and removed from training and test folds; App K's logistic (z-scored on the training folds)
      on 256 / 1024 neurons; LDA and QDA in 16 standardized PCs of 256 neurons (QDA regularization by inner 3-fold CV), the
      h3_v1_decode.lda_qda pipeline, under the same per-fold removal.
  (b) the shift in App I's design: the subspace of the half's own class means removed (sample class means exactly zero),
      against 200 permutations each removing its own permuted subspace; with the within / class-mean split.
Out: data_canonical/h4_r1c_crossfit.json (smoke: h4_r1c_crossfit_smoke.json)"""
import importlib.util
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
DATA = HERE.parent / "data_canonical"
OUT = DATA / "h4_r1c_crossfit.json"
NS = (256, 1024)
N_DRAW = 5


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R1 = _load("h4r1", "h4_r1_meanfree.py")


def folds(y, seed):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def proj_cols(X, Q, cols):
    """Columns `cols` of X projected off the subspace Q (all neurons): X[:, cols] - (X Q) Q[cols]^T."""
    return X[:, cols] - (X @ Q.astype(np.float32)) @ Q[cols].T.astype(np.float32)


def logistic_cf(X, y, cols, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    hits = 0
    for tr, te in folds(y, seed):
        Q = R1.mean_subspace(X[tr], y[tr])
        Xc = proj_cols(X, Q, cols)
        sc = StandardScaler().fit(Xc[tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xc[tr]), y[tr])
        hits += int((m.predict(sc.transform(Xc[te])) == y[te]).sum())
    return hits / len(y)


def lda_qda_cf(X, y, cols, seed):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    hl = hq = 0
    for tr, te in folds(y, seed):
        Q = R1.mean_subspace(X[tr], y[tr])
        Xd = proj_cols(X, Q, cols)
        sc = StandardScaler().fit(Xd[tr])
        pca = PCA(16, random_state=seed).fit(sc.transform(Xd[tr]))
        Ptr, Pte = pca.transform(sc.transform(Xd[tr])), pca.transform(sc.transform(Xd[te]))
        s2 = StandardScaler().fit(Ptr)
        Ptr, Pte = s2.transform(Ptr), s2.transform(Pte)
        hl += int((LinearDiscriminantAnalysis().fit(Ptr, y[tr]).predict(Pte) == y[te]).sum())
        g = GridSearchCV(QuadraticDiscriminantAnalysis(), {"reg_param": [0.01, 0.1, 0.3, 0.6, 0.9]}, cv=3).fit(Ptr, y[tr])
        hq += int((g.predict(Pte) == y[te]).sum())
    return hl / len(y), hq / len(y)


def one_half(X, y, sel, ow, rng_perm, n_perm):
    t0 = time.time()
    XB, yB = X[sel], y[sel]
    nB = len(yB)
    out = {"n": int(nB), "chance_band": float(1 / 8 + 2 * np.sqrt((1 / 8) * (7 / 8) / nB))}
    rng = np.random.default_rng(0)                                   # R1's neuron draws, in R1's order
    dec = {}
    for N in NS:
        acc = []
        for d in range(N_DRAW):
            cols = rng.choice(X.shape[1], N, replace=False)
            acc.append(logistic_cf(XB, yB, cols, d))
        dec[str(N)] = {"mean": float(np.mean(acc)), "draws": acc}
    out["decode_cf"] = dec
    lq = []
    for d in range(N_DRAW):
        cols = rng.choice(X.shape[1], 256, replace=False)
        lq.append(lda_qda_cf(XB, yB, cols, d))
    out["lda_cf"], out["qda_cf"] = float(np.mean([a[0] for a in lq])), float(np.mean([a[1] for a in lq]))
    out["lda_qda_draws"] = lq
    print(f"    decoders done ({time.time() - t0:.0f}s): log256 {dec['256']['mean']:.3f} log1024 {dec['1024']['mean']:.3f} "
          f"LDA {out['lda_cf']:.3f} QDA {out['qda_cf']:.3f} band {out['chance_band']:.3f}", flush=True)
    GB = (XB @ XB.T).astype(np.float64)

    def proj_gram(lab):
        Q = R1.mean_subspace(XB, lab)
        P = (XB @ Q.astype(np.float32)).astype(np.float64)
        return GB - P @ P.T
    out["insample"] = ow.ladder(proj_gram(yB), yB, R1.rung_sets(yB), np.random.default_rng(42))
    nd, nw = [], []
    for i in range(n_perm):
        yp = yB[rng_perm.permutation(nB)]
        r = ow.ladder(proj_gram(yp), yp, R1.rung_sets(yp), np.random.default_rng(42))
        nd.append(r["delta"])
        nw.append(r["within"])
        if (i + 1) % 50 == 0:
            print(f"    perm {i + 1}/{n_perm} null mean {np.mean(nd):+.4f} ({time.time() - t0:.0f}s)", flush=True)
    nd = np.array(nd)
    obs = out["insample"]["delta"]
    out["null"] = {"delta": nd.tolist(), "within": nw, "mean": float(nd.mean()), "sd": float(nd.std()), "max": float(nd.max()),
                   "p_one": float((1 + (nd >= obs).sum()) / (n_perm + 1)), "z": float((obs - nd.mean()) / nd.std())}
    return out


def run_one(args):
    fp, n_perm = args
    warnings.filterwarnings("ignore")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    X /= X.std() + 1e-9
    y = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    del dat
    half = np.zeros(len(y), int)
    for k in range(8):
        idx = np.where(y == k)[0]
        half[idx[1::2]] = 1
    row = {"name": fp.stem}
    for h in (0, 1):
        r = one_half(X, y, np.where(half == h)[0], ow, np.random.default_rng(2000 + h), n_perm)
        row[f"half{h}"] = r
        print(f"{fp.stem[:32]} half{h}: log256 {r['decode_cf']['256']['mean']:.3f} log1024 {r['decode_cf']['1024']['mean']:.3f} "
              f"(band {r['chance_band']:.3f}) LDA {r['lda_cf']:.3f} QDA {r['qda_cf']:.3f} | in-sample proj shift {r['insample']['delta']:+.3f} "
              f"(w {r['insample']['within']:+.3f} cm {r['insample']['class_means']:+.3f}) null {r['null']['mean']:+.4f} max "
              f"{r['null']['max']:+.4f} p {r['null']['p_one']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return row


def score(rows):
    cells = [(r["name"], h, r[h]) for r in rows for h in ("half0", "half1")]
    v = {"n_cells": len(cells),
         "R1c_a_log256_at_chance": sum(c[2]["decode_cf"]["256"]["mean"] <= c[2]["chance_band"] for c in cells),
         "log1024_at_chance": sum(c[2]["decode_cf"]["1024"]["mean"] <= c[2]["chance_band"] for c in cells),
         "R1c_b_shift_p_le_0.05": sum(c[2]["null"]["p_one"] <= 0.05 for c in cells),
         "log256_range": [min(c[2]["decode_cf"]["256"]["mean"] for c in cells), max(c[2]["decode_cf"]["256"]["mean"] for c in cells)],
         "log1024_range": [min(c[2]["decode_cf"]["1024"]["mean"] for c in cells), max(c[2]["decode_cf"]["1024"]["mean"] for c in cells)],
         "lda_range": [min(c[2]["lda_cf"] for c in cells), max(c[2]["lda_cf"] for c in cells)],
         "qda_range": [min(c[2]["qda_cf"] for c in cells), max(c[2]["qda_cf"] for c in cells)],
         "qda_above_band_per_half": {h: sum(r[h]["qda_cf"] > r[h]["chance_band"] for r in rows) for h in ("half0", "half1")},
         "lda_above_band": sum(c[2]["lda_cf"] > c[2]["chance_band"] for c in cells),
         "shift_range": [min(c[2]["insample"]["delta"] for c in cells), max(c[2]["insample"]["delta"] for c in cells)],
         "within_range": [min(c[2]["insample"]["within"] for c in cells), max(c[2]["insample"]["within"] for c in cells)],
         "null_max_range": [min(c[2]["null"]["max"] for c in cells), max(c[2]["null"]["max"] for c in cells)],
         "chance_band_range": [min(c[2]["chance_band"] for c in cells), max(c[2]["chance_band"] for c in cells)]}
    v["verdict"] = "PASS" if v["R1c_a_log256_at_chance"] == v["n_cells"] and v["R1c_b_shift_p_le_0.05"] == v["n_cells"] else "MISS"
    return v


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    files = sorted(RAW.glob("gratings*.npy"))
    if smoke:
        files = [f for f in files if "drifting_GT3" in f.name]
    n_perm = 4 if smoke else 200
    with ProcessPoolExecutor(1 if smoke else 4, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(run_one, [(f, n_perm) for f in files]))
    out = {"registration": "H4-R1c (ex0_REGISTERED.md, d4f222cd)", "n_perm": n_perm, "rows": rows}
    if not smoke:
        out["score"] = score(rows)
        print("SCORE", json.dumps(out["score"], indent=1), flush=True)
    json.dump(out, open(OUT.with_name("h4_r1c_crossfit_smoke.json") if smoke else OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
