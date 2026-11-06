"""H4-R5 (Oct 25; registration H4-R5 in data_canonical/ex0_REGISTERED.md, commit fb8af68d): is the shift an easier detector of
class-dependent fluctuation structure than the covariance tools a referee would name? Imaging, 8 grating recordings, input as
Table 9. Per n per class in {8, 16, 32, 64}: 5 draws of n trials per class and 256 random neurons. Statistics on the same trials
and neurons, each against 50 label permutations refitting everything label-dependent (p = (1 + #null >= obs) / 51):
  shift   mean-free shift, R1c design (the subsample's own 7-dim class-mean span removed; rungs 1,2,3,4,6,8; floor rng 42)
  frob    sum_{c<d} ||W_c - W_d||_F^2 with Tr W_c^2 from independent split halves of each class (unbiased) and Tr(W_c W_d) from
          the class sample covariances (the L2 family of Schott 2007 and Li and Chen 2012)
  sigma   1 - sigma, sigma = mean off-diagonal / mean diagonal of K_cd = Tr(W_c W_d) / (Tr W_c Tr W_d) (diagonal split-half)
  pa_k    1 - mean over class pairs of ||U_c^T U_d||_F^2 / k, U_c the top-k residual subspace of class c (k = 2, 5, 10; capped
          at n - 1)
  qda     QDA accuracy under fold-wise mean-span removal (h4_r1c_crossfit.lda_qda_cf)
Out: data_canonical/h4_r5_power.json (smoke: _smoke.json)"""
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
OUT = DATA / "h4_r5_power.json"
NS = (8, 16, 32, 64)
N_DRAW = 5
N_NEUR = 256
KS = (2, 5, 10)
STATS = ("shift", "frob", "sigma") + tuple(f"pa_{k}" for k in KS) + ("qda",)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cov_stats(X, y):
    """frob, 1 - sigma and principal-angle dissimilarities from class residual covariances (neurons x neurons)."""
    R = [X[y == c] - X[y == c].mean(0) for c in range(8)]
    W = [r.T @ r / (len(r) - 1) for r in R]
    t = np.array([np.trace(w) for w in W])
    T = np.array([[np.sum(W[c] * W[d]) for d in range(8)] for c in range(8)])
    for c in range(8):                                     # unbiased Tr W_c^2 from independent split halves
        a, b = X[y == c][0::2], X[y == c][1::2]
        a, b = a - a.mean(0), b - b.mean(0)
        T[c, c] = np.sum((a.T @ a / (len(a) - 1)) * (b.T @ b / (len(b) - 1)))
    frob = float(sum(T[c, c] + T[d, d] - 2 * T[c, d] for c in range(8) for d in range(c + 1, 8)))
    K = T / np.outer(t, t)
    off = ~np.eye(8, dtype=bool)
    out = {"frob": frob, "sigma": float(1 - K[off].mean() / np.diag(K).mean())}
    for k in KS:
        kk = min(k, min(len(r) for r in R) - 1)
        U = [np.linalg.svd(r, full_matrices=False)[2][:kk].T for r in R]
        ov = [np.sum((U[c].T @ U[d]) ** 2) / kk for c in range(8) for d in range(c + 1, 8)]
        out[f"pa_{k}"] = float(1 - np.mean(ov))
    return out


def all_stats(X, y, ow, R1, R1c, with_qda=True):
    t = {}
    t0 = time.perf_counter()
    G = (X @ X.T).astype(np.float64)
    Q = R1.mean_subspace(X, y)
    P = (X @ Q.astype(np.float32)).astype(np.float64)
    s = {"shift": ow.ladder(G - P @ P.T, y, R1.rung_sets(y), np.random.default_rng(42))["delta"]}
    t["shift"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    s.update(cov_stats(X.astype(np.float64), y))
    t["cov"] = time.perf_counter() - t0
    if with_qda:
        t0 = time.perf_counter()
        try:
            s["qda"] = R1c.lda_qda_cf(X, y, np.arange(X.shape[1]), 0)[1]
        except Exception:                                  # too few trials for the inner CV: recorded as missing
            s["qda"] = float("nan")
        t["qda"] = time.perf_counter() - t0
    return s, t


def run_one(args):
    fp, n_perm = args
    warnings.filterwarnings("ignore")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    R1 = _load("h4r1", "h4_r1_meanfree.py")
    R1c = _load("h4r1c", "h4_r1c_crossfit.py")
    dat = np.load(fp, allow_pickle=True).item()
    Xf = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    Xf /= Xf.std() + 1e-9
    yf = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    del dat
    rows = []
    for n in NS:
        for d in range(N_DRAW):
            t0 = time.time()
            rng = np.random.default_rng(10000 * n + d)
            tr = np.concatenate([rng.choice(np.where(yf == c)[0], n, replace=False) for c in range(8)])
            cols = rng.choice(Xf.shape[1], N_NEUR, replace=False)
            X = np.ascontiguousarray(Xf[np.ix_(tr, cols)])
            y = yf[tr]
            obs, tim = all_stats(X, y, ow, R1, R1c)
            prng = np.random.default_rng(20000 * n + d)
            null = {k: [] for k in STATS}
            for _ in range(n_perm):
                sp, _ = all_stats(X, y[prng.permutation(len(y))], ow, R1, R1c)
                for k in STATS:
                    null[k].append(sp[k])
            pv = {}
            for k in STATS:
                a = np.array(null[k])
                pv[k] = float("nan") if not np.isfinite(obs[k]) else float((1 + np.nansum(a >= obs[k])) / (np.isfinite(a).sum() + 1))
            rows.append({"name": fp.stem, "n": n, "draw": d, "obs": obs, "p": pv, "seconds": tim,
                         "null_mean": {k: float(np.nanmean(null[k])) for k in STATS}})
            print(f"{fp.stem[:28]} n {n:2d} d {d}: " + " ".join(f"{k} p {pv[k]:.3f}" for k in STATS) +
                  f" | s {tim['shift']:.2f} cov {tim['cov']:.2f} qda {tim.get('qda', 0):.2f} ({time.time() - t0:.0f}s)", flush=True)
    return rows


def score(rows):
    v = {}
    for n in NS:
        r = [x for x in rows if x["n"] == n]
        v[str(n)] = {k: {"detected": int(sum(x["p"][k] <= 0.05 for x in r if np.isfinite(x["p"][k]))),
                         "valid": int(sum(np.isfinite(x["p"][k]) for x in r))} for k in STATS}
    sec = {k: float(np.median([x["seconds"][k] for x in rows])) for k in ("shift", "cov", "qda")}
    v["median_seconds"] = sec
    return v


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    files = sorted(RAW.glob("gratings*.npy"))
    global NS, N_DRAW
    if smoke:
        files = [f for f in files if "drifting_GT3" in f.name]
        NS, N_DRAW = (8, 32), 1
    args = [(f, 3 if smoke else 50) for f in files]
    with ProcessPoolExecutor(1 if smoke else 4, mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(run_one, args) for x in part]
    out = {"registration": "H4-R5 (ex0_REGISTERED.md, fb8af68d)", "rows": rows}
    if not smoke:
        out["score"] = score(rows)
        print("SCORE", json.dumps(out["score"], indent=1), flush=True)
    json.dump(out, open(OUT.with_name("h4_r5_power_smoke.json") if smoke else OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
