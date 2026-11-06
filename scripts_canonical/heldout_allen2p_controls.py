"""Runs on data the paper never saw: Section 5's two controls and the spread over class orders on Allen two-photon experiments.

For each experiment (arrays written by heldout_allen2p_extract.py), rung.zoom with 50 random class orders and no permutations on
  data      the release's per-presentation responses;
  shape     rung.bias_null(mode='shape'): class means and sizes kept, one covariance shape for every class;
  meanfree  every class's residuals around the grand mean: class means made equal, each class's fluctuations kept;
and each class's own log PR. Records the declared shift, the order-averaged shift and its SD over orders per arm.
The spread over orders follows how much the classes' own fluctuation dimensions differ; one shape for every class removes it.

Usage: python scripts_canonical/heldout_allen2p_controls.py raw/allen2p [workers]
Out: data_canonical/heldout_allen2p_controls.json
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data_canonical", "heldout_allen2p_controls.json")


def meanfree(X, y):
    Y = X.copy(); g = X.mean(0)
    for c in np.unique(y):
        Y[y == c] = X[y == c] - X[y == c].mean(0) + g
    return Y


def one(args):
    arr_dir, eid = args
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    sys.path.insert(0, os.path.join(HERE, ".."))
    import rung
    X = np.load(os.path.join(arr_dir, eid, "X.npy")).astype(np.float64)
    y = np.load(os.path.join(arr_dir, eid, "labels.npy"))
    lp = [float(np.log(v)) for v in rung.class_pr(X, y).values()]
    out = dict(id=eid, class_logpr=lp, class_logpr_sd=float(np.std(lp)))
    Ys, ys = rung.bias_null(X, y, mode="shape", seed=0)
    for tag, Z, yy in (("data", X, y), ("shape", Ys, ys), ("meanfree", meanfree(X, y), y)):
        r = rung.zoom(Z, yy, n_perm=0, k_orders=50, seed=0)
        out[tag] = dict(delta=r["delta"], delta_orderavg=r["delta_orderavg"], delta_orderavg_sd=r["delta_orderavg_sd"])
    print(eid, " ".join(f"{t} {out[t]['delta_orderavg']:+.3f} (sd {out[t]['delta_orderavg_sd']:.3f})" for t in ("data", "shape", "meanfree")),
          flush=True)
    return out


def main():
    arr_dir = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ids = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(arr_dir, "*", "X.npy")))
    import multiprocessing as mp
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(one, [(arr_dir, i) for i in ids]))
    sd = {t: np.array([r[t]["delta_orderavg_sd"] for r in rows]) for t in ("data", "shape", "meanfree")}
    lpsd = np.array([r["class_logpr_sd"] for r in rows])
    rk = lambda v: np.argsort(np.argsort(v))
    summ = dict(n=len(rows), median_sd_data=float(np.median(sd["data"])),
                spearman_sd_class_logpr_sd=float(np.corrcoef(rk(sd["data"]), rk(lpsd))[0, 1]))
    for t in ("shape", "meanfree"):
        summ[t] = dict(median_sd=float(np.median(sd[t])), halved=int((sd[t] <= 0.5 * sd["data"]).sum()),
                       median_delta_orderavg=float(np.median([r[t]["delta_orderavg"] for r in rows])))
    print(json.dumps(summ))
    json.dump(dict(summary=summ, rows=rows), open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
