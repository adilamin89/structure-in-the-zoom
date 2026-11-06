"""A covariance-specific null and a
fine-tuning control for the mean-free shift (Codex audit item 18). Imaging, 8 recordings, S1 halves (odd / even presentation).
  (a) mean-preserving residual null: residual vectors pooled across classes (withinclass_shuffle_identification.pool_residuals: class means
      kept, every class given the pooled residual distribution), then the half's own class-mean span removed and the mean-free
      shift measured (R1c / R11 design); 100 draws.
  (b) within-bin tuning removed first (cr_order_and_withinbin_controls.remove_withinbin, 7.5-degree sub-bins), then the
      mean-free shift against 100 label permutations (each removing its own permuted span).
Out: data_canonical/covariance_null.json"""
import importlib.util
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
DATA = HERE.parent / "data_canonical"
N_DRAW = 100


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def meanfree_shift(X, y, ow, R1):
    Q = R1.mean_subspace(X, y)
    P = (X @ Q.astype(np.float32)).astype(np.float64)
    G = (X @ X.T).astype(np.float64) - P @ P.T
    return ow.ladder(G, y, R1.rung_sets(y), np.random.default_rng(42))


def run_one(fp):
    warnings.filterwarnings("ignore")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    R1 = _load("h4r1", "meanfree_shift.py")
    ci = _load("ci", "withinclass_shuffle_identification.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel()
    y = ow.bins(phi, 8)
    del dat
    h = np.arange(len(y)) % 2
    row = {"name": fp.stem}
    for hv in (0, 1):
        sel = np.where(h == hv)[0]
        XB, yB, pB = X[sel], y[sel], phi[sel]
        obs = meanfree_shift(XB, yB, ow, R1)
        pooled = np.array([meanfree_shift(ci.pool_residuals(XB, yB, np.random.default_rng(100 + i)).astype(np.float32), yB, ow, R1)["delta"]
                           for i in range(N_DRAW)])
        Yw = ow.remove_withinbin(XB, yB, pB, 6, 8).astype(np.float32)
        obs_w = meanfree_shift(Yw, yB, ow, R1)
        rng = np.random.default_rng(300 + hv)
        perm_w = np.array([meanfree_shift(Yw, yp, ow, R1)["delta"] for yp in (yB[rng.permutation(len(yB))] for _ in range(N_DRAW))])
        row[f"half{hv}"] = {"meanfree": obs, "pooled_null": {"delta": pooled.tolist(), "mean": float(pooled.mean()), "max": float(pooled.max()),
                                                           "p_one": float((1 + (pooled >= obs["delta"]).sum()) / (N_DRAW + 1))},
                            "withinbin_removed": obs_w, "withinbin_perm": {"mean": float(perm_w.mean()), "max": float(perm_w.max()),
                                                                           "p_one": float((1 + (perm_w >= obs_w["delta"]).sum()) / (N_DRAW + 1))},
                            "withinbin_ratio": float(obs_w["delta"] / obs["delta"])}
        r = row[f"half{hv}"]
        print(f"{fp.stem[:32]} half{hv}: mean-free {obs['delta']:+.3f} pooled null {r['pooled_null']['mean']:+.4f}/{r['pooled_null']['max']:+.4f} "
              f"p {r['pooled_null']['p_one']:.3f} | within-bin removed {obs_w['delta']:+.3f} (ratio {r['withinbin_ratio']:.2f}) perm max "
              f"{r['withinbin_perm']['max']:+.4f} p {r['withinbin_perm']['p_one']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return row


def main():
    files = sorted(RAW.glob("gratings*.npy"))
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(run_one, files))
    cells = [r[h] for r in rows for h in ("half0", "half1")]
    sc = {"n_cells": len(cells), "R12a_pooled_p_le_0.05": sum(c["pooled_null"]["p_one"] <= 0.05 for c in cells),
          "R12a_conjunction_p": max(c["pooled_null"]["p_one"] for c in cells),
          "pooled_null_mean_range": [min(c["pooled_null"]["mean"] for c in cells), max(c["pooled_null"]["mean"] for c in cells)],
          "R12b_withinbin_positive_and_p": sum(c["withinbin_removed"]["delta"] > 0 and c["withinbin_perm"]["p_one"] <= 0.05 for c in cells),
          "R12b_conjunction_p": max(c["withinbin_perm"]["p_one"] for c in cells),
          "withinbin_ratio_range": [min(c["withinbin_ratio"] for c in cells), max(c["withinbin_ratio"] for c in cells)],
          "withinbin_falls_less_than_half": sum(c["withinbin_ratio"] > 0.5 for c in cells)}
    print("SCORE", json.dumps(sc, indent=1), flush=True)
    json.dump({"rows": rows, "score": sc}, open(DATA / "covariance_null.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
