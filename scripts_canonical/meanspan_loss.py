"""Why removing the class-mean span
costs within-class structure (a gain prediction the data refuted). Imaging, the eight grating recordings, input as meanfree_shift_exact (globally
standardized, 8 direction bins), S1 halves (odd / even presentation index). In each half: the within-class part of the ladder after
removing a 7-dim span, as a fraction of the raw within part (retention), for the half's own class-mean span (real), 20 label-permuted
class-mean spans (perm), 20 sets of 7 directions drawn from W as R^T z (W), and the other half's class-mean span (other); with
beta = Tr B / Tr W and the alignment a of the half (sign_law terms, as allen_modes_removed).
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler meanspan_loss.py [n_workers] [smoke]   Out: data_canonical/meanspan_loss.json"""
import importlib.util
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
DATA = HERE.parent / "data_canonical"
N_DRAW = 20


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def half(X, y, sel, other_sel, ow, R1, E, n_draw):
    XB, yB = X[sel], y[sel]
    GB = (XB @ XB.T).astype(np.float64)
    sets = R1.rung_sets(yB)
    raw = ow.ladder(GB, yB, sets, np.random.default_rng(42))

    def retention(Q):
        P = (XB @ Q.astype(np.float32)).astype(np.float64)
        lad = ow.ladder(GB - P @ P.T, yB, sets, np.random.default_rng(42))
        return float(lad["within"] / raw["within"]), lad

    r_real, lad_real = retention(R1.mean_subspace(XB, yB))
    r_other, _ = retention(R1.mean_subspace(X[other_sel], y[other_sel]))
    r_perm = [retention(R1.mean_subspace(XB, yB[np.random.default_rng(11 + k).permutation(len(yB))]))[0] for k in range(n_draw)]
    M = np.stack([XB[yB == c].mean(0) for c in range(8)])
    R = (XB - M[yB]).astype(np.float64)
    r_w = []
    for k in range(n_draw):
        V = R.T @ np.random.default_rng(100 + k).standard_normal((len(yB), 7))   # 7 draws with covariance n W
        r_w.append(retention(np.linalg.qr(V)[0])[0])
    sg = E.sign_law(XB.astype(np.float64), yB)
    a = sg["x_share"] * sg["trB_trW"] / (2 * (1 - sg["x_share"]) * sg["PR_B"]) * sg["PR_W"]
    return {"n": int(len(yB)), "n_per_class": np.bincount(yB, minlength=8).tolist(), "raw": raw, "real_ladder": lad_real,
            "retention_real": r_real, "retention_other": r_other, "retention_perm": r_perm, "retention_W": r_w,
            "loss_real": 1 - r_real, "loss_other": 1 - r_other, "loss_perm": float(1 - np.mean(r_perm)), "loss_W": float(1 - np.mean(r_w)),
            "loss_perm_sd": float(np.std(r_perm)), "loss_W_sd": float(np.std(r_w)),
            "beta": sg["trB_trW"], "a": float(a), "PR_W": sg["PR_W"], "PR_B": sg["PR_B"]}


def run_one(args):
    fp, n_draw = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    R1 = _load("h4r1", "meanfree_shift.py")
    E = _load("E", "allen_engine.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    X /= X.std() + 1e-9
    y = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    t = np.asarray(dat["stimtimes"], float).ravel()
    del dat
    assert len(t) == len(y) and np.all(np.diff(t) > 0), fp.stem
    h = np.arange(len(y)) % 2
    row = {"name": fp.stem, "n": int(len(y)), "N": int(X.shape[1])}
    for hv in (0, 1):
        row[f"half{hv}"] = half(X, y, np.where(h == hv)[0], np.where(h == 1 - hv)[0], ow, R1, E, n_draw)
        q = row[f"half{hv}"]
        print(f"{fp.stem[9:34]} half{hv}: loss real {q['loss_real']:.3f} perm {q['loss_perm']:.3f}+-{q['loss_perm_sd']:.3f} "
              f"W {q['loss_W']:.3f}+-{q['loss_W_sd']:.3f} other {q['loss_other']:.3f} | beta {q['beta']:.3f} a {q['a']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return row


def score(rows):
    cells = [r[f"half{h}"] | {"name": r["name"]} for r in rows for h in (0, 1)]
    g = lambda k: np.array([c[k] for c in cells])  # noqa: E731
    sp = lambda x, y: float(stats.spearmanr(x, y)[0])  # noqa: E731
    out = {"n_cells": len(cells),
           "P1_spearman_loss_real_beta": sp(g("loss_real"), g("beta")),
           "P2_perm_vs_W_within_0.10": int(np.sum(np.abs(g("loss_perm") - g("loss_W")) <= 0.10)),
           "P3_spearman_excess_beta": sp(g("loss_real") - g("loss_perm"), g("beta")),
           "spearman_loss_real_a": sp(g("loss_real"), g("a")),
           "P4_beta_beats_a": bool(abs(sp(g("loss_real"), g("beta"))) > abs(sp(g("loss_real"), g("a")))),
           "spearman_loss_real_loss_perm": sp(g("loss_real"), g("loss_perm")),
           "spearman_loss_real_loss_W": sp(g("loss_real"), g("loss_W")),
           "spearman_loss_other_beta": sp(g("loss_other"), g("beta")),
           "loss_real_range": [float(g("loss_real").min()), float(g("loss_real").max())],
           "loss_perm_range": [float(g("loss_perm").min()), float(g("loss_perm").max())],
           "loss_W_range": [float(g("loss_W").min()), float(g("loss_W").max())],
           "loss_other_range": [float(g("loss_other").min()), float(g("loss_other").max())],
           "beta_range": [float(g("beta").min()), float(g("beta").max())]}
    out["P1"] = out["P1_spearman_loss_real_beta"] <= -0.5
    out["P2"] = out["P2_perm_vs_W_within_0.10"] >= 12
    out["P3"] = out["P3_spearman_excess_beta"] <= -0.5
    out["P4"] = out["P4_beta_beats_a"]
    out["cells"] = [{k: c[k] for k in ("name", "loss_real", "loss_perm", "loss_W", "loss_other", "beta", "a")} for c in cells]
    out["pipeline_check_vs_h4_r4"] = pipeline_check(rows)
    return out


def pipeline_check(rows):
    """The real-span retention is meanfree_shift_exact's (same span, ladder and rng): max abs difference over the cells present in both."""
    r4 = {r["name"]: r for r in json.load(open(DATA / "meanfree_shift_exact.json"))["rows"]}
    d = [abs(r[f"half{h}"]["retention_real"] - r4[r["name"]][f"S1_half{h}"]["retention"]) for r in rows if r["name"] in r4 for h in (0, 1)]
    return {"n": len(d), "max_abs_diff": float(max(d)) if d else None}


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4
    smoke = "smoke" in sys.argv
    files = sorted(RAW.glob("gratings*.npy"))
    if smoke:
        files, n_draw = files[:1], 2
    else:
        n_draw = N_DRAW
    with ProcessPoolExecutor(nw, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(run_one, [(f, n_draw) for f in files]))
    out = {"n_draw": n_draw, "rows": rows, "pipeline_check_vs_h4_r4": pipeline_check(rows)}
    if not smoke:
        out["score"] = score(rows)
        print("SCORE", json.dumps({k: v for k, v in out["score"].items() if k != "cells"}, indent=1), flush=True)
    json.dump(out, open(DATA / ("meanspan_loss_smoke.json" if smoke else "meanspan_loss.json"), "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
