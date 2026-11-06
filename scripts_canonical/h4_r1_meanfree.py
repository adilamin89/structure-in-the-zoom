"""H4-R1 (Oct 24; registration H4-R1 in data_canonical/ex0_REGISTERED.md, commit 8cba0800): what the direction shift reads that
class-mean measures miss (reviewers tYDC W4, 1GCj W3). The cortical form of App I's projection test, held out.

Input as Table 9 (deconvolved responses / global SD, eight 45-degree classes, run72.bins); halves alternate within each class
(h3_v1_decode). Per recording and both ways (estimate on half A, measure on half B; and swapped):
  Q_A   the 7 right singular vectors of half A's centred class means (all neurons); half B projected off Q_A.
  (i)   App K's linear decoder (h3_v1_decode.decode_acc) on 256 / 1024 random neurons of half B (5 draws), raw and projected
        (same neurons); (ii) beta = Tr B / Tr W of half B, raw and projected; (iii) the shift and its split on half B
        (cr_order_and_withinbin_controls.ladder: rungs 1,2,3,4,6,8 classes from origin 0, ten-draw floor rng 42), raw and
        projected; (iv) the projected shift against 200 label permutations (halves fixed; labels permuted inside each half at
        fixed class sizes; each draw projects half B off ITS OWN permuted half-A mean subspace and reruns the ladder with the
        permuted labels); (v) the projected shift at all 8 origins; LDA vs QDA (h3_v1_decode.lda_qda) raw and projected.
Pipeline check: the full-data ladder reproduces Table 9's delta (cr_withinclass_controls.json orig) to 1e-4 (asserted).
Out: data_canonical/h4_r1_meanfree.json (smoke: h4_r1_meanfree_smoke.json)"""
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
OUT = DATA / "h4_r1_meanfree.json"
RUNGS = (1, 2, 3, 4, 6, 8)
NS = (256, 1024)
N_DRAW = 5


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mean_subspace(X, y, nc=8):
    """Orthonormal basis (neurons x nc-1) of the span of the centred class means (h3_probes.mean_subspace)."""
    M = np.stack([X[y == c].mean(0) for c in range(nc)]).astype(np.float64)
    M -= M.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[: nc - 1].T


def rung_sets(y, origin=0):
    m = [np.where(y == b)[0] for b in range(8)]
    return [np.concatenate([m[(origin + j) % 8] for j in range(c)]) for c in RUNGS]


def beta(X, y):
    mu = X.mean(0)
    trb = trw = 0.0
    for c in range(8):
        Xc = X[y == c].astype(np.float64)
        mc = Xc.mean(0)
        trb += len(Xc) * float(((mc - mu) ** 2).sum())
        trw += float(((Xc - mc) ** 2).sum())
    return trb / trw


def one_way(X, y, est, meas, ow, h3, rng_perm, n_perm):
    """Estimate the subspace on half `est`, measure on half `meas` (index arrays into X / y)."""
    t0 = time.time()
    XA, yA = X[est], y[est]
    XB, yB = X[meas], y[meas]
    nB = len(yB)
    GB = (XB @ XB.T).astype(np.float64)
    Q = mean_subspace(XA, yA)
    PB = (XB @ Q.astype(np.float32)).astype(np.float64)
    GBp = GB - PB @ PB.T
    XBp = XB - (PB @ Q.T).astype(np.float32)
    out = {"n_meas": int(nB), "chance_band": float(1 / 8 + 2 * np.sqrt((1 / 8) * (7 / 8) / nB))}
    out["beta_raw"], out["beta_proj"] = beta(XB, yB), beta(XBp, yB)
    out["raw"] = ow.ladder(GB, yB, rung_sets(yB), np.random.default_rng(42))
    out["proj"] = ow.ladder(GBp, yB, rung_sets(yB), np.random.default_rng(42))
    out["proj_origins"] = [ow.ladder(GBp, yB, rung_sets(yB, o), np.random.default_rng(42))["delta"] for o in range(8)]
    # decoders, same neurons raw and projected
    rng = np.random.default_rng(0)
    dec = {}
    for N in NS:
        acc_r, acc_p = [], []
        for d in range(N_DRAW):
            cols = rng.choice(X.shape[1], N, replace=False)
            acc_r.append(h3.decode_acc(XB[:, cols], yB, seed=d))
            acc_p.append(h3.decode_acc(XBp[:, cols], yB, seed=d))
        dec[str(N)] = {"raw": float(np.mean(acc_r)), "proj": float(np.mean(acc_p)), "raw_draws": acc_r, "proj_draws": acc_p}
    out["decode"] = dec
    lq_r, lq_p = [], []
    for d in range(N_DRAW):
        cols = rng.choice(X.shape[1], 256, replace=False)
        lq_r.append(h3.lda_qda(XB[:, cols], yB, d))
        lq_p.append(h3.lda_qda(XBp[:, cols], yB, d))
    out["lda_qda"] = {"raw_lda": float(np.mean([a[0] for a in lq_r])), "raw_qda": float(np.mean([a[1] for a in lq_r])),
                      "proj_lda": float(np.mean([a[0] for a in lq_p])), "proj_qda": float(np.mean([a[1] for a in lq_p])),
                      "proj_draws": lq_p}
    # null: each permutation removes its own permuted half-A subspace
    nd, nw = [], []
    for i in range(n_perm):
        yAp = yA[rng_perm.permutation(len(yA))]
        yBp = yB[rng_perm.permutation(nB)]
        Qp = mean_subspace(XA, yAp)
        Pp = (XB @ Qp.astype(np.float32)).astype(np.float64)
        r = ow.ladder(GB - Pp @ Pp.T, yBp, rung_sets(yBp), np.random.default_rng(42))
        nd.append(r["delta"])
        nw.append(r["within"])
        if (i + 1) % 50 == 0:
            print(f"    perm {i + 1}/{n_perm} null mean {np.mean(nd):+.4f} ({time.time() - t0:.0f}s)", flush=True)
    nd = np.array(nd)
    obs = out["proj"]["delta"]
    out["null"] = {"delta": nd.tolist(), "within": nw, "mean": float(nd.mean()) if n_perm else None,
                   "sd": float(nd.std()) if n_perm else None, "max": float(nd.max()) if n_perm else None,
                   "p_one": float((1 + (nd >= obs).sum()) / (n_perm + 1)) if n_perm else None,
                   "z": float((obs - nd.mean()) / nd.std()) if n_perm else None}
    return out


def run_one(args):
    fp, n_perm = args
    warnings.filterwarnings("ignore")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    h3 = _load("h3", "h3_v1_decode.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    X /= X.std() + 1e-9
    y = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    del dat
    G = (X @ X.T).astype(np.float64)
    full = ow.ladder(G, y, rung_sets(y), np.random.default_rng(42))
    del G
    t9 = {r["name"]: r["orig"] for r in json.load(open(DATA / "cr_withinclass_controls.json"))["rows"]}[fp.stem]
    assert abs(full["delta"] - t9["delta"]) < 1e-4, (fp.stem, full["delta"], t9["delta"])
    print(f"{fp.stem[:32]} CHECK full delta {full['delta']:.5f} = Table 9 {t9['delta']:.5f} ({time.time() - t0:.0f}s)", flush=True)
    half = np.zeros(len(y), int)
    for k in range(8):
        idx = np.where(y == k)[0]
        half[idx[1::2]] = 1
    row = {"name": fp.stem, "n_trials": int(len(y)), "n_neurons": int(X.shape[1]), "full": full}
    for w, (e, m) in enumerate(((0, 1), (1, 0))):
        r = one_way(X, y, np.where(half == e)[0], np.where(half == m)[0], ow, h3, np.random.default_rng(1000 + w), n_perm)
        row[f"est{e}_meas{m}"] = r
        print(f"{fp.stem[:32]} est{e}->meas{m}: beta {r['beta_raw']:.4f}->{r['beta_proj']:.5f} | dec256 {r['decode']['256']['raw']:.3f}"
              f"->{r['decode']['256']['proj']:.3f} (band {r['chance_band']:.3f}) dec1024 {r['decode']['1024']['raw']:.3f}->"
              f"{r['decode']['1024']['proj']:.3f} | shift raw {r['raw']['delta']:+.3f} (w {r['raw']['within']:+.3f}) proj "
              f"{r['proj']['delta']:+.3f} (w {r['proj']['within']:+.3f} cm {r['proj']['class_means']:+.3f}) null "
              f"{r['null']['mean']} max {r['null']['max']} p {r['null']['p_one']} | QDA proj {r['lda_qda']['proj_qda']:.3f} LDA "
              f"{r['lda_qda']['proj_lda']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return row


def score(rows):
    cells = [(r["name"], k, r[k]) for r in rows for k in ("est0_meas1", "est1_meas0")]
    a = [c for c in cells if c[2]["decode"]["256"]["proj"] <= c[2]["chance_band"]]
    b = [c for c in cells if c[2]["null"]["p_one"] <= 0.05]
    v = {"n_cells": len(cells),
         "R1a_dec256_at_chance": len(a), "R1a_dec1024_at_chance": sum(c[2]["decode"]["1024"]["proj"] <= c[2]["chance_band"] for c in cells),
         "R1b_proj_shift_p_le_0.05": len(b),
         "dec256_raw_range": [min(c[2]["decode"]["256"]["raw"] for c in cells), max(c[2]["decode"]["256"]["raw"] for c in cells)],
         "dec256_proj_range": [min(c[2]["decode"]["256"]["proj"] for c in cells), max(c[2]["decode"]["256"]["proj"] for c in cells)],
         "dec1024_proj_range": [min(c[2]["decode"]["1024"]["proj"] for c in cells), max(c[2]["decode"]["1024"]["proj"] for c in cells)],
         "chance_band_range": [min(c[2]["chance_band"] for c in cells), max(c[2]["chance_band"] for c in cells)],
         "proj_shift_range": [min(c[2]["proj"]["delta"] for c in cells), max(c[2]["proj"]["delta"] for c in cells)],
         "null_max_range": [min(c[2]["null"]["max"] for c in cells), max(c[2]["null"]["max"] for c in cells)],
         "proj_gt_raw_shift_recordings_both_ways": sum(all(r[k]["proj"]["delta"] > r[k]["raw"]["delta"] for k in ("est0_meas1", "est1_meas0")) for r in rows),
         "proj_shift_positive_all_origins_cells": sum(min(c[2]["proj_origins"]) > 0 for c in cells),
         "beta_proj_over_raw_range": [min(c[2]["beta_proj"] / c[2]["beta_raw"] for c in cells), max(c[2]["beta_proj"] / c[2]["beta_raw"] for c in cells)],
         "qda_proj_range": [min(c[2]["lda_qda"]["proj_qda"] for c in cells), max(c[2]["lda_qda"]["proj_qda"] for c in cells)],
         "lda_proj_range": [min(c[2]["lda_qda"]["proj_lda"] for c in cells), max(c[2]["lda_qda"]["proj_lda"] for c in cells)],
         "qda_proj_above_band": sum(c[2]["lda_qda"]["proj_qda"] > c[2]["chance_band"] for c in cells)}
    v["verdict"] = "PASS" if v["R1a_dec256_at_chance"] == v["n_cells"] and v["R1b_proj_shift_p_le_0.05"] == v["n_cells"] else "MISS"
    return v


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    files = sorted(RAW.glob("gratings*.npy"))
    if smoke:
        files = [f for f in files if "drifting_GT3" in f.name]
    n_perm = 4 if smoke else 200
    with ProcessPoolExecutor(1 if smoke else 4, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(run_one, [(f, n_perm) for f in files]))
    out = {"registration": "H4-R1 (ex0_REGISTERED.md, 8cba0800)", "n_perm": n_perm, "rows": rows}
    if not smoke:
        out["score"] = score(rows)
        print("SCORE", json.dumps(out["score"], indent=1), flush=True)
    json.dump(out, open(OUT.with_name("h4_r1_meanfree_smoke.json") if smoke else OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
