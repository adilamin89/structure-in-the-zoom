"""The mean-free shift with
label-independent halves and time-aware nulls, and the physics of what it reads.

Imaging, the eight grating recordings, input as Table 9; presentations in time order (stimtimes asserted increasing).
Splits fixed without labels: S1 = odd / even presentation index, S2 = first / second half in time. For every measured half:
  R11  the mean-free shift (R1c design: the half's own 7-dim class-mean span removed) against 200 unrestricted and 200
       epoch-restricted permutations (labels permuted inside contiguous 48-presentation epochs, as time_blocked_null); each draw
       removes its own permuted span. QDA under fold-wise removal (meanfree_crossfit.lda_qda_cf, neuron draw 0) against 50
       permutations of the whole decoding procedure (fixed neurons).
  R4   (S1 halves) gain test: variance of residuals along each unit class-mean direction (means from the OTHER half) for that
       class's trials over other classes' trials; retention = mean-free within part / raw within part; mean-field sigma law
       on the projected residuals (split halves alternating within class inside the half) against the mean-free shift.
Out: data_canonical/meanfree_shift_exact.json (smoke: _smoke.json)"""
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
OUT = DATA / "meanfree_shift_exact.json"
EPOCH = 48


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def perm_within(lab, blocks, rng):
    out = lab.copy()
    for b in np.unique(blocks):
        i = np.where(blocks == b)[0]
        out[i] = lab[i][rng.permutation(len(i))]
    return out


def measured_half(X, y, ep, sel, ow, R1, R1c, n_perm, n_perm_qda, seed, other=None, E=None, om=None):
    t0 = time.time()
    XB, yB, eB = X[sel], y[sel], ep[sel]
    GB = (XB @ XB.T).astype(np.float64)

    def proj_gram(lab):
        Q = R1.mean_subspace(XB, lab)
        P = (XB @ Q.astype(np.float32)).astype(np.float64)
        return GB - P @ P.T, Q

    Gp, Q = proj_gram(yB)
    obs = ow.ladder(Gp, yB, R1.rung_sets(yB), np.random.default_rng(42))
    raw = ow.ladder(GB, yB, R1.rung_sets(yB), np.random.default_rng(42))
    out = {"n": int(len(yB)), "class_counts": np.bincount(yB, minlength=8).tolist(), "meanfree": obs, "raw": raw}
    for name, blocks in (("unrestricted", np.zeros(len(yB), int)), ("epoch", eB)):
        rng = np.random.default_rng(seed + (0 if name == "unrestricted" else 1))
        nd = []
        for _ in range(n_perm):
            yp = perm_within(yB, blocks, rng)
            nd.append(ow.ladder(proj_gram(yp)[0], yp, R1.rung_sets(yp), np.random.default_rng(42))["delta"])
        nd = np.array(nd)
        out[f"null_{name}"] = {"delta": nd.tolist(), "mean": float(nd.mean()), "sd": float(nd.std()), "max": float(nd.max()),
                               "p_one": float((1 + (nd >= obs["delta"]).sum()) / (n_perm + 1))}
    print(f"    shift {obs['delta']:+.3f} null unr {out['null_unrestricted']['mean']:+.4f}/{out['null_unrestricted']['max']:+.4f} "
          f"epoch {out['null_epoch']['mean']:+.4f}/{out['null_epoch']['max']:+.4f} ({time.time() - t0:.0f}s)", flush=True)
    cols = np.random.default_rng(0).choice(X.shape[1], 256, replace=False)
    lda, qda = R1c.lda_qda_cf(XB, yB, cols, 0)
    rng = np.random.default_rng(seed + 2)
    nq = np.array([R1c.lda_qda_cf(XB, yB[rng.permutation(len(yB))], cols, 0)[1] for _ in range(n_perm_qda)])
    out["qda"] = {"lda": lda, "qda": qda, "null": nq.tolist(), "null_mean": float(nq.mean()), "null_max": float(nq.max()),
                  "p_one": float((1 + (nq >= qda).sum()) / (n_perm_qda + 1)),
                  "prior": float(np.bincount(yB).max() / len(yB))}
    print(f"    QDA {qda:.3f} null {nq.mean():.3f}/{nq.max():.3f} ({time.time() - t0:.0f}s)", flush=True)
    if other is not None:
        Xo, yo = other
        mo = np.stack([Xo[yo == c].mean(0) for c in range(8)]).astype(np.float64)
        mo -= mo.mean(0, keepdims=True)
        U = mo / np.linalg.norm(mo, axis=1, keepdims=True)
        R = XB - np.stack([XB[yB == c].mean(0) for c in range(8)])[yB]
        S = (R.astype(np.float64) @ U.T) ** 2                       # trials x classes: squared projection on each mean direction
        ratios = []
        for c in range(8):
            own = S[yB == c, c].mean()
            oth = S[yB != c, c].mean()
            ratios.append(float(own / oth))
        out["gain_ratio_per_class"] = ratios
        out["gain_ratio"] = float(np.mean(ratios))
        out["retention"] = float(obs["within"] / raw["within"])
        Z = XB - (XB @ Q.astype(np.float32)) @ Q.T.astype(np.float32)
        A, B = [], []
        for c in range(8):
            Zc = Z[yB == c]
            a, b = Zc[0::2], Zc[1::2]
            A.append((a - a.mean(0)).astype(np.float32))
            B.append((b - b.mean(0)).astype(np.float32))
        sig = om.sigma(om.split_half_K(A, B))
        out["sigma_meanfree"] = sig
        out["sigma_law_pred"] = E.mean_field(sig)
        print(f"    gain ratio {out['gain_ratio']:.3f} retention {out['retention']:.3f} sigma {sig:.3f} pred {out['sigma_law_pred']:+.3f} "
              f"vs mean-free shift {obs['delta']:+.3f} ({time.time() - t0:.0f}s)", flush=True)
    return out


def run_one(args):
    fp, n_perm, n_perm_qda = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    R1 = _load("h4r1", "meanfree_shift.py")
    R1c = _load("h4r1c", "meanfree_crossfit.py")
    om = _load("om", "overlap_identity_checks.py")
    E = _load("E", "allen_engine.py")
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    X /= X.std() + 1e-9
    y = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    t = np.asarray(dat["stimtimes"], float).ravel()
    del dat
    assert len(t) == len(y) and np.all(np.diff(t) > 0), fp.stem
    n = len(y)
    ep = np.arange(n) // EPOCH
    splits = {"S1": np.arange(n) % 2, "S2": (np.arange(n) >= n // 2).astype(int)}
    row = {"name": fp.stem, "n": int(n)}
    for s, h in splits.items():
        for hv in (0, 1):
            print(f"{fp.stem[:32]} {s} half{hv}", flush=True)
            other = (X[h == 1 - hv], y[h == 1 - hv]) if s == "S1" else None
            row[f"{s}_half{hv}"] = measured_half(X, y, ep, np.where(h == hv)[0], ow, R1, R1c, n_perm, n_perm_qda,
                                                 7000 + 10 * hv + (0 if s == "S1" else 100), other, E, om)
    return row


def score(rows):
    cells = [(r["name"], k, r[k]) for r in rows for k in r if k.startswith("S")]
    pu = [c[2]["null_unrestricted"]["p_one"] for c in cells]
    pe = [c[2]["null_epoch"]["p_one"] for c in cells]
    pq = [c[2]["qda"]["p_one"] for c in cells]
    s1 = [c[2] for c in cells if c[1].startswith("S1")]
    from scipy import stats
    v = {"n_cells": len(cells), "R11a_both_nulls_p_le_0.05": sum(a <= 0.05 and b <= 0.05 for a, b in zip(pu, pe)),
         "conjunction_p_unrestricted": max(pu), "conjunction_p_epoch": max(pe),
         "meanfree_shift_range": [min(c[2]["meanfree"]["delta"] for c in cells), max(c[2]["meanfree"]["delta"] for c in cells)],
         "epoch_minus_unrestricted_null_mean_range": [min(c[2]["null_epoch"]["mean"] - c[2]["null_unrestricted"]["mean"] for c in cells),
                                                      max(c[2]["null_epoch"]["mean"] - c[2]["null_unrestricted"]["mean"] for c in cells)],
         "R11b_qda_p_le_0.05": sum(p <= 0.05 for p in pq), "conjunction_p_qda": max(pq),
         "qda_range": [min(c[2]["qda"]["qda"] for c in cells), max(c[2]["qda"]["qda"] for c in cells)],
         "qda_null_max_range": [min(c[2]["qda"]["null_max"] for c in cells), max(c[2]["qda"]["null_max"] for c in cells)],
         "S1_vs_S2_shift": {k: [c[2]["meanfree"]["delta"] for c in cells if c[1].startswith(k)] for k in ("S1", "S2")},
         "R4_gain_ratio_gt1": sum(x["gain_ratio"] > 1 for x in s1), "R4_gain_ratio_range": [min(x["gain_ratio"] for x in s1), max(x["gain_ratio"] for x in s1)],
         "R4_retention_range": [min(x["retention"] for x in s1), max(x["retention"] for x in s1)],
         "R4_spearman_gain_vs_loss": float(stats.spearmanr([x["gain_ratio"] for x in s1], [1 - x["retention"] for x in s1])[0]),
         "R4_sigma_law_pearson": float(np.corrcoef([x["sigma_law_pred"] for x in s1], [x["meanfree"]["delta"] for x in s1])[0, 1]),
         "R4_sigma_law_median_abs_err": float(np.median([abs(x["sigma_law_pred"] - x["meanfree"]["delta"]) for x in s1]))}
    v["R11_verdict"] = "PASS" if v["R11a_both_nulls_p_le_0.05"] == len(cells) and v["R11b_qda_p_le_0.05"] == len(cells) else "MISS"
    return v


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    files = sorted(RAW.glob("gratings*.npy"))
    if smoke:
        files = [f for f in files if "drifting_GT3" in f.name]
    args = [(f, 3, 2) if smoke else (f, 200, 50) for f in files]
    with ProcessPoolExecutor(1 if smoke else 4, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(run_one, args))
    out = {"rows": rows}
    if not smoke:
        out["score"] = score(rows)
        print("SCORE", json.dumps(out["score"], indent=1), flush=True)
    json.dump(out, open(OUT.with_name("meanfree_shift_exact_smoke.json") if smoke else OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
