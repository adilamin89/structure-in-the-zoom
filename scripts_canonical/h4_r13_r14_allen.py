"""H4-R13 + H4-R14 (Oct 25; registrations in data_canonical/ex0_REGISTERED.md, commit 16e4797d). Allen, all 167 populations
(E.populations, all units), raw counts, full session, eight direction classes.
R13  a standard covariance tool against its own label-blind floor: principal-angle dissimilarity (k = 5) and Frobenius
     heterogeneity (h4_r5_power.cov_stats), observed and their mean over 50 label permutations; corrected = observed - floor.
R14  means first, then the shift: LDA accuracy (h3_v1_decode.lda_qda); the mean-free shift (own class-mean span removed) as z
     against 100 label permutations; target = QDA under fold-wise mean-span removal (h4_r1c_crossfit.lda_qda_cf) against 50
     permutations of the whole procedure; seconds per population for each.
Out: data_canonical/h4_r13_r14_allen.json"""
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
OUT = HERE.parent / "data_canonical" / "h4_r13_r14_allen.json"
N_FLOOR, N_SHIFT, N_QDA = 50, 100, 50


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _try(f):
    """H5b: a decoder that cannot run at a masked design (a class covariance singular inside a CV fold: 743475441 VISp under B,
    ~31 presentations per class) is recorded as NaN; the caller still draws its permutation, so the random stream is unchanged."""
    try:
        return f()
    except Exception:
        return float("nan")


def run_session(s):
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    E = _load("E", "h1_allen_engine.py")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    R1 = _load("h4r1", "h4_r1_meanfree.py")
    R1c = _load("h4r1c", "h4_r1c_crossfit.py")
    R5 = _load("h4r5", "h4_r5_power.py")
    h3 = _load("h3", "h3_v1_decode.py")
    S = E.load_session(s["session"])
    rows = []
    for area, idx in E.populations(S).items():
        t0 = time.time()
        P = E.pop_view(S, area, idx)   # H5-R1 masking path
        X = np.ascontiguousarray(P["X"].astype(np.float32))
        y = P["lab"]
        rng = np.random.default_rng(int(s["session"]) % 100000)
        cs = R5.cov_stats(X.astype(np.float64), y)
        fl = [R5.cov_stats(X.astype(np.float64), y[rng.permutation(len(y))]) for _ in range(N_FLOOR)]
        r13 = {k: {"obs": cs[k], "floor": float(np.mean([f[k] for f in fl])), "floor_sd": float(np.std([f[k] for f in fl]))}
               for k in ("pa_5", "frob")}
        t1 = time.time()
        obs = R5.all_stats(X, y, ow, R1, R1c, with_qda=False)[0]["shift"]
        null = np.array([R5.all_stats(X, y[rng.permutation(len(y))], ow, R1, R1c, with_qda=False)[0]["shift"] for _ in range(N_SHIFT)])
        t_shift = time.time() - t1
        t2 = time.time()
        cols = np.arange(X.shape[1])
        lda_raw = _try(lambda: h3.lda_qda(X, y, 0)[0])
        q_obs = _try(lambda: R1c.lda_qda_cf(X, y, cols, 0)[1])
        q_null = np.array([_try(lambda yp=y[rng.permutation(len(y))]: R1c.lda_qda_cf(X, yp, cols, 0)[1]) for _ in range(N_QDA)])
        t_qda = time.time() - t2
        rows.append({"session": s["session"], "area": str(area), "N": int(X.shape[1]), "r13": r13,
                     "lda_raw": float(lda_raw), "shift_meanfree": float(obs), "shift_null_mean": float(null.mean()),
                     "shift_z": float((obs - null.mean()) / null.std()), "shift_p": float((1 + (null >= obs).sum()) / (N_SHIFT + 1)),
                     "qda_meanfree": float(q_obs), "qda_null_mean": float(q_null.mean()),
                     "qda_p": float((1 + (q_null >= q_obs).sum()) / (np.isfinite(q_null).sum() + 1)) if np.isfinite(q_obs) else float("nan"), "sec_shift": t_shift, "sec_qda": t_qda})
        r = rows[-1]
        print(f"{s['session']} {area:6s} N {r['N']:3d} | PA {r13['pa_5']['obs']:.3f} floor {r13['pa_5']['floor']:.3f} | LDA {lda_raw:.3f} | "
              f"shift {obs:+.3f} z {r['shift_z']:+.1f} | QDA-mf {q_obs:.3f} p {r['qda_p']:.3f} | s {t_shift:.0f}/{t_qda:.0f} ({time.time() - t0:.0f}s)", flush=True)
    return rows


def score(rows):
    from scipy import stats
    N = np.array([r["N"] for r in rows])
    v = {"n": len(rows)}
    for k in ("pa_5", "frob"):
        raw = np.array([r["r13"][k]["obs"] for r in rows])
        cor = raw - np.array([r["r13"][k]["floor"] for r in rows])
        v[k] = {"spearman_raw_N": float(stats.spearmanr(raw, N)[0]), "spearman_corrected_N": float(stats.spearmanr(cor, N)[0]),
                "floor_share_median": float(np.median(np.array([r["r13"][k]["floor"] for r in rows]) / raw))}
        areas = sorted({r["area"] for r in rows if sum(x["area"] == r["area"] for x in rows) >= 10})
        mr = [np.median([r["r13"][k]["obs"] for r in rows if r["area"] == a]) for a in areas]
        mc = [np.median([r["r13"][k]["obs"] - r["r13"][k]["floor"] for r in rows if r["area"] == a]) for a in areas]
        v[k]["areas"] = areas
        v[k]["area_median_raw"], v[k]["area_median_corrected"] = [float(x) for x in mr], [float(x) for x in mc]
        v[k]["kendall_raw_vs_corrected"] = float(stats.kendalltau(mr, mc)[0]) if len(areas) > 2 else None
    excl = [f'{r["session"]} {r["area"]}' for r in rows if not (np.isfinite(r["lda_raw"]) and np.isfinite(r["qda_meanfree"]))]
    rows = [r for r in rows if np.isfinite(r["lda_raw"]) and np.isfinite(r["qda_meanfree"])]   # H5b: R14 over populations whose decoders ran
    v["R14_excluded_decoder_undefined"] = excl
    tgt = np.array([r["qda_p"] <= 0.05 for r in rows])
    z = np.array([r["shift_z"] for r in rows])
    lda = np.array([r["lda_raw"] for r in rows])
    from sklearn.metrics import roc_auc_score
    v["R14"] = {"target_positive": int(tgt.sum()),
                "auc_shift_z": float(roc_auc_score(tgt, z)) if 0 < tgt.sum() < len(tgt) else None,
                "auc_lda": float(roc_auc_score(tgt, lda)) if 0 < tgt.sum() < len(tgt) else None,
                "spearman_z_qda": float(stats.spearmanr(z, [r["qda_meanfree"] for r in rows])[0]),
                "spearman_lda_qda": float(stats.spearmanr(lda, [r["qda_meanfree"] for r in rows])[0]),
                "spearman_z_lda": float(stats.spearmanr(z, lda)[0])}
    q = np.array([r["qda_meanfree"] for r in rows])
    rz = stats.rankdata(z); rq = stats.rankdata(q); rl = stats.rankdata(lda)
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)  # noqa: E731
    v["R14"]["partial_z_qda_given_lda"] = float(np.corrcoef(res(rz, rl), res(rq, rl))[0, 1])
    hi_l, hi_z = lda > np.median(lda), z >= 2
    v["R14"]["quadrants"] = {"means+_fluct+": int((hi_l & hi_z).sum()), "means+_fluct-": int((hi_l & ~hi_z).sum()),
                             "means-_fluct+": int((~hi_l & hi_z).sum()), "means-_fluct-": int((~hi_l & ~hi_z).sum())}
    v["R14"]["median_seconds"] = {"shift_screen": float(np.median([r["sec_shift"] for r in rows])), "qda_pipeline": float(np.median([r["sec_qda"] for r in rows]))}
    return v


def main():
    sess = json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]
    only = [int(a) for a in sys.argv[1:] if a.isdigit()]   # H5b: rerun only these sessions (H5-R1: the rest are copied from 'none')
    if only:
        sess = [x for x in sess if x["session"] in only]
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        global N_FLOOR, N_SHIFT, N_QDA
        N_FLOOR, N_SHIFT, N_QDA = 3, 3, 2
        run_session(sess[0])
        return
    with ProcessPoolExecutor(3, mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(run_session, sess) for x in part]
    out = {"registration": "H4-R13 + H4-R14 (ex0_REGISTERED.md, 16e4797d)", "rows": rows, "score": score(rows)}
    print("SCORE", json.dumps(out["score"], indent=1), flush=True)
    json.dump(out, open(OUT.with_name(OUT.stem + "_partial.json") if only else OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
