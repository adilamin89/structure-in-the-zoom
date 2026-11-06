"""Three cloud-run (Modal) Allen generators recomputed from the local counts, each population through
allen_engine.pop_view (ALLEN_VARIANT); estimator code verbatim from each source:
  allen_multipoles.py            -> data_canonical/allen_multipoles_all_sessions.json (class-mean correlation harmonics a, c, b, b4,
                                    J1/J2, fit r2, cardinal fraction, C(dphi))
  modal_allen_withinclass_pr.py  -> data_canonical/allen_withinclass_pr.json (PR of each direction class, rung 1, within-class
                                    trial correlation)
  modal_allen_crossfit.py (Modal) -> data_canonical/allen_crossfit_split.json (harmonics from one half of each class,
                                    delta_dir8 from the other, own floors rng 101/102; full-session delta rng 103 + 10 shuffles
                                    with the reused floor; summary r A->B, B->A)
float32 counts as on Modal. 'none' must reproduce the canonical JSONs (1e-6).
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler allen_classstats_local.py [n_workers]"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
BIN_COUNTS_DIR = [1, 2, 3, 4, 6, 8]
N_NULL = 10
N_SHUFFLE = 10


def pr_trace(X):
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    if n < 3:
        return 1.0
    G = (Xc @ Xc.T).astype(np.float64) / (n - 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.maximum(np.asarray(prs, dtype=float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


# ---------------------------------------------------------------------------- allen_multipoles.py
def fit_multipoles(C, dphi):
    A = np.vstack([np.ones_like(dphi), np.cos(dphi), np.cos(2 * dphi), np.cos(4 * dphi)]).T
    coef, *_ = np.linalg.lstsq(A, C, rcond=None)
    resid = C - A @ coef
    r2 = 1 - resid.var() / C.var() if C.var() > 0 else None
    return coef, r2


def multipoles_area(X, ori, dirs):
    M = np.stack([X[ori == d].mean(axis=0) for d in dirs])
    Mc = M - M.mean(axis=1, keepdims=True)
    Mn = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-9)
    Cfull = Mn @ Mn.T
    dmat = np.abs(np.radians(dirs)[:, None] - np.radians(dirs)[None, :])
    dmat = np.minimum(dmat, 2 * np.pi - dmat)
    pairs = {}
    for i in range(8):
        for j in range(8):
            key = round(float(dmat[i, j]), 6)
            pairs.setdefault(key, []).append(float(Cfull[i, j]))
    dphi = np.array(sorted(pairs))
    Cbar = np.array([np.mean(pairs[k]) for k in sorted(pairs)])
    (a0, c1, b2, b4), r2 = fit_multipoles(Cbar, dphi)
    power = (Mc ** 2).sum(axis=1)
    card = float(power[np.isin(dirs, [0, 90, 180, 270])].sum() / power.sum())
    return {"n_units": int(X.shape[1]), "a": float(a0), "c_dipole": float(c1), "b_quadrupole": float(b2), "b4": float(b4),
            "J1_over_J2": float(c1 / b2) if abs(b2) > 1e-6 else None, "fit_r2": r2 if r2 is None else float(r2),
            "cardinal_fraction": card, "C_dphi": {str(round(float(k), 3)): float(np.mean(v)) for k, v in pairs.items()}}


# ---------------------------------------------------------------------------- modal_allen_withinclass_pr.py
def withinclass_area(X, ori, dirs):
    cls_prs, cls_cors = [], []
    for d in dirs:
        idx = np.where(ori == d)[0]
        V = X[idx]
        cls_prs.append(pr_trace(V))
        Vn = V - V.mean(axis=1, keepdims=True)
        Vn /= np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-9
        Cm = Vn @ Vn.T
        m = len(idx)
        cls_cors.append(float((Cm.sum() - m) / (m * (m - 1))))
    return {"n_units": int(X.shape[1]), "pr_within_rung1": cls_prs[0], "pr_within_mean": float(np.mean(cls_prs)),
            "pr_within_per_class": cls_prs, "corr_within_mean": float(np.mean(cls_cors))}


# ---------------------------------------------------------------------------- modal_allen_crossfit.py
def crossfit_area(X, ori, dirs):
    def ladder_delta(X, labels, rng, floor_logs=None):
        members = [np.where(labels == c)[0] for c in dirs]
        sizes, prs = [], []
        for c in BIN_COUNTS_DIR:
            sel = np.concatenate(members[:c])
            if len(sel) < 10:
                continue
            sizes.append(len(sel))
            prs.append(pr_trace(X[sel]))
        if len(sizes) < 3:
            return None, None, None
        th_o = slope(sizes, prs)
        if floor_logs is None:
            floor_logs = np.zeros((N_NULL, len(sizes)))
            for d in range(N_NULL):
                for k, s in enumerate(sizes):
                    floor_logs[d, k] = np.log(max(pr_trace(X[rng.choice(len(X), s, replace=False)]), 1e-9))
        th_f = slope(sizes, np.exp(floor_logs.mean(axis=0)))
        return th_o - th_f, floor_logs, sizes

    def multipoles(X, labels):
        M = np.stack([X[labels == c].mean(axis=0) for c in dirs])
        Mc = M - M.mean(axis=1, keepdims=True)
        Mc /= np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-9
        C = Mc @ Mc.T
        ang = np.deg2rad(dirs)
        dphi = np.abs(ang[:, None] - ang[None, :])
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        iu = np.triu_indices(len(dirs), k=1)
        d, c = dphi[iu], C[iu]
        A = np.vstack([np.ones_like(d), np.cos(d), np.cos(2 * d), np.cos(4 * d)]).T
        coef, *_ = np.linalg.lstsq(A, c, rcond=None)
        return {"a": float(coef[0]), "c_dipole": float(coef[1]), "b_quadrupole": float(coef[2]), "b4": float(coef[3])}

    half = np.zeros(len(ori), dtype=bool)
    for c in dirs:
        m = np.where(ori == c)[0]
        half[m[::2]] = True
    XA, oA = X[half], ori[half]
    XB, oB = X[~half], ori[~half]
    dA, _, _ = ladder_delta(XA, oA, np.random.default_rng(101))
    dB, _, _ = ladder_delta(XB, oB, np.random.default_rng(102))
    mpA, mpB = multipoles(XA, oA), multipoles(XB, oB)
    d_full, floor_logs, _ = ladder_delta(X, ori, np.random.default_rng(103))
    shuf = []
    for s in range(N_SHUFFLE):
        srng = np.random.default_rng(500 + s)
        ds, _, _ = ladder_delta(X, ori[srng.permutation(len(ori))], None, floor_logs=floor_logs)
        if ds is not None:
            shuf.append(ds)
    return {"n_units": int(X.shape[1]), "delta_A": dA, "delta_B": dB, "b_quad_A": mpA["b_quadrupole"], "b_quad_B": mpB["b_quadrupole"],
            "c_dip_A": mpA["c_dipole"], "c_dip_B": mpB["c_dipole"], "delta_full": d_full,
            "shuffled_mean_10": float(np.mean(shuf)) if shuf else None, "shuffled_sd_10": float(np.std(shuf)) if shuf else None}


def session(sid):
    sys.path.insert(0, str(HERE))
    import allen_engine as E
    t0 = time.time()
    S = E.load_session(sid)
    counts = S["X"].astype(np.float32)
    out = {k: {"session": sid, "status": "ok", "areas": {}, "h5_variant": E.VARIANT} for k in ("mp", "wc", "cf")}
    for area, idx in E.populations(S).items():
        P = E.pop_view(S, area, idx)
        X = counts[np.ix_(P["rows"], P["cols"])]
        ori, dirs = S["dirs"][P["lab"]], S["dirs"]
        out["mp"]["areas"][area] = multipoles_area(X, ori, dirs)
        out["wc"]["areas"][area] = withinclass_area(X, ori, dirs)
        out["cf"]["areas"][area] = crossfit_area(X, ori, dirs)
    print(f"[{sid}] {len(out['mp']['areas'])} areas ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    sids = [s["session"] for s in json.load(open(DATA / "allen_expansion_all_sessions.json"))["results"]]
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(session, sids))
    json.dump({"results": [r["mp"] for r in res]}, open(DATA / "allen_multipoles_all_sessions.json", "w"), indent=1)
    json.dump({"results": [r["wc"] for r in res]}, open(DATA / "allen_withinclass_pr.json", "w"), indent=1)
    rows = [dict(session=r["cf"]["session"], area=a, **v) for r in res for a, v in r["cf"]["areas"].items()
            if v["delta_A"] is not None and v["delta_B"] is not None]

    def corr(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y)))
    r1 = corr([r["b_quad_A"] for r in rows], [r["delta_B"] for r in rows])
    r2 = corr([r["b_quad_B"] for r in rows], [r["delta_A"] for r in rows])
    json.dump({"registration": "modal_allen_crossfit.py docstring (pre-run); local port allen_classstats_local.py",
               "n_populations": len(rows), "crossfit_r_AtoB": r1, "crossfit_r_BtoA": r2, "crossfit_r_mean": (r1 + r2) / 2,
               "results": [r["cf"] for r in res]}, open(DATA / "allen_crossfit_split.json", "w"), indent=1)
    print("crossfit r", r1, r2, "DONE", flush=True)


if __name__ == "__main__":
    main()
