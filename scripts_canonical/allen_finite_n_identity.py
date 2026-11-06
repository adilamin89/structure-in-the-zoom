"""Allen's two-term sign misses against the finite-sample identity (Section 6, Appendix H, Appendix K's finite-sample form).
The within part of the TF-residualized arm (variant B), per population, from the release's own engine (allen_engine.py in a release clone:
load_session, populations, pop_view, tf_residualize, ladder). The arm is linear, Y = A X (A = tf_residualize applied to the identity), so
for any set of rows of class c the within scatter is X_c^T Q X_c with Q = A_rows^T C A_rows; for rows iid within their cell (covariance
Sigma_c), E Tr W = sum_c q1_c t_c, E Tr W^2 = sum_cd q1_c q1_d T_cd + sum_c [q2_c (t_c^2 + T_cc) + q3_c (m4_c - t_c^2 - 2 T_cc)],
q1 = tr Q, q2 = tr Q^2, q3 = sum_i Q_ii^2. Arms: W0 printed mean field, W1 population identity on the same rungs, W2 finite-n identity
(ladder all rows; floor 200 random row subsets), J from 50 Gaussian surrogates through the release ladder, W2J = W2 + J.
The paper prints the W2 column: the two terms with W2 give the shift's sign in 163 of 167 populations (r 0.94 on the shift, 0.95 on the
within part; the 15 misses' mean W2 -0.083 against the measured -0.071).
Run (raw/allen_counts from allen_counts_local.py): VECLIB_MAXIMUM_THREADS=1 python -X faulthandler scripts_canonical/allen_finite_n_identity.py [repository] [n_workers]
Out: data_canonical/allen_finite_n_identity.json"""
import glob
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
N_FLOOR, N_SIM, N_SPLIT = 200, 50, 10


def engine(clone):
    spec = importlib.util.spec_from_file_location("allen_engine", Path(clone) / "scripts_canonical" / "allen_engine.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def cell_resid(X, cell):
    R = X.copy(); k = 0
    for b in np.unique(cell):
        m = cell == b; R[m] = X[m] - X[m].mean(0); k += 1
    return R, len(X) - k


def moments(X, lab, cell, rng):
    t, m4, E, dof = np.zeros(8), np.zeros(8), [], np.zeros(8)
    for c in range(8):
        R, d = cell_resid(X[lab == c], cell[lab == c]); E.append(R); dof[c] = d
        t[c] = (R * R).sum() / d
        m4[c] = (((R * R).sum(1)) ** 2).mean() * (len(R) / d) ** 2
    T = np.zeros((8, 8))
    for c in range(8):
        for d in range(c + 1, 8):
            T[c, d] = T[d, c] = ((E[c] @ E[d].T) ** 2).sum() / (dof[c] * dof[d])
    for c in range(8):
        Xc, cc = X[lab == c], cell[lab == c]; acc, n_ok = 0.0, 0
        for _ in range(N_SPLIT):
            h = np.zeros(len(Xc), int)
            for b in np.unique(cc):
                ix = np.where(cc == b)[0]; h[rng.permutation(ix)[::2]] = 1
            Ra, da = cell_resid(Xc[h == 0], cc[h == 0]); Rb, db = cell_resid(Xc[h == 1], cc[h == 1])
            if da > 0 and db > 0:
                acc += ((Ra @ Rb.T) ** 2).sum() / (da * db); n_ok += 1
        T[c, c] = acc / n_ok
    return {"t": t, "T": T, "m4": m4, "E": E, "dof": dof}


def qs(A, rows, cols):
    Ac = A[np.ix_(rows, cols)]
    Ac = Ac - Ac.mean(0)                                  # C A_rows
    Q = Ac.T @ Ac
    return float(np.trace(Q)), float((Q * Q).sum()), float((np.diag(Q) ** 2).sum())


def lpr(classes, mom, gauss=False):
    """classes: list of (c, q1, q2, q3)."""
    num = sum(q1 * mom["t"][c] for c, q1, _, _ in classes) ** 2
    den = sum(a[1] * b[1] * mom["T"][a[0], b[0]] for a in classes for b in classes)
    for c, q1, q2, q3 in classes:
        t, Tc = mom["t"][c], mom["T"][c, c]
        m4 = t * t + 2 * Tc if gauss else mom["m4"][c]
        den += q2 * (t * t + Tc) + q3 * (m4 - t * t - 2 * Tc)
    return float(np.log(num / den))


def lpr_pop(counts, mom):
    w = np.asarray(counts, float)
    return float(np.log((w @ mom["t"]) ** 2 / (w @ mom["T"] @ w)))


def predict(A, lab, mom, sets, rng, gauss=False):
    cols = [np.where(lab == c)[0] for c in range(8)]
    sizes = [len(s) for s in sets]
    lad, flo, lad1, flo1 = [], [], [], []
    for s in sets:
        cl = [(c, *qs(A, np.intersect1d(s, cols[c]), cols[c])) for c in range(8) if np.isin(cols[c], s).sum() >= 2]
        lad.append(lpr(cl, mom, gauss)); lad1.append(lpr_pop(np.bincount(lab[s], minlength=8), mom))
        f, f1 = [], []
        for _ in range(N_FLOOR):
            sub = rng.choice(len(lab), len(s), replace=False)
            cl = [(c, *qs(A, sub[lab[sub] == c], cols[c])) for c in range(8) if (lab[sub] == c).sum() >= 2]
            f.append(lpr(cl, mom, gauss)); f1.append(lpr_pop(np.bincount(lab[sub], minlength=8), mom))
        flo.append(np.mean(f)); flo1.append(np.mean(f1))
    x = np.log(sizes)
    return float(np.polyfit(x, np.subtract(lad, flo), 1)[0]), float(np.polyfit(x, np.subtract(lad1, flo1), 1)[0])


def run_session(args):
    clone, sid = args
    os.environ.setdefault("ALLEN_VARIANT", "B")
    E = engine(clone)
    stored = json.load(open(Path(clone) / "data_canonical" / "allen_sessions" / f"session_{sid}.json"))["areas"]
    S = E.load_session(sid); out = []
    for area, idx in E.populations(S).items():
        P = E.pop_view(S, area, idx); X, lab, cell = P["X"], P["lab"], P["cell"]
        Y = E.tf_residualize(X, lab, cell)
        meas = E.ladder(Y, lab)["within"]
        assert abs(meas - stored[area]["tfres"]["within"]) < 1e-9, (sid, area, meas, stored[area]["tfres"]["within"])
        A = E.tf_residualize(np.eye(len(X)), lab, cell)
        assert np.allclose(A @ X, Y, atol=1e-8 * np.abs(Y).max()), (sid, area)
        rng = np.random.default_rng(sid % 100000 + len(X))
        mom = moments(X, lab, cell, rng)
        sets = E.class_sets(lab, list(range(8)))
        W2, W1 = predict(A, lab, mom, sets, rng)
        # curvature: Gaussian surrogates with each class's within-cell covariance, through the release ladder
        sur = {"t": mom["t"].copy(), "T": np.array([[((mom["E"][c] @ mom["E"][d].T) ** 2).sum() / (mom["dof"][c] * mom["dof"][d])
                                                      for d in range(8)] for c in range(8)]), "m4": None}
        W2s, _ = predict(A, lab, sur, sets, rng, gauss=True)
        sims = []
        for k in range(N_SIM):
            Xs = np.zeros_like(X)
            for c in range(8):
                m = lab == c
                Xs[m] = rng.normal(size=(m.sum(), len(mom["E"][c]))) @ mom["E"][c] / np.sqrt(mom["dof"][c])
            sims.append(E.ladder(A @ Xs, lab, seed=1000 + k)["within"])
        J = float(np.mean(sims) - W2s)
        out.append({"session": sid, "area": area, "N": int(X.shape[1]), "n_rows": int(len(X)), "measured_within": meas,
                    "W1": W1, "W2": W2, "J": J, "W2J": W2 + J, "sim_within_mean": float(np.mean(sims)),
                    "sim_within_sd": float(np.std(sims, ddof=1)), "W2_surrogate": W2s,
                    "PR_class_pop": float(np.mean(mom["t"] ** 2 / np.diag(mom["T"]))), "per_class": float(len(X) / 8)})
        print(f"{sid} {area:6s} N {X.shape[1]:3d} rows {len(X)}: meas {meas:+.3f} W1 {W1:+.3f} W2 {W2:+.3f} J {J:+.3f} -> {W2 + J:+.3f}", flush=True)
    return out


def main():
    clone = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent); nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    sids = sorted(int(Path(f).stem.split("_")[1]) for f in glob.glob(str(Path(clone) / "data_canonical" / "allen_sessions" / "session_*.json")))
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        rows = [r for rs in ex.map(run_session, [(clone, s) for s in sids]) for r in rs]
    pred = {(p["session"], p["area"]): p for p in json.load(open(Path(clone) / "data_canonical" / "allen_twoterm_predictions.json"))}
    st = {}
    for f in glob.glob(str(Path(clone) / "data_canonical" / "allen_sessions" / "session_*.json")):
        d = json.load(open(f))
        for a, v in d["areas"].items():
            st[(d["session"], a)] = v["tfres"]["delta"]
    for r in rows:
        p = pred[(r["session"], r["area"])]
        r.update(W0=p["within_pred"], classmean_pred=p["classmean_pred"], delta_measured=st[(r["session"], r["area"])],
                 miss=bool(np.sign(p["delta_pred"]) != np.sign(st[(r["session"], r["area"])])))
    mi = [r for r in rows if r["miss"]]; mm = np.array([r["measured_within"] for r in mi]); wj = np.array([r["W2J"] for r in mi])
    se = mm.std(ddof=1) / np.sqrt(len(mm))
    neg = [r for r in mi if r["measured_within"] < 0]
    v = {"n": len(rows), "n_miss": len(mi), "miss_measured_mean": float(mm.mean()), "miss_W2J_mean": float(wj.mean()),
         "miss_W2_mean": float(np.mean([r["W2"] for r in mi])), "miss_W0_mean": float(np.mean([r["W0"] for r in mi])), "miss_se": float(se),
         "miss_mean_within_2se": bool(abs(wj.mean() - mm.mean()) <= 2 * se),
         "W2J_negative_of_measured_negative": int(sum(r["W2J"] < 0 for r in neg)), "n_measured_negative": len(neg),
         "mean_err_W2J": float(np.mean([r["W2J"] - r["measured_within"] for r in rows])),
         "mean_err_W0": float(np.mean([r["W0"] - r["measured_within"] for r in rows])),
         "mean_err_W2": float(np.mean([r["W2"] - r["measured_within"] for r in rows])),
         "sign_agree_W2J": int(sum(np.sign(r["W2J"] + r["classmean_pred"]) == np.sign(r["delta_measured"]) for r in rows)),
         "sign_agree_W0": int(sum(np.sign(r["W0"] + r["classmean_pred"]) == np.sign(r["delta_measured"]) for r in rows)),
         "pearson_W2J": float(np.corrcoef([r["W2J"] for r in rows], [r["measured_within"] for r in rows])[0, 1]),
         "pearson_W0": float(np.corrcoef([r["W0"] for r in rows], [r["measured_within"] for r in rows])[0, 1])}
    v["sign_agree_W2"] = int(sum(np.sign(r["W2"] + r["classmean_pred"]) == np.sign(r["delta_measured"]) for r in rows))
    v["pearson_W2"] = float(np.corrcoef([r["W2"] for r in rows], [r["measured_within"] for r in rows])[0, 1])
    v["pearson_delta_W2"] = float(np.corrcoef([r["W2"] + r["classmean_pred"] for r in rows], [r["delta_measured"] for r in rows])[0, 1])
    print("VERDICT", json.dumps(v), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(DATA / "allen_finite_n_identity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
