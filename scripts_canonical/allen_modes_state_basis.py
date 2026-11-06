"""Are Allen VISp's held-out state modes measured state once pupil
and running enter lagged and nonlinearly? The 26 sessions with eye tracking, masking path (ALLEN_VARIANT, default B);
regressors from allen_state_extract.py; presentations with any NaN regressor dropped from every arm.
Basis (13): pupil and speed at lags 0, 0.5, 1.0, 1.5 s; pupil change; pupil(0.5)^2; speed(0)^2; pupil(0) x speed(0); running (speed(0) > 1).
Columns are z-scored with the estimator half's mean and SD; loadings fitted on the estimator half's residuals and applied with the
measured half's regressors (allen_modes_pupil.regress_out). Arms: linear0 (pupil(0) and speed(0): R3's 'joint', pipeline check), basis,
and k = 1..5 modes from the other half (h4_r3.measure). All trials: canonical correlations between the top 3 mode scores and the basis.
Out: data_canonical/allen_modes_state_basis.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import allen_engine as E  # noqa: E402

spec = importlib.util.spec_from_file_location("r3", HERE / "allen_heldout_modes.py")
R3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R3)
spec = importlib.util.spec_from_file_location("pu", HERE / "allen_modes_pupil.py")
PU = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PU)
ST = HERE.parent / "raw" / "allen_state"
LAGS = (0, 500, 1000, 1500)


def basis(z):
    p0, s0 = z["pupil_L0"], z["speed_L0"]
    cols = [z[f"pupil_L{L}"] for L in LAGS] + [z[f"speed_L{L}"] for L in LAGS] + [z["dpupil"], z["pupil_L500"] ** 2, s0 ** 2,
                                                                                  p0 * s0, (s0 > 1).astype(float)]
    return np.column_stack(cols)


def zscore_by(Q, ref):
    mu, sd = Q[ref].mean(0), Q[ref].std(0)
    sd[sd == 0] = 1.0
    return (Q - mu) / sd


def cancorr(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    qa, _ = np.linalg.qr(A); qb, _ = np.linalg.qr(B)
    return np.linalg.svd(qa.T @ qb, compute_uv=False).tolist()


def main():
    rows = []
    for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]:
        sid = s["session"]
        if not (ST / f"session_{sid}.npz").exists():
            continue
        S = E.load_session(sid)
        idx = E.populations(S).get("VISp")
        if idx is None:
            continue
        z = np.load(ST / f"session_{sid}.npz"); c = np.load(E.COUNTS / f"session_{sid}.npz")
        assert np.allclose(z["start"], c["start"]), sid
        V = E.pop_view(S, "VISp", idx, extra={k: z[k] for k in z.files if k != "start"})   # masking path
        Qall = basis(V)
        ok = np.isfinite(Qall).all(1)
        X, lab, cell, Q = V["X"][ok], V["lab"][ok], V["cell"][ok], Qall[ok]
        lin = np.column_stack([V["pupil_L0"][ok], V["speed_L0"][ok]])
        row = {"session": sid, "n_used": int(ok.sum()), "n_dropped": int((~ok).sum())}
        Y = E.tf_residualize(X, lab, cell); R = Y - np.stack([Y[lab == k].mean(0) for k in range(8)])[lab]
        _, W = np.linalg.eigh(R.T @ R); W = W[:, ::-1]
        row["cancorr_modes_basis"] = cancorr(R @ W[:, :3], zscore_by(Q, np.ones(len(Q), bool)))[:2]
        row["var_explained_basis"] = PU.frac_explained(R, zscore_by(Q, np.ones(len(Q), bool)))
        h = E.halves(cell)
        P = [R3.half_parts(X, lab, cell, h, 0), R3.half_parts(X, lab, cell, h, 1)]
        for est, meas in ((0, 1), (1, 0)):
            Pe, Pm = P[est], P[meas]
            w = {f"k{k}": R3.measure(Pm, Pe["V"], k) for k in range(6)}
            for name, q in (("linear0", lin), ("basis", zscore_by(Q, h == est))):
                Rr = PU.regress_out(Pe["R"], q[h == est], Pm["R"], q[h == meas])
                w[name] = E.ladder(Pm["M"][Pm["lab"]] + Rr, Pm["lab"])
            row[f"est{est}_meas{meas}"] = w
        rows.append(row)
        x = row["est0_meas1"]
        print(sid, f"n {row['n_used']} | cancorr {np.round(row['cancorr_modes_basis'], 2)} | cm k0 {x['k0']['class_means']:+.3f} "
              f"linear0 {x['linear0']['class_means']:+.3f} basis {x['basis']['class_means']:+.3f} k1 {x['k1']['class_means']:+.3f} "
              f"k3 {x['k3']['class_means']:+.3f}", flush=True)
    summ = {"n": len(rows), "n_dropped_total": int(sum(r["n_dropped"] for r in rows)),
            "cancorr1_median": float(np.median([r["cancorr_modes_basis"][0] for r in rows])),
            "cancorr2_median": float(np.median([r["cancorr_modes_basis"][1] for r in rows])),
            "var_explained_basis_median": float(np.median([r["var_explained_basis"] for r in rows]))}
    for way in ("est0_meas1", "est1_meas0"):
        summ[way] = {arm: {"cm_negative": int(sum(r[way][arm]["class_means"] < 0 for r in rows)),
                           "cm_median": float(np.median([r[way][arm]["class_means"] for r in rows])),
                           "within_change_median": float(np.median([r[way][arm]["within"] - r[way]["k0"]["within"] for r in rows]))}
                     for arm in ["linear0", "basis"] + [f"k{k}" for k in range(6)]}
    W2 = ("est0_meas1", "est1_meas0")
    summ["predictions"] = {"P1": all(summ[w]["basis"]["cm_negative"] <= 12 for w in W2),
                           "P2": all(summ[w]["basis"]["cm_negative"] - summ[w]["linear0"]["cm_negative"] <= 4 for w in W2)}
    print("SUMMARY", json.dumps(summ, indent=1), flush=True)
    json.dump({"variant": E.VARIANT, "rows": rows, "summary": summ},
              open(HERE.parent / "data_canonical" / "allen_modes_state_basis.json", "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
