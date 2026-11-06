"""H5b-R3 (Oct 26; registration in data_canonical/ex0_REGISTERED.md): Allen VISp state modes against pupil (ledger J14), beside running.
The 26 sessions whose NWB carries eye tracking; masking path (variant from the sandbox, registered B); presentations with NaN pupil
(h5b_allen_pupil_local.py) or NaN speed are dropped from every arm of that session.
(a) all trials (as cr_h1_r33b): TF-residualized within-class residuals R; top 3 eigenmodes of W; Pearson r of each mode score with pupil
    and with speed; partial r with pupil given speed; fraction of the residual variance explained by pupil, speed and both; r(pupil, speed).
(b) held out (as h4_r3 / h4_r9, both ways, the same halves and per-half TF residualization): loadings of the residuals on the centred
    regressor(s) fitted on half A, removed from half B with half B's regressor(s): pupil, speed, both; and k3 (three modes of half A).
Out: data_canonical/h5b_r3_pupil.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402

spec = importlib.util.spec_from_file_location("r3", HERE / "h4_r3_heldout_modes.py")
R3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R3)
RUN = HERE.parent / "raw" / "allen_running"
PUP = HERE.parent / "raw" / "allen_pupil"


def regress_out(RA, QA, RB, QB):
    """Loadings of RA on the centred regressors QA (n x p), removed from RB with RB's own centred regressors."""
    QA = QA - QA.mean(0); QB = QB - QB.mean(0)
    B = np.linalg.lstsq(QA, RA, rcond=None)[0]
    return RB - QB @ B


def frac_explained(R, Q):
    Q = Q - Q.mean(0)
    fit = Q @ np.linalg.lstsq(Q, R, rcond=None)[0]
    return float((fit ** 2).sum() / (R ** 2).sum())


def partial(x, y, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    rows = []
    for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]:
        sid = s["session"]
        if not (PUP / f"session_{sid}.npz").exists():
            continue
        S = E.load_session(sid)
        idx = E.populations(S).get("VISp")
        if idx is None:
            continue
        r = np.load(RUN / f"session_{sid}.npz"); p = np.load(PUP / f"session_{sid}.npz"); c = np.load(E.COUNTS / f"session_{sid}.npz")
        assert np.allclose(r["start"], c["start"]) and np.allclose(p["start"], c["start"]), sid
        V = E.pop_view(S, "VISp", idx, extra={"speed": r["speed"], "pupil": p["pupil"]})   # H5-R1 masking path
        ok = np.isfinite(V["pupil"]) & np.isfinite(V["speed"])
        X, lab, cell = V["X"][ok], V["lab"][ok], V["cell"][ok]
        sp, pu = V["speed"][ok].astype(np.float64), V["pupil"][ok].astype(np.float64)
        row = {"session": sid, "N": int(X.shape[1]), "n_used": int(ok.sum()), "n_dropped_nan": int((~ok).sum()),
               "r_pupil_speed": float(np.corrcoef(pu, sp)[0, 1])}
        # (a) all trials
        Y = E.tf_residualize(X, lab, cell); R = Y - np.stack([Y[lab == k].mean(0) for k in range(8)])[lab]
        lam, W = np.linalg.eigh(R.T @ R); W = W[:, ::-1]
        sc = R @ W[:, :3]
        row["r_mode_pupil"] = [float(np.corrcoef(sc[:, k], pu)[0, 1]) for k in range(3)]
        row["r_mode_speed"] = [float(np.corrcoef(sc[:, k], sp)[0, 1]) for k in range(3)]
        row["partial_mode_pupil_given_speed"] = [partial(sc[:, k], pu, sp) for k in range(3)]
        row["max_abs_r_pupil"] = float(np.max(np.abs(row["r_mode_pupil"])))
        row["max_abs_partial_pupil"] = float(np.max(np.abs(row["partial_mode_pupil_given_speed"])))
        row["var_explained"] = {"pupil": frac_explained(R, pu[:, None]), "speed": frac_explained(R, sp[:, None]),
                                "both": frac_explained(R, np.c_[pu, sp])}
        # (b) held out
        h = E.halves(cell)
        P = [R3.half_parts(X, lab, cell, h, 0), R3.half_parts(X, lab, cell, h, 1)]
        Q = {"pupil": pu[:, None], "speed": sp[:, None], "joint": np.c_[pu, sp]}
        for est, meas in ((0, 1), (1, 0)):
            Pe, Pm = P[est], P[meas]
            w = {"k0": R3.measure(Pm, Pe["V"], 0), "k3": R3.measure(Pm, Pe["V"], 3)}
            for name, q in Q.items():
                Rr = regress_out(Pe["R"], q[h == est], Pm["R"], q[h == meas])
                Z = Pm["M"][Pm["lab"]] + Rr
                w[f"regress_{name}"] = E.ladder(Z, Pm["lab"])
            row[f"est{est}_meas{meas}"] = w
        rows.append(row)
        x = row["est0_meas1"]
        print(sid, f"n {row['n_used']} | r(pu,sp) {row['r_pupil_speed']:+.2f} | modes~pupil {np.round(row['r_mode_pupil'], 2)} "
              f"partial {np.round(row['partial_mode_pupil_given_speed'], 2)} | cm k0 {x['k0']['class_means']:+.3f} k3 {x['k3']['class_means']:+.3f} "
              f"pupil {x['regress_pupil']['class_means']:+.3f} speed {x['regress_speed']['class_means']:+.3f} joint {x['regress_joint']['class_means']:+.3f}",
              flush=True)
    summ = {"n": len(rows), "n_dropped_nan_total": int(sum(r["n_dropped_nan"] for r in rows)),
            "max_abs_r_pupil_ge_0.3": int(sum(r["max_abs_r_pupil"] >= 0.3 for r in rows)),
            "median_max_abs_r_pupil": float(np.median([r["max_abs_r_pupil"] for r in rows])),
            "median_max_abs_r_speed": float(np.median([np.max(np.abs(r["r_mode_speed"])) for r in rows])),
            "median_max_abs_partial_pupil": float(np.median([r["max_abs_partial_pupil"] for r in rows])),
            "median_r_pupil_speed": float(np.median([r["r_pupil_speed"] for r in rows])),
            "median_var_explained": {k: float(np.median([r["var_explained"][k] for r in rows])) for k in ("pupil", "speed", "both")}}
    for way in ("est0_meas1", "est1_meas0"):
        d = {}
        for arm in ("k0", "k3", "regress_pupil", "regress_speed", "regress_joint"):
            cm = np.array([r[way][arm]["class_means"] for r in rows]); wi = np.array([r[way][arm]["within"] for r in rows])
            w0 = np.array([r[way]["k0"]["within"] for r in rows])
            d[arm] = {"cm_negative": int((cm < 0).sum()), "cm_median": float(np.median(cm)), "within_median": float(np.median(wi)),
                      "within_change_median": float(np.median(wi - w0))}
        summ[way] = d
    pred = {"P1": summ["max_abs_r_pupil_ge_0.3"] >= 13,
            "P2": all(summ[w]["regress_pupil"]["cm_negative"] < summ[w]["k3"]["cm_negative"] for w in ("est0_meas1", "est1_meas0")),
            "P3": all(summ[w]["regress_joint"]["cm_negative"] >= summ[w]["regress_speed"]["cm_negative"] for w in ("est0_meas1", "est1_meas0"))
            and any(summ[w]["regress_joint"]["cm_negative"] > summ[w]["regress_speed"]["cm_negative"] for w in ("est0_meas1", "est1_meas0")),
            "P4": summ["median_max_abs_partial_pupil"] >= 0.2}
    summ["predictions"] = {k: bool(v) for k, v in pred.items()}
    print("SUMMARY", json.dumps(summ, indent=1), flush=True)
    json.dump({"registration": "H5b-R3 (ex0_REGISTERED.md)", "variant": E.VARIANT, "rows": rows, "summary": summ},
              open(HERE.parent / "data_canonical" / "h5b_r3_pupil.json", "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
