"""H4-R3 (Oct 24; registration H4-R3 in data_canonical/ex0_REGISTERED.md, commit 8cba0800; ledger J1, GLM cold read 6 E1).

(a) Allen's state modes held out. R33a (cr_h1_r33_topmodes.py) estimated W's top modes on one half and removed them from both
    halves, then measured on all trials. Here each half (E.halves: alternate within each direction x TF cell) is TF-residualized on
    its own trials (E.tf_residualize inside the half); W's top k = 1, 3, 5 eigenmodes come from half A's class residuals; half B's
    residuals are projected off them (B's class means kept exactly) and the Table 9 ladder and split (E.ladder), c_full and the
    alignment a are measured on half B only; and swapped. k = 0 is each half's baseline. Beside it: modes from the measured half
    itself (in-sample).
(b) Running against direction, per session: correlation ratio eta^2 of the presentation's mean running speed over the 8 direction
    classes, one-way F-test p, 2000-permutation p (rng 0), circular-linear correlation with (cos, sin) of direction; the same
    eta^2 against the 5 temporal frequencies.
Out: data_canonical/h4_r3_heldout_modes.json"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402

RUN = HERE.parent / "raw" / "allen_running"
KS = (0, 1, 3, 5)


def alignment(sg):
    """R33a's alignment a = PR(W) x / (w1 b1), from the sign-law quantities."""
    return float(sg["x_share"] * sg["trB_trW"] / (2 * (1 - sg["x_share"]) * sg["PR_B"]) * sg["PR_W"])


def half_parts(X, lab, cell, h, s):
    m = h == s
    Y = E.tf_residualize(X[m], lab[m], cell[m])
    M = np.stack([Y[lab[m] == c].mean(0) for c in range(8)])
    R = Y - M[lab[m]]
    _, V = np.linalg.eigh(R.T @ R)
    return {"lab": lab[m], "M": M, "R": R, "V": V}


def measure(P, V, k):
    Vk = V[:, -k:] if k else np.zeros((P["R"].shape[1], 0))
    Z = P["M"][P["lab"]] + P["R"] - (P["R"] @ Vk) @ Vk.T
    sg = E.sign_law(Z, P["lab"])
    return {**E.ladder(Z, P["lab"]), "c_full": sg["c_full"], "PR_W": sg["PR_W"], "a": alignment(sg)}


def eta2(v, g):
    mu = v.mean()
    ssb = sum((g == c).sum() * (v[g == c].mean() - mu) ** 2 for c in np.unique(g))
    sst = ((v - mu) ** 2).sum()
    return float(ssb / sst)


def running_lock(sp, lab, tfl, dirs_deg, rng):
    if sp.std() == 0:
        return None
    e = eta2(sp, lab)
    null = np.array([eta2(sp, lab[rng.permutation(len(lab))]) for _ in range(2000)])
    th = np.deg2rad(dirs_deg[lab])
    rc, rs = np.corrcoef(sp, np.cos(th))[0, 1], np.corrcoef(sp, np.sin(th))[0, 1]
    rcs = np.corrcoef(np.cos(th), np.sin(th))[0, 1]
    rcl = np.sqrt(max((rc ** 2 + rs ** 2 - 2 * rc * rs * rcs) / (1 - rcs ** 2), 0.0))
    et = eta2(sp, tfl)
    nt = np.array([eta2(sp, tfl[rng.permutation(len(tfl))]) for _ in range(2000)])
    return {"eta2_dir": e, "p_F_dir": float(stats.f_oneway(*[sp[lab == c] for c in range(8)]).pvalue),
            "p_perm_dir": float((1 + (null >= e).sum()) / 2001), "eta2_null_mean": float(null.mean()),
            "r_circlin_dir": float(rcl), "eta2_tf": et, "p_perm_tf": float((1 + (nt >= et).sum()) / 2001),
            "speed_mean": float(sp.mean()), "frac_running": float((sp > 1).mean())}


def main():
    rows = []
    for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]:
        sid = s["session"]
        S = E.load_session(sid)
        idx = E.populations(S).get("VISp")
        if idx is None:
            continue
        r = np.load(RUN / f"session_{sid}.npz")
        c = np.load(E.COUNTS / f"session_{sid}.npz")
        assert np.allclose(r["start"], c["start"]), sid
        V = E.pop_view(S, "VISp", idx, extra={"speed": r["speed"]})   # H5-R1 masking path
        X, lab, cell = V["X"], V["lab"], V["cell"]
        h = E.halves(cell)
        P = [half_parts(X, lab, cell, h, 0), half_parts(X, lab, cell, h, 1)]
        row = {"session": sid, "N": int(X.shape[1]), "n_half": [int((h == 0).sum()), int((h == 1).sum())]}
        for est, meas in ((0, 1), (1, 0)):
            row[f"est{est}_meas{meas}"] = {f"k{k}": measure(P[meas], P[est]["V"], k) for k in KS}
            row[f"insample_meas{meas}"] = {f"k{k}": measure(P[meas], P[meas]["V"], k) for k in KS}
        row["running"] = running_lock(V["speed"].astype(np.float64), lab, V["tf"], S["dirs"], np.random.default_rng(0))
        rows.append(row)
        ho = row["est0_meas1"]
        print(sid, "B<-A " + " ".join(f"k{k}: cm {ho[f'k{k}']['class_means']:+.3f} w {ho[f'k{k}']['within']:+.3f}" for k in KS),
              "| A<-B k3 cm %+.3f w %+.3f" % (row["est1_meas0"]["k3"]["class_means"], row["est1_meas0"]["k3"]["within"]),
              "| run eta2 %s p %s" % ((None, None) if row["running"] is None else (round(row["running"]["eta2_dir"], 4), round(row["running"]["p_perm_dir"], 4))),
              flush=True)
    summ = {}
    for way in ("est0_meas1", "est1_meas0", "insample_meas1", "insample_meas0"):
        d = {}
        for k in KS:
            cm = np.array([r[way][f"k{k}"]["class_means"] for r in rows])
            w = np.array([r[way][f"k{k}"]["within"] for r in rows])
            d[f"k{k}"] = {"cm_negative": int((cm < 0).sum()), "cm_median": float(np.median(cm)), "within_median": float(np.median(w)),
                          "within_positive": int((w > 0).sum()),
                          "delta_median": float(np.median([r[way][f"k{k}"]["delta"] for r in rows])),
                          "c_negative": int(sum(r[way][f"k{k}"]["c_full"] < 0 for r in rows)),
                          "a_median": float(np.median([r[way][f"k{k}"]["a"] for r in rows]))}
        k0 = lambda q: np.array([r[way]["k0"][q] for r in rows])  # noqa: E731
        k3 = lambda q: np.array([r[way]["k3"][q] for r in rows])  # noqa: E731
        d["median_change_k0_k3"] = {"class_means": float(np.median(k3("class_means") - k0("class_means"))),
                                    "within": float(np.median(k3("within") - k0("within")))}
        summ[way] = d
    for way in ("est0_meas1", "est1_meas0"):
        d = summ[way]
        d["R3a_pass"] = bool(d["k3"]["cm_negative"] >= 22 and abs(d["median_change_k0_k3"]["within"]) < abs(d["median_change_k0_k3"]["class_means"]))
    ok = [r["running"] for r in rows if r["running"] is not None]
    summ["running"] = {"n": len(rows), "n_with_speed_variance": len(ok),
                       "p_perm_dir_lt_0.05": int(sum(x["p_perm_dir"] < 0.05 for x in ok)),
                       "p_F_dir_lt_0.05": int(sum(x["p_F_dir"] < 0.05 for x in ok)),
                       "eta2_dir_median": float(np.median([x["eta2_dir"] for x in ok])),
                       "eta2_dir_max": float(np.max([x["eta2_dir"] for x in ok])),
                       "eta2_null_mean_median": float(np.median([x["eta2_null_mean"] for x in ok])),
                       "r_circlin_median": float(np.median([x["r_circlin_dir"] for x in ok])),
                       "p_perm_tf_lt_0.05": int(sum(x["p_perm_tf"] < 0.05 for x in ok)),
                       "eta2_tf_median": float(np.median([x["eta2_tf"] for x in ok]))}
    summ["running"]["R3b_locked"] = bool(summ["running"]["p_perm_dir_lt_0.05"] >= 5)
    summ["r33a_alltrials_reference"] = json.load(open(HERE.parent / "data_canonical" / "cr_h1_r33a.json"))["summary"]
    print("SUMMARY", json.dumps({k: v for k, v in summ.items() if k != "r33a_alltrials_reference"}, indent=1), flush=True)
    json.dump({"registration": "H4-R3 (ex0_REGISTERED.md, 8cba0800)", "rows": rows, "summary": summ},
              open(HERE.parent / "data_canonical" / "h4_r3_heldout_modes.json", "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
