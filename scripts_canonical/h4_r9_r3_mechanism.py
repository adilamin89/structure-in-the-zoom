"""H4-R9 (Oct 25; registration H4-R9 in data_canonical/ex0_REGISTERED.md, commit a2c751a8): R3's statistics and mechanism.
Same sessions, halves, per-half TF residualization and held-out modes as h4_r3_heldout_modes.py (both ways).
(a) n per class vs PR(W) on the measured half (k = 0, 3); the pooled-fluctuation control (cr_c8_identification.pool_residuals,
    seeds 7..16) on the measured half's input: the class-mean part net of the control's mean.
(b) running regressed out, held out: loadings b_A of each unit's residual on centred speed (half A); on half B the component
    (speed_B - mean) x b_A removed ('regress'), or the residuals projected off b_A / |b_A| ('project'); class means kept.
(c) one-factor attribution of c = log[(1+beta)^2/(1+beta^2 r+2 a beta)], r = PR(W)/PR(B): each factor alone from k = 3.
(d) session bootstrap (10,000) of the median change k0 -> k3 of the within and class-mean parts; Wilcoxon signed-rank.
Out: data_canonical/h4_r9_r3_mechanism.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402

spec = importlib.util.spec_from_file_location("ci", HERE / "cr_c8_identification.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
spec = importlib.util.spec_from_file_location("r3", HERE / "h4_r3_heldout_modes.py")
R3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R3)
RUN = HERE.parent / "raw" / "allen_running"
SEEDS = tuple(range(7, 17))


def cterm(beta, r, a):
    return float(np.log((1 + beta) ** 2 / (1 + beta ** 2 * r + 2 * a * beta)))


def factors(Z, lab):
    sg = E.sign_law(Z, lab)
    return {"beta": sg["trB_trW"], "r": sg["PR_W"] / sg["PR_B"], "a": R3.alignment(sg), "c_full": sg["c_full"], "PR_W": sg["PR_W"]}


def inputs(P, V, k):
    Vk = V[:, -k:] if k else np.zeros((P["R"].shape[1], 0))
    return P["M"][P["lab"]] + P["R"] - (P["R"] @ Vk) @ Vk.T


def measured_with_control(Z, lab):
    lad = E.ladder(Z, lab)
    ctl = [E.ladder(ci.pool_residuals(Z, lab, np.random.default_rng(s)), lab) for s in SEEDS]
    cm_ctl = float(np.mean([c["class_means"] for c in ctl]))
    return {**lad, "ctl_class_means": cm_ctl, "ctl_within": float(np.mean([c["within"] for c in ctl])),
            "net_class_means": lad["class_means"] - cm_ctl}


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
        sp = V["speed"].astype(np.float64)
        P = [R3.half_parts(X, lab, cell, h, 0), R3.half_parts(X, lab, cell, h, 1)]
        row = {"session": sid, "N": int(X.shape[1])}
        for est, meas in ((0, 1), (1, 0)):
            Pm, Pe = P[meas], P[est]
            n_per_class = float(np.bincount(Pm["lab"]).mean())
            w = {}
            for k in (0, 3):
                Z = inputs(Pm, Pe["V"], k)
                w[f"k{k}"] = {**measured_with_control(Z, Pm["lab"]), **factors(Z, Pm["lab"])}
                w[f"k{k}"]["n_over_PRW"] = n_per_class / w[f"k{k}"]["PR_W"]
            spA = sp[h == est] - sp[h == est].mean()
            spB = sp[h == meas] - sp[h == meas].mean()
            bA = (spA @ Pe["R"]) / max(spA @ spA, 1e-12)
            Rreg = Pm["R"] - np.outer(spB, bA)
            u = bA / max(np.linalg.norm(bA), 1e-12)
            Rproj = Pm["R"] - np.outer(Pm["R"] @ u, u)
            for name, R in (("regress", Rreg), ("project", Rproj)):
                Z = Pm["M"][Pm["lab"]] + R
                w[name] = {**E.ladder(Z, Pm["lab"]), **factors(Z, Pm["lab"])}
            f0, f3 = w["k0"], w["k3"]
            w["one_factor_c"] = {"beta": cterm(f3["beta"], f0["r"], f0["a"]), "r": cterm(f0["beta"], f3["r"], f0["a"]),
                                 "a": cterm(f0["beta"], f0["r"], f3["a"]), "k0": cterm(f0["beta"], f0["r"], f0["a"]),
                                 "k3": cterm(f3["beta"], f3["r"], f3["a"])}
            for k, f in (("k0", f0), ("k3", f3)):            # Eq. 3 identity: c from (beta, r, a) = sign_law's c_full
                assert abs(w["one_factor_c"][k] - f["c_full"]) < 1e-8, (sid, k, w["one_factor_c"][k], f["c_full"])
            w["n_per_class"] = n_per_class
            row[f"est{est}_meas{meas}"] = w
        rows.append(row)
        x = row["est0_meas1"]
        print(sid, f"n/PRW k0 {x['k0']['n_over_PRW']:.2f} | cm k0 {x['k0']['class_means']:+.3f} (net {x['k0']['net_class_means']:+.3f}) k3 "
              f"{x['k3']['class_means']:+.3f} (net {x['k3']['net_class_means']:+.3f}) | regress cm {x['regress']['class_means']:+.3f} "
              f"project cm {x['project']['class_means']:+.3f} | 1-factor c: beta {x['one_factor_c']['beta']:+.3f} r {x['one_factor_c']['r']:+.3f} "
              f"a {x['one_factor_c']['a']:+.3f} (k0 {x['one_factor_c']['k0']:+.3f} k3 {x['one_factor_c']['k3']:+.3f})", flush=True)
    summ = {}
    rng = np.random.default_rng(0)
    for way in ("est0_meas1", "est1_meas0"):
        W = [r[way] for r in rows]
        d = {"n_over_PRW_k0_range": [min(w["k0"]["n_over_PRW"] for w in W), max(w["k0"]["n_over_PRW"] for w in W)],
             "n_over_PRW_k3_range": [min(w["k3"]["n_over_PRW"] for w in W), max(w["k3"]["n_over_PRW"] for w in W)],
             "n_per_class_median": float(np.median([w["n_per_class"] for w in W])),
             "net_cm_k0_positive": int(sum(w["k0"]["net_class_means"] > 0 for w in W)),
             "net_cm_k3_negative": int(sum(w["k3"]["net_class_means"] < 0 for w in W)),
             "ctl_cm_k0_median": float(np.median([w["k0"]["ctl_class_means"] for w in W])),
             "ctl_cm_k3_median": float(np.median([w["k3"]["ctl_class_means"] for w in W])),
             "ctl_within_k0_median": float(np.median([w["k0"]["ctl_within"] for w in W])),
             "regress_cm_negative": int(sum(w["regress"]["class_means"] < 0 for w in W)),
             "project_cm_negative": int(sum(w["project"]["class_means"] < 0 for w in W)),
             "regress_a_median": float(np.median([w["regress"]["a"] for w in W])),
             "one_factor_flips": {f: int(sum(w["one_factor_c"]["k0"] >= 0 and w["one_factor_c"][f] < 0 for w in W)) for f in ("beta", "r", "a")},
             "c_k0_nonneg_k3_neg": int(sum(w["one_factor_c"]["k0"] >= 0 and w["one_factor_c"]["k3"] < 0 for w in W))}
        for part in ("within", "class_means"):
            ch = np.array([w["k3"][part] - w["k0"][part] for w in W])
            boots = [np.median(ch[rng.integers(0, len(ch), len(ch))]) for _ in range(10000)]
            d[f"{part}_change_median"] = float(np.median(ch))
            d[f"{part}_change_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            d[f"{part}_change_wilcoxon_p"] = float(stats.wilcoxon(ch).pvalue)
        summ[way] = d
    print("SUMMARY", json.dumps(summ, indent=1), flush=True)
    json.dump({"registration": "H4-R9 (ex0_REGISTERED.md, a2c751a8)", "rows": rows, "summary": summ},
              open(HERE.parent / "data_canonical" / "h4_r9_r3_mechanism.json", "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
