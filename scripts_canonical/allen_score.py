"""The Allen predictions scored from data_canonical/allen_sessions/session_*.json.
Every criterion's number is printed before its verdict. Out: data_canonical/allen_verdicts.json"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
R66 = DATA / "run66_allen_deficit_decomposition.json"
MP = DATA / "allen_multipoles_all_sessions.json"


def load():
    pops, qual, rep = [], [], []
    for fp in sorted((DATA / "allen_sessions").glob("session_*.json")):
        s = json.load(open(fp))
        for a, v in s["areas"].items():
            pops.append({"session": s["session"], "area": a, **v})
        if s["quality_VISp"]:
            qual.append({"session": s["session"], "area": "VISp", **s["quality_VISp"]})
        rep += [(s["session"], a, v["recomputed"], v["stored"]) for a, v in s["reproduce_stored"].items()]
    return pops, qual, rep


def mean_seeds(v, key, part):
    return float(np.mean([r[part] for r in v[key]]))


def order_mean(v, part):
    return float(np.mean([o[part] for o in v["order_tfres"]]))


def law_neg(sg):
    return sg["ratio"] > sg["threshold"]


def score_block(P, label, verdict, printv):
    n = len(P); out = {}
    # R3 = P1
    dpos = sum(v["tfres"]["delta"] > 0 for v in P)
    out["P1_tfres_delta_pos"] = f"{dpos}/{n}"
    if "perm_tfres" in P[0]:
        out["p_le_005_tfres"] = f"{sum(v['perm_tfres']['p'] <= 0.05 for v in P)}/{n}"
        out["p_le_005_pooled"] = f"{sum(v['perm_pooled']['p'] <= 0.05 for v in P)}/{n}"
        out["p_min_tfres"] = f"{sum(v['perm_tfres']['p'] <= 1 / 201 + 1e-12 for v in P)}/{n}"
    # R4 = P2
    pw = [mean_seeds(v, "resid_pool", "within") for v in P]; pdl = [mean_seeds(v, "resid_pool", "delta") for v in P]
    c1 = sum(abs(w) <= 0.03 for w in pw); c2 = sum(d < v["tfres"]["delta"] for d, v in zip(pdl, P))
    c12 = sum(abs(w) <= 0.03 and d < v["tfres"]["delta"] for w, d, v in zip(pw, pdl, P))
    out["P2_pool_within_abs_le_003"] = f"{c1}/{n}"; out["P2_pool_delta_below_tfres"] = f"{c2}/{n}"; out["P2_both"] = f"{c12}/{n}"
    out["P2_pool_delta_negative"] = f"{sum(d < 0 for d in pdl)}/{n}"
    out["P2_pool_within_median"] = float(np.median(pw)); out["P2_pool_delta_median"] = float(np.median(pdl))
    # R5 = P3
    sdl = [mean_seeds(v, "resid_shuf", "delta") for v in P]
    out["P3_shuf_below_tfres"] = f"{sum(d < v['tfres']['delta'] for d, v in zip(sdl, P))}/{n}"
    out["P3_shuf_negative"] = f"{sum(d < 0 for d in sdl)}/{n}"; out["P3_shuf_delta_median"] = float(np.median(sdl))
    # R6 = P4
    for frame, sk, lk in (("tfres", "sign_tfres", "tfres"), ("pooled", "sign_pooled", "pooled"), ("tfres_z", "sign_tfres_z", "tfres_z")):
        cmn = [v[lk]["class_means"] < 0 for v in P]
        lawn = [law_neg(v[sk]) for v in P]; cfn = [v[sk]["c_full"] < 0 for v in P]
        out[f"P4_{frame}_classmean_positive"] = f"{sum(not c for c in cmn)}/{n}"
        out[f"P4_{frame}_law_agrees"] = f"{sum(a == b for a, b in zip(cmn, lawn))}/{n}"
        out[f"P4_{frame}_cfull_agrees"] = f"{sum(a == b for a, b in zip(cmn, cfn))}/{n}"
        out[f"P4_{frame}_PR_W_range"] = [float(np.min([v[sk]["PR_W"] for v in P])), float(np.max([v[sk]["PR_W"] for v in P]))]
        out[f"P4_{frame}_PR_B_range"] = [float(np.min([v[sk]["PR_B"] for v in P])), float(np.max([v[sk]["PR_B"] for v in P]))]
        out[f"P4_{frame}_ratio_range"] = [float(np.min([v[sk]["ratio"] for v in P])), float(np.max([v[sk]["ratio"] for v in P]))]
        out[f"P4_{frame}_threshold_range"] = [float(np.min([v[sk]["threshold"] for v in P])), float(np.max([v[sk]["threshold"] for v in P]))]
        out[f"P4_{frame}_trB_trW_range"] = [float(np.min([v[sk]["trB_trW"] for v in P])), float(np.max([v[sk]["trB_trW"] for v in P]))]
    # R7 = P5
    mf = lambda s: np.log(8 / (1 + 7 * s)) / np.log(8)
    pred = [mf(v["sigma_noise"]) for v in P]; wit = [v["tfres"]["within"] for v in P]
    out["P5a_same_data_r"] = float(stats.pearsonr(pred, wit)[0]); out["P5a_mae"] = float(np.mean(np.abs(np.array(pred) - wit)))
    ph = [np.mean([h["pred"] for h in v["sigma_heldout"]]) for v in P]; wh = [np.mean([h["within"] for h in v["sigma_heldout"]]) for v in P]
    out["P5b_heldout_r"] = float(stats.pearsonr(ph, wh)[0]); out["P5b_heldout_mae"] = float(np.mean(np.abs(np.array(ph) - wh)))
    out["sigma_noise_range"] = [float(np.min([v["sigma_noise"] for v in P])), float(np.max([v["sigma_noise"] for v in P]))]
    out["sigma_raw_range"] = [float(np.min([v["sigma_raw"] for v in P])), float(np.max([v["sigma_raw"] for v in P]))]
    # R8 = P6
    el = [v for v in P if v["ds_thirds_tfres"]]
    for k in ("sel0_meas1", "sel1_meas0"):
        out[f"P6_{k}"] = f"{sum(v['ds_thirds_tfres'][k]['DS'] > v['ds_thirds_tfres'][k]['nonDS'] for v in el)}/{len(el)}"
    # R9 = P7
    dK = np.array([v["dK_tfres"] for v in P]); oew = np.array([order_mean(v, "within") for v in P])
    out["P7_sign_match"] = f"{int(np.sum(np.sign(oew) == -np.sign(dK)))}/{n}"; out["P7_r"] = float(stats.pearsonr(dK, oew)[0])
    out["P7_dK_negative"] = f"{int((dK < 0).sum())}/{n}"
    out["order_total_mean"] = float(np.mean([order_mean(v, "delta") for v in P]))
    out["order_within_mean"] = float(oew.mean()); out["order_classmeans_mean"] = float(np.mean([order_mean(v, "class_means") for v in P]))
    # R10 = P8
    for a in ("pooled", "tfres"):
        ra = [v[f"random_axis_{a}"] for v in P]
        out[f"P8_{a}_mean"] = float(np.mean(ra)); out[f"P8_{a}_sd"] = float(np.std(ra))
    # R2
    out["tf_share_reliable_median"] = float(np.median([v["tf_share_reliable"] for v in P]))
    out["tf_share_reliable_range"] = [float(np.min([v["tf_share_reliable"] for v in P])), float(np.max([v["tf_share_reliable"] for v in P]))]
    out["tfres_minus_pooled_abs_le_005"] = f"{sum(abs(v['tfres']['delta'] - v['pooled']['delta']) <= 0.05 for v in P)}/{n}"
    for arm in ("pooled", "tfres", "tfstrat", "tfres_z"):
        for part in ("delta", "within", "class_means"):
            x = [v[arm][part] for v in P]
            out[f"{arm}_{part}_mean"] = float(np.mean(x)); out[f"{arm}_{part}_sd"] = float(np.std(x))
            out[f"{arm}_{part}_pos"] = f"{sum(t > 0 for t in x)}/{n}"
    if printv:
        print(f"===== {label} (n = {n})")
        for k, v in out.items():
            print(f"  {k}: {v}")
    verdict[label] = out
    return out


def p9(pops):
    import pandas as pd
    import statsmodels.formula.api as smf
    mp = {(s["session"], a): v for s in json.load(open(MP))["results"] for a, v in s.get("areas", {}).items()}
    rows = [{"session": v["session"], "area": v["area"], "delta": v["tfres"]["delta"], "ds": v["ds_fraction"],
             "quad": mp[(v["session"], v["area"])]["b_quadrupole"], "logN": np.log(v["N"])} for v in pops
            if np.isfinite(v["ds_fraction"]) and (v["session"], v["area"]) in mp]
    df = pd.DataFrame(rows)
    m = smf.mixedlm("delta ~ ds + quad + C(area) + logN", df, groups=df["session"]).fit(reml=True)
    o = smf.ols("delta ~ ds + quad + C(area) + logN", df).fit(cov_type="cluster", cov_kwds={"groups": df["session"]})
    rx = smf.ols("ds ~ quad + C(area) + logN", df).fit().resid; ry = smf.ols("delta ~ quad + C(area) + logN", df).fit().resid
    pr = stats.pearsonr(rx, ry)
    res = {"n": len(df), "mixed_coef": float(m.params["ds"]), "mixed_se": float(m.bse["ds"]), "mixed_p": float(m.pvalues["ds"]),
           "ols_cluster_coef": float(o.params["ds"]), "ols_cluster_p": float(o.pvalues["ds"]), "partial_r": float(pr[0]),
           "partial_p": float(pr[1]), "raw_r": float(stats.pearsonr(df["ds"], df["delta"])[0]),
           "ds_range": [float(df["ds"].min()), float(df["ds"].max())]}
    res["pass"] = bool(res["mixed_coef"] > 0 and res["mixed_p"] < 0.05)
    return res


def r1(pops, rep):
    d = [abs(a - b) for _, _, a, b in rep]
    stored = {(s, a): b for s, a, _, b in rep}
    eng = [abs(v["pooled"]["delta"] - stored[(v["session"], v["area"])]) for v in pops]
    r66 = {(s["session"], a): v for s in json.load(open(R66))["results"] if s.get("status") == "ok" for a, v in s["areas"].items()}
    dD, sumchk = [], []
    for v in pops:
        q = r66.get((v["session"], v["area"]))
        if q:
            f = q["delta_split_free"]; dD.append(abs(f["D"] - v["pooled"]["within"]))
            sumchk.append(abs(f["D"] + f["Tb"] + f["Cb"] + f["Cx"] - q["delta"]))
    return {"exact_max_absdiff": float(max(d)), "n": len(d), "engine_vs_stored_median": float(np.median(eng)),
            "engine_vs_stored_max": float(max(eng)), "run66_D_vs_engine_within_median": float(np.median(dD)),
            "run66_D_vs_engine_within_max": float(max(dD)), "run66_sum_check_max": float(max(sumchk)), "n_run66": len(dD)}


def main():
    pops, qual, rep = load()
    print(f"{len(pops)} populations, {len({v['session'] for v in pops})} sessions, quality VISp {len(qual)}")
    verdict = {"R1": r1(pops, rep)}
    print("R1", verdict["R1"])
    visp = [v for v in pops if v["area"] == "VISp"]
    score_block(visp, "VISp", verdict, True)
    score_block(pops, "all", verdict, True)
    if qual:
        score_block(qual, "quality_VISp", verdict, True)
        verdict["quality_VISp"]["N_range"] = [min(v["N"] for v in qual), max(v["N"] for v in qual)]
    verdict["P9"] = p9(pops); print("P9", verdict["P9"])
    # B1d: Allen VISp PR_W range per frame for the imaging comparison
    verdict["VISp_N_range"] = [min(v["N"] for v in visp), max(v["N"] for v in visp)]
    json.dump(verdict, open(DATA / "allen_verdicts.json", "w"), indent=1)
    print("wrote allen_verdicts.json")


if __name__ == "__main__":
    main()
