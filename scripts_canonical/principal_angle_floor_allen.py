"""Held-out test of the principal-angle floor law (principal_angle_floor_law)
on Allen Neuropixels, the 167 populations of allen_covariance_floors_decoders.py (masking path, variant B; raw counts, all units, eight
direction classes). Per population: C = covariance of all presentations (labels ignored); gain kurtosis as h5c_r1 (radius kurtosis over
1 + 2/d_eff); M at each permuted class's own size n_c; floor = 1 - mean over class pairs of sum_a m_a(n_c) m_a(n_d) / k,
k = min(5, min n_c - 1) as covariance_tools_power.cov_stats. Arms law_A (spectrum + gain kurtosis), law_T (spectrum alone).
Target: the 50-permutation floor in data_canonical/allen_covariance_floors_decoders.json.
Out: data_canonical/principal_angle_floor_allen.json"""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "principal_angle_floor_allen.json"
K, GRID = 5, 2000


def law(lam, sizes, gain):
    import principal_angle_floor_law as FL
    kk = min(K, min(sizes) - 1)
    m = {n: FL.projector_weights(lam, n, kk, gain, n_grid=GRID)[0] for n in sorted(set(sizes))}
    ov = [m[sizes[c]] @ m[sizes[d]] / kk for c in range(8) for d in range(c + 1, 8)]
    return float(1 - np.mean(ov))


def run_session(s):
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import allen_engine as E
    import principal_angle_floor_law as FL
    S = E.load_session(s)
    out = []
    for area, idx in E.populations(S).items():
        P = E.pop_view(S, area, idx)
        X = P["X"].astype(np.float64); y = P["lab"]
        sizes = [int((y == c).sum()) for c in range(8)]
        Xc = X - X.mean(0)
        lam = np.clip(np.linalg.eigvalsh(Xc.T @ Xc / (len(Xc) - 1)), 0, None)
        kap, deff = FL.kappa_gain(Xc, lam)
        out.append({"session": s, "area": str(area), "N": int(X.shape[1]), "sizes": sizes, "kappa": kap, "deff": deff,
                    "law_A": law(lam, sizes, ("gamma", kap)), "law_T": law(lam, sizes, "one")})
        print(s, area, f"N {X.shape[1]} n {min(sizes)}-{max(sizes)} kappa {kap:.2f} | A {out[-1]['law_A']:.3f} T {out[-1]['law_T']:.3f}", flush=True)
    return out


def main():
    sess = [x["session"] for x in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
    with ProcessPoolExecutor(8, mp_context=get_context("spawn")) as ex:
        rows = [r for part in ex.map(run_session, sess) for r in part]
    r13 = {(r["session"], r["area"]): r for r in json.load(open(HERE.parent / "data_canonical" / "allen_covariance_floors_decoders.json"))["rows"]}
    for r in rows:
        t = r13[(r["session"], r["area"])]
        assert t["N"] == r["N"], (r["session"], r["area"])
        r["floor_perm"] = t["r13"]["pa_5"]["floor"]; r["floor_perm_sd"] = t["r13"]["pa_5"]["floor_sd"]; r["obs"] = t["r13"]["pa_5"]["obs"]
    fp = np.array([r["floor_perm"] for r in rows]); A = np.array([r["law_A"] for r in rows]); T = np.array([r["law_T"] for r in rows])
    obs = np.array([r["obs"] for r in rows])
    v = {"n": len(rows), "law_A_relerr_median": float(np.median(np.abs(A - fp) / fp)), "law_A_pearson": float(np.corrcoef(A, fp)[0, 1]),
         "law_A_signed_median": float(np.median((A - fp) / fp)), "law_A_relerr_p90": float(np.percentile(np.abs(A - fp) / fp, 90)),
         "law_T_relerr_median": float(np.median(np.abs(T - fp) / fp)), "law_T_pearson": float(np.corrcoef(T, fp)[0, 1]),
         "share_measured_median": float(np.median(fp / obs)), "share_law_median": float(np.median(A / obs)),
         "share_spearman": float(stats.spearmanr(A / obs, fp / obs)[0]), "kappa_median": float(np.median([r["kappa"] for r in rows])),
         "perm_sd_over_floor_median": float(np.median([r["floor_perm_sd"] / r["floor_perm"] for r in rows]))}
    v["predictions"] = {"P1": v["law_A_relerr_median"] <= 0.05 and v["law_A_pearson"] >= 0.9,
                        "P2": v["law_T_relerr_median"] >= 3 * v["law_A_relerr_median"],
                        "P3": abs(v["share_law_median"] - v["share_measured_median"]) <= 0.05}
    print("SUMMARY", json.dumps(v, indent=1), flush=True)
    json.dump({"rows": rows, "summary": v}, open(OUT, "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
