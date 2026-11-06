"""Do the standard covariance tools carry the floor logic the shift
makes explicit? Imaging, the 8 grating recordings, input as covariance_tools_power (globally standardized, 8 direction bins), 256 random neurons,
5 draws per trial count n per class in {8, 16, 32, 64, 128, 256}. Statistics covariance_tools_power.cov_stats: principal-angle dissimilarity
(pa_5) and Frobenius heterogeneity (frob). Per cell: observed (true labels); permutation floor = mean over 20 label permutations of the
same trials; Gaussian floor = the statistic on 8 classes of n draws from N(0, W_pool), W_pool the pooled within-class covariance of the
same 256 neurons over all the recording's trials (20 draws). Exponent split per recording: slopes over log n of log(observed) and
log(floor).
Out: data_canonical/covariance_floors_trial_ladder.json"""
import importlib.util
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
OUT = HERE.parent / "data_canonical" / "covariance_floors_trial_ladder.json"
NS = (8, 16, 32, 64, 128, 256)
N_DRAW, N_NEUR, N_PERM, N_GAUSS = 5, 256, 20, 20
KEYS = ("pa_5", "frob")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_one(fp):
    warnings.filterwarnings("ignore")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    R5 = _load("h4r5", "covariance_tools_power.py")
    dat = np.load(fp, allow_pickle=True).item()
    Xf = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    Xf /= Xf.std() + 1e-9
    yf = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    del dat
    rows = []
    for n in NS:
        for d in range(N_DRAW):
            rng = np.random.default_rng(30000 * n + d)
            tr = np.concatenate([rng.choice(np.where(yf == c)[0], n, replace=False) for c in range(8)])
            cols = rng.choice(Xf.shape[1], N_NEUR, replace=False)
            X = Xf[np.ix_(tr, cols)].astype(np.float64); y = yf[tr]
            Xa = Xf[:, cols].astype(np.float64)
            Ra = Xa - np.stack([Xa[yf == c].mean(0) for c in range(8)])[yf]
            lam, V = np.linalg.eigh(Ra.T @ Ra / (len(Ra) - 8))
            A = V * np.sqrt(np.clip(lam, 0, None))                                  # W_pool = A A^T
            obs = R5.cov_stats(X, y)
            perm = [R5.cov_stats(X, y[rng.permutation(len(y))]) for _ in range(N_PERM)]
            ygs = np.repeat(np.arange(8), n)
            gau = [R5.cov_stats(rng.standard_normal((8 * n, N_NEUR)) @ A.T, ygs) for _ in range(N_GAUSS)]
            rows.append({"name": fp.stem, "n": n, "draw": d,
                         **{k: {"obs": obs[k], "floor_perm": float(np.mean([p[k] for p in perm])),
                                "floor_perm_sd": float(np.std([p[k] for p in perm])),
                                "floor_gauss": float(np.mean([g[k] for g in gau]))} for k in KEYS}})
        r = [x for x in rows if x["n"] == n]
        print(fp.stem[:30], f"n {n:3d} | " + " | ".join(
            f"{k} obs {np.mean([x[k]['obs'] for x in r]):.4g} perm {np.mean([x[k]['floor_perm'] for x in r]):.4g} "
            f"gauss {np.mean([x[k]['floor_gauss'] for x in r]):.4g}" for k in KEYS), flush=True)
    return rows


def score(rows):
    names = sorted({r["name"] for r in rows})
    mean = lambda nm, n, k, f: float(np.mean([r[k][f] for r in rows if r["name"] == nm and r["n"] == n]))  # noqa: E731
    v = {}
    mf = json.load(open(HERE.parent / "data_canonical" / "meanfree_shift_exact.json"))["rows"]
    shift = {r["name"]: np.mean([r["S1_half0"]["meanfree"]["delta"], r["S1_half1"]["meanfree"]["delta"]]) for r in mf}
    for k in KEYS:
        per = {}
        for nm in names:
            raw = np.array([mean(nm, n, k, "obs") for n in NS]); fl = np.array([mean(nm, n, k, "floor_perm") for n in NS])
            cor = raw - fl
            ln = np.log(NS)
            th_raw = float(np.polyfit(ln, np.log(raw), 1)[0]); th_fl = float(np.polyfit(ln, np.log(fl), 1)[0])
            per[nm] = {"raw": raw.tolist(), "floor_perm": fl.tolist(), "corrected": cor.tolist(),
                       "floor_gauss": [mean(nm, n, k, "floor_gauss") for n in NS],
                       "flat": bool(abs(cor[-1] - cor[0]) < 0.5 * abs(raw[-1] - raw[0])),
                       "theta_raw": th_raw, "theta_floor": th_fl, "delta_S": th_raw - th_fl,
                       "floor_carries": bool(abs(th_fl) >= 0.7 * abs(th_raw))}
        fp = np.array([c for nm in names for c in per[nm]["floor_perm"]]); fg = np.array([c for nm in names for c in per[nm]["floor_gauss"]])
        rel = np.abs(fg - fp) / np.abs(fp)
        cor256 = [per[nm]["corrected"][-1] for nm in names]
        v[k] = {"per_recording": per, "P1_flat": int(sum(per[nm]["flat"] for nm in names)),
                "P2_gauss_relerr_median": float(np.median(rel)), "P2_gauss_pearson": float(np.corrcoef(fp, fg)[0, 1]),
                "P3_floor_carries": int(sum(per[nm]["floor_carries"] for nm in names)),
                "P4_spearman_corrected256_vs_meanfree_shift": float(stats.spearmanr(cor256, [shift[nm] for nm in names])[0]),
                "floor_share_at_n": {str(n): float(np.median([per[nm]["floor_perm"][i] / per[nm]["raw"][i] for nm in names])) for i, n in enumerate(NS)}}
    pa = v["pa_5"]
    v["predictions"] = {"P1": pa["P1_flat"] >= 6,
                        "P2_pa_5": pa["P2_gauss_relerr_median"] <= 0.10 and pa["P2_gauss_pearson"] >= 0.9,
                        "P2_frob": v["frob"]["P2_gauss_relerr_median"] <= 0.10 and v["frob"]["P2_gauss_pearson"] >= 0.9,
                        "P3": pa["P3_floor_carries"] >= 6, "P4": pa["P4_spearman_corrected256_vs_meanfree_shift"] >= 0.6}
    return v


def main():
    files = sorted(RAW.glob("gratings*.npy"))
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(run_one, files) for x in part]
    out = {"rows": rows, "score": score(rows)}
    print("SCORE", json.dumps({k: (v if k == "predictions" else {a: b for a, b in v.items() if a != "per_recording"}) for k, v in out["score"].items()}, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
