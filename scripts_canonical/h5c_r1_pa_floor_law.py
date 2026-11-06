"""H5c-R1 (Oct 26; registration in data_canonical/ex0_REGISTERED.md; ledger J15): the analytic principal-angle floor law
(h5c_floor_law.py) against H5b-R4b's permutation floors, and the ceiling-normalized structure part (H5b-R7) predicted from class
covariances. Same recordings, n, draws and neurons as h5b_r4_cov_ladder.py (its rng stream replayed; the observed pa_5 asserted
against its JSON). Per draw:
  law_A  total covariance T's spectrum + Gamma gain with the deconvolved radius kurtosis   (spectrum + gain kurtosis)
  law_E  T's spectrum + the empirical r^2 distribution (64 quantiles) as the gain           (counterpart of R4b 'ell')
  law_T  T's spectrum, no gain (R4b 'g_T');   law_W  pooled within-class W's spectrum, no gain (R4b 'g_W')
  s_pred = 1 - O_obs / O_floor: O_obs from each class's complement covariance and gain (Tr M_c M_d / k over pairs), O_floor from
         8 pseudo-classes drawn from the pooled complements with the classes' complement sizes (matched estimation noise)
Out: data_canonical/h5c_r1_pa_floor_law.json"""
import importlib.util
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
OUT = HERE.parent / "data_canonical" / "h5c_r1_pa_floor_law.json"
NS = (8, 16, 32, 64, 128, 256)
N_DRAW, N_NEUR, K, GRID = 5, 256, 5, 2000


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gain_and_spectrum(Xc):
    """Eigen-decomposition of the covariance of centred rows Xc, the deconvolved gain kurtosis and the r^2 quantiles."""
    C = Xc.T @ Xc / (len(Xc) - 1)
    lam, V = np.linalg.eigh(C)
    lam = np.clip(lam, 0, None)
    FL = sys.modules["h5c_floor_law"]
    kap, deff = FL.kappa_gain(Xc, lam)
    r2 = np.sum(Xc * Xc, axis=1)
    q = np.quantile(r2 / r2.mean(), (np.arange(64) + 0.5) / 64)
    return lam, V, kap, deff, q


def M_of(Xc, n):
    FL = sys.modules["h5c_floor_law"]
    lam, V, kap, _, _ = gain_and_spectrum(Xc)
    m, info = FL.projector_weights(lam, n, K, ("gamma", kap), n_grid=GRID)
    return V, m, kap, info["k"]


def run_one(fp):
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import h5c_floor_law as FL  # noqa: F401
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    R5 = _load("h4r5", "h4_r5_power.py")
    r4 = {(x["n"], x["draw"]): x for x in json.load(open(HERE.parent / "data_canonical" / "h5b_r4_cov_ladder.json"))["rows"]
          if x["name"] == fp.stem}
    dat = np.load(fp, allow_pickle=True).item()
    Xf = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    Xf /= Xf.std() + 1e-9
    yf = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    del dat
    rows = []
    for n in NS:
        for d in range(N_DRAW):
            rng = np.random.default_rng(30000 * n + d)                         # H5b-R4's stream, replayed
            tr = np.concatenate([rng.choice(np.where(yf == c)[0], n, replace=False) for c in range(8)])
            cols = rng.choice(Xf.shape[1], N_NEUR, replace=False)
            X = Xf[np.ix_(tr, cols)].astype(np.float64); y = yf[tr]
            obs = R5.cov_stats(X, y)["pa_5"]
            assert abs(obs - r4[(n, d)]["pa_5"]["obs"]) < 1e-9, (fp.stem, n, d)
            Xa = Xf[:, cols].astype(np.float64)
            Ta = Xa - Xa.mean(0)
            lamT, _, kapT, deffT, qT = gain_and_spectrum(Ta)
            Ra = Xa - np.stack([Xa[yf == c].mean(0) for c in range(8)])[yf]
            lamW = np.clip(np.linalg.eigvalsh(Ra.T @ Ra / (len(Ra) - 8)), 0, None)
            law = {"law_A": FL.floor_pa(lamT, n, K, ("gamma", kapT), n_grid=GRID)[0],
                   "law_E": FL.floor_pa(lamT, n, K, qT, n_grid=GRID)[0],
                   "law_T": FL.floor_pa(lamT, n, K, "one", n_grid=GRID)[0],
                   "law_W": FL.floor_pa(lamW, n, K, "one", n_grid=GRID)[0]}
            # ceiling-normalized structure part: class complements and matched pseudo-classes
            insel = np.zeros(len(yf), bool); insel[tr] = True
            comp = [np.where((yf == c) & ~insel)[0] for c in range(8)]
            Mc = []
            for c in range(8):
                Z = Xa[comp[c]]; V, m, kap, kk = M_of(Z - Z.mean(0), n)
                Mc.append((V, m))
            r2 = np.random.default_rng(90000 * n + d)
            pool = np.concatenate(comp)
            Mp = []
            for c in range(8):
                Z = Xa[r2.choice(pool, len(comp[c]), replace=False)]; V, m, _, _ = M_of(Z - Z.mean(0), n)
                Mp.append((V, m))
            kk = min(K, n - 1)
            O_obs = FL.overlap_labelled(Mc) / kk
            O_fl = FL.overlap_labelled(Mp) / kk
            rows.append({"name": fp.stem, "n": n, "draw": d, "obs": float(obs), "perm": r4[(n, d)]["pa_5"]["floor_perm"],
                         **{k: float(v) for k, v in law.items()}, "kappa_T": kapT, "deff_T": deffT,
                         "O_obs_pred": O_obs, "O_floor_pred": O_fl, "s_pred": 1 - O_obs / O_fl,
                         "s_pred_unmatched": 1 - O_obs / (1 - law["law_A"]),
                         "comp_min": int(min(len(x) for x in comp))})
        r = [x for x in rows if x["n"] == n]
        print(fp.stem[:30], f"n {n:3d} | perm {np.mean([x['perm'] for x in r]):.3f} A {np.mean([x['law_A'] for x in r]):.3f} "
              f"E {np.mean([x['law_E'] for x in r]):.3f} T {np.mean([x['law_T'] for x in r]):.3f} W {np.mean([x['law_W'] for x in r]):.3f} "
              f"kappa {np.mean([x['kappa_T'] for x in r]):.2f} | s_pred {np.mean([x['s_pred'] for x in r]):.3f}", flush=True)
    return rows


def score(rows):
    r4b = json.load(open(HERE.parent / "data_canonical" / "h5b_r4b_floor_cause.json"))["score"]["cells"]
    cells = {}
    for x in rows:
        cells.setdefault((x["name"], x["n"]), []).append(x)
    keys = ("obs", "perm", "law_A", "law_E", "law_T", "law_W", "s_pred", "s_pred_unmatched", "kappa_T")
    agg = {k: {q: float(np.mean([x[q] for x in v])) for q in keys} for k, v in cells.items()}
    for (nm, n), c in agg.items():
        c["s_meas"] = (c["obs"] - c["perm"]) / (1 - c["perm"])
        sim = r4b[f"{nm}|{n}"]
        c.update({"sim_g_T": sim["g_T"], "sim_g_W": sim["g_W"], "sim_ell": sim["ell"]})
    C = list(agg.values())
    rel = lambda a, b: float(np.median([abs(c[a] - c[b]) / abs(c[b]) for c in C]))  # noqa: E731
    v = {"P0_law_T_vs_g_T": rel("law_T", "sim_g_T"), "P0_law_W_vs_g_W": rel("law_W", "sim_g_W"), "P0_law_E_vs_ell": rel("law_E", "sim_ell"),
         "law_A_relerr": rel("law_A", "perm"), "law_A_pearson": float(np.corrcoef([c["law_A"] for c in C], [c["perm"] for c in C])[0, 1]),
         "law_E_relerr": rel("law_E", "perm"), "law_T_relerr": rel("law_T", "perm"), "law_W_relerr": rel("law_W", "perm"),
         "A_minus_E_over_perm": float(np.median([abs(c["law_A"] - c["law_E"]) / c["perm"] for c in C])),
         "relerr_A_by_n": {str(n): float(np.median([abs(c["law_A"] - c["perm"]) / c["perm"] for (nm, nn), c in agg.items() if nn == n])) for n in NS},
         "signed_A_by_n": {str(n): float(np.median([(c["law_A"] - c["perm"]) / c["perm"] for (nm, nn), c in agg.items() if nn == n])) for n in NS},
         "kappa_T_median": float(np.median([c["kappa_T"] for c in C]))}
    sp, sm = np.array([c["s_pred"] for c in C]), np.array([c["s_meas"] for c in C])
    v["P4_pearson"] = float(np.corrcoef(sp, sm)[0, 1]); v["P4_median_abs_err"] = float(np.median(np.abs(sp - sm)))
    v["P4_spearman"] = float(stats.spearmanr(sp, sm)[0])
    su = np.array([c["s_pred_unmatched"] for c in C])
    v["unmatched_pearson"] = float(np.corrcoef(su, sm)[0, 1]); v["unmatched_median_abs_err"] = float(np.median(np.abs(su - sm)))
    flat = {}
    for nm in sorted({k[0] for k in agg}):
        p = np.array([agg[(nm, n)]["s_pred"] for n in NS]); m = np.array([agg[(nm, n)]["s_meas"] for n in NS])
        flat[nm] = {"cv_pred": float(p.std() / p.mean()), "cv_meas": float(m.std() / m.mean()),
                    "agree": bool((p.std() / p.mean() <= 0.25) == (m.std() / m.mean() <= 0.25))}
    v["P5_flat_calls"] = flat; v["P5_agree"] = int(sum(f["agree"] for f in flat.values()))
    v["predictions"] = {"P0": all(v[k] <= 0.03 for k in ("P0_law_T_vs_g_T", "P0_law_W_vs_g_W", "P0_law_E_vs_ell")),
                        "P1": v["law_A_relerr"] <= 0.05 and v["law_A_pearson"] >= 0.98,
                        "P2": v["law_T_relerr"] >= 3 * v["law_A_relerr"], "P3": v["A_minus_E_over_perm"] <= 0.02,
                        "P4": v["P4_pearson"] >= 0.8 and v["P4_median_abs_err"] <= 0.05, "P5": v["P5_agree"] >= 6}
    v["cells"] = {f"{k[0]}|{k[1]}": c for k, c in agg.items()}
    return v


def main():
    files = sorted(RAW.glob("gratings*.npy"))
    with ProcessPoolExecutor(8, mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(run_one, files) for x in part]
    out = {"registration": "H5c-R1 (ex0_REGISTERED.md)", "rows": rows, "score": score(rows)}
    print("SCORE", json.dumps({k: v for k, v in out["score"].items() if k != "cells"}, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
