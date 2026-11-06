"""Why the Gaussian floor of the principal-angle dissimilarity
(pa_5) sits below its permutation floor in covariance_floors_trial_ladder.py. Same recordings, neurons, trials, n and random stream as covariance_floors_trial_ladder.py (the
permutation floor and the within-covariance Gaussian floor are replayed and checked against its JSON). Candidate floors, 20 draws each:
  g_W   Gaussian, pooled within-class covariance of the 256 neurons over all trials (= covariance_floors_trial_ladder's)
  g_T   Gaussian, total covariance of all trials (class-mean spread included)
  boot  8n real trials drawn at random from the whole recording, random labels (the exact label-blind distribution)
  ell   g_T with each sample scaled by a per-trial radius factor resampled from the real trials (radial non-Gaussianity)
Out: data_canonical/principal_angle_floor_cause.json"""
import importlib.util
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
KEY = sys.argv[1] if len(sys.argv) > 1 else "pa_5"   # "frob": the Frobenius version (same draws, arms and seeds)
OUT = HERE.parent / "data_canonical" / ("principal_angle_floor_cause.json" if KEY == "pa_5" else "frobenius_floor_cause.json")
NS = (8, 16, 32, 64, 128, 256)
N_DRAW, N_NEUR, N_PERM, N_G = 5, 256, 20, 20


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sqrtm_psd(C):
    lam, V = np.linalg.eigh(C)
    return V * np.sqrt(np.clip(lam, 0, None))


def run_one(fp):
    warnings.filterwarnings("ignore")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    R5 = _load("h4r5", "covariance_tools_power.py")
    dat = np.load(fp, allow_pickle=True).item()
    Xf = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    Xf /= Xf.std() + 1e-9
    yf = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    del dat
    pa = lambda X, y: R5.cov_stats(X, y)[KEY]  # noqa: E731
    rows = []
    for n in NS:
        for d in range(N_DRAW):
            rng = np.random.default_rng(30000 * n + d)                         # covariance_floors_trial_ladder's stream, replayed
            tr = np.concatenate([rng.choice(np.where(yf == c)[0], n, replace=False) for c in range(8)])
            cols = rng.choice(Xf.shape[1], N_NEUR, replace=False)
            X = Xf[np.ix_(tr, cols)].astype(np.float64); y = yf[tr]
            Xa = Xf[:, cols].astype(np.float64)
            Ra = Xa - np.stack([Xa[yf == c].mean(0) for c in range(8)])[yf]
            lamW, VW = np.linalg.eigh(Ra.T @ Ra / (len(Ra) - 8))
            AW = VW * np.sqrt(np.clip(lamW, 0, None))
            R5.cov_stats(X, y)                                                  # observed (not stored; consumes nothing)
            perm = [pa(X, y[rng.permutation(len(y))]) for _ in range(N_PERM)]
            ygs = np.repeat(np.arange(8), n)
            gW = [pa(rng.standard_normal((8 * n, N_NEUR)) @ AW.T, ygs) for _ in range(N_G)]
            r2 = np.random.default_rng(70000 * n + d)                           # new arms: their own stream
            Ta = Xa - Xa.mean(0)
            AT = sqrtm_psd(Ta.T @ Ta / (len(Ta) - 1))
            rad = np.linalg.norm(Ta, axis=1) / np.sqrt(np.sum(Ta * Ta) / len(Ta))
            gT = [pa(r2.standard_normal((8 * n, N_NEUR)) @ AT.T, ygs) for _ in range(N_G)]
            boot = [pa(Xa[r2.choice(len(Xa), 8 * n, replace=False)], ygs) for _ in range(N_G)]
            ell = []
            for _ in range(N_G):
                Z = r2.standard_normal((8 * n, N_NEUR))
                Z /= np.linalg.norm(Z, axis=1, keepdims=True) / np.sqrt(N_NEUR)    # unit-radius directions, Gaussian shape kept
                ell.append(pa((Z * r2.choice(rad, 8 * n)[:, None]) @ AT.T, ygs))
            rows.append({"name": fp.stem, "n": n, "draw": d, "perm": float(np.mean(perm)), "g_W": float(np.mean(gW)),
                         "g_T": float(np.mean(gT)), "boot": float(np.mean(boot)), "ell": float(np.mean(ell))})
        r = [x for x in rows if x["n"] == n]
        print(fp.stem[:30], f"n {n:3d} | " + " ".join(f"{k} {np.mean([x[k] for x in r]):.3f}" for k in ("perm", "g_W", "g_T", "boot", "ell")), flush=True)
    return rows


def score(rows):
    r4 = json.load(open(HERE.parent / "data_canonical" / "covariance_floors_trial_ladder.json"))["rows"]
    ref = {(x["name"], x["n"], x["draw"]): x[KEY] for x in r4}
    chk = max(max(abs(x["perm"] - ref[(x["name"], x["n"], x["draw"])]["floor_perm"]),
                  abs(x["g_W"] - ref[(x["name"], x["n"], x["draw"])]["floor_gauss"])) for x in rows)
    cells = {}
    for x in rows:
        cells.setdefault((x["name"], x["n"]), []).append(x)
    agg = {k: {q: float(np.mean([x[q] for x in v])) for q in ("perm", "g_W", "g_T", "boot", "ell")} for k, v in cells.items()}
    gap = lambda c, q: (c[q] - c["g_W"]) / (c["perm"] - c["g_W"])  # noqa: E731
    rel = lambda c, q: abs(c[q] - c["perm"]) / c["perm"]  # noqa: E731
    v = {"pipeline_max_abs_diff_vs_r4": chk,
         "gT_closes_gap_median": float(np.median([gap(c, "g_T") for c in agg.values()])),
         "boot_relerr_median": float(np.median([rel(c, "boot") for c in agg.values()])),
         "gT_relerr_median": float(np.median([rel(c, "g_T") for c in agg.values()])),
         "gW_relerr_median": float(np.median([rel(c, "g_W") for c in agg.values()])),
         "ell_moves_remaining_median": float(np.median([(c["ell"] - c["g_T"]) / (c["perm"] - c["g_T"]) for c in agg.values()
                                                       if abs(c["perm"] - c["g_T"]) > 1e-3])),
         "ell_relerr_median": float(np.median([rel(c, "ell") for c in agg.values()])),
         "cells": {f"{k[0]}|{k[1]}": c for k, c in agg.items()}}
    v["predictions"] = {"P1": v["gT_closes_gap_median"] >= 0.7, "P2": v["boot_relerr_median"] <= 0.05,
                        "P3": v["ell_moves_remaining_median"] <= 0.3}
    if KEY != "pa_5":   # the Frobenius version's scoring
        v["ell_closes_gap_median"] = float(np.median([gap(c, "ell") for c in agg.values()]))
        cause = "g_T" if v["gT_closes_gap_median"] > v["ell_closes_gap_median"] else "ell"
        v["predictions"] = {"P1_boot_within_25pct": v["boot_relerr_median"] <= 0.25, "P2_cause": cause,
                            "P2_cause_closes_half": max(v["gT_closes_gap_median"], v["ell_closes_gap_median"]) >= 0.5}
    return v


def main():
    files = sorted(RAW.glob("gratings*.npy"))
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        rows = [x for part in ex.map(run_one, files) for x in part]
    out = {"statistic": KEY,
           "rows": rows, "score": score(rows)}
    print("SCORE", json.dumps({k: v for k, v in out["score"].items() if k != "cells"}, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
