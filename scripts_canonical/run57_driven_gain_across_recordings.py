"""Run 57 (S77, 2026-09-05) - The gain field without its baseline: does the
symmetry-allowed model order GT1, GT2, GT3?

PHYSICS (written before the run). Decompose the class mean m_j = mu0 + t_j
into its untuned (l = 0) part mu0 and its tuned part t_j, whose kernel is
b2 cos 2dphi + c1 cos dphi + b4 cos 4dphi (l = 2, 4 even under the antipodal
Z2 = the four-class shift, l = 1 odd). The calibrated model modulates K shared
within-class modes by a gain field h_j; its within-class subspace is
span{U_k * h_j}, so the alignment between classes is a monotone function of
the gain field's kernel C_h(dphi). The l = 0 component of h is inert (the
same vector for every class; it rotates nothing) and caps the rotation. With
h_j = m_j (the current model) C_h(pi) = (a + b2 - c1 + b4)/(a + b2 + c1 + b4)
= 0.70 / 0.65 / 0.59 on GT3 / GT2 / GT1; with h_j = t_j (gain multiplies the
driven response, not the baseline) C_h(pi) = (b2 - c1 + b4)/(b2 + c1 + b4) =
0.48 / 0.34 / 0.22, a drop that GROWS with the dipole share c1/b2, i.e. the
correction is largest where run48's one-parameter model under-predicts most
(GT1 -0.118, GT2 -0.100, GT3 +0.116), and the measured 180-degree alignment
falls the same way (0.19, 0.14, 0.10). Run 13's odd field D was an ad hoc
route to the same effect; the symmetry-allowed term is the tuned response
itself and its only coupling constant is the odd share.

DESIGN: run48's per-recording calibration (coefficients, within-class
correlation and PR, measured 4-point alignment profile, observed shift read
from run48's artifact); gain field h_j = lambda * mu0 + t_j with mu0 the
class-averaged mean vector and t_j = m_j - mu0; lambda in {0, 0.5, 1}
(lambda = 1 reproduces run48's model up to the normalization); s in {0.5,
0.65, 0.8, 0.9, 1.0}; three seeds; 4000 synthetic neurons; s* = least squares
to the measured profile at each lambda; the ridge (cells within 1.5x the
minimum error) reported.

REGISTERED EXPECTATIONS (before the run):
D1: at lambda = 0 the model's 180-degree alignment at s* is within 0.05 of
    the measured value on all three recordings (the over-recovery removed
    without an odd field), and it falls with the dipole share as measured.
D2: at lambda = 0 the predicted shifts order GT1 > GT2 > GT3 (observed
    0.347 > 0.280 > 0.240) and |error| <= 0.06 on all three at s*.
D3: the improvement over lambda = 1 is monotone in the dipole share (the
    error on GT1 shrinks most, GT3 least or reverses least).
D4: lambda* (the lambda with the smallest profile error) is 0 on every
    recording.
RIDGE RULE as in run 56. A miss is reported at full volume.

Out: ../data_canonical/run57_driven_gain_across_recordings.json (+ .log)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = DATA / "run57_driven_gain_across_recordings.json"
spec = importlib.util.spec_from_file_location("run2", HERE / "run2_calibrated_corotating.py")
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
spec6 = importlib.util.spec_from_file_location("run6", HERE / "run6_corotation_calibration.py")
r6 = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(r6)
N_BINS = r2.N_BINS
S_VALS = [0.5, 0.65, 0.8, 0.9, 1.0]
LAMBDAS = [0.0, 0.5, 1.0]
SEEDS = (1, 2, 3)
N_SYN = 4000
RIDGE_FACTOR = 1.5


def labels(name):
    dat = np.load(RAW / f"{name}.npy", allow_pickle=True).item()
    istim = np.asarray(dat["istim"], float); n_stim = np.asarray(dat["sresp"]).shape[1]
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
    del dat
    return n_stim, bl


def make_driven(n_stim, n_neur, bl, r_within, pr_within, s, lam, coef, seed=1):
    """Class means from the recording's kernel (they carry the dipole);
    gain field h_j = lam * mu0 + t_j, mu0 = class-averaged mean, t_j = m_j - mu0;
    within-class variability = K shared modes modulated by normalize(h_j),
    s interpolating shared (0) to modulated (1)."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, N_BINS, endpoint=False)
    dphi = np.abs(ang[:, None] - ang[None, :])
    a0, b2, c1, b4 = coef["a"], coef["b2"], coef["c1"], coef["b4"]
    Cm = a0 + b2 * np.cos(2 * dphi) + c1 * np.cos(dphi) + b4 * np.cos(4 * dphi)
    np.fill_diagonal(Cm, a0 + b2 + c1 + b4)
    Cm = Cm / Cm[0, 0]
    w, V = np.linalg.eigh(Cm); L = V @ np.diag(np.sqrt(np.maximum(w, 0)))
    M = L @ rng.standard_normal((N_BINS, n_neur))
    mu0 = M.mean(axis=0, keepdims=True)
    H = lam * mu0 + (M - mu0)
    Hhat = H / (np.sqrt((H ** 2).mean(axis=1, keepdims=True)) + 1e-9)
    s2 = float((M ** 2).mean()); sig2_total = s2 * (1 - r_within) / max(r_within, 1e-3)
    K = max(int(round(pr_within)), 2); frac_iso = 0.1
    U = np.linalg.qr(rng.standard_normal((n_neur, K)))[0]
    g = rng.standard_normal((n_stim, K)) * np.sqrt(sig2_total * (1 - frac_iso) * n_neur / K)
    W = (g @ U.T) * ((1 - s) + s * Hhat[bl])
    Xs = M[bl] + W + rng.standard_normal((n_stim, n_neur)) * np.sqrt(sig2_total * frac_iso)
    return np.ascontiguousarray(Xs.astype(np.float32))


def run_cell(n_stim, bl, rw, pw, s, lam, coef, measured):
    ds, profs = [], []
    for seed in SEEDS:
        Xs = make_driven(n_stim, N_SYN, bl, rw, pw, s, lam, coef, seed=seed)
        d, _ = r2.ladder_delta(Xs, bl, np.random.default_rng(300 + seed)); ds.append(d)
        profs.append(r6.alignment_profile(Xs, bl, np.random.default_rng(400 + seed)))
        del Xs
    prof = {sep: float(np.mean([p[sep] for p in profs])) for sep in profs[0]}
    err = float(np.mean([(prof[sep] - measured[sep]) ** 2 for sep in (45, 90, 135, 180)]))
    return {"delta": float(np.mean(ds)), "delta_sd": float(np.std(ds)), "profile": {str(k): v for k, v in prof.items()}, "sq_err": err}


def summarize(cells, obs, measured):
    best_key = min(cells, key=lambda k: cells[k]["sq_err"]); best = cells[best_key]; emin = best["sq_err"]
    ridge = {k: v for k, v in cells.items() if v["sq_err"] <= RIDGE_FACTOR * emin}; dr = [v["delta"] for v in ridge.values()]
    return {"s_star": float(best_key), "delta_pred": best["delta"], "error": best["delta"] - obs, "sq_err": emin,
            "model_180": best["profile"]["180"], "measured_180": measured[180], "ridge_cells": sorted(ridge),
            "ridge_delta_min": min(dr), "ridge_delta_max": max(dr), "ridge_spans_observed": bool(min(dr) <= obs <= max(dr))}


def main():
    t0 = time.time()
    r48 = json.load(open(DATA / "run48_overshoot_across_recordings.json"))["rows"]
    out = {"design": {"S": S_VALS, "lambdas": LAMBDAS, "seeds": list(SEEDS), "n_syn": N_SYN, "ridge_factor": RIDGE_FACTOR}, "rows": {}}
    for tag in ["GT1", "GT2", "GT3"]:
        r = r48[tag]; coef = r["coef"]; share = float(r["odd_share_c1_over_b2"])
        rw, pw, obs = float(r["within_corr"]), float(r["within_pr"]), float(r["observed_delta"])
        measured = {int(k): float(v) for k, v in r["measured_profile"].items()}
        a0, b2, c1, b4 = coef["a"], coef["b2"], coef["c1"], coef["b4"]
        ch = {"with_baseline": (a0 + b2 - c1 + b4) / (a0 + b2 + c1 + b4), "tuned_only": (b2 - c1 + b4) / (b2 + c1 + b4)}
        n_stim, bl = labels(r["name"])
        print(f"[{tag}] share {share:.2f} C_h(180) with baseline {ch['with_baseline']:.2f} tuned-only {ch['tuned_only']:.2f} | measured 180 {measured[180]:.3f} | obs {obs:+.3f}", flush=True)
        row = {"name": r["name"], "share": share, "observed_delta": obs, "one_param_error_run48": float(r["delta_pred"]) - obs,
               "C_h_180": ch, "by_lambda": {}, "cells": {}}
        for lam in LAMBDAS:
            cells = {}
            for s in S_VALS:
                c = run_cell(n_stim, bl, rw, pw, s, lam, coef, measured); cells[str(s)] = c; row["cells"][f"{s}_{lam}"] = c
                print(f"  lam={lam:.1f} s={s:.2f}: delta {c['delta']:+.3f}±{c['delta_sd']:.3f} | align {c['profile']['45']:.3f}/{c['profile']['90']:.3f}/{c['profile']['135']:.3f}/{c['profile']['180']:.3f} | err {c['sq_err']:.5f} | {time.time()-t0:.0f}s", flush=True)
            sm = summarize(cells, obs, measured); row["by_lambda"][str(lam)] = sm
            print(f"  => lam={lam:.1f}: s*={sm['s_star']} delta_pred {sm['delta_pred']:+.3f} (error {sm['error']:+.3f}) 180 model {sm['model_180']:.3f} vs measured {measured[180]:.3f}; ridge [{sm['ridge_delta_min']:+.3f},{sm['ridge_delta_max']:+.3f}] spans obs {sm['ridge_spans_observed']}", flush=True)
        row["lambda_star"] = min(LAMBDAS, key=lambda l: row["by_lambda"][str(l)]["sq_err"])
        out["rows"][tag] = row
        json.dump(out, open(OUT, "w"), indent=1)
    rows = out["rows"]; order = ["GT1", "GT2", "GT3"]
    l0 = [rows[t]["by_lambda"]["0.0"] for t in order]; l1 = [rows[t]["by_lambda"]["1.0"] for t in order]
    gap0 = [abs(x["model_180"] - x["measured_180"]) for x in l0]
    e0 = [x["error"] for x in l0]; e1 = [x["error"] for x in l1]
    improve = [abs(b) - abs(a) for a, b in zip(e0, e1)]  # positive = lambda 0 better
    verdict = {"D1_180_within_0.05_all_lambda0": bool(all(g <= 0.05 for g in gap0)), "model_180_lambda0": [x["model_180"] for x in l0],
               "measured_180": [x["measured_180"] for x in l0],
               "D1_180_falls_with_share": bool(l0[0]["model_180"] < l0[1]["model_180"] < l0[2]["model_180"]),
               "D2_orders_lambda0": bool(l0[0]["delta_pred"] > l0[1]["delta_pred"] > l0[2]["delta_pred"]),
               "D2_errors_le_0.06_lambda0": bool(all(abs(e) <= 0.06 for e in e0)), "delta_pred_lambda0": [x["delta_pred"] for x in l0],
               "error_lambda0": e0, "error_lambda1": e1, "improvement_lambda0_over_1": improve,
               "D3_improvement_monotone_in_share": bool(improve[0] >= improve[1] >= improve[2]),
               "D4_lambda_star_is_0_all": bool(all(rows[t]["lambda_star"] == 0.0 for t in order)), "lambda_star": [rows[t]["lambda_star"] for t in order],
               "ridge_spans_observed_lambda0": [x["ridge_spans_observed"] for x in l0]}
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
