"""Run 56 (S77, 2026-09-05) - Does an odd-sector admixture in the WITHIN-CLASS
pathway order the calibrated model across GT1, GT2, GT3?

Facts that motivate it (all released):
  * run48: the one-parameter alignment-calibrated model (s fitted to the
    measured principal-angle profile) gives signed errors -0.118, -0.100,
    +0.116 on GT1, GT2, GT3, monotone in the dipole share c1/b2 of each
    recording's code (0.84, 0.62, 0.43): it under-predicts in proportion to
    how much direction the code carries.
  * The model's CLASS MEANS already carry each recording's dipole (the means
    are drawn from the full kernel a + b2 cos2 + c1 cos + b4 cos4; the paper's
    old phrase "exactly pi-symmetric class means" was wrong). What follows the
    even-dominated class-mean profile is the class-dependent GAIN modulation of
    the within-class modes, so the model's within-class subspaces at opposite
    directions are more aligned than the measured ones (GT3: 0.25-0.30 model
    vs 0.19 measured; and the measured 180-degree alignment falls with the
    dipole share: 0.19, 0.14, 0.10 on GT3, GT2, GT1).
  * run13 (GT3 only): giving the gain pathway an odd-harmonic admixture gamma
    (modulation pattern = normalize(M + gamma D), D a pure cos(dphi) field)
    lowers the 180-degree alignment and admits delta = +0.24 near the minimum
    profile error, along a (s, gamma) ridge the four-point profile does not
    resolve.

DESIGN: run13's two-parameter model generalized per recording (run48's
per-recording coefficients, within-class correlation and PR, and measured
4-point alignment profile, all read from run48's artifact so the calibration
is identical); grid s in {0.5, 0.65, 0.8, 0.9, 1.0} x gamma in {0, 0.5, 1,
2}, three seeds, 4000 synthetic neurons; (s*, gamma*) = least squares to the
measured profile; prediction delta(s*, gamma*) against the observed
direction-aligned shift. Ridge = all cells within 1.5x the minimum error;
their delta range is reported. Tied variant: kappa = gamma*(GT3) / share(GT3);
gamma_tied(rec) = kappa * share(rec) added as an extra gamma column for GT1
and GT2, s fitted per recording, so GT1 and GT2 carry no free odd parameter.

REGISTERED EXPECTATIONS (written before the run):
R1 (free fit): gamma* orders with the dipole share, gamma*(GT1) >= gamma*(GT2)
    >= gamma*(GT3) with at least one strict inequality: the odd admixture the
    profile demands grows with the code's direction content.
R2 (free fit): the predicted shifts order GT1 > GT2 > GT3 (observed 0.347 >
    0.280 > 0.240) and |error| <= 0.06 on all three at the minimum-error cell.
R3 (tied gamma): the same ordering with s fitted per recording, and error
    magnitudes on GT1 and GT2 below the one-parameter values (0.118, 0.100).
R4: the model's 180-degree alignment at the best fit is within 0.05 of the
    measured value on all three (the over-recovery removed).
RIDGE RULE: if the ridge's delta range spans the observed value on a
    recording, the profile does not discriminate there; R2/R3 are judged at
    the minimum-error cell and the range is stated. A miss is reported at
    full volume.

Out: ../data_canonical/run56_odd_gain_across_recordings.json (+ .log)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = DATA / "run56_odd_gain_across_recordings.json"

spec = importlib.util.spec_from_file_location("run2", HERE / "run2_calibrated_corotating.py")
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
spec6 = importlib.util.spec_from_file_location("run6", HERE / "run6_corotation_calibration.py")
r6 = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(r6)
N_BINS = r2.N_BINS
S_VALS = [0.5, 0.65, 0.8, 0.9, 1.0]
G_VALS = [0.0, 0.5, 1.0, 2.0]
SEEDS = (1, 2, 3)
N_SYN = 4000
RIDGE_FACTOR = 1.5


def labels(name):
    dat = np.load(RAW / f"{name}.npy", allow_pickle=True).item()
    istim = np.asarray(dat["istim"], float)
    n_stim = np.asarray(dat["sresp"]).shape[1]
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
    del dat
    return n_stim, bl


def make_two_param(n_stim, n_neur, bl, r_within, pr_within, s, gamma, coef, seed=1):
    """run13's model with the recording's own kernel: class means from
    a + b2 cos2 + c1 cos + b4 cos4 (they carry the dipole); within-class
    variability = K shared low-rank modes modulated per class by
    normalize(M + gamma D), D a pure dipole field; s interpolates shared
    (0) to class-modulated (1)."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, N_BINS, endpoint=False)
    dphi = np.abs(ang[:, None] - ang[None, :])
    a0, b2, c1, b4 = coef["a"], coef["b2"], coef["c1"], coef["b4"]
    Cm = a0 + b2 * np.cos(2 * dphi) + c1 * np.cos(dphi) + b4 * np.cos(4 * dphi)
    np.fill_diagonal(Cm, a0 + b2 + c1 + b4)
    Cm = Cm / Cm[0, 0]
    w, V = np.linalg.eigh(Cm); L = V @ np.diag(np.sqrt(np.maximum(w, 0)))
    M = L @ rng.standard_normal((N_BINS, n_neur))
    Cd = 0.5 + 0.5 * np.cos(dphi)
    wd, Vd = np.linalg.eigh(Cd); Ld = Vd @ np.diag(np.sqrt(np.maximum(wd, 0)))
    D = Ld @ rng.standard_normal((N_BINS, n_neur))
    G = M + gamma * D
    Ghat = G / (np.sqrt((G ** 2).mean(axis=1, keepdims=True)) + 1e-9)
    s2 = float((M ** 2).mean()); sig2_total = s2 * (1 - r_within) / max(r_within, 1e-3)
    K = max(int(round(pr_within)), 2); frac_iso = 0.1
    U = np.linalg.qr(rng.standard_normal((n_neur, K)))[0]
    g = rng.standard_normal((n_stim, K)) * np.sqrt(sig2_total * (1 - frac_iso) * n_neur / K)
    W = (g @ U.T) * ((1 - s) + s * Ghat[bl])
    Xs = M[bl] + W + rng.standard_normal((n_stim, n_neur)) * np.sqrt(sig2_total * frac_iso)
    return np.ascontiguousarray(Xs.astype(np.float32))


def run_cell(n_stim, bl, rw, pw, s, gamma, coef, measured):
    ds, profs = [], []
    for seed in SEEDS:
        Xs = make_two_param(n_stim, N_SYN, bl, rw, pw, s, gamma, coef, seed=seed)
        d, _ = r2.ladder_delta(Xs, bl, np.random.default_rng(300 + seed)); ds.append(d)
        profs.append(r6.alignment_profile(Xs, bl, np.random.default_rng(400 + seed)))
        del Xs
    prof = {sep: float(np.mean([p[sep] for p in profs])) for sep in profs[0]}
    err = float(np.mean([(prof[sep] - measured[sep]) ** 2 for sep in (45, 90, 135, 180)]))
    return {"delta": float(np.mean(ds)), "delta_sd": float(np.std(ds)),
            "profile": {str(k): v for k, v in prof.items()}, "sq_err": err}


def summarize(cells, obs, measured):
    best_key = min(cells, key=lambda k: cells[k]["sq_err"]); best = cells[best_key]
    emin = best["sq_err"]
    ridge = {k: v for k, v in cells.items() if v["sq_err"] <= RIDGE_FACTOR * emin}
    dr = [v["delta"] for v in ridge.values()]
    s_star, g_star = (float(x) for x in best_key.split("_"))
    return {"s_star": s_star, "gamma_star": g_star, "delta_pred": best["delta"], "error": best["delta"] - obs,
            "model_180": best["profile"]["180"], "measured_180": measured[180],
            "ridge_cells": sorted(ridge), "ridge_delta_min": min(dr), "ridge_delta_max": max(dr),
            "ridge_spans_observed": bool(min(dr) <= obs <= max(dr))}


def main():
    t0 = time.time()
    r48 = json.load(open(DATA / "run48_overshoot_across_recordings.json"))["rows"]
    out = {"design": {"S": S_VALS, "G": G_VALS, "seeds": list(SEEDS), "n_syn": N_SYN, "ridge_factor": RIDGE_FACTOR}, "rows": {}}
    order = ["GT3", "GT1", "GT2"]  # GT3 first so kappa exists for the tied columns
    kappa = None
    for tag in order:
        r = r48[tag]; coef = r["coef"]; share = float(r["odd_share_c1_over_b2"])
        rw, pw, obs = float(r["within_corr"]), float(r["within_pr"]), float(r["observed_delta"])
        measured = {int(k): float(v) for k, v in r["measured_profile"].items()}
        n_stim, bl = labels(r["name"])
        print(f"[{tag}] n_stim {n_stim} share {share:.2f} corr {rw:.3f} PR {pw:.1f} obs {obs:+.3f} "
              f"measured 45/90/135/180 {measured[45]:.3f}/{measured[90]:.3f}/{measured[135]:.3f}/{measured[180]:.3f}", flush=True)
        cells = {}
        gammas = list(G_VALS)
        if kappa is not None:
            gammas.append(round(kappa * share, 3))
        for gamma in gammas:
            for s in S_VALS:
                c = run_cell(n_stim, bl, rw, pw, s, gamma, coef, measured)
                cells[f"{s}_{gamma}"] = c
                print(f"  s={s:.2f} g={gamma:.3f}: delta {c['delta']:+.3f}±{c['delta_sd']:.3f} | "
                      f"align {c['profile']['45']:.3f}/{c['profile']['90']:.3f}/{c['profile']['135']:.3f}/{c['profile']['180']:.3f} "
                      f"| err {c['sq_err']:.5f} | {time.time()-t0:.0f}s", flush=True)
        free = summarize({k: v for k, v in cells.items() if float(k.split('_')[1]) in G_VALS}, obs, measured)
        row = {"name": r["name"], "share": share, "observed_delta": obs, "one_param_error_run48": float(r["delta_pred"]) - obs,
               "cells": cells, "free": free}
        if kappa is None:
            kappa = free["gamma_star"] / share
            row["kappa_from_gt3"] = kappa
        else:
            gt = round(kappa * share, 3)
            tied = {k: v for k, v in cells.items() if float(k.split('_')[1]) == gt}
            row["tied"] = summarize(tied, obs, measured) | {"gamma_tied": gt}
        out["rows"][tag] = row
        print(f"  => FREE s*={free['s_star']} gamma*={free['gamma_star']} delta_pred {free['delta_pred']:+.3f} vs obs {obs:+.3f} "
              f"(error {free['error']:+.3f}; run48 one-param {row['one_param_error_run48']:+.3f}); 180 model {free['model_180']:.3f} "
              f"vs measured {measured[180]:.3f}; ridge delta [{free['ridge_delta_min']:+.3f}, {free['ridge_delta_max']:+.3f}] "
              f"spans obs: {free['ridge_spans_observed']}", flush=True)
        if "tied" in row:
            t = row["tied"]; print(f"  => TIED gamma={t['gamma_tied']} s*={t['s_star']} delta_pred {t['delta_pred']:+.3f} (error {t['error']:+.3f})", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    rows = out["rows"]; byshare = sorted(rows, key=lambda t: -rows[t]["share"])  # GT1, GT2, GT3
    gs = [rows[t]["free"]["gamma_star"] for t in byshare]; dp = [rows[t]["free"]["delta_pred"] for t in byshare]
    er = [rows[t]["free"]["error"] for t in byshare]; a180 = [abs(rows[t]["free"]["model_180"] - rows[t]["free"]["measured_180"]) for t in byshare]
    tied_ok = all("tied" in rows[t] for t in ("GT1", "GT2"))
    verdict = {"share_order": byshare, "gamma_star": gs, "delta_pred_free": dp, "error_free": er,
               "R1_gamma_orders_with_share": bool(gs[0] >= gs[1] >= gs[2] and (gs[0] > gs[1] or gs[1] > gs[2])),
               "R2_pred_orders_GT1_GT2_GT3": bool(dp[0] > dp[1] > dp[2]), "R2_all_errors_le_0.06": bool(all(abs(e) <= 0.06 for e in er)),
               "R4_180_within_0.05_all": bool(all(a <= 0.05 for a in a180)), "abs_180_gap": a180,
               "ridge_spans_observed": {t: rows[t]["free"]["ridge_spans_observed"] for t in byshare}}
    if tied_ok:
        dt = [rows["GT1"]["tied"]["delta_pred"], rows["GT2"]["tied"]["delta_pred"], rows["GT3"]["free"]["delta_pred"]]
        et = [rows["GT1"]["tied"]["error"], rows["GT2"]["tied"]["error"]]
        verdict["R3_tied_orders"] = bool(dt[0] > dt[1] > dt[2])
        verdict["R3_tied_errors_below_one_param"] = bool(abs(et[0]) < 0.118 and abs(et[1]) < 0.100)
        verdict["delta_pred_tied"] = dt; verdict["error_tied"] = et; verdict["kappa"] = rows["GT3"]["kappa_from_gt3"]
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
