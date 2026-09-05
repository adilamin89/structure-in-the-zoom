"""Run 60b (S78, 2026-09-05) - Run 60's mixture model at matched within-class
dimensionality and ten seeds.

WHY: run 60 (three seeds) found that the resampled-DSI mixture (arm B) lands
within 0.07 of the observed direction-aligned shift on all eight grating
recordings with no fitted parameter, orders the three drifting recordings
(D1 > D2 > D3) and puts the localized recordings at the top, while the
two-species idealization (arm A, one direction-selectivity value per species)
over-predicts GT3 by 0.27. Two things weaken that verdict: the seed-to-seed
spread (SD up to 0.17 on one recording at three seeds), and the model's
realized within-class participation ratio, which fell below the measured
target on every recording (13-23 against 18-46; multiplying shared gain modes
by the rate concentrates them on the neurons that fire), so the calibration
target was not met and, by the sign rule of Section 7, a too-low within-class
dimensionality inflates the shift.

DESIGN: run 60's arms A and B at s = 1 with (i) the number of gain modes K
chosen per recording and arm by bisection so that the REALIZED within-class PR
(run 48's estimator on the synthetic data, seed 1) matches the measured value
to within 5%, and (ii) ten seeds. Everything else as in run 60 (measured f,
DSI thirds, amplitudes, within-class correlation, the same ladder, floors and
per-third subsets). Reported per recording and arm: mean, SD and standard
error over seeds; K_matched against round(PR_measured); the realized PR; the
180-degree alignment; the per-third shifts. One arm per process
(`--arm A|B`).

REGISTERED EXPECTATIONS (written before the run):
K1: matching K lowers the predicted shift relative to run 60's three-seed
    mean on every recording (run 60's realized PR fell below the target on all
    eight), by the most on the two low-contrast recordings (realized 16-23
    against measured 38-46).
K2 (the claim, arm B at matched K, ten seeds): |predicted - observed| <= 0.06
    on 8 of 8 recordings (run 56's criterion), D1 > D2 > D3, Spearman
    rho(predicted, observed) >= 0.7 over the eight, and the two largest
    predictions are localized recordings.
K3: arm A's over-prediction of D3 persists (error > +0.15 at matched K).
K4: the direction-selective third's shift exceeds the orientation-selective
    third's in 8 of 8 recordings (arm B).
A miss is reported at full volume.

Out: ../data_canonical/run60b_mixture_matchedK_arm{A,B}.json (+ .log)
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
spec = importlib.util.spec_from_file_location("run60", HERE / "run60_mixture_model.py")
r60 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r60)
r2, r6, r59 = r60.r2, r60.r6, r60.r59
SEEDS = tuple(range(1, 11))
N_SYN = r60.N_SYN
TOL = 0.05


def d_vector(arm, meas, arrays, rng):
    if arm == "A":
        d_DS, d_OS = float(r60.d_from_dsi(meas["median_dsi_DS_third"])), float(r60.d_from_dsi(meas["median_dsi_OS_third"]))
        n_ds = int(round(meas["f_ds"] * N_SYN))
        d_vec = np.concatenate([np.full(n_ds, d_DS), np.full(N_SYN - n_ds, d_OS)]); rng.shuffle(d_vec)
        return d_vec
    return r60.d_from_dsi(rng.choice(arrays["dsi_tuned"], N_SYN, replace=True))


def realized_pr(n_stim, bl, d_vec, amp, rw, K, seed):
    Xs, _ = r60.make_mixture(n_stim, bl, d_vec, amp, rw, K, 1.0, np.random.default_rng(100 * seed + 7))
    return r60.within_stats(Xs, bl, np.random.default_rng(0))[1]


def match_K(n_stim, bl, d_vec, amp, rw, target, seed=1, lo=2, hi=400):
    """Smallest K whose realized within-class PR is within TOL of the target
    (the realized PR grows with K); bisection on the integer K."""
    trace = []
    f_lo, f_hi = realized_pr(n_stim, bl, d_vec, amp, rw, lo, seed), realized_pr(n_stim, bl, d_vec, amp, rw, hi, seed)
    trace += [(lo, f_lo), (hi, f_hi)]
    if f_hi < target:
        return hi, f_hi, trace
    while hi - lo > 1:
        mid = (lo + hi) // 2
        f_mid = realized_pr(n_stim, bl, d_vec, amp, rw, mid, seed); trace.append((mid, f_mid))
        if abs(f_mid - target) <= TOL * target:
            return mid, f_mid, trace
        if f_mid < target:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return (hi, f_hi, trace) if abs(f_hi - target) < abs(f_lo - target) else (lo, f_lo, trace)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["A", "B"], required=True); a = ap.parse_args()
    arm = a.arm; OUT = DATA / f"run60b_mixture_matchedK_arm{arm}.json"
    t0 = time.time()
    r60_art = json.load(open(DATA / "run60_mixture_model.json"))["rows"]
    oz = {r["name"]: r for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    names = [n for n in oz if "gratings_" in n and "static" not in n]
    out = {"design": {"arm": arm, "seeds": list(SEEDS), "n_syn": N_SYN, "s": 1.0, "tol": TOL}, "rows": {}}
    for name in names:
        tag = r60.short(name)
        bl, meas, arrays = r60.measure(name); r60.bl_global[0] = bl
        n_stim = len(bl); obs = float(oz[name]["delta"])
        rw, target = meas["within_corr"], meas["within_pr"]
        d1 = d_vector(arm, meas, arrays, np.random.default_rng(107))
        K, pr_at_K, trace = match_K(n_stim, bl, d1, arrays["amp_tuned"], rw, target)
        run60_key = "A_two_species_s1.0" if arm == "A" else "B_resampled_s1.0"
        print(f"[{tag}] target PR {target:.1f} (round {int(round(target))}) -> K {K} realized {pr_at_K:.1f} | run60 delta {r60_art[tag]['arms'][run60_key]['delta']:+.3f} "
              f"(realized PR {r60_art[tag]['arms'][run60_key]['realized_within_pr']:.1f}) | obs {obs:+.3f} | {time.time()-t0:.0f}s", flush=True)
        cells = []
        for seed in SEEDS:
            rng = np.random.default_rng(100 * seed + 7)
            d_vec = d_vector(arm, meas, arrays, rng)
            Xs, R = r60.make_mixture(n_stim, bl, d_vec, arrays["amp_tuned"], rw, K, 1.0, rng)
            d, _ = r2.ladder_delta(Xs, bl, np.random.default_rng(300 + seed))
            prof = r6.alignment_profile(Xs, bl, np.random.default_rng(400 + seed))
            rw_r, pw_r = r60.within_stats(Xs, bl, np.random.default_rng(0))
            cell = {"delta": float(d), "profile": {str(int(k)): float(v) for k, v in prof.items()}, "realized_within_corr": rw_r,
                    "realized_within_pr": pw_r, "b2_over_c1": float(r59.profile_b2_c1(R)[0]), "thirds": r60.thirds_report(Xs, R, rng)}
            cells.append(cell); del Xs
            print(f"  seed {seed:2d}: delta {d:+.3f} | PR {pw_r:.1f} | align180 {cell['profile']['180']:.3f} | thirds DS/rand/OS "
                  f"{cell['thirds']['DS']['delta']:+.3f}/{cell['thirds']['random']['delta']:+.3f}/{cell['thirds']['OS']['delta']:+.3f} | {time.time()-t0:.0f}s", flush=True)
        agg = r60.mean_cells(cells, obs)
        agg["delta_se"] = agg["delta_sd"] / np.sqrt(len(SEEDS))
        row = {"name": name, "observed_delta": obs, "measured": {k: v for k, v in meas.items()}, "K_matched": int(K), "K_round_target": int(round(target)),
               "realized_pr_at_K_seed1": pr_at_K, "bisection_trace": trace, "run60_delta": r60_art[tag]["arms"][run60_key]["delta"],
               "run60_realized_pr": r60_art[tag]["arms"][run60_key]["realized_within_pr"], "seeds": cells} | agg
        out["rows"][tag] = row
        print(f"  => {tag}: delta {agg['delta']:+.3f} ± {agg['delta_sd']:.3f} (SE {row['delta_se']:.3f}) vs obs {obs:+.3f}, err {agg['error']:+.3f}; run60 {row['run60_delta']:+.3f} | "
              f"align180 {agg['profile']['180']:.3f} (meas {meas['profile']['180']:.3f}) | thirds DS/rand/OS {agg['thirds']['DS']['delta']:+.3f}/{agg['thirds']['random']['delta']:+.3f}/{agg['thirds']['OS']['delta']:+.3f}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    rows = out["rows"]; tags = list(rows); drift = [t for t in tags if t.startswith("D")]
    pred = [rows[t]["delta"] for t in tags]; obsv = [rows[t]["observed_delta"] for t in tags]
    lowered = [rows[t]["delta"] < rows[t]["run60_delta"] for t in tags]
    drop = {t: rows[t]["run60_delta"] - rows[t]["delta"] for t in tags}
    top2 = sorted(tags, key=lambda t: -rows[t]["delta"])[:2]
    verdict = {"arm": arm, "predicted": pred, "observed": obsv, "errors": [rows[t]["error"] for t in tags], "se": [rows[t]["delta_se"] for t in tags],
               "K_matched": {t: rows[t]["K_matched"] for t in tags},
               "K1_lowered_count": int(sum(lowered)), "K1_drop": drop,
               "K1": bool(all(lowered) and sorted(tags, key=lambda t: -drop[t])[:2] == sorted(["C1", "C2"], key=lambda t: -drop[t])),
               "K2_errors_le_0.06_count": int(sum(abs(rows[t]["error"]) <= 0.06 for t in tags)),
               "K2_orders_D1_D2_D3": bool(rows["D1"]["delta"] > rows["D2"]["delta"] > rows["D3"]["delta"]),
               "K2_rho": float(spearmanr(pred, obsv)[0]), "K2_top2": top2}
    verdict["K2"] = bool(verdict["K2_errors_le_0.06_count"] == 8 and verdict["K2_orders_D1_D2_D3"] and verdict["K2_rho"] >= 0.7 and all(t.startswith("L") for t in top2))
    verdict["K3_D3_error"] = rows["D3"]["error"]; verdict["K3"] = bool(rows["D3"]["error"] > 0.15)
    verdict["K4_DS_gt_OS_count"] = int(sum(rows[t]["thirds"]["DS"]["delta"] > rows[t]["thirds"]["OS"]["delta"] for t in tags)); verdict["K4"] = verdict["K4_DS_gt_OS_count"] == 8
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
