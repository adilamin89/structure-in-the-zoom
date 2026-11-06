"""H1-R41 (C11, Oct 23): the frozen two-term prediction (h2_xy_twoterm.predict, read-only) on the POSITIVE, polar-ordered XY
side: L 32, J1/J2 0.42, T 0.6/0.7/0.8/0.85, cold start, n_equil 1500, 8 ensembles x 40 chains x 5 samples. Ensembles regenerated
exactly as h2_lattice.xy_point (same seeds and calls); delta_director asserted against the H2 artifact to 1e-9. Measured parts with
H2's estimator (h2_xy_asymptote.split on the globally standardized features, director-sorted rungs, floor rng 900 + e) and per-class
PR / n_over_d from h2_xy_asymptote.class_pr_and_overlap. Out: data_canonical/h1_xy_polar_twoterm.json"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TS = (0.60, 0.70, 0.80, 0.85)
L, J1, N_ENS, CHAINS, SAMPLES, SPACING, NEQ = 32, 0.42, 8, 40, 5, 10, 1500


def one_T(T):
    import h2_lattice as h
    import h2_xy_asymptote as xa
    import h2_xy_twoterm as tt
    import nematic_polar_delta as nem
    art = json.load(open(HERE.parent / "data_canonical" / f"h2_lattice_xy_L32_T{T:.2f}_J0.42_cold1500.json"))
    assert art["start"] == "cold" and art["n_equil"] == NEQ and art["chains_per_ens"] == CHAINS
    rows = []
    for e in range(N_ENS):
        t0 = time.time(); cfgs = []
        for c in range(CHAINS):
            seed = 77_000_000 + 100_000 * int(round(J1 * 100)) + 1000 * L + 40 * e + c + int(round(T * 1000)) * 7
            x, _ = h.xy_chain_w(L, T, J1, 1.0, seed, False, NEQ, SAMPLES, SPACING)
            cfgs.append(x)
        cfg = np.concatenate(cfgs)
        _, _, psi = h.order_params(cfg)
        d_dir, sizes, counts = h.director_ladder(nem.features(cfg), psi, np.random.default_rng(1_000 + e))
        stored = art["ensembles"][e]["delta_director"]
        assert abs(d_dir - stored) < 1e-9, (T, e, d_dir, stored)
        p = tt.predict(cfg, 800 + e)
        lab = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
        F = nem.features(cfg); F = F - F.mean(0); F /= F.std()
        order = np.argsort(lab, kind="stable"); cnt = np.bincount(lab, minlength=8)
        szs = sorted({int(cnt[:k].sum()) for k in nem.BIN_COUNTS if cnt[:k].sum() >= 10})
        rng = np.random.default_rng(900 + e)
        sp = xa.split(F, lab, [order[:s] for s in szs], rng)
        b1, w1, b2, w2, xx = xa.traces_fs(F, lab)
        pr_class, prof, sigma = xa.class_pr_and_overlap(F, lab, rng)
        per = len(lab) / 8
        rows.append({"T": T, "ens": e, "per_class": per, "bin_counts": cnt.tolist(), "PR_class": pr_class, "n_over_d": per / pr_class,
                     "PR_W": w1 ** 2 / w2, "PR_B": b1 ** 2 / b2, **p, "measured_delta_director": d_dir, "measured_delta": sp["logpr"],
                     "measured_within": sp["noise"], "measured_class_mean": sp["probe"]})
        r = rows[-1]
        print(f"T {T} e{e}: pred d {r['pred_delta']:+.3f} (w {r['pred_within']:+.3f} m {r['pred_class_mean']:+.3f}) | meas d_dir {d_dir:+.3f} "
              f"split d {r['measured_delta']:+.3f} (w {r['measured_within']:+.3f} m {r['measured_class_mean']:+.3f}) | PR_W {r['PR_W']:.1f} "
              f"n/d {r['n_over_d']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    return rows


def main():
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as ex:
        rows = [r for rs in ex.map(one_T, TS) for r in rs]
    summ = {}
    for T in TS:
        rs = [r for r in rows if r["T"] == T]
        summ[f"{T:.2f}"] = {"sign_agree_delta": int(sum(np.sign(r["pred_delta"]) == np.sign(r["measured_delta_director"]) for r in rs)),
                            "sign_agree_class_mean": int(sum(np.sign(r["pred_class_mean"]) == np.sign(r["measured_class_mean"]) for r in rs)),
                            "n": len(rs), **{f"mae_{k}": float(np.mean([abs(r[f"pred_{k}"] - r[f"measured_{m}"]) for r in rs]))
                                              for k, m in (("delta", "delta_director"), ("within", "within"), ("class_mean", "class_mean"))},
                            "mean_pred_delta": float(np.mean([r["pred_delta"] for r in rs])), "mean_meas_delta": float(np.mean([r["measured_delta_director"] for r in rs])),
                            "mean_pred_within": float(np.mean([r["pred_within"] for r in rs])), "mean_meas_within": float(np.mean([r["measured_within"] for r in rs])),
                            "mean_pred_cm": float(np.mean([r["pred_class_mean"] for r in rs])), "mean_meas_cm": float(np.mean([r["measured_class_mean"] for r in rs])),
                            "mean_PR_W": float(np.mean([r["PR_W"] for r in rs])), "mean_n_over_d": float(np.mean([r["n_over_d"] for r in rs]))}
    summ["R41_pass"] = all(summ[f"{T:.2f}"]["sign_agree_delta"] >= 7 for T in TS if T <= 0.8)
    print("SUMMARY", json.dumps(summ, indent=1), flush=True)
    json.dump({"rows": rows, "summary": summ}, open(HERE.parent / "data_canonical" / "h1_xy_polar_twoterm.json", "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
