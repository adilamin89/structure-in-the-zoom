"""Run 64b v2 (S78b, 2026-09-06; v1 failed its construction check B5 on the first recording, top-mode PR 2 against 60, because gating by the rate hands the mode to the largest-amplitude member) - The assembly model: within-class fluctuation
modes localized on co-tuned groups of neurons, plus the measured heavy-tailed
independent part. The population model the diagnostics point to.

WHY: run 63 puts the direction-aligned shift in the fluctuation-pooling term
and shows that on the direction-selective third the pooled within-class second
moment follows log(k/8) rung by rung: the fluctuations of direction cells are
private to their direction class (the block-mixture prediction). Run 64a shows
what those fluctuations are in neuron space: the top within-class modes of the
data are localized on 50-120 neurons (0.3-0.9% of the population), carry
28-50% of the within-class variance in the top ten, and sit on an independent
part whose own effective dimension (280-690) is 7-19 times the full
within-class PR (22-78); the previous population model's modes were dense
patterns dominated by one to four giant-amplitude neurons, which is why its
pooling term was half the data's on the recordings it under-predicted and its
antipodal alignment too high. A mode that lives on a group of co-tuned
neurons fires with them, so it is private to the classes at which the group
fires, without any per-class construction.

MODEL: as run 60 (resampled arm): 4000 neurons, preferred directions uniform,
tuning T(theta; d) with d from the recording's DSI distribution, amplitudes
from its response ranges, class means R. Within-class variability, total
variance V set by the measured within-class correlation as before, split as
  (i) assembly modes, a fraction q of V (q = the recording's measured top-10
      within-class variance fraction): K modes, mode k supported with equal
      weights 1/sqrt(m) on the m neurons whose preferred direction lies within
      +/- w of a random center, w chosen so that m matches the recording's
      measured neuron-space participation ratio of its top within-class mode;
      each mode gated by the neuron's tuning at the trial's direction (peak
      one; v2), so the assembly fluctuates when it fires and every member
      weighs alike;
  (ii) an independent part, a fraction 1 - q of V, with per-neuron variance
      proportional to amplitude squared times a log-normal factor whose spread
      brings the model's log-variance spread to the recording's measured one.
K is chosen by bisection so that the realized within-class PR matches the
measured one (run 60b's rule). Nothing is fitted to the shift or to the
alignment. Ten seeds; the ladder, floors, alignment profile, thirds and the
run-63 decomposition (seed 1) as before.

REGISTERED EXPECTATIONS (written before the run):
B1: |predicted - observed| <= 0.06 on 8 of 8 recordings.
B2: the model's 180-degree within-class alignment lies within 0.05 of the
    measured value on 8 of 8.
B3: the model's full-population pooling term P(k) has a slope against
    log(k/8) within 0.15 of the data's on 6 of 8 (run 63's numbers).
B4: the direction-selective third shifts more than the orientation-selective
    third in 8 of 8.
B5 (construction check): the model's top-mode neuron-space PR is within a
    factor of 2 of the data's, and its diagonal-to-full PR ratio within a
    factor of 2 of the data's, on 8 of 8.
A miss is reported at full volume.

Out: ../data_canonical/run64b_assembly_model.json (+ .log)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUT = DATA / "run64b_assembly_model.json"


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


r2 = load_module("run2", "run2_calibrated_corotating.py")
r6 = load_module("run6", "run6_corotation_calibration.py")
r59 = load_module("run59", "run59_shift_by_direction_selectivity.py")
r60 = load_module("run60", "run60_mixture_model.py")
r60b = load_module("run60b", "run60b_mixture_matchedK.py")
r63 = load_module("run63", "run63_deficit_decomposition.py")
r64a = load_module("run64a", "run64a_within_class_localization.py")
NB, N_SYN, SEEDS, TOL = 8, 4000, tuple(range(1, 11)), 0.05
ANG = r60.ANG


def make_assembly(n_stim, bl, d_vec, amp_pool, rw, K, q, m_target, s_target, rng):
    N = len(d_vec)
    phi0 = rng.uniform(0, 2 * np.pi, N)
    amp = rng.choice(amp_pool, N, replace=True)
    R = amp[None, :] * r60.tuning(ANG[:, None] - phi0[None, :], d_vec[None, :])
    S = float(((R - R.mean(1, keepdims=True)) ** 2).mean())
    V = S * (1 - rw) / max(rw, 1e-3)
    # (i) assembly modes: equal weights on the m neurons nearest (in preferred direction) to a random center
    w = np.pi * m_target / N            # half-width giving m_target neurons on average for uniform phi0
    centers = rng.uniform(0, 2 * np.pi, K)
    U = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        dphi = np.abs((phi0 - centers[k] + np.pi) % (2 * np.pi) - np.pi)
        members = np.where(dphi < w)[0]
        if len(members) < 5:
            members = np.argsort(dphi)[:max(5, int(m_target))]
        U[members, k] = 1.0 / np.sqrt(len(members))
    # v2: the assembly's gate is the neuron's tuning (peak one), not its rate: an assembly fluctuates when it
    # fires, with every member weighted alike, so the mode lives on the group and not on its largest member
    Tn = r60.tuning(ANG[:, None] - phi0[None, :], d_vec[None, :])            # classes x neurons, in [0, 1]
    That = Tn / (np.sqrt((Tn ** 2).mean(axis=1, keepdims=True)) + 1e-9)     # unit mean square per class
    amp_scale = float(np.sqrt((amp ** 2).mean()))                          # one population amplitude for the modes
    g = rng.standard_normal((n_stim, K)) * np.sqrt(q * V * N / K)          # per-mode power so the class-average variance is q V
    W = (g @ U.T) * That[bl] * amp_scale / (amp_scale + 1e-12) * 1.0
    W = W * np.sqrt(q * V) / (np.sqrt((W ** 2).mean()) + 1e-12)            # renormalize the mode part to q V exactly
    # (ii) independent part with the measured log-variance spread
    s_amp = float(np.std(np.log(amp ** 2 + 1e-12)))
    s_extra = float(np.sqrt(max(s_target ** 2 - s_amp ** 2, 0.0)))
    var_j = amp ** 2 * np.exp(s_extra * rng.standard_normal(N) - s_extra ** 2 / 2)
    var_j = var_j / var_j.mean() * (1 - q) * V
    E = rng.standard_normal((n_stim, N)) * np.sqrt(var_j)[None, :]
    Xs = R[bl] + W + E
    return np.ascontiguousarray(Xs.astype(np.float32)), R


def realized(n_stim, bl, d_vec, amp, rw, K, q, m, s, seed):
    Xs, _ = make_assembly(n_stim, bl, d_vec, amp, rw, K, q, m, s, np.random.default_rng(100 * seed + 7))
    return r60.within_stats(Xs, bl, np.random.default_rng(0))[1]


def match_K(n_stim, bl, d_vec, amp, rw, q, m, s, target, lo=2, hi=400):
    f_lo = realized(n_stim, bl, d_vec, amp, rw, lo, q, m, s, 1); f_hi = realized(n_stim, bl, d_vec, amp, rw, hi, q, m, s, 1)
    trace = [(lo, f_lo), (hi, f_hi)]
    if f_hi < target:
        return hi, f_hi, trace
    if f_lo > target:
        return lo, f_lo, trace
    while hi - lo > 1:
        mid = (lo + hi) // 2; f_mid = realized(n_stim, bl, d_vec, amp, rw, mid, q, m, s, 1); trace.append((mid, f_mid))
        if abs(f_mid - target) <= TOL * target:
            return mid, f_mid, trace
        if f_mid < target:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return (hi, f_hi, trace) if abs(f_hi - target) < abs(f_lo - target) else (lo, f_lo, trace)


def main():
    t0 = time.time()
    oz = {r["name"]: r for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    names = [n for n in oz if "gratings_" in n and "static" not in n]
    loc = json.load(open(DATA / "run64a_within_class_localization.json"))["rows"]
    dec = json.load(open(DATA / "run63_deficit_decomposition.json"))["rows"]
    ks = [1, 2, 3, 4, 6, 8]; ref = np.log(np.array(ks) / 8.0)
    out = {"design": {"n_syn": N_SYN, "seeds": list(SEEDS), "tol": TOL}, "rows": {}}
    for name in names:
        tag = r63.short(name)
        bl, meas, arrays = r60.measure(name); r60.bl_global[0] = bl
        n_stim = len(bl); obs = float(oz[name]["delta"]); rw, target = meas["within_corr"], meas["within_pr"]
        ds = loc[tag]["data"]["summary"]
        q, m, s = ds["top10_var_frac_mean"], ds["pr_neuron_top1_mean"], ds["log_var_spread_mean"]
        d1 = r60b.d_vector("B", meas, arrays, np.random.default_rng(107))
        K, prK, trace = match_K(n_stim, bl, d1, arrays["amp_tuned"], rw, q, m, s, target)
        print(f"[{tag}] targets: PR {target:.1f} q {q:.2f} m {m:.0f} spread {s:.2f} corr {rw:.3f} -> K {K} (realized {prK:.1f}) | obs {obs:+.3f} | {time.time()-t0:.0f}s", flush=True)
        cells = []
        for seed in SEEDS:
            rng = np.random.default_rng(100 * seed + 7)
            d_vec = r60b.d_vector("B", meas, arrays, rng)
            Xs, R = make_assembly(n_stim, bl, d_vec, arrays["amp_tuned"], rw, K, q, m, s, rng)
            d, _ = r2.ladder_delta(Xs, bl, np.random.default_rng(300 + seed))
            prof = r6.alignment_profile(Xs, bl, np.random.default_rng(400 + seed))
            rw_r, pw_r = r60.within_stats(Xs, bl, np.random.default_rng(0))
            cell = {"delta": float(d), "profile": {str(int(k)): float(v) for k, v in prof.items()}, "realized_within_corr": rw_r, "realized_within_pr": pw_r,
                    "b2_over_c1": float(r59.profile_b2_c1(R)[0]), "thirds": r60.thirds_report(Xs, R, rng)}
            if seed == 1:
                full = r63.decompose(Xs, bl, np.random.default_rng(42))
                P = [x["P"] for x in full["rungs"]]
                cell["decomposition_full"] = {"at_four": full["at_four"], "P_by_rung": P, "P_slope_vs_logk8": float(np.polyfit(ref[:-1], P[:-1], 1)[0])}
                cls = r64a.class_stats(Xs, bl, np.arange(N_SYN)); cell["localization"] = r64a.summarize(cls, N_SYN)
            cells.append(cell); del Xs
            print(f"  seed {seed:2d}: delta {d:+.3f} | PR {pw_r:.1f} corr {rw_r:.3f} | align180 {cell['profile']['180']:.3f} (meas {meas['profile']['180']:.3f}) | thirds DS/rand/OS "
                  f"{cell['thirds']['DS']['delta']:+.3f}/{cell['thirds']['random']['delta']:+.3f}/{cell['thirds']['OS']['delta']:+.3f} | {time.time()-t0:.0f}s", flush=True)
        agg = r60.mean_cells(cells, obs); agg["delta_se"] = agg["delta_sd"] / np.sqrt(len(SEEDS))
        dataP = [x["P"] for x in dec[tag]["data"]["full"]["rungs"]]
        row = {"name": name, "observed_delta": obs, "measured": meas, "targets": {"q": q, "m": m, "spread": s, "pr": target}, "K_matched": int(K), "realized_pr_seed1": prK,
               "bisection_trace": trace, "seeds": cells, "data_P_slope": float(np.polyfit(ref[:-1], dataP[:-1], 1)[0]),
               "data_localization": ds} | agg
        out["rows"][tag] = row
        c1 = cells[0]
        print(f"  => {tag}: delta {agg['delta']:+.3f} ± {agg['delta_sd']:.3f} (SE {row['delta_se']:.3f}) vs obs {obs:+.3f}, err {agg['error']:+.3f} | align180 {agg['profile']['180']:.3f} vs {meas['profile']['180']:.3f} "
              f"| P slope {c1['decomposition_full']['P_slope_vs_logk8']:.2f} vs data {row['data_P_slope']:.2f} | top-mode PR {c1['localization']['pr_neuron_top1_mean']:.0f} vs data {m:.0f} | diag/full {c1['localization']['ratio_mean']:.1f} vs data {ds['ratio_mean']:.1f} "
              f"| thirds DS/rand/OS {agg['thirds']['DS']['delta']:+.3f}/{agg['thirds']['random']['delta']:+.3f}/{agg['thirds']['OS']['delta']:+.3f}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    rows = out["rows"]; tags = list(rows)
    pred = [rows[t]["delta"] for t in tags]; obsv = [rows[t]["observed_delta"] for t in tags]
    v = {"predicted": pred, "observed": obsv, "errors": [rows[t]["error"] for t in tags], "se": [rows[t]["delta_se"] for t in tags], "rho": float(spearmanr(pred, obsv)[0]),
         "B1_count": int(sum(abs(rows[t]["error"]) <= 0.06 for t in tags)),
         "B2_count": int(sum(abs(rows[t]["profile"]["180"] - rows[t]["measured"]["profile"]["180"]) <= 0.05 for t in tags)),
         "B3_count": int(sum(abs(rows[t]["seeds"][0]["decomposition_full"]["P_slope_vs_logk8"] - rows[t]["data_P_slope"]) <= 0.15 for t in tags)),
         "B4_count": int(sum(rows[t]["thirds"]["DS"]["delta"] > rows[t]["thirds"]["OS"]["delta"] for t in tags)),
         "B5_count": int(sum((0.5 <= rows[t]["seeds"][0]["localization"]["pr_neuron_top1_mean"] / rows[t]["targets"]["m"] <= 2.0) and
                             (0.5 <= rows[t]["seeds"][0]["localization"]["ratio_mean"] / rows[t]["data_localization"]["ratio_mean"] <= 2.0) for t in tags)),
         "order_D1_D2_D3": bool(rows["D1"]["delta"] > rows["D2"]["delta"] > rows["D3"]["delta"])}
    v["B1"], v["B2"], v["B3"], v["B4"], v["B5"] = v["B1_count"] == 8, v["B2_count"] == 8, v["B3_count"] >= 6, v["B4_count"] == 8, v["B5_count"] == 8
    out["verdict"] = v
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
