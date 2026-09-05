"""Run 60 (S78, 2026-09-05) - The mixture model: does a population made of a
direction-selective fraction and an orientation-selective fraction, with
non-negative rates and gain variability that follows the rate, predict the
direction-aligned shift across the eight grating recordings with no fitted
parameter?

WHY: run 59 located the cross-recording variation of the shift in the
direction-selective subpopulation (DS third > orientation-only third in 8 of
8; the full shift tracks the DS fraction at rho = 0.86), and runs 56-58 showed
that repairing the homogeneous calibrated model's within-class variability
does not order the three drifting recordings. The homogeneous model draws
signed Gaussian class means and modulates its gain modes by that signed
pattern, so no neuron's variability is confined to where it fires. Here the
population is heterogeneous and the rates are non-negative: a neuron's
trial-to-trial gain variability is proportional to its rate, which for a
direction-selective neuron is small on the half-circle opposite its
preferred direction.

MODEL (arms A and B share everything but how direction selectivity is
assigned): N = 4000 neurons with preferred directions uniform on the circle;
tuning T(theta; d) = |cos theta| (1 + d sgn cos theta) / (1 + d), one cosine
rectified two ways: d = 0 is a pure orientation cell (period pi), d = 1 a
half-wave rectified direction cell; amplitudes resampled from the measured
response ranges (max minus min over the eight class means) of the recording's
tuned neurons; class means R_j(phi_c) = A_j T(phi_c - phi_j; d_j) on the
recording's eight directions. Within-class variability: K shared latent gain
modes (K = the measured within-class PR) multiplied elementwise by the
neuron's rate at the trial's direction (normalized to unit mean square per
class), plus 10% isotropic noise, with the total within-class variance set by
the measured within-class trial correlation exactly as in runs 2/48/56;
s = 1 (all structured variability is rate-modulated) is the claim, s = 0.8
(the homogeneous model's fitted value) a descriptive row. No parameter is
fitted to the shift or to the alignment profile.
  Arm A (two species): a fraction f of neurons carry d_DS and the rest d_OS,
  where f = the measured fraction of tuned neurons with DSI > 0.3 and d_DS,
  d_OS are set so that each species' DSI (vector strength on the eight class
  means after subtracting the minimum, run 59's definition) equals the
  measured median DSI of the recording's top and bottom DSI thirds.
  Arm B (resampled): each neuron's d_j is set so that its DSI equals the DSI
  of a tuned neuron drawn at random from the recording (d clipped at 1 where
  the measured DSI exceeds the tuning shape's maximum; the clipped fraction is
  reported).
  Arm C (homogeneous control): run 48's model at s = 0.8 with the recording's
  own kernel coefficients (a, b2, c1, b4), within-class correlation and PR;
  reproduces run 48 on the three drifting recordings and extends it to the
  other five.
Measured per recording: trial labels; tuned neurons (F-test p < 0.01); DSI
and OSI; f; median DSI of the thirds; amplitude ranges; within-class
correlation and PR (run 48's estimator, 120 trials per class, seed 0); the
principal-angle alignment profile at 45/90/135/180 degrees (run 6's
estimator, rank 10, 120 trials per class); the observed shift
(orientation_zoom.json). Per arm and seed (three seeds): the eight-class
direction ladder against ten-draw floors (run 2's ladder_delta), the model's
alignment profile, the model's sector balance b2/c1 (run 59's profile
estimator), realized within-class correlation and PR, and, for arms A and B at
s = 1, the per-third ladders (neurons sorted by their model DSI, thirds of
equal size, plus a random third) with per-rung deficits and late fractions
(run 59b's definitions).

REGISTERED EXPECTATIONS (written before the run; arm A at s = 1 is the claim,
arm B at s = 1 the check):
M1: the predicted shift orders the three drifting recordings GT1 > GT2 > GT3
    (observed 0.347 > 0.280 > 0.240) with no fitted parameter.
M2: across the eight recordings the predicted shift tracks the observed one
    at Spearman rho >= 0.7, and at least two of the model's three largest
    predictions are localized-grating recordings.
M3: |predicted - observed| <= 0.06 on all three drifting recordings (run 56's
    criterion; the homogeneous model's errors are -0.12, -0.10, +0.12).
M4: in the model the direction-selective third shifts more than the
    orientation-selective third in 8 of 8 recordings, and its late fraction
    is the larger in 8 of 8 (runs 59 and 59b reproduced).
M5: the model's 180-degree within-class alignment lies within 0.05 of the
    measured value in 8 of 8 recordings (the homogeneous model over-recovers
    it by 0.05-0.12 on the drifting three).
A miss is reported at full volume.

Out: ../data_canonical/run60_mixture_model.json (+ .log)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = DATA / "run60_mixture_model.json"


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


r2 = load_module("run2", "run2_calibrated_corotating.py")
r6 = load_module("run6", "run6_corotation_calibration.py")
r48 = load_module("run48", "run48_overshoot_across_recordings.py")
r59 = load_module("run59", "run59_shift_by_direction_selectivity.py")
r59b = load_module("run59b", "run59b_per_rung_thirds.py")

NB = 8
N_SYN = 4000
SEEDS = (1, 2, 3)
FRAC_ISO = 0.1
S_PRIMARY, S_SECONDARY = 1.0, 0.8
DSI_THRESHOLD = 0.3
ANG = 2 * np.pi * np.arange(NB) / NB


def short(name):
    kind = "D" if "drifting" in name else "L" if "local" in name else "C"
    return kind + name.split("GT")[1][0]


def tuning(theta, d):
    """One cosine, rectified two ways: d = 0 is |cos| (orientation, period
    pi), d = 1 is the half-wave rectified cosine (direction)."""
    c = np.cos(theta)
    return np.abs(c) * (1 + d * np.sign(c)) / (1 + d)


def dsi_osi(R):
    """Run 59's estimator on classes x neurons: vector strengths at the first
    and second harmonic after subtracting each neuron's minimum."""
    R = R - R.min(0, keepdims=True) + 1e-9
    dsi = np.abs((R * np.exp(1j * ANG[:, None])).sum(0) / R.sum(0))
    osi = np.abs((R * np.exp(2j * ANG[:, None])).sum(0) / R.sum(0))
    return dsi, osi


_D_GRID = np.linspace(0, 1, 2001)
_DSI_GRID = dsi_osi(np.stack([tuning(ANG, d) for d in _D_GRID], axis=1))[0]
assert np.all(np.diff(_DSI_GRID) >= -1e-12), "DSI(d) must be monotone for the inversion"
DSI_MAX = float(_DSI_GRID[-1])


def d_from_dsi(x):
    """Invert DSI(d) on the grid; values above the shape's maximum clip to d = 1."""
    return np.interp(np.asarray(x, float), _DSI_GRID, _D_GRID)


def within_stats(Xt, bl, rng):
    """Run 48's within-class estimator: mean pairwise trial correlation and
    median within-class PR over 120-trial draws per class."""
    cors, wc = [], []
    for b in range(NB):
        idx = rng.choice(np.where(bl == b)[0], 120, replace=False)
        V = Xt[idx]; wc.append(r2.pr_c(V))
        Vn = V - V.mean(axis=1, keepdims=True); Vn /= np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-9
        Cm = Vn @ Vn.T; cors.append((Cm.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
    return float(np.mean(cors)), float(np.median(wc))


def measure(name):
    dat = np.load(RAW / f"{name}.npy", allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel() % (2 * np.pi)
    bl = np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, NB + 1)) - 1, 0, NB - 1)
    Xt = np.ascontiguousarray(X.T); del dat, X
    n, N = Xt.shape
    counts = np.bincount(bl, minlength=NB)
    M = np.stack([Xt[bl == k].mean(0) for k in range(NB)])
    grand = Xt.mean(0)
    ssb = sum(counts[k] * (M[k] - grand) ** 2 for k in range(NB))
    ssw = sum(((Xt[bl == k] - M[k]) ** 2).sum(0) for k in range(NB))
    F = (ssb / (NB - 1)) / (ssw / (n - NB) + 1e-12)
    tuned = (1 - stats.f.cdf(F, NB - 1, n - NB)) < 0.01
    dsi, osi = dsi_osi(M)
    tidx = np.where(tuned)[0]; nsub = len(tidx) // 3
    order = tidx[np.argsort(dsi[tidx])]
    rw, pw = within_stats(Xt, bl, np.random.default_rng(0))
    prof = r6.alignment_profile(Xt, bl, np.random.default_rng(3))
    b2c1_tuned, _ = r59.profile_b2_c1(M[:, tidx])
    meas = {"n_trials": int(n), "n_neurons": int(N), "n_tuned": int(len(tidx)), "n_sub": int(nsub),
            "f_ds": float((dsi[tidx] > DSI_THRESHOLD).mean()), "median_dsi_tuned": float(np.median(dsi[tidx])),
            "median_osi_tuned": float(np.median(osi[tidx])), "median_dsi_DS_third": float(np.median(dsi[order[-nsub:]])),
            "median_dsi_OS_third": float(np.median(dsi[order[:nsub]])), "frac_dsi_above_shape_max": float((dsi[tidx] > DSI_MAX).mean()),
            "within_corr": rw, "within_pr": pw, "profile": {str(int(k)): float(v) for k, v in prof.items()},
            "b2_over_c1_tuned": float(b2c1_tuned)}
    arrays = {"dsi_tuned": dsi[tidx].astype(float), "amp_tuned": (M.max(0) - M.min(0))[tidx].astype(float)}
    del Xt
    return bl, meas, arrays


def make_mixture(n_stim, bl, d_vec, amp_pool, rw, pw, s, rng):
    N = len(d_vec)
    phi0 = rng.uniform(0, 2 * np.pi, N)
    amp = rng.choice(amp_pool, N, replace=True)
    R = amp[None, :] * tuning(ANG[:, None] - phi0[None, :], d_vec[None, :])
    S = float(((R - R.mean(1, keepdims=True)) ** 2).mean())
    sig2 = S * (1 - rw) / max(rw, 1e-3)
    K = max(int(round(pw)), 2)
    U = np.linalg.qr(rng.standard_normal((N, K)))[0]
    g = rng.standard_normal((n_stim, K)) * np.sqrt(sig2 * (1 - FRAC_ISO) * N / K)
    Rhat = R / (np.sqrt((R ** 2).mean(axis=1, keepdims=True)) + 1e-9)
    W = (g @ U.T) * ((1 - s) + s * Rhat[bl])
    Xs = R[bl] + W + rng.standard_normal((n_stim, N)) * np.sqrt(sig2 * FRAC_ISO)
    return np.ascontiguousarray(Xs.astype(np.float32)), R


def thirds_report(Xs, R, rng):
    mdsi = dsi_osi(R)[0]; order = np.argsort(mdsi); nsub = len(mdsi) // 3
    idxs = {"DS": order[-nsub:], "OS": order[:nsub], "random": rng.choice(len(mdsi), nsub, replace=False)}
    rep = {}
    for tn, idx in idxs.items():
        sizes, ob, fl, dd = r59b.ladder_rungs(np.ascontiguousarray(Xs[:, idx]), bl_global[0], np.random.default_rng(42))
        deficit = [o - f for o, f in zip(ob, fl)]
        rep[tn] = {"delta": dd, "deficit": deficit, "late_fraction": float(deficit[3] / deficit[0]) if deficit[0] != 0 else None,
                   "median_model_dsi": float(np.median(mdsi[idx]))}
    return rep


bl_global = [None]


def mean_cells(cells, obs):
    out = {"delta": float(np.mean([c["delta"] for c in cells])), "delta_sd": float(np.std([c["delta"] for c in cells]))}
    out["error"] = out["delta"] - obs
    out["profile"] = {k: float(np.mean([c["profile"][k] for c in cells])) for k in cells[0]["profile"]}
    out["realized_within_corr"] = float(np.mean([c["realized_within_corr"] for c in cells]))
    out["realized_within_pr"] = float(np.mean([c["realized_within_pr"] for c in cells]))
    if "b2_over_c1" in cells[0]:
        out["b2_over_c1"] = float(np.mean([c["b2_over_c1"] for c in cells]))
    if "thirds" in cells[0]:
        out["thirds"] = {tn: {"delta": float(np.mean([c["thirds"][tn]["delta"] for c in cells])),
                              "late_fraction": float(np.mean([c["thirds"][tn]["late_fraction"] for c in cells])),
                              "deficit": [float(np.mean([c["thirds"][tn]["deficit"][k] for c in cells])) for k in range(len(cells[0]["thirds"][tn]["deficit"]))]}
                         for tn in cells[0]["thirds"]}
    return out


def main():
    t0 = time.time()
    oz = {r["name"]: r for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    mh = {r["name"]: r for r in json.load(open(DATA / "multipole_harmonics_8dir.json"))["rows"]}
    names = [n for n in oz if "gratings_" in n and "static" not in n]
    out = {"design": {"n_syn": N_SYN, "seeds": list(SEEDS), "frac_iso": FRAC_ISO, "s_primary": S_PRIMARY, "s_secondary": S_SECONDARY,
                      "dsi_threshold": DSI_THRESHOLD, "dsi_shape_max": DSI_MAX, "n_null": r2.N_NULL}, "rows": {}}
    for name in names:
        tag = short(name)
        bl, meas, arrays = measure(name)
        bl_global[0] = bl
        obs = float(oz[name]["delta"]); coef = mh[name]["coef"]
        n_stim = len(bl)
        row = {"name": name, "measured": meas, "observed_delta": obs, "share_c1_over_b2": float(coef["c1"] / coef["b2"]), "arms": {}}
        d_DS, d_OS = float(d_from_dsi(meas["median_dsi_DS_third"])), float(d_from_dsi(meas["median_dsi_OS_third"]))
        row["d_DS"], row["d_OS"] = d_DS, d_OS
        print(f"[{tag}] trials {n_stim} tuned {meas['n_tuned']} | f {meas['f_ds']:.2f} DSI thirds {meas['median_dsi_DS_third']:.2f}/{meas['median_dsi_OS_third']:.2f} "
              f"-> d {d_DS:.2f}/{d_OS:.2f} (clip frac {meas['frac_dsi_above_shape_max']:.2f}) | corr {meas['within_corr']:.3f} PR {meas['within_pr']:.1f} "
              f"| align 45/90/135/180 {meas['profile']['45']:.3f}/{meas['profile']['90']:.3f}/{meas['profile']['135']:.3f}/{meas['profile']['180']:.3f} "
              f"| b2/c1 tuned {meas['b2_over_c1_tuned']:.2f} | obs {obs:+.3f} | {time.time()-t0:.0f}s", flush=True)
        for arm, s_list in (("A_two_species", (S_PRIMARY, S_SECONDARY)), ("B_resampled", (S_PRIMARY, S_SECONDARY)), ("C_homogeneous", (0.8,))):
            for s in s_list:
                cells = []
                for seed in SEEDS:
                    rng = np.random.default_rng(100 * seed + 7)
                    R = None
                    if arm == "A_two_species":
                        n_ds = int(round(meas["f_ds"] * N_SYN))
                        d_vec = np.concatenate([np.full(n_ds, d_DS), np.full(N_SYN - n_ds, d_OS)]); rng.shuffle(d_vec)
                        Xs, R = make_mixture(n_stim, bl, d_vec, arrays["amp_tuned"], meas["within_corr"], meas["within_pr"], s, rng)
                    elif arm == "B_resampled":
                        d_vec = d_from_dsi(rng.choice(arrays["dsi_tuned"], N_SYN, replace=True))
                        Xs, R = make_mixture(n_stim, bl, d_vec, arrays["amp_tuned"], meas["within_corr"], meas["within_pr"], s, rng)
                    else:
                        Xs = r48.make_partial(n_stim, N_SYN, bl, meas["within_corr"], meas["within_pr"], s, coef, seed=seed)
                    d, _ = r2.ladder_delta(Xs, bl, np.random.default_rng(300 + seed))
                    prof = r6.alignment_profile(Xs, bl, np.random.default_rng(400 + seed))
                    rw_r, pw_r = within_stats(Xs, bl, np.random.default_rng(0))
                    cell = {"delta": float(d), "profile": {str(int(k)): float(v) for k, v in prof.items()},
                            "realized_within_corr": rw_r, "realized_within_pr": pw_r}
                    if R is not None:
                        cell["b2_over_c1"] = float(r59.profile_b2_c1(R)[0])
                        if s == S_PRIMARY:
                            cell["thirds"] = thirds_report(Xs, R, rng)
                    cells.append(cell); del Xs
                key = f"{arm}_s{s}"
                row["arms"][key] = mean_cells(cells, obs) | {"seeds": cells}
                a = row["arms"][key]
                msg = (f"  {key:22s}: delta {a['delta']:+.3f}±{a['delta_sd']:.3f} (obs {obs:+.3f}, err {a['error']:+.3f}) | align180 {a['profile']['180']:.3f} "
                       f"(meas {meas['profile']['180']:.3f}) | realized corr {a['realized_within_corr']:.3f} PR {a['realized_within_pr']:.1f}")
                if "b2_over_c1" in a:
                    msg += f" | model b2/c1 {a['b2_over_c1']:.2f}"
                if "thirds" in a:
                    t = a["thirds"]
                    msg += (f" | thirds DS/rand/OS {t['DS']['delta']:+.3f}/{t['random']['delta']:+.3f}/{t['OS']['delta']:+.3f}"
                            f" late {t['DS']['late_fraction']:.2f}/{t['random']['late_fraction']:.2f}/{t['OS']['late_fraction']:.2f}")
                print(msg + f" | {time.time()-t0:.0f}s", flush=True)
        out["rows"][tag] = row
        json.dump(out, open(OUT, "w"), indent=1)
    rows = out["rows"]; tags = list(rows); drift = [t for t in tags if t.startswith("D")]
    verdict = {"tags": tags, "observed": [rows[t]["observed_delta"] for t in tags], "arms": {}}
    for key in rows[tags[0]]["arms"]:
        pred = [rows[t]["arms"][key]["delta"] for t in tags]
        pd = {t: rows[t]["arms"][key]["delta"] for t in drift}
        rho = float(spearmanr(pred, verdict["observed"])[0])
        top3 = sorted(tags, key=lambda t: -rows[t]["arms"][key]["delta"])[:3]
        v = {"predicted": pred, "errors": [rows[t]["arms"][key]["error"] for t in tags],
             "M1_orders_D1_D2_D3": bool(pd["D1"] > pd["D2"] > pd["D3"]), "M2_rho": rho, "M2_top3": top3,
             "M2": bool(rho >= 0.7 and sum(t.startswith("L") for t in top3) >= 2),
             "M3_drifting_errors": [rows[t]["arms"][key]["error"] for t in drift],
             "M3": bool(all(abs(rows[t]["arms"][key]["error"]) <= 0.06 for t in drift)),
             "M5_abs_180_gap": [abs(rows[t]["arms"][key]["profile"]["180"] - rows[t]["measured"]["profile"]["180"]) for t in tags]}
        v["M5"] = bool(all(g <= 0.05 for g in v["M5_abs_180_gap"]))
        if "thirds" in rows[tags[0]]["arms"][key]:
            th = {t: rows[t]["arms"][key]["thirds"] for t in tags}
            v["M4_DS_gt_OS_count"] = int(sum(th[t]["DS"]["delta"] > th[t]["OS"]["delta"] for t in tags))
            v["M4_DS_late_gt_OS_count"] = int(sum(th[t]["DS"]["late_fraction"] > th[t]["OS"]["late_fraction"] for t in tags))
            v["M4"] = bool(v["M4_DS_gt_OS_count"] == 8 and v["M4_DS_late_gt_OS_count"] == 8)
        verdict["arms"][key] = v
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
