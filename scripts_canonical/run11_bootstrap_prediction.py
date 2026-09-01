"""Run 11 - Proper uncertainty for the alignment-calibrated delta prediction
(round-4 item 4) + covariance-preserving permutation null for the angular
profile (item 6).

Part A: bootstrap the GT3 alignment profile (resample the 120 trials per
class with replacement, recompute subspaces and the 45/90/135/180 profile),
refit s on each realization against the run6 model curves (linear
interpolation of profile(s) and delta(s) over the s-grid), and propagate to a
predicted distribution p(delta_model | alignment data). The observed delta
(+0.237) never enters the fit.

Part B: class-label permutation preserving the real neuronal covariance
(trials reassigned to pseudo-classes of the same sizes), recompute the
angular profile. Under the null the "classes" are random subsets of one
covariance, so the profile should be FLAT and HIGH relative to the measured
angular modulation.

REGISTERED EXPECTATIONS (before run):
B1: the bootstrap-propagated prediction interval for delta contains the
    independently measured +0.237.
B2: the permutation-null profile is flat (no 90-deg minimum / 180-deg
    recovery): angular modulation range under the null is small compared to
    the measured modulation (0.19 - 0.05 = 0.14).

Out: feedback_runs/run11_bootstrap_prediction.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
N_BINS = 8
RANK = 10
N_PER = 120
N_BOOT = 200
N_PERM = 20
SEPS = [45, 90, 135, 180]

run6 = json.load(open(HERE / "run6_corotation_calibration.json"))
S_GRID = sorted(float(s) for s in run6["sweep"])
MODEL_PROF = {s: {int(float(k)): v for k, v in
                  run6["sweep"][str(s)]["profile"].items()} for s in S_GRID}
MODEL_DELTA = {s: run6["sweep"][str(s)]["delta"] for s in S_GRID}


def profile_of(subs):
    prof = {sep: [] for sep in SEPS}
    for i in range(N_BINS):
        for j in range(i + 1, N_BINS):
            sep = min((j - i) % N_BINS, (i - j) % N_BINS) * 45
            sv = np.linalg.svd(subs[i].T @ subs[j], compute_uv=False)
            prof[sep].append(float(np.mean(np.clip(sv, 0, 1) ** 2)))
    return {sep: float(np.mean(v)) for sep, v in prof.items()}


def subspace(V):
    Vc = V - V.mean(axis=0)
    G = Vc @ Vc.T
    w, U = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:RANK]
    B = Vc.T @ U[:, order]
    B /= np.linalg.norm(B, axis=0, keepdims=True) + 1e-12
    return B


def fit_s(prof):
    """Least-squares s on a dense interpolation of the model curves."""
    s_dense = np.linspace(S_GRID[0], S_GRID[-1], 101)
    errs = []
    for s in s_dense:
        e = 0.0
        for sep in SEPS:
            mvals = [MODEL_PROF[g][sep] for g in S_GRID]
            m = np.interp(s, S_GRID, mvals)
            e += (m - prof[sep]) ** 2
        errs.append(e)
    s_star = float(s_dense[int(np.argmin(errs))])
    d_star = float(np.interp(s_star, S_GRID,
                             [MODEL_DELTA[g] for g in S_GRID]))
    return s_star, d_star


dat = np.load(DATA / "gratings_drifting_GT3_2019_04_05_1.npy",
              allow_pickle=True).item()
X = np.asarray(dat["sresp"], np.float32)
X /= X.std() + 1e-9
istim = np.asarray(dat["istim"], float)
bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
             0, N_BINS - 1)
Xt = np.ascontiguousarray(X.T)
rng = np.random.default_rng(0)
class_trials = [rng.choice(np.where(bl == b)[0], N_PER, replace=False)
                for b in range(N_BINS)]

# point profile + fit (sanity)
subs0 = [subspace(Xt[idx]) for idx in class_trials]
prof0 = profile_of(subs0)
s0, d0 = fit_s(prof0)
print(f"point profile: " + " ".join(f"{k}={v:.3f}" for k, v in prof0.items())
      + f" | s*={s0:.2f} delta_pred={d0:+.3f} (observed +0.237)", flush=True)

# Part A: bootstrap
boot_d = []
for b in range(N_BOOT):
    brng = np.random.default_rng(1000 + b)
    subs = [subspace(Xt[idx[brng.integers(0, N_PER, N_PER)]])
            for idx in class_trials]
    _, d = fit_s(profile_of(subs))
    boot_d.append(d)
    if b % 50 == 0:
        print(f"boot {b}: delta_pred={d:+.3f}", flush=True)
boot_d = np.array(boot_d)
lo, hi = np.percentile(boot_d, [2.5, 97.5])
print(f"p(delta_model | alignment): mean={boot_d.mean():+.3f} "
      f"95% [{lo:+.3f}, {hi:+.3f}] | observed +0.237 "
      f"{'INSIDE' if lo <= 0.237 <= hi else 'OUTSIDE'}", flush=True)

# Part B: covariance-preserving label permutation
perm_ranges = []
all_idx = np.concatenate(class_trials)
for p in range(N_PERM):
    prng = np.random.default_rng(2000 + p)
    perm = prng.permutation(all_idx)
    psubs = [subspace(Xt[perm[b * N_PER:(b + 1) * N_PER]])
             for b in range(N_BINS)]
    pprof = profile_of(psubs)
    perm_ranges.append(max(pprof.values()) - min(pprof.values()))
perm_ranges = np.array(perm_ranges)
meas_range = max(prof0.values()) - min(prof0.values())
print(f"angular modulation: measured={meas_range:.3f} | permutation null "
      f"{perm_ranges.mean():.3f} ± {perm_ranges.std():.3f} "
      f"(max {perm_ranges.max():.3f})", flush=True)

json.dump({
    "point_profile": prof0, "s_star": s0, "delta_pred_point": d0,
    "boot_mean": float(boot_d.mean()),
    "boot_ci95": [float(lo), float(hi)], "n_boot": N_BOOT,
    "observed_delta": 0.237,
    "observed_inside": bool(lo <= 0.237 <= hi),
    "measured_modulation_range": float(meas_range),
    "perm_null_range_mean": float(perm_ranges.mean()),
    "perm_null_range_sd": float(perm_ranges.std()),
    "perm_null_range_max": float(perm_ranges.max()),
}, open(HERE / "run11_bootstrap_prediction.json", "w"), indent=1)
print("DONE run11", flush=True)
