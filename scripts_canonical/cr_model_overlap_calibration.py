"""C6 (Oct 20): why does the calibrated generative model miss, and does calibrating it to the right statistic fix it?

Facts (released): the one-parameter model of Sec 6 (run48 construction: class means from each recording's multipole
kernel; within-class fluctuations = K shared modes whose loadings are scaled neuron by neuron by (1 - s) + s * the
class's normalized mean, so s interpolates shared -> class-dependent gain), calibrated by fitting s to the measured
top-10 principal-angle alignment profile, errs by -0.118 (GT1), -0.100 (GT2), +0.116 (GT3), in proportion to each
code's dipole share (run48). Adding an odd harmonic to the gain (run56) fixed the 180-degree alignment in all three but
not the shift (R1-R3 failed). C6 found the within-class part is set by the variance-weighted overlap K_cd of the class
covariances (mean field r = 0.92), of which the principal-angle alignment is an unweighted proxy.

HYPOTHESIS: the model misses because it is calibrated to the proxy. Calibrate s to the measured shape-overlap profile
Kn(dphi) = K_cd / sqrt(K_cc K_dd) at 45/90/135/180 degrees (eight classes, split halves: alternate presentations within
each class, so the diagonal carries no sampling term), with the identical estimator on model and data, and predict delta.
REGISTERED EXPECTATIONS (written before the run):
M1: at the overlap-calibrated s*, |delta_pred - delta_obs| <= 0.06 in all three drifting recordings, and the predictions
    order GT1 > GT2 > GT3 as observed.
M2: at run48's alignment-calibrated s*, the model's sigma (mean off-diagonal / mean diagonal Kn) errs in the direction
    of its shift error: below the measured sigma in GT3 (over-predicted shift), above it in GT1 and GT2.
M3 (named failure mode): if no s matches the measured overlap profile within the profile's own split-half noise, the
    model family (shared modes, neuron-wise gain) cannot make classes as distinct as cortex's; then report the best s,
    its sigma gap, and delta there.
Out: data_canonical/cr_model_overlap_calibration.json"""
import importlib.util
import json
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUT = DATA / "cr_model_overlap_calibration.json"


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod


R48 = load("r48", "run48_overshoot_across_recordings.py"); R2 = R48.r2
S_VALS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = (1, 2, 3)


def overlap_profile(X, bl):
    """Split-half shape-overlap profile Kn at separations 1..4 classes (45..180 deg) and sigma, eight classes."""
    A, B = [], []
    for c in range(8):
        idx = np.where(bl == c)[0]; a, b = idx[0::2], idx[1::2]
        Za = X[a] - X[a].mean(0); Zb = X[b] - X[b].mean(0); A.append(Za); B.append(Zb)
    tr = lambda Z: float((Z * Z).sum() / len(Z))
    K = np.zeros((8, 8))
    for c in range(8):
        for d in range(8):
            ab = A[c] @ B[d].T; ba = B[c] @ A[d].T
            K[c, d] = 0.5 * ((ab * ab).sum() / (len(A[c]) * len(B[d]) * tr(A[c]) * tr(B[d]))
                             + (ba * ba).sum() / (len(B[c]) * len(A[d]) * tr(B[c]) * tr(A[d])))
    Kn = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))
    sep = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
    prof = {45 * k: float(Kn[sep == k].mean()) for k in range(1, 5)}
    off = ~np.eye(8, dtype=bool)
    return prof, float(K[off].mean() / np.diag(K).mean())


def main():
    mh = {r["name"]: r for r in json.load(open(DATA / "multipole_harmonics_8dir.json"))["rows"]}
    oz = {r["name"]: r for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    r48 = json.load(open(DATA / "run48_overshoot_across_recordings.json"))["rows"]
    out = {}
    for tag, name in R48.RECS.items():
        t0 = time.time()
        coef = mh[name]["coef"]; obs = float(oz[name]["delta"])
        (n_stim, n_neur), bl, rw, pw, _ = R48.measure(name)
        dat = np.load(R48.RAW / f"{name}.npy", allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9; del dat
        meas_prof, meas_sigma = overlap_profile(X, bl); del X
        sweep = {}
        for s in S_VALS:
            ds, profs, sigs = [], [], []
            for seed in SEEDS:
                Xs = R48.make_partial(n_stim, 4000, bl, rw, pw, s, coef, seed=seed)
                d, _ = R2.ladder_delta(Xs, bl, np.random.default_rng(300 + seed)); ds.append(d)
                p, sg = overlap_profile(Xs, bl); profs.append(p); sigs.append(sg)
            prof = {k: float(np.mean([p[k] for p in profs])) for k in profs[0]}
            err = float(np.mean([(prof[k] - meas_prof[k]) ** 2 for k in prof]))
            sweep[str(s)] = {"delta": float(np.mean(ds)), "delta_sd": float(np.std(ds)), "profile": prof,
                             "sigma": float(np.mean(sigs)), "sq_err": err}
            print(f"[{tag}] s {s:.1f}: delta {np.mean(ds):+.3f} sigma {np.mean(sigs):.3f} Kn " +
                  "/".join(f"{prof[k]:.3f}" for k in prof) + f" err {err:.5f}", flush=True)
        best = min(sweep, key=lambda k: sweep[k]["sq_err"])
        s48 = str(r48[tag]["s_star"])
        out[tag] = {"observed_delta": obs, "measured_profile": meas_prof, "measured_sigma": meas_sigma, "sweep": sweep,
                    "s_star_overlap": float(best), "delta_pred_overlap": sweep[best]["delta"],
                    "error_overlap": sweep[best]["delta"] - obs, "s_star_alignment_run48": r48[tag]["s_star"],
                    "sigma_at_alignment_s": sweep[s48]["sigma"] if s48 in sweep else None}
        print(f"[{tag}] measured sigma {meas_sigma:.3f} Kn " + "/".join(f"{meas_prof[k]:.3f}" for k in meas_prof)
              + f" | overlap-calibrated s* {best} delta {sweep[best]['delta']:+.3f} vs obs {obs:+.3f}"
              f" (error {sweep[best]['delta'] - obs:+.3f}) | run48 s* {s48}: sigma {out[tag]['sigma_at_alignment_s']} ({time.time() - t0:.0f}s)", flush=True)
        json.dump({"rows": out}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
