"""Run 48 (S74, 2026-09-02) - Does the calibrated model's overshoot track the
odd-sector share? The run6/run11b alignment-calibrated prediction overshoots
GT3 (+0.32 vs +0.237). The paper attributes the overshoot to the model's
exactly pi-symmetric within-class gain (over-recovered 180-degree alignment,
0.25-0.30 model vs 0.19 measured). If that is the cause, the overshoot should
be LARGER in recordings whose code carries more odd (direction) structure.
The exact 8-class decomposition (multipole_harmonics_8dir) gives odd shares
c1/b2 = 0.84 (drifting GT1), 0.62 (GT2), 0.43 (GT3).

DESIGN: the run6 procedure, generalized per recording: class-mean correlation
built from the recording's own (a, b2, c1, b4); within-class trial correlation
and within-class PR measured from the recording (run2 estimator, 120 trials per
class); the measured principal-angle profile at 45/90/135/180 deg computed with
run6's alignment_profile on the real data (rank 10, 120 trials per class); s
swept over {0, 0.2, ..., 1.0} with three seeds at 4000 synthetic neurons;
s* = least squares to the measured profile; prediction delta(s*) against the
observed direction-aligned shift of the same 8-class ladder (orientation_zoom).

REGISTERED PREDICTIONS (written before the run):
P1: delta(s*) - delta_obs > 0 in all three drifting recordings.
P2: the overshoot, absolute and relative, orders GT1 > GT2 > GT3 (with the
    odd share). A reversed or flat ordering means the pi-symmetric-gain account
    is incomplete and the overshoot has another source (to be reported as such).
P3: the model's 180-degree alignment exceeds the measured value in all three,
    by more where the odd share is larger.

Out: ../data_canonical/run48_overshoot_across_recordings.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = DATA / "run48_overshoot_across_recordings.json"

spec = importlib.util.spec_from_file_location("run2", HERE / "run2_calibrated_corotating.py")
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
spec6 = importlib.util.spec_from_file_location("run6", HERE / "run6_corotation_calibration.py")
r6 = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(r6)
N_BINS = r2.N_BINS
RECS = {"GT1": "gratings_drifting_GT1_2019_04_12_1", "GT2": "gratings_drifting_GT2_2019_04_05_1",
        "GT3": "gratings_drifting_GT3_2019_04_05_1"}


def measure(name):
    dat = np.load(RAW / f"{name}.npy", allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32); X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)
    rng = np.random.default_rng(0)
    cors, wc = [], []
    for b in range(N_BINS):
        idx = rng.choice(np.where(bl == b)[0], 120, replace=False)
        V = Xt[idx]; wc.append(r2.pr_c(V))
        Vn = V - V.mean(axis=1, keepdims=True); Vn /= np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-9
        Cm = Vn @ Vn.T; cors.append((Cm.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
    prof = r6.alignment_profile(Xt, bl, np.random.default_rng(3))
    return Xt.shape, bl, float(np.mean(cors)), float(np.median(wc)), {int(k): v for k, v in prof.items()}


def make_partial(n_stim, n_neur, bl, r_within, pr_within, s, coef, seed=1):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, N_BINS, endpoint=False)
    dphi = np.abs(ang[:, None] - ang[None, :])
    a0, b2, c1, b4 = coef["a"], coef["b2"], coef["c1"], coef["b4"]
    Cm = a0 + b2 * np.cos(2 * dphi) + c1 * np.cos(dphi) + b4 * np.cos(4 * dphi)
    np.fill_diagonal(Cm, a0 + b2 + c1 + b4)
    Cm = Cm / Cm[0, 0]
    w, V = np.linalg.eigh(Cm); L = V @ np.diag(np.sqrt(np.maximum(w, 0)))
    M = L @ rng.standard_normal((N_BINS, n_neur))
    s2 = float((M ** 2).mean()); sig2_total = s2 * (1 - r_within) / max(r_within, 1e-3)
    K = max(int(round(pr_within)), 2); frac_iso = 0.1
    U = np.linalg.qr(rng.standard_normal((n_neur, K)))[0]
    g = rng.standard_normal((n_stim, K)) * np.sqrt(sig2_total * (1 - frac_iso) * n_neur / K)
    Mhat = M / (np.sqrt((M ** 2).mean(axis=1, keepdims=True)) + 1e-9)
    W = (g @ U.T) * ((1 - s) + s * Mhat[bl])
    Xs = M[bl] + W + rng.standard_normal((n_stim, n_neur)) * np.sqrt(sig2_total * frac_iso)
    return np.ascontiguousarray(Xs.astype(np.float32))


def main():
    mh = {r["name"]: r for r in json.load(open(DATA / "multipole_harmonics_8dir.json"))["rows"]}
    oz = {r["name"]: r for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    out = {}
    for tag, name in RECS.items():
        coef = mh[name]["coef"]; odd_share = coef["c1"] / coef["b2"]
        (n_stim, n_neur), bl, rw, pw, measured = measure(name)
        obs = float(oz[name]["delta"])
        print(f"[{tag}] n_stim {n_stim} corr {rw:.3f} PR {pw:.1f} odd share {odd_share:.2f} | measured align 45/90/135/180 "
              f"{measured[45]:.3f}/{measured[90]:.3f}/{measured[135]:.3f}/{measured[180]:.3f} | observed delta {obs:+.3f}", flush=True)
        sweep = {}; best = None
        for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            ds, profs = [], []
            for seed in (1, 2, 3):
                Xs = make_partial(n_stim, 4000, bl, rw, pw, s, coef, seed=seed)
                d, _ = r2.ladder_delta(Xs, bl, np.random.default_rng(300 + seed)); ds.append(d)
                profs.append(r6.alignment_profile(Xs, bl, np.random.default_rng(400 + seed)))
            prof = {sep: float(np.mean([p[sep] for p in profs])) for sep in profs[0]}
            err = float(np.mean([(prof[sep] - measured[sep]) ** 2 for sep in (45, 90, 135, 180)]))
            sweep[str(s)] = {"delta": float(np.mean(ds)), "delta_sd": float(np.std(ds)), "profile": {str(k): v for k, v in prof.items()}, "sq_err": err}
            print(f"  s={s:.1f}: delta {np.mean(ds):+.3f}±{np.std(ds):.3f} | align {prof[45]:.3f}/{prof[90]:.3f}/{prof[135]:.3f}/{prof[180]:.3f} | err {err:.5f}", flush=True)
            if best is None or err < best[1]:
                best = (s, err, float(np.mean(ds)), prof[180])
        out[tag] = {"name": name, "coef": coef, "odd_share_c1_over_b2": odd_share, "within_corr": rw, "within_pr": pw,
                    "measured_profile": measured, "observed_delta": obs, "sweep": sweep, "s_star": best[0],
                    "delta_pred": best[2], "overshoot_abs": best[2] - obs, "overshoot_rel": (best[2] - obs) / obs,
                    "model_180_at_s_star": best[3], "measured_180": measured[180], "over_recovery_180": best[3] - measured[180]}
        print(f"  => s* {best[0]} delta_pred {best[2]:+.3f} vs obs {obs:+.3f}: overshoot {best[2]-obs:+.3f} ({100*(best[2]-obs)/obs:+.0f}%); "
              f"180-deg model {best[3]:.3f} vs measured {measured[180]:.3f}", flush=True)
    order = sorted(RECS, key=lambda t: -out[t]["odd_share_c1_over_b2"])
    ov = [out[t]["overshoot_abs"] for t in order]; ovr = [out[t]["overshoot_rel"] for t in order]; rec180 = [out[t]["over_recovery_180"] for t in order]
    verdict = {"odd_share_order": order,
               "P1_overshoot_positive_all": bool(all(v > 0 for v in ov)),
               "P2_overshoot_abs_orders_with_odd_share": bool(ov[0] > ov[1] > ov[2]),
               "P2_overshoot_rel_orders_with_odd_share": bool(ovr[0] > ovr[1] > ovr[2]),
               "P3_over_recovery_180_positive_all": bool(all(v > 0 for v in rec180)),
               "P3_over_recovery_orders_with_odd_share": bool(rec180[0] > rec180[1] > rec180[2]),
               "overshoot_abs_by_odd_share_order": [round(v, 3) for v in ov], "overshoot_rel": [round(v, 3) for v in ovr],
               "over_recovery_180": [round(v, 3) for v in rec180]}
    print("VERDICT", json.dumps(verdict, indent=1))
    json.dump({"rows": out, "verdict": verdict}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
