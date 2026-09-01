"""Run 8 - Ladder-design robustness (referee minor 5 / feedback point 10).

On all eight grating recordings, the direction-aligned decomposition under:
  (a) baseline rungs [1,2,3,4,6,8]
  (b) drop first rung [2,3,4,6,8]
  (c) drop last rung  [1,2,3,4,6]
  (d) coarse schedule [1,2,4,8]
  (e) integrated residual A = trapz(logPR_aligned - logPR_floor, dlog n)
      on the baseline rungs (a slope-free summary).

REGISTERED EXPECTATION: the biological conclusion (positive aligned residual)
survives every variant on every grating recording; magnitudes may shift.

Out: feedback_runs/run8_ladder_robustness.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
N_BINS = 8
ALL_COUNTS = [1, 2, 3, 4, 6, 8]
VARIANTS = {
    "baseline": [1, 2, 3, 4, 6, 8],
    "drop_first": [2, 3, 4, 6, 8],
    "drop_last": [1, 2, 3, 4, 6],
    "coarse": [1, 2, 4, 8],
}
N_NULL = 10

FILES = {
    "drifting_GT1": "gratings_drifting_GT1_2019_04_12_1.npy",
    "drifting_GT2": "gratings_drifting_GT2_2019_04_05_1.npy",
    "drifting_GT3": "gratings_drifting_GT3_2019_04_05_1.npy",
    "local_GT1": "gratings_local_GT1_2019_04_27_2.npy",
    "local_GT2": "gratings_local_GT2_2019_04_23_2.npy",
    "local_GT3": "gratings_local_GT3_2019_04_24_2.npy",
    "lowc_GT1": "gratings_low_contrast_GT1_2019_04_09_1.npy",
    "lowc_GT2": "gratings_low_contrast_GT2_2019_04_12_2.npy",
}


def pr_c(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def slope(sizes, logs):
    x = np.log(np.asarray(sizes, float))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, np.asarray(logs, float), rcond=None)[0][1])


def run_recording(tag, fname):
    dat = np.load(DATA / fname, allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
                 0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)
    members = [np.where(bl == b)[0] for b in range(N_BINS)]
    rng = np.random.default_rng(0)

    # observed log-PR and mean floor log-PR at every rung count, computed once
    sizes, obs_log, floor_log = {}, {}, {}
    for c in ALL_COUNTS:
        sel = np.concatenate(members[:c])
        n = len(sel)
        sizes[c] = n
        obs_log[c] = np.log(max(pr_c(Xt[sel]), 1e-9))
        draws = [np.log(max(pr_c(Xt[rng.choice(len(Xt), n, replace=False)]),
                            1e-9)) for _ in range(N_NULL)]
        floor_log[c] = float(np.mean(draws))

    res = {}
    for name, counts in VARIANTS.items():
        s = [sizes[c] for c in counts]
        d = slope(s, [obs_log[c] for c in counts]) \
            - slope(s, [floor_log[c] for c in counts])
        res[name] = round(d, 4)
    x = np.log([sizes[c] for c in ALL_COUNTS])
    resid = np.array([obs_log[c] - floor_log[c] for c in ALL_COUNTS])
    res["integrated_A"] = round(float(np.trapezoid(resid, x)), 4)
    print(f"{tag}: " + " ".join(f"{k}={v:+.3f}" for k, v in res.items()),
          flush=True)
    return res


out = {}
for tag, fname in FILES.items():
    out[tag] = run_recording(tag, fname)
allpos = all(v > 0 for r in out.values() for v in r.values())
out["_all_positive"] = allpos
print(f"ALL POSITIVE: {allpos}", flush=True)
json.dump(out, open(HERE / "run8_ladder_robustness.json", "w"), indent=1)
print("DONE run8", flush=True)
