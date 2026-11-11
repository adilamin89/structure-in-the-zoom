"""rung 1.14's drift check, its null calibrated over seeds before any flag is read. Stationary sessions (8 classes x 40 rows in random order, each class
stretching its own 6 of 48 features by 2, no drift) and drifting ones (the stretch growing by 3 over the session): per statistic
(declared shift, order-averaged shift, mean-free shift) the SD of z = (late - early) / interleaved spread and the flag rate at |z| > 2.
A t with 19 degrees of freedom (the spread from 20 splits) exceeds 2 about 6% of the time. Out: data_canonical/drift_calibration.json.
Run: python scripts_canonical/drift_calibration.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rung  # noqa: E402

OUT = HERE.parent / "data_canonical" / "drift_calibration.json"


def session(seed, drift):
    g = np.random.default_rng(seed)
    y = g.permutation(np.repeat(np.arange(8), 40))
    t = np.linspace(0, 1, len(y))
    X = g.standard_normal((len(y), 48))
    X *= 1 + (1.0 + drift * t)[:, None] * (y[:, None] == np.arange(48) // 6)
    return X, y


def main():
    out = {"rung": rung.__version__}
    for name, drift, n_seeds in (("stationary", 0.0, 200), ("drift3", 3.0, 50)):
        z = {k: [] for k in ("shift", "orderavg", "meanfree")}
        for s in range(n_seeds):
            X, y = session(1000 + s, drift)
            dr = rung.drift_check(X, y, n_splits=20, n_floor_draws=10, seed=s)
            for k in z:
                z[k].append(dr["stats"][k]["z"])
            if s % 50 == 49:
                print(name, s + 1, {k: round(float(np.mean(np.abs(v) > 2)), 3) for k, v in z.items()}, flush=True)
        out[name] = {k: {"z_mean": float(np.mean(v)), "z_sd": float(np.std(v)), "flag_rate": float(np.mean(np.abs(v) > 2)),
                         "n": len(v)} for k, v in z.items()}
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
