"""Run 2c - 3-seed sweep of the co-rotating GAIN-AMPLITUDE cells (R4 arm of
run2_calibrated_corotating.py). Registered expectation (before run): raising
the co-rotating gain amplitude raises delta (the biological covariation path),
with or without doubling b2. Out: feedback_runs/run2c_gain_seeds.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "run2", HERE / "run2_calibrated_corotating.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)  # __name__ != "__main__", so main() does not run

(n_stim, n_neur), bl, rw, pw = r2.measure_gt3()
out = {}
for name, kw in [
    ("corotating_gainx2", dict(within="corotating", gain_scale=2.0)),
    ("corotating_b2x2_gainx2", dict(within="corotating", b2=2 * r2.B2,
                                    gain_scale=2.0)),
]:
    ds = []
    for seed in (1, 2, 3):
        Xs = r2.make_synthetic(n_stim, 4000, bl, rw, pw, seed=seed, **kw)
        d, _ = r2.ladder_delta(Xs, bl, np.random.default_rng(200 + seed))
        ds.append(d)
        print(f"{name} seed{seed}: {d:+.4f}", flush=True)
    out[name] = {"mean": float(np.mean(ds)), "sd": float(np.std(ds)),
                 "values": ds}
    print(f"{name}: {np.mean(ds):+.4f} ± {np.std(ds):.4f}", flush=True)

json.dump(out, open(HERE / "run2c_gain_seeds.json", "w"), indent=1)
print("DONE run2c", flush=True)
