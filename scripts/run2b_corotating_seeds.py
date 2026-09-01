"""Run 2b - 5-seed robustness sweep of the run-2 co-rotating results.
No number enters the tex on two seeds. Sweeps seeds 1..5 for the four key
cells: shared, corotating, shared_b2x2, corotating_b2x2.
Out: feedback_runs/run2b_corotating_seeds.json
"""
import json
from pathlib import Path

import numpy as np

import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "run2", HERE / "run2_calibrated_corotating.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)  # __name__ != "__main__", so main() does not run

(n_stim, n_neur), bl, r_within, pr_within = r2.measure_gt3()
print(f"GT3 calibration: corr={r_within:.3f} PR={pr_within:.1f}", flush=True)

cells = {
    "shared": dict(within="shared"),
    "corotating": dict(within="corotating"),
    "shared_b2x2": dict(within="shared", b2=2 * r2.B2),
    "corotating_b2x2": dict(within="corotating", b2=2 * r2.B2),
}
out = {}
for name, kw in cells.items():
    ds = []
    for seed in range(1, 6):
        Xs = r2.make_synthetic(n_stim, 4000, bl, r_within, pr_within,
                               seed=seed, **kw)
        d, _ = r2.ladder_delta(Xs, bl, np.random.default_rng(100 + seed))
        ds.append(d)
        print(f"  {name} seed{seed}: {d:+.4f}", flush=True)
    out[name] = {"mean": float(np.mean(ds)), "sd": float(np.std(ds)),
                 "values": ds}
    print(f"{name}: {np.mean(ds):+.4f} ± {np.std(ds):.4f}", flush=True)

json.dump(out, open(HERE / "run2b_corotating_seeds.json", "w"), indent=1)
print("DONE run2b", flush=True)
