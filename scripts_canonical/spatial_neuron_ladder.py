"""Spatially structured neuron ladder - the REPAIRED neuron-axis test.

The prior implementation (neuron_zoom_all.py) compared random neuron subsets
to random neuron subsets (exchangeable; expected delta = 0), so its negative
mean could not test spatial redundancy. This script implements the genuine
structured ladder:

  observed  - spatially contiguous subsets: for each rung size ns, take the ns
              nearest neighbors (Euclidean, suite2p 'med' positions) around a
              random anchor cell; rung value = mean PR over N_DRAWS anchors.
  floor     - random same-size neuron subsets; rung value = mean PR over
              N_DRAWS draws (SAME aggregation as observed - no Jensen asymmetry).
  control   - random-vs-random with disjoint seeds, same estimator; expected
              delta = 0 (empirical check of the exchangeability diagnosis).

delta_spatial < 0 (spatially local subsets more redundant than random) is the
hypothesis the paper's text claims. Output: data/stringer_neuron_zoom_spatial.json
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stringer_v1" / "natimg"
OUT = HERE / "data" / "stringer_neuron_zoom_spatial.json"

SIZES = [50, 100, 200, 500, 1000, 2000, 5000]
N_DRAWS = 50


def pr_trace(X):
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    if n < 3:
        return 1.0
    G = Xc @ Xc.T / (n - 1)
    tr = np.trace(G)
    tr2 = float((G * G).sum())
    return float(tr * tr / tr2) if tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.maximum(np.asarray(prs, dtype=float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def ladder(X, subsets_per_size):
    sizes, prs = [], []
    for ns, subsets in subsets_per_size:
        sizes.append(ns)
        prs.append(float(np.mean([pr_trace(X[idx]) for idx in subsets])))
    return slope(sizes, prs)


def main():
    rows = []
    for fp in sorted(DATA.glob("*.npy")):
        t0 = time.time()
        dat = np.load(fp, allow_pickle=True).item()
        X = np.asarray(dat["sresp"], dtype=np.float64)  # neurons x stimuli
        pos = np.array([s["med"] for s in dat["stat"]], dtype=float)
        n_neur = X.shape[0]
        sizes = [s for s in SIZES if s <= min(n_neur, 5000)]

        rng = np.random.default_rng(42)
        spatial = []
        for ns in sizes:
            subs = []
            for _ in range(N_DRAWS):
                anchor = rng.integers(n_neur)
                d2 = ((pos - pos[anchor]) ** 2).sum(axis=1)
                subs.append(np.argsort(d2)[:ns])
            spatial.append((ns, subs))
        theta_spatial = ladder(X, spatial)

        rng_f = np.random.default_rng(1000)
        floor = [(ns, [rng_f.choice(n_neur, ns, replace=False) for _ in range(N_DRAWS)])
                 for ns in sizes]
        theta_floor = ladder(X, floor)

        rng_a = np.random.default_rng(2000)
        rng_b = np.random.default_rng(3000)
        ctrl_a = [(ns, [rng_a.choice(n_neur, ns, replace=False) for _ in range(N_DRAWS)])
                  for ns in sizes]
        ctrl_b = [(ns, [rng_b.choice(n_neur, ns, replace=False) for _ in range(N_DRAWS)])
                  for ns in sizes]
        delta_control = ladder(X, ctrl_a) - ladder(X, ctrl_b)

        row = {
            "name": fp.stem, "n_neurons": n_neur, "sizes": sizes,
            "theta_spatial": theta_spatial, "theta_floor": theta_floor,
            "delta_spatial": theta_spatial - theta_floor,
            "delta_control_rand_vs_rand": delta_control,
        }
        rows.append(row)
        print(f"{fp.stem}: delta_spatial={row['delta_spatial']:+.4f} "
              f"control={delta_control:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        with OUT.open("w") as f:
            json.dump({"construction": "spatial kNN anchors vs random subsets, "
                       f"{N_DRAWS} draws/rung both arms, identical aggregation",
                       "rows": rows}, f, indent=1)
    ds = [r["delta_spatial"] for r in rows]
    cs = [r["delta_control_rand_vs_rand"] for r in rows]
    with OUT.open("w") as f:
        json.dump({"construction": "spatial kNN anchors vs random subsets, "
                   f"{N_DRAWS} draws/rung both arms, identical aggregation",
                   "rows": rows,
                   "mean_delta_spatial": float(np.mean(ds)),
                   "n_negative": int(sum(d < 0 for d in ds)),
                   "mean_delta_control": float(np.mean(cs))}, f, indent=1)
    print(f"ALL: mean delta_spatial={np.mean(ds):+.4f} ({sum(d<0 for d in ds)}/{len(ds)} neg), "
          f"mean control={np.mean(cs):+.4f}")


if __name__ == "__main__":
    main()
