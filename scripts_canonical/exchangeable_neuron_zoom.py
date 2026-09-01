"""Neuron-subsampling zoom on ALL 10 natimg recordings.
Replicates the n=1 neuron-zoom result (delta = -0.033 on GT3) across
every stimulus-driven Stringer recording. Kill-safe incremental saves.
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stringer_v1" / "natimg"
OUT = HERE / "data" / "stringer_neuron_zoom_all.json"


def pr_fast(X):
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    if n < 3:
        return 1.0
    G = Xc @ Xc.T / (n - 1)
    lam = np.linalg.eigvalsh(G)
    lam = lam[lam > 1e-10]
    return float((lam.sum()) ** 2 / (lam ** 2).sum()) if len(lam) else 1.0


def fit_theta(units):
    x = np.log(np.array([u[1] for u in units], dtype=float))
    y = np.log(np.array([max(u[2], 1e-9) for u in units], dtype=float))
    A = np.vstack([np.ones(len(x)), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def main():
    rows = []
    for fp in sorted(DATA.glob("*.npy")):
        t0 = time.time()
        dat = np.load(fp, allow_pickle=True).item()
        X = dat["sresp"]  # neurons x stimuli
        n_neur = X.shape[0]
        # Cap the top rung at 5000 neurons: the full-size Gram
        # eigendecomposition (20K x 20K) costs ~1hr/recording for a rung
        # whose PR is within noise of the 5000 rung; the ladder slope is
        # insensitive to the cap (checked on GT3).
        sizes = [s for s in [50, 100, 200, 500, 1000, 2000, 5000]
                 if s <= min(n_neur, 5000)]

        units = []
        for ns in sizes:
            prs = [pr_fast(X[np.random.default_rng(42 + s).choice(
                n_neur, ns, replace=False)]) for s in range(3)]
            units.append(("n", ns, float(np.mean(prs))))
        theta_obs = fit_theta(units)

        nulls = []
        for ni in range(5):
            rng = np.random.default_rng(1000 + ni)
            nu = [("x", ns, pr_fast(X[rng.choice(n_neur, ns, replace=False)]))
                  for _, ns, _ in units]
            nulls.append(fit_theta(nu))
        theta_floor = float(np.mean(nulls))
        delta = theta_obs - theta_floor
        rows.append({
            "name": fp.stem, "n_neurons": n_neur,
            "theta_obs": theta_obs, "theta_floor": theta_floor,
            "delta": delta,
            "floor_frac": (theta_floor / theta_obs
                           if abs(theta_obs) > 0.01 else None),
        })
        print(f"{fp.stem}: theta={theta_obs:.4f} floor={theta_floor:.4f} "
              f"delta={delta:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        with OUT.open("w") as f:
            json.dump({"rows": rows}, f, indent=1)

    deltas = [r["delta"] for r in rows]
    summary = {"rows": rows, "mean_delta": float(np.mean(deltas)),
               "n_negative": int(sum(1 for d in deltas if d < 0)),
               "n_recordings": len(rows)}
    with OUT.open("w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nALL {len(rows)}: mean delta={np.mean(deltas):+.4f}, "
          f"negative in {summary['n_negative']}/{len(rows)}")


if __name__ == "__main__":
    main()
