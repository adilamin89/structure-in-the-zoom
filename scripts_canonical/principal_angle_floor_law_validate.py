"""Validate principal_angle_floor_law against Monte Carlo on SYNTHETIC populations (no recording data): two independent classes of
n samples from x = sqrt(tau) C^(1/2) z, centred, top-k right singular vectors, pa = 1 - ||U1'U2||^2 / k, 400 pairs per cell.
Spectra: power laws lambda_a = a^-alpha (alpha 0.7, 1.0, 1.3) with p = 256; gains: tau = 1, Gamma kappa 1.5 and 2.5.
Out: data_canonical/principal_angle_floor_law_validate.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import principal_angle_floor_law as FL  # noqa: E402

P, K, NPAIR = 256, 5, 400
NS = (8, 16, 32, 64, 128, 256)


def mc(lam, n, kappa, rng):
    A = np.sqrt(lam)
    vals = []
    for _ in range(NPAIR):
        U = []
        for _c in range(2):
            Z = rng.standard_normal((n, P)) * A
            if kappa > 1:
                s = 1 / (kappa - 1); Z *= np.sqrt(rng.gamma(s, 1 / s, n))[:, None]
            Z -= Z.mean(0)
            U.append(np.linalg.svd(Z, full_matrices=False)[2][:min(K, n - 1)].T)
        vals.append(1 - np.sum((U[0].T @ U[1]) ** 2) / min(K, n - 1))
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(NPAIR))


def main():
    rng = np.random.default_rng(0)
    rows = []
    for alpha in (0.7, 1.0, 1.3):
        lam = np.arange(1, P + 1) ** -alpha
        for kappa in (1.0, 1.5, 2.5):
            for n in NS:
                t0 = time.time()
                m, se = mc(lam, n, kappa, rng)
                pred, info = FL.floor_pa(lam, n, K, "one" if kappa == 1 else ("gamma", kappa))
                rows.append({"alpha": alpha, "kappa": kappa, "n": n, "mc": m, "mc_se": se, "law": pred, "info": info})
                print(f"alpha {alpha} kappa {kappa} n {n:3d}: MC {m:.4f} +- {se:.4f}  law {pred:.4f}  rel {(pred - m) / m:+.3f}  "
                      f"sum_m {info['sum_m']:.3f} nonconv {info['nonconverged']} ({time.time() - t0:.1f}s)", flush=True)
    rel = np.array([abs(r["law"] - r["mc"]) / r["mc"] for r in rows])
    summ = {"median_rel_err": float(np.median(rel)), "max_rel_err": float(rel.max()),
            "by_n": {str(n): float(np.median([abs(r["law"] - r["mc"]) / r["mc"] for r in rows if r["n"] == n])) for n in NS}}
    print("SUMMARY", json.dumps(summ, indent=1))
    json.dump({"note": "floor-law validation on synthetic data", "rows": rows, "summary": summ},
              open(HERE.parent / "data_canonical" / "principal_angle_floor_law_validate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
