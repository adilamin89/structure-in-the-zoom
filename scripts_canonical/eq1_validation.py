"""Eq. (1) correction and simulation validation - non-interpolating expectation.

The paper's Eq. (1), PR_null(n) = (n-1)d/((n-1)+d), is an interpolant. Here we
compare candidate closed forms against the SIMULATED expectation E[PR_n] for
isotropic populations (the regime Eq. 1 addresses), across the ladder range:
  f1: (n-1)d/((n-1)+d)                    (paper's Eq. 1)
  f2: ratio-of-expectations E[TrG]^2/E[TrG^2] computed exactly for isotropic
      Wishart: with d population dimensions, centered n-sample Gram,
      E[TrG] = d, E[TrG^2] = d(d+n)/(n-1)  ->  f2 = d(n-1)/(d+n)
  f3: delta-method correction of the ratio (adds Var/Cov terms measured from
      theory of Wishart traces; here estimated numerically once per (n,d))

REGISTERED EXPECTATION: f2 (exact ratio-of-expectations) tracks the simulated
mean PR to within ~1-2% over the ladder range 50 <= n <= 4000 for d in
{50, 200, 1000}; f1 is close to f2 (they differ at O(1/n)); the delta-method
f3 closes most of the residual.

Output: data/eq1_validation.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "eq1_validation.json"


def pr_c(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if tr2 > 0 else 1.0


def main():
    rng = np.random.default_rng(0)
    rows = []
    for d in [50, 200, 1000]:
        D = d  # isotropic population: PR = D exactly
        for n in [50, 100, 200, 500, 1000, 2000, 4000]:
            reps = 60 if n <= 1000 else 20
            prs, trs, tr2s = [], [], []
            for _ in range(reps):
                X = rng.standard_normal((n, D))
                Xc = X - X.mean(axis=0)
                G = Xc @ Xc.T / (n - 1)
                tr, tr2 = float(np.trace(G)), float((G * G).sum())
                prs.append(tr * tr / tr2)
                trs.append(tr)
                tr2s.append(tr2)
            sim = float(np.mean(prs))
            f1 = (n - 1) * d / ((n - 1) + d)
            EA, EB = np.mean(np.square(trs)), np.mean(tr2s)
            f2 = d * (n - 1) / (d + n)
            # delta method with empirical moments (A = Tr^2, B = TrG2)
            A = np.square(trs)
            covAB = float(np.cov(A, tr2s)[0, 1])
            varB = float(np.var(tr2s))
            f3 = (EA / EB) * (1 + varB / EB ** 2 - covAB / (EA * EB))
            rows.append({"d": d, "n": n, "sim_mean_pr": sim,
                         "f1_paper": f1, "f2_ratio_exp": f2, "f3_delta": f3,
                         "err_f1_pct": 100 * (f1 - sim) / sim,
                         "err_f2_pct": 100 * (f2 - sim) / sim,
                         "err_f3_pct": 100 * (f3 - sim) / sim})
            print(f"d={d:4d} n={n:5d}: sim={sim:8.2f}  "
                  f"f1 {rows[-1]['err_f1_pct']:+.2f}%  "
                  f"f2 {rows[-1]['err_f2_pct']:+.2f}%  "
                  f"f3 {rows[-1]['err_f3_pct']:+.2f}%", flush=True)
    worst = {k: max(abs(r[f"err_{k}_pct"]) for r in rows) for k in ("f1", "f2", "f3")}
    print("worst |err| %:", worst)
    with OUT.open("w") as f:
        json.dump({"rows": rows, "worst_abs_err_pct": worst}, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
