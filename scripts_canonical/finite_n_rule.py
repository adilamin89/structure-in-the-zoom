"""The convergence rule of the within-class part. With finite_n_identity.pr_fin, the sampling deficit D = W2 - W1 for mean-field
classes (eight equal classes of n samples, one overlap sigma: T_cd = sigma / PR_class off the diagonal, 1 / PR_class on it, Gaussian
fourth moments) on a grid of n / PR_class and sigma, at PR_class = 5, 30 and 100. Leading order (n >> PR_W): D ~ -g(sigma) PR_class / n,
g = minus the least-squares slope on log k (k = 1, 2, 3, 4, 6, 8) of h_k = PR_W / (k PR_class) - 1 / (1 + (k-1) sigma),
PR_W / PR_class = m = 8 / (1 + 7 sigma); its endpoint form is (m - 1) / ln 8. The floor's first rungs estimate the pooled fluctuations
(dimension m PR_class) from as many samples as the declared rungs use for k classes.
Out: data_canonical/finite_n_rule.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import finite_n_identity as P  # noqa: E402

DATA = HERE.parent / "data_canonical"
KS = np.array([1, 2, 3, 4, 6, 8])


def g_leading(sigma):
    h = 8 / (KS * (1 + 7 * sigma)) - 1 / (1 + (KS - 1) * sigma)
    return float(-np.polyfit(np.log(KS), h, 1)[0])


def deficit(sigma, pr_class, n, seed=0):
    mom = {"t": np.ones(8), "T": np.full((8, 8), sigma / pr_class) + np.eye(8) * (1 - sigma) / pr_class}
    mom["m4"] = mom["t"] ** 2 + 2 * np.diag(mom["T"])
    lab = np.repeat(np.arange(8), n)
    a = P.arms(mom, np.full(8, n), lab, sigma, seed)
    return a["W2"] - a["W1"], a["W1"]


def main():
    grid = []
    for sigma in (0.15, 0.2, 0.3, 0.44, 0.54, 0.7, 0.9):
        g = g_leading(sigma)
        for pr in (5, 30, 100):
            for r in (1, 2, 4, 8, 16, 32, 64):
                n = int(round(r * pr))
                if n < 3:
                    continue
                D, W1 = deficit(sigma, pr, n)
                grid.append({"sigma": sigma, "PR_class": pr, "n_over_PR": n / pr, "n": n, "D": D, "W1": W1, "g": g, "D_leading": -g * pr / n})
        print(f"sigma {sigma}: g {g:.3f}; D at n/PR 1,2,4,8,16,32,64 (PR 30): " +
              " ".join(f"{q['D']:+.3f}" for q in grid if q["sigma"] == sigma and q["PR_class"] == 30), flush=True)
    # the rule: smallest n / PR_class with |D| <= eps, per sigma (PR_class 30), interpolated on log n
    rule = {}
    for sigma in sorted({q["sigma"] for q in grid}):
        rows = [q for q in grid if q["sigma"] == sigma and q["PR_class"] == 30]
        for eps in (0.05, 0.03):
            x = np.log([q["n_over_PR"] for q in rows]); y = np.array([abs(q["D"]) for q in rows])
            ok = np.where(y <= eps)[0]
            if len(ok) == 0 or ok[0] == 0:
                rule[f"{sigma}_{eps}"] = None if len(ok) == 0 else float(np.exp(x[0])); continue
            i = ok[0]
            f = (np.log(y[i - 1]) - np.log(eps)) / (np.log(y[i - 1]) - np.log(y[i]))
            rule[f"{sigma}_{eps}"] = float(np.exp(x[i - 1] + f * (x[i] - x[i - 1])))
    print("RULE n/PR_class for |D| <= eps:", json.dumps(rule), flush=True)
    json.dump({"grid": grid, "rule": rule}, open(DATA / "finite_n_rule.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
