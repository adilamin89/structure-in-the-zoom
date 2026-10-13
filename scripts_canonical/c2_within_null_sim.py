"""The finite-sample null of the within-class part of delta (Section 7, Appendix K).

Claim checked: "if every class fluctuated alike, the within-class part would be zero in expectation". At the population
level it is (a common W gives a constant within-class PR). With estimated class means it is not: each rung subtracts the
mean of every class it occupies, so an early declared rung (few classes) and its floor (all classes) keep different
residual degrees of freedom. This simulation measures the sign and size of that bias for identical classes, using the
same estimator as run72 (run63_deficit_decomposition.traces: class-size-weighted B, pooled W, both over n).

Design: K = 8 identical Gaussian classes (identity or power-law spectrum), n_c trials per class, d features; declared
ladder 1, 2, 3, 4, 6, 8 classes in order; floor = ten random subsets of the same sizes (labels kept, so their W is pooled
over the classes each subset happens to hold); within-class part = slope of log(w1^2/w2) on log n, ladder minus floor.
Configurations span the paper's regime direction (n_c in the hundreds, d up to a few hundred) within laptop time.
Out: data_canonical/c2_within_null_sim.json
"""
import json
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data_canonical"
RUNGS = [1, 2, 3, 4, 6, 8]
K = 8


def within_logpr(Y, lab):
    R = np.empty_like(Y)
    for c in np.unique(lab):
        m = lab == c
        R[m] = Y[m] - Y[m].mean(0)
    G = R @ R.T                      # n x n Gram of residuals; PR from traces
    w1 = np.trace(G); w2 = float((G * G).sum())
    return np.log(w1 ** 2 / w2)


def one_population(rng, n_c, d, spectrum):
    sd = np.ones(d) if spectrum == "iso" else np.arange(1, d + 1) ** -0.5
    X = rng.standard_normal((K * n_c, d)) * sd
    lab = np.repeat(np.arange(K), n_c)
    sizes, ys, fs = [], [], []
    for k in RUNGS:
        idx = np.where(lab < k)[0]
        n = len(idx); sizes.append(n)
        ys.append(within_logpr(X[idx], lab[idx]))
        fl = []
        for _ in range(10):
            j = rng.choice(len(lab), n, replace=False)
            fl.append(within_logpr(X[j], lab[j]))
        fs.append(np.mean(fl))
    x = np.log(sizes)
    return np.polyfit(x, np.array(ys) - np.array(fs), 1)[0]


if __name__ == "__main__":
    rng = np.random.default_rng(197)
    out = {"design": {"K": K, "rungs": RUNGS, "n_floor": 10}, "rows": []}
    for n_c, d, spec, reps in [(30, 20, "iso", 300), (100, 50, "iso", 200), (300, 100, "iso", 100),
                               (300, 300, "powerlaw", 60), (500, 200, "powerlaw", 40)]:
        t0 = time.time()
        v = np.array([one_population(rng, n_c, d, spec) for _ in range(reps)])
        row = {"n_per_class": n_c, "d": d, "spectrum": spec, "reps": reps, "mean": float(v.mean()),
               "se": float(v.std(ddof=1) / np.sqrt(reps))}
        out["rows"].append(row)
        print(f"n_c={n_c:4d} d={d:4d} {spec:8s}: within-class part {row['mean']:+.4f} (SE {row['se']:.4f})  {time.time()-t0:.0f}s",
              flush=True)
    json.dump(out, open(DATA / "c2_within_null_sim.json", "w"), indent=1)
