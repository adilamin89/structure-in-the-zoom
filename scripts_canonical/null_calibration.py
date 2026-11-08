"""The arms' nulls calibrated under each arm's own claim, on synthetic populations with a known answer (rung 1.13).

Populations: K = 8 classes of 40 rows in 192 features. Fluctuations r-dimensional and isotropic in a random subspace plus 5%
isotropic noise; each row optionally times a lognormal gain. Regimes: samples per class against each class's own fluctuation
dimension, n / PR_class = 0.5, 1, 4, 16 (r = 80, 40, 10, 3), and n / PR_class = 4 with heavy-tailed gains (log SD 0.7).
  between arm, its claim 'the class means differ': H0 = equal means, each class its own subspace (fluctuations differ).
    classmean_share(null='permutation') (rung 1.12) and classmean_share(null='wild') (1.13 default), 100 draws each.
  within arm, its claim 'the fluctuations differ with the means set aside': H0 = one shared subspace, class means present
    (Tr B / Tr W about 0.5). within_arm(null='permutation') (1.12, the paper's) and within_arm(null='freedman-lane') (1.13
    default), 50 draws each, half 0 of the odd/even split, a floor of four draws.
  power of the between arm: class means at Tr B / Tr W = 0.002, 0.004, 0.008 (100 seeds), detection at one-sided p <= 0.05.
Reported per regime and null: z mean (SE), z SD, one-sided type-I rate at 0.05, mean one-sided p (0.5 when calibrated).
Out: data_canonical/null_calibration.json. Run: python scripts_canonical/null_calibration.py [N_SEEDS] (default 200; about 40 CPU
minutes; RUNG_QUICK=1 for 10 seeds)."""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rung as r  # noqa: E402

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else (10 if os.environ.get("RUNG_QUICK") else 200)
K, N_PER, D = 8, 40, 192
REGIMES = [("n/PR 0.5", 80, 0.0), ("n/PR 1", 40, 0.0), ("n/PR 4", 10, 0.0), ("n/PR 16", 3, 0.0), ("n/PR 4, heavy gains", 10, 0.7)]
POWER = [("n/PR 1", 40, 0.0), ("n/PR 4", 10, 0.0), ("n/PR 4, heavy gains", 10, 0.7)]
BETAS = [0.002, 0.004, 0.008]


def population(seed, r_dim, gain_sd, shared, beta):
    g = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), N_PER); g.shuffle(y)
    bases = [np.linalg.qr(g.standard_normal((D, r_dim)))[0] for _ in range(1 if shared else K)]
    X = np.empty((len(y), D))
    for i, c in enumerate(y):
        X[i] = bases[0 if shared else c] @ g.standard_normal(r_dim) + 0.05 * np.sqrt(r_dim / D) * g.standard_normal(D)
    if gain_sd > 0:
        X *= np.exp(gain_sd * g.standard_normal(len(y)) - gain_sd ** 2 / 2)[:, None]
    if beta > 0:
        M = g.standard_normal((K, D)); M -= M.mean(0)
        M *= np.sqrt(beta * float(np.mean(np.sum(X ** 2, axis=1))) / np.mean(np.sum(M ** 2, axis=1)))
        X += M[y]
    return X, y


def summarize(rows):
    z = np.array([q["z"] for q in rows]); p = np.array([q["p"] for q in rows])
    return {"z_mean": float(z.mean()), "z_se": float(z.std(ddof=1) / np.sqrt(len(z))), "z_sd": float(z.std(ddof=1)),
            "typeI_p05": float((p <= 0.05).mean()), "p_mean": float(p.mean()), "n": len(z)}


def within_score(X, y, null, s):
    w = r.within_arm(X, y, halves="oddeven", n_perm=50, n_null=4, seed=42, perm_seed=30_000 + s, null=null)["half0"]
    nl = w["null_unrestricted"]
    return {"z": (w["meanfree_shift"] - nl["mean"]) / nl["sd"] if nl["sd"] > 0 else 0.0, "p": nl["p_one"]}


def main():
    t0 = time.time()
    out = {"design": {"K": K, "n_per": N_PER, "d": D, "regimes": REGIMES, "seeds": N_SEEDS, "between_draws": 100, "within_draws": 50},
           "calibration": {}, "power": {}}
    for name, r_dim, gain in REGIMES:
        res = {"between_permutation": [], "between_wild": [], "within_permutation": [], "within_freedman_lane": []}
        for s in range(N_SEEDS):
            X, y = population(s, r_dim, gain, shared=False, beta=0.0)
            for null in ("permutation", "wild"):
                b = r.classmean_share(X, y, n_perm=100, seed=10_000 + s, null=null)
                res[f"between_{null}"].append({"z": b["z"], "p": b["p_one"]})
            X, y = population(50_000 + s, r_dim, gain, shared=True, beta=0.5)
            res["within_permutation"].append(within_score(X, y, "permutation", s))
            res["within_freedman_lane"].append(within_score(X, y, "freedman-lane", s))
        out["calibration"][name] = {k: summarize(v) for k, v in res.items()}
        print(f"{name} ({time.time() - t0:.0f} s): " + " | ".join(f"{k} z SD {v['z_sd']:.2f} type-I {v['typeI_p05']:.3f}"
                                                                for k, v in out["calibration"][name].items()), flush=True)
    for name, r_dim, gain in POWER:
        for beta in BETAS:
            hits = {"permutation": 0, "wild": 0}
            n_pow = max(N_SEEDS // 2, 5)
            for s in range(n_pow):
                X, y = population(70_000 + s, r_dim, gain, shared=False, beta=beta)
                for null in hits:
                    hits[null] += r.classmean_share(X, y, n_perm=100, seed=20_000 + s, null=null)["p_one"] <= 0.05
            out["power"][f"{name}, beta {beta}"] = {k: v / n_pow for k, v in hits.items()}
            print(f"power {name}, beta {beta}: " + ", ".join(f"{k} {v / n_pow:.2f}" for k, v in hits.items()), flush=True)
    path = ROOT / "data_canonical" / ("null_calibration.json" if N_SEEDS >= 200 else "null_calibration_quick.json")
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path.name, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
