"""L=128 Ising at criticality - registered test of delta scale-invariance.

REGISTERED PREDICTION (2026-08-22, before run): delta is a log-log ladder slope,
an intensive diagnostic, NOT an extensive susceptibility. If the ladder samples
the scale-invariant critical regime, delta(Tc, L) should be approximately
L-independent (within finite-size drift ~15%), NOT grow as L^{gamma/nu} = L^1.75.
Observed so far (uncentered_signed): L=32 +0.401, L=64 +0.346.
Prediction: L=128 in [0.28, 0.40] (continued mild finite-size drift, no growth).

Output: data/ising_L128_tc.json
"""
import json
import time
from pathlib import Path

import numpy as np

import ising_wolff_matrix as iwm
from ising_wolff_matrix import T_C, ARMS, delta_arm, wolff_chain, N_CHAINS

# L=128 needs far more equilibration than 300 steps (first run: U(Tc) = -0.31,
# under-equilibrated; Binder caught it). Override module constants.
iwm.N_EQUIL = 5000
iwm.N_SPACING = 10

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "ising_L128_tc.json"

TEMPS = [0.947 * T_C, T_C, 1.058 * T_C]
L = 128


def main():
    results = []
    for T in TEMPS:
        t0 = time.time()
        per_arm = {name: [] for name, _, _ in ARMS}
        ms = []
        for c in range(N_CHAINS):
            configs, rng = wolff_chain(L, T, seed=90_000 + 10 * c + int(T * 7))
            ms.append(configs.mean(axis=1))
            for name, cent, sk in ARMS:
                per_arm[name].append(delta_arm(configs, rng, cent, sk))
        m = np.concatenate(ms)
        row = {"L": L, "T": T, "T_over_Tc": T / T_C,
               "binder_U": float(1 - np.mean(m**4) / (3 * np.mean(m**2) ** 2))}
        for name in per_arm:
            row[name] = {"mean": float(np.mean(per_arm[name])),
                         "sd": float(np.std(per_arm[name]))}
        results.append(row)
        print(f"L={L} T/Tc={T/T_C:.3f}: " +
              " ".join(f"{n}={row[n]['mean']:+.3f}±{row[n]['sd']:.3f}" for n in per_arm) +
              f" U={row['binder_U']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        with OUT.open("w") as f:
            json.dump({"registered_prediction": "delta(Tc,128) in [0.28,0.40] "
                       "uncentered_signed; L-independence not L^1.75 growth",
                       "results": results}, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
