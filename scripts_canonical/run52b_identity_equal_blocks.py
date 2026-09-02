"""Run 52b (S75) - the blocking identity checked the way the algebra states it.

run52's R1 compared the blocked sector powers with P_1 computed over ALL neurons,
while the sorted arms drop the n mod K remainder and the k-means arm has unequal
block sizes; the 0.10 "identity deviation" was that mismatch, not the identity.
Here: equal-size blocks only (random, orientation-sorted, direction-sorted at
K in {4, 16, 32, 64}), P_1 restricted to the neurons the blocks cover.

REGISTERED PREDICTION (before the run):
R1b: P_K(l) * K / P_1,covered(l) = 1/K + (1 - 1/K) rho_l to < 1e-10 in every
     cell, rho_l the power-weighted mean pairwise normalized product of the
     sector-l harmonics within a block.
Out: ../data_canonical/run52b_identity_equal_blocks.json
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
DATA = HERE.parent / "data_canonical"
OUT = DATA / "run52b_identity_equal_blocks.json"
NB = 8; KS = [4, 16, 32, 64]; ANG = 2 * np.pi * np.arange(NB) / NB


def harmonics(M):
    Mc = M - M.mean(0, keepdims=True)
    return np.stack([(Mc * np.exp(1j * l * ANG[:, None])).sum(0) for l in (1, 2)])


def main():
    rows = []; worst = 0.0
    for f in sorted(RAW.glob("gratings_*.npy")):
        dat = np.load(f, allow_pickle=True).item()
        X = dat["sresp"].T.astype(np.float64)
        phi = np.asarray(dat["istim"]).ravel() % (2 * np.pi)
        b = np.floor(phi / (2 * np.pi) * NB).astype(int) % NB
        if np.bincount(b, minlength=NB).min() < 10:
            continue
        M = np.stack([X[b == k].mean(0) for k in range(NB)]); z = harmonics(M)
        orders = {"random": np.random.default_rng(1).permutation(M.shape[1]),
                  "ori_sorted": np.argsort((np.angle(z[1]) / 2) % np.pi),
                  "dir_sorted": np.argsort(np.angle(z[0]) % (2 * np.pi))}
        rec = {"name": f.stem, "cells": {}}
        for arm, order in orders.items():
            for K in KS:
                nb = M.shape[1] // K; idx = order[:nb * K].reshape(nb, K)
                for l in (0, 1):
                    zz = z[l][idx]                                    # nb x K
                    P1 = float(np.sum(np.abs(zz) ** 2))               # covered neurons only
                    PK = float(np.sum(np.abs(zz.mean(1)) ** 2))       # blocked (K-averaged) power
                    s_all = np.abs(zz.sum(1)) ** 2; s_self = np.sum(np.abs(zz) ** 2, 1)
                    rho = float(np.sum(s_all - s_self) / np.sum((K - 1) * s_self))
                    lhs = PK * K / P1; rhs = 1 / K + (1 - 1 / K) * rho
                    dev = abs(lhs - rhs); worst = max(worst, dev)
                    rec["cells"][f"{arm}_K{K}_l{l+1}"] = {"rho": rho, "lhs": lhs, "rhs": rhs, "dev": dev}
        rows.append(rec)
        print(f"  {f.stem[:28]:28s} max dev {max(c['dev'] for c in rec['cells'].values()):.2e}", flush=True)
    verdict = {"R1b_max_dev": worst, "R1b_pass": bool(worst < 1e-10), "n_recordings": len(rows)}
    print("VERDICT", json.dumps(verdict))
    json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1); print("wrote", OUT)


if __name__ == "__main__":
    main()
