"""What the class means alone do to the topic shift, under the pooled-fluctuation control of Section 5 (Section 9 and Figure 8B
of the extended version).

The control used for V1 imaging and Allen: keep every class mean exactly, pool the within-class residual rows of all classes, deal
them back at random (class sizes kept) and re-centre them within each class, so the fluctuations keep their correlations and lose
their dependence on the class while every class's realized mean stays its own. Applied to Pythia-2.8B-deduped's topic states
(8 x 32, last token), every layer, covariance and z-scored frames: rung.zoom (declared order and the average over 50 random class
orders) on the data and on N_DRAWS control draws. Computed with rung 1.13.0; rung 1.14 replays it with order_average='random'
(its default for eight classes is the exact average over all 40,320 orders).
Needs raw/llm_states/pythia-2.8b-deduped__main__topics.npz (see _common.states). Out: v3/lm_pooled_control_recentered.json.
Run: python v3/lm_pooled_control_recentered.py [N_DRAWS=10]"""
import json
import sys
import time

import numpy as np

from _common import HERE, rung, states, zframe

N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def pooled(X, lab, rng):
    """Class means kept exactly; residual rows of all classes pooled, dealt back at random (class sizes kept) and re-centred within
    each class, so every class's realized mean is its own (without the re-centring the dealt residuals carry class means of their
    own and Tr B grows)."""
    mu = {c: X[lab == c].mean(0) for c in np.unique(lab)}
    M = np.stack([mu[c] for c in lab])
    D = (X - M)[rng.permutation(len(X))]
    for c in np.unique(lab):
        m = lab == c
        D[m] -= D[m].mean(0)
    Y = M + D
    for c in np.unique(lab):
        assert np.allclose(Y[lab == c].mean(0), mu[c], atol=1e-8 * (1 + np.abs(mu[c]).max()))
    return Y


def zoom(X, lab):
    return rung.zoom(X, lab, n_perm=0, k_orders=50, seed=0, order_average="random")


def main():
    H, lab, name, _, _ = states()
    out = {"states": name, "n_draws": N_DRAWS, "rung": rung.__version__, "frames": {}}
    for frame in ("cov", "z"):
        rows = []
        for l in range(H.shape[0]):
            t0 = time.time()
            X = H[l].astype(np.float64)
            if frame == "z":
                X = zframe(X)
            d = zoom(X, lab)
            rng = np.random.default_rng(1000 + l)
            sur = [zoom(pooled(X, lab, rng), lab) for _ in range(N_DRAWS)]
            row = {"layer": l, "delta": d["delta"], "delta_orderavg": d["delta_orderavg"],
                   "pooled_delta": float(np.mean([s["delta"] for s in sur])), "pooled_delta_sd": float(np.std([s["delta"] for s in sur], ddof=1)),
                   "pooled_orderavg": float(np.mean([s["delta_orderavg"] for s in sur]))}
            rows.append(row)
            print(f"{frame} L{l:2d}: declared {row['delta']:+.3f} -> pooled {row['pooled_delta']:+.3f} (SD {row['pooled_delta_sd']:.3f}) | "
                  f"order-avg {row['delta_orderavg']:+.3f} -> pooled {row['pooled_orderavg']:+.3f} | {time.time() - t0:.0f} s", flush=True)
        out["frames"][frame] = rows
        json.dump(out, open(HERE / "lm_pooled_control_recentered.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
