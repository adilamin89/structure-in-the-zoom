"""What carries a language model's topic lift, averaged over class orders and read net of the split's own estimation bias
(Section 9 and Appendix I of the extended version).

Pythia-2.8B-deduped on the topic axis (8 topics x 32 prompts, last-token states), every layer, covariance and z-scored frames. For
each of N_ORD random class orders, rung.decompose on that order with n_bias shape surrogates (each topic keeps its mean, size and
variance; all topics share one fluctuation shape): the raw split (within-class part D, class-mean part Tb + Cb + Cx), the
surrogates' split and the data's split net of them. Rows give the mean over orders and the SD over orders, and each topic's own
fluctuation dimension from split halves (rung.class_pr(estimate='split')).
rung's own default nets the declared order only; this script averages over orders. The extended version's numbers were computed
with rung 1.13.1; rung 1.14 returns the same values (decompose with an explicit order and n_bias is unchanged).
Needs raw/llm_states/pythia-2.8b-deduped__main__topics.npz (see _common.states). Out: v3/lm_split_orderavg.json.
Run: python v3/lm_split_orderavg.py [N_ORD=20] [N_BIAS=10] [LAYERS, e.g. 0,24]"""
import json
import sys
import time

import numpy as np

from _common import HERE, rung, states, zframe

N_ORD = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_BIAS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
LAYERS = [int(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else None


def cm(f):
    return f["Tb"] + f["Cb"] + f["Cx"]


def main():
    H, lab, name, dtype, _ = states()
    rng = np.random.default_rng(2026)
    orders = [[int(c) for c in rng.permutation(8)] for _ in range(N_ORD)]
    out = {"states": name, "dtype": dtype, "rung": rung.__version__, "n_orders": N_ORD, "n_bias": N_BIAS, "n_null": 20,
           "orders": orders, "frames": {}}
    layers = LAYERS if LAYERS is not None else list(range(H.shape[0]))
    for frame in ("cov", "z"):
        rows = []
        for l in layers:
            t0 = time.time()
            X = H[l].astype(np.float64)
            if frame == "z":
                X = zframe(X)
            per = []
            for o in orders:
                r = rung.decompose(X, lab, order=o, n_null=20, seed=42, n_bias=N_BIAS)
                raw, nb, bn = r["delta_split_free"], r["net_of_bias"]["delta_split_free"], r["bias_null"]["delta_split_free"]
                per.append({"delta": r["delta"], "D": raw["D"], "cm": cm(raw), "bias_D": bn["D"], "bias_cm": cm(bn),
                            "net_delta": r["net_of_bias"]["delta"], "net_D": nb["D"], "net_cm": cm(nb)})
            cpr = rung.class_pr(X, lab, estimate="split")
            row = {"layer": l}
            for k in per[0]:
                v = np.array([p[k] for p in per])
                row[k] = float(v.mean()); row[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            row["class_dim_split"] = [float(cpr[k]) for k in sorted(cpr)]
            row["n_per_class"] = 32
            rows.append(row)
            print(f"{frame} L{l:2d}: shift {row['delta']:+.3f} | raw D {row['D']:+.3f} cm {row['cm']:+.3f} | net D {row['net_D']:+.3f} "
                  f"cm {row['net_cm']:+.3f} (SD over orders {row['net_D_sd']:.3f}) | class dim {min(row['class_dim_split']):.0f}-"
                  f"{max(row['class_dim_split']):.0f} | {time.time() - t0:.1f}s", flush=True)
        out["frames"][frame] = rows
    fname = "lm_split_orderavg.json" if LAYERS is None else "lm_split_orderavg_smoke.json"
    json.dump(out, open(HERE / fname, "w"), indent=1)
    for frame, rows in out["frames"].items():
        nd = [r["net_D"] for r in rows]; nc = [r["net_cm"] for r in rows]; cd = [d for r in rows for d in r["class_dim_split"]]
        print(f"SUMMARY {frame}: net D {min(nd):+.3f}..{max(nd):+.3f}, positive {sum(x > 0 for x in nd)}/{len(nd)}; net class-mean "
              f"{min(nc):+.3f}..{max(nc):+.3f}; class dimension {min(cd):.0f}..{max(cd):.0f} (median {np.median(cd):.0f}) vs 32 prompts")
    print("DONE")


if __name__ == "__main__":
    main()
