"""The topic split in the declared order, raw, net of the shape surrogates and in excess of label shuffles (Appendix I's table of
the extended version, declared-order columns).

Pythia-2.8B-deduped on the topic axis (8 topics x 32 prompts, last-token states), every layer, covariance and z-scored frames:
rung.decompose(n_null=20, seed=layer, n_bias=N_BIAS, n_shuffle=N_SHUF) gives the shift, its raw split, the split net of the shape
surrogates (each topic keeps its mean, size and variance, all topics share one fluctuation shape) and the split's excess over label
shuffles. Computed with rung 1.13.0; rung 1.14 returns the same values with n_bias and n_shuffle passed explicitly.
Needs raw/llm_states/pythia-2.8b-deduped__main__topics.npz (see _common.states). Out: v3/lm_split_bias.json.
Run: python v3/lm_split_bias.py [N_BIAS=10] [N_SHUF=10]"""
import json
import sys
import time

import numpy as np

from _common import HERE, rung, states, zframe

N_BIAS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_SHUF = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def main():
    H, lab, name, dtype, device = states()
    out = {"states": name, "dtype": dtype, "device": device, "n_bias": N_BIAS, "n_shuffle": N_SHUF, "rung": rung.__version__, "frames": {}}
    for frame in ("cov", "z"):
        rows = []
        for l in range(H.shape[0]):
            t0 = time.time()
            X = H[l].astype(np.float64)
            if frame == "z":
                X = zframe(X)
            r = rung.decompose(X, lab, n_null=20, seed=l, n_bias=N_BIAS, n_shuffle=N_SHUF)
            raw = r["delta_split_free"]
            row = {"layer": l, "delta": r["delta"], "D": raw["D"], "class_means": raw["Tb"] + raw["Cb"] + raw["Cx"]}
            if "net_of_bias" in r:
                nb = r["net_of_bias"]["delta_split_free"]
                row.update(bias_D=r["bias_null"]["delta_split_free"]["D"], net_D=nb["D"], net_class_means=nb["Tb"] + nb["Cb"] + nb["Cx"],
                           net_delta=r["net_of_bias"]["delta"])
            if "excess" in r:
                ex = r["excess"]["delta_split"]
                row.update(shuffle_delta=r["shuffle"]["delta"], excess_D=ex["D"], excess_class_means=ex["Tb"] + ex["Cb"] + ex["Cx"])
            rows.append(row)
            print(f"{frame} L{l:2d}: shift {row['delta']:+.3f} | raw D {row['D']:+.3f} cm {row['class_means']:+.3f} | net of bias D "
                  f"{row.get('net_D', float('nan')):+.3f} cm {row.get('net_class_means', float('nan')):+.3f} | excess over shuffle D "
                  f"{row.get('excess_D', float('nan')):+.3f} cm {row.get('excess_class_means', float('nan')):+.3f} | {time.time() - t0:.0f} s",
                  flush=True)
        out["frames"][frame] = rows
        json.dump(out, open(HERE / "lm_split_bias.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
