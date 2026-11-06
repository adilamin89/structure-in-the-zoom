"""Order-averaged shift and its own null (200 permutations x 50 orders) for the spectrum-repaired Grams (z-score,
top-4-PC removed), on cached states. Out: data_canonical/llm_repaired_nulls.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_battery as hb  # noqa: E402


def main():
    path = hb.OUTDIR / "llm_repaired_nulls.json"
    out = json.load(open(path)) if path.exists() else {}
    for mkey in sys.argv[1:]:
        out[mkey] = {}
        for ax in ("world_knowledge", "language_type"):
            z = np.load(hb.STATES / f"{mkey}__{ax}.npz")
            H, labels = z["H"], z["labels"]
            nc = labels.max() + 1
            counts = [1, 2, 3, 4, 6, 8]
            orng = np.random.default_rng(3701)
            orders = [orng.permutation(nc).tolist() for _ in range(50)]
            rng = np.random.default_rng(4800)
            draws = [(hb.null_labels("perm", labels, rng), [rng.permutation(nc).tolist() for _ in range(50)]) for _ in range(200)]
            res = {}
            for v in ("zscore", "pc4"):
                rows = []
                for l in range(1, H.shape[0]):
                    K = hb.gram(H[l], v)
                    mem = hb.members_of(labels, nc)
                    sizes = hb.ladder(K, mem, list(range(nc)), counts)[1]
                    th_f = hb.floor(K, len(labels), sizes, l * 100 + 1)
                    obs = hb.order_avg(K, mem, orders, counts, th_f)
                    null = np.array([hb.order_avg(K, hb.members_of(lab, nc), o, counts, th_f) for lab, o in draws])
                    p = float((1 + (null >= obs).sum()) / 201)
                    rows.append({"layer": l, "dbar50": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()), "p_one": p})
                    print(f"{mkey}/{ax}/{v} L{l:02d} dbar {obs:+.3f} null {null.mean():+.3f}±{null.std():.3f} p {p:.3f}", flush=True)
                res[v] = {"rows": rows, "n_above": int(sum(r["p_one"] <= 0.05 for r in rows)), "n": len(rows)}
            out[mkey][ax] = res
            json.dump(out, open(path, "w"), indent=1)
            print(mkey, ax, {v: (res[v]["n_above"], res[v]["n"]) for v in res}, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
