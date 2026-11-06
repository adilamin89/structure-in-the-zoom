"""Baroni 64 pairs with the pair-preserving order average (48 orders) against within-pair swaps averaged over the
same orders. Estimator = llm_battery (run37). Out: data_canonical/llm_baroni_pairs.json"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_battery as hb  # noqa: E402

N_NULL = 200
ORDERS = [sum(([2 * c + m, 2 * c + 1 - m] for c, m in zip(perm, mem)), [])
          for perm in itertools.permutations(range(3)) for mem in itertools.product((0, 1), repeat=3)]


def main():
    out = json.load(open(hb.OUTDIR / "llm_baroni_pairs.json")) if (hb.OUTDIR / "llm_baroni_pairs.json").exists() else {}
    for mkey in sys.argv[1:]:
        z = np.load(hb.STATES / f"{mkey}__baroni64.npz")
        H, labels = z["H"], z["labels"]
        counts = [1, 2, 3, 4, 6]
        rng = np.random.default_rng(1700)
        draws = [hb.null_labels("swap", labels, rng) for _ in range(N_NULL)]
        rows = []
        for l in range(H.shape[0]):
            K = hb.gram(H[l], "linear")
            mem0 = hb.members_of(labels, 6)
            sizes = hb.ladder(K, mem0, list(range(6)), counts)[1]
            th_f = hb.floor(K, len(labels), sizes, l * 100 + 1)
            obs = hb.order_avg(K, mem0, ORDERS, counts, th_f)
            null = np.array([hb.order_avg(K, hb.members_of(d, 6), ORDERS, counts, th_f) for d in draws])
            nm = null.mean()
            p = float((1 + (np.abs(null - nm) >= abs(obs - nm)).sum()) / (N_NULL + 1))
            rows.append({"layer": l, "pairavg": obs, "swap_null_mean": float(nm), "swap_null_sd": float(null.std()), "p_two": p})
            print(f"{mkey} L{l:02d} pair-avg {obs:+.4f} swap null {nm:+.4f}±{null.std():.4f} excess {obs - nm:+.4f} p {p:.3f}", flush=True)
        n = len(rows)
        late = sorted([r for r in rows if r["layer"] >= n / 2], key=lambda r: r["p_two"])
        bh = 0
        for i, r in enumerate(late):
            if r["p_two"] <= 0.05 * (i + 1) / len(late):
                bh = i + 1
        out[mkey] = {"rows": rows, "late_beyond": int(sum(r["p_two"] < 0.05 for r in late)), "n_late": len(late), "bh_count": bh}
        json.dump(out, open(hb.OUTDIR / "llm_baroni_pairs.json", "w"), indent=1)
        print(mkey, "late beyond swap", out[mkey]["late_beyond"], "/", len(late), "BH", bh, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
