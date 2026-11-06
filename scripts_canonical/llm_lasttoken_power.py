"""Power control for the last-token test (llm_lasttoken.py). (i) size-matched random subsamples without token matching; (ii) movable fraction of the
within-last-token-strata null. Out: data_canonical/llm_lasttoken_power_<model>.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_lasttoken as lt  # noqa: E402
hb = lt.hb


def run(mkey, tok_name, batteries):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_name)
    path = hb.OUTDIR / f"llm_lasttoken_power_{mkey}.json"
    out = json.load(open(path)) if path.exists() else {}      # merge: keep earlier batteries
    for b in batteries:
        H, labels, names, prompts, counts = lt.load(mkey, b)
        nc = len(names)
        last = np.array([tok(p).input_ids[-1] for p in prompts])
        m = min(len(set(last[labels == c])) for c in range(nc))
        multi = [t for t in set(last.tolist()) if len(set(labels[last == t])) >= 2]
        movable = float(np.isin(last, multi).mean())
        rng = np.random.default_rng(1800)
        subs = [np.sort(np.concatenate([rng.choice(np.where(labels == c)[0], m, replace=False) for c in range(nc)]))
                for _ in range(lt.N_SUB)]
        layers = []
        for l in range(H.shape[0]):
            vals = []
            for kidx in subs:
                lab = labels[kidx]
                K = hb.gram(H[l][kidx], "linear")
                vals.append(lt.stat_with_null(K, lab, nc, counts, rng, lambda r, lab=lab: lab[r.permutation(len(lab))]))
            v = np.array(vals)
            layers.append({"layer": l, "dbar_median": float(np.median(v[:, 0])), "p_median": float(np.median(v[:, 1])),
                           "frac_p05": float((v[:, 1] <= 0.05).mean())})
            print(f"{mkey}/{b} random-{m}/class L{l:02d} dbar {layers[-1]['dbar_median']:+.3f} p_med {layers[-1]['p_median']:.3f}",
                  flush=True)
        out[b] = {"m": m, "movable_fraction": movable, "random_matched_size": layers,
                  "n_pass_ge1": int(sum(x["p_median"] <= 0.05 for x in layers[1:])), "n_layers_ge1": len(layers) - 1}
        print(b, "movable", movable, "random-size pass", out[b]["n_pass_ge1"], "/", out[b]["n_layers_ge1"], flush=True)
        json.dump(out, open(hb.OUTDIR / f"llm_lasttoken_power_{mkey}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3:])
