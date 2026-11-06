"""H2-R15: last-token lexical diversity as a confound of the language-model shift (C10's request; Osman, Baroni &
Macocco 2026, arXiv 2608.14361: last-token diversity moves layer-wise ID estimates).

The read-out is the last token's hidden state, so a class whose prompts end on few distinct tokens starts its ladder on a
low-dimensional rung at the embedding layer (App I: arithmetic ends on 'is' / 'equals'). For each cached battery:
 (a) distinct last tokens per class; Spearman across classes with the class's own PR at the embedding and middle layers;
 (b) matched variant: per class m prompts with m distinct last tokens (m = min distinct count; only if m >= 6), 20 random
     matched subsamples, each with the 50-order shift and its null (100 permutations x 20 orders);
 (c) where >= 3 last tokens are shared by >= 2 classes: labels permuted within last-token strata (100 x 20) against the
     observed 50-order shift of the full battery.
Estimator = run37's (h2_llm_battery). Out: data_canonical/h2_lasttoken_<model>.json
Usage: python h2_lasttoken.py <model_key> <hf_tokenizer> [battery ...]   (battery 'pilot' = App I's 160m states)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h2_llm_battery as hb  # noqa: E402

N_SUB, N_PERM, N_ORD = 20, 100, 20
PILOT = hb.OUTDIR / "llm_states_paper_battery.npz"


def load(mkey, b):
    if b == "pilot":
        z = np.load(PILOT)
        return z["H"], z["labels"], [str(x) for x in z["names"]], [str(p) for p in z["prompts"]], [1, 2, 3, 4, 6, 8]
    z = np.load(hb.STATES / f"{mkey}__{b}.npz")
    _, _, names, counts = hb.battery_prompts(b)
    return z["H"], z["labels"], [str(x) for x in z["names"]], [str(p) for p in z["prompts"]], counts


def stat_with_null(K, labels, nc, counts, rng, label_draw):
    mem = hb.members_of(labels, nc)
    sizes = hb.ladder(K, mem, list(range(nc)), counts)[1]
    th_f = hb.floor(K, len(labels), sizes, int(rng.integers(1 << 30)))
    obs = hb.order_avg(K, mem, [rng.permutation(nc).tolist() for _ in range(50)], counts, th_f)
    null = np.array([hb.order_avg(K, hb.members_of(label_draw(rng), nc),
                                  [rng.permutation(nc).tolist() for _ in range(N_ORD)], counts, th_f)
                     for _ in range(N_PERM)])
    return obs, float((1 + (null >= obs).sum()) / (N_PERM + 1)), float(null.mean()), float(null.std())


def run(mkey, tok_name, batteries):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_name)
    out_path = hb.OUTDIR / f"h2_lasttoken_{mkey}.json"
    out = json.load(open(out_path)) if out_path.exists() else {"model_key": mkey, "batteries": {}}
    for b in batteries:
        H, labels, names, prompts, counts = load(mkey, b)
        nc = len(names)
        last = np.array([tok(p).input_ids[-1] for p in prompts])
        distinct = [int(len(set(last[labels == c]))) for c in range(nc)]
        L = H.shape[0]
        mid = L // 2
        class_pr = {l: [hb.r17.pr_c(H[l][labels == c].astype(np.float64)) for c in range(nc)] for l in (0, mid)}
        rec = {"class_names": names, "distinct_last_tokens": distinct,
               "last_tokens_per_class": [[tok.decode([t]) for t in sorted(set(last[labels == c]))][:20] for c in range(nc)],
               "spearman_distinct_vs_classPR": {str(l): float(spearmanr(distinct, class_pr[l])[0])
                                                if len(set(distinct)) > 1 else None for l in (0, mid)},
               "class_PR": {str(l): class_pr[l] for l in (0, mid)}, "mid_layer": mid}
        print(f"[{mkey}/{b}] distinct last tokens {distinct}; rho(distinct, class PR) L0 "
              f"{rec['spearman_distinct_vs_classPR']['0']} L{mid} {rec['spearman_distinct_vs_classPR'][str(mid)]}",
              flush=True)
        m = min(distinct)
        rec["m"] = m
        # (b) matched subsamples
        if m >= 6:
            rng = np.random.default_rng(1500)
            subs = []
            for _ in range(N_SUB):
                keep = []
                for c in range(nc):
                    idx = np.where(labels == c)[0]
                    toks = rng.permutation(sorted(set(last[idx])))[:m]
                    for t in toks:
                        keep.append(rng.choice(idx[last[idx] == t]))
                subs.append(np.array(sorted(keep)))
            layers = []
            for l in range(L):
                vals = []
                for kidx in subs:
                    lab = labels[kidx]
                    K = hb.gram(H[l][kidx], "linear")
                    vals.append(stat_with_null(K, lab, nc, counts, rng, lambda r, lab=lab: lab[r.permutation(len(lab))]))
                v = np.array(vals)
                layers.append({"layer": l, "dbar_median": float(np.median(v[:, 0])), "p_median": float(np.median(v[:, 1])),
                               "frac_p05": float((v[:, 1] <= 0.05).mean())})
                print(f"  matched L{l:02d} dbar {layers[-1]['dbar_median']:+.3f} p_med {layers[-1]['p_median']:.3f} "
                      f"frac(p<=.05) {layers[-1]['frac_p05']:.2f}", flush=True)
            rec["matched"] = layers
        else:
            rec["matched"] = "cannot match (m < 6)"
        # (c) within-last-token strata
        shared = [t for t in set(last.tolist()) if len(set(labels[last == t])) >= 2]
        rec["n_shared_tokens"] = len(shared)
        if len(shared) >= 3:
            rng = np.random.default_rng(1600)
            strata = [np.where(last == t)[0] for t in set(last.tolist())]

            def draw(r):
                sl = labels.copy()
                for idx in strata:
                    if len(idx) > 1:
                        sl[idx] = sl[idx[r.permutation(len(idx))]]
                return sl
            layers = []
            for l in range(L):
                K = hb.gram(H[l], "linear")
                obs, p, nm, ns = stat_with_null(K, labels, nc, counts, rng, draw)
                layers.append({"layer": l, "dbar": obs, "p_one": p, "null_mean": nm, "null_sd": ns})
                print(f"  strata L{l:02d} dbar {obs:+.3f} p {p:.3f} null {nm:+.3f}±{ns:.3f}", flush=True)
            rec["within_lasttoken_strata"] = layers
        out["batteries"][b] = rec
        json.dump(out, open(out_path, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3:])
