"""Declared-axis ladder on a language model - the paper's ACTUAL construction.

Replaces the analytic-floor Pythia measurements with the real observable:
last-layer hidden states over a prompt battery of 8 TOPIC CLASSES x 8 prompts,
class-accumulation ladder (counts 1,2,3,4,6,8) against an EMPIRICAL matched
floor (10 random same-size prompt subsets), plus the shuffled-label control.

REGISTERED EXPECTATIONS (before run):
E1: delta_topic > 0 (topics occupy distinct subspaces = sector accumulation).
E2: shuffled-label control near zero.
E3: per-layer profile: delta_topic small at embedding layer, grows with depth.

Model: EleutherAI/pythia-160m (local HF cache). CPU.
Output: data/llm_declared_axis.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "llm_declared_axis.json"

TOPICS = {
    "arithmetic": ["Two plus two equals", "Seven times eight is", "The square root of sixteen is",
                   "Half of ninety is", "Ten minus three equals", "Five squared is",
                   "The sum of one through ten is", "Twelve divided by four is"],
    "geography": ["The capital of France is", "The longest river in the world is",
                  "Mount Everest is located in", "The largest ocean on Earth is",
                  "The Sahara desert lies in", "Japan is an island nation in",
                  "The Amazon rainforest spans", "Norway borders Sweden and"],
    "biology": ["The mitochondria is the", "DNA is composed of", "Photosynthesis converts sunlight into",
                "The human heart pumps", "Neurons communicate through", "Enzymes catalyze chemical",
                "Red blood cells carry", "The immune system defends"],
    "history": ["The Roman Empire fell in", "World War Two ended in",
                "The French Revolution began in", "Ancient Egypt built the",
                "The printing press was invented by", "The Cold War divided",
                "The Industrial Revolution started in", "Napoleon was defeated at"],
    "cooking": ["To bake bread you need", "The sauce thickens when you",
                "Simmer the broth until", "Knead the dough for",
                "Season the steak with", "Whisk the eggs before",
                "Preheat the oven to", "Chop the onions and"],
    "sports": ["The striker scored a", "The marathon runner paced",
               "In tennis a tiebreak occurs", "The goalkeeper saved the",
               "Basketball teams have five", "The relay team passed",
               "The boxer dodged the", "Olympic swimmers train"],
    "law": ["The defendant pleaded not", "The contract becomes void when",
            "The jury deliberated for", "Copyright protects original",
            "The witness testified under", "The appeal was filed in",
            "The statute of limitations", "The judge overruled the"],
    "weather": ["The hurricane made landfall near", "Snow accumulated overnight in",
                "The forecast predicts heavy", "Lightning struck the tall",
                "The drought lasted three", "Fog rolled in from",
                "The temperature dropped below", "Monsoon season brings"],
}
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_NULL = 10
N_SHUF = 5


def pr_c(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, float))
    y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def ladder_delta(H, labels, rng, n_shuf=0):
    classes = sorted(set(labels))
    members = [np.where(np.asarray(labels) == c)[0] for c in classes]
    sizes, prs = [], []
    for k in BIN_COUNTS:
        sel = np.concatenate(members[:k])
        if len(sel) < 3:
            continue
        sizes.append(len(sel))
        prs.append(pr_c(H[sel]))
    th_o = slope(sizes, prs)
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(pr_c(H[rng.choice(len(H), s, replace=False)]), 1e-9))
    th_f = slope(sizes, np.exp(nl.mean(axis=0)))
    shufs = []
    lab = np.asarray(labels)
    for s in range(n_shuf):
        srng = np.random.default_rng(700 + s)
        perm = lab[srng.permutation(len(lab))]
        m2 = [np.where(perm == c)[0] for c in classes]
        sz2, pr2 = [], []
        for k in BIN_COUNTS:
            sel = np.concatenate(m2[:k])
            if len(sel) < 3:
                continue
            sz2.append(len(sel))
            pr2.append(pr_c(H[sel]))
        shufs.append(slope(sz2, pr2) - th_f)
    return th_o, th_f, th_o - th_f, shufs


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "EleutherAI/pythia-160m"
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()

    prompts, labels = [], []
    for t, ps in TOPICS.items():
        prompts += ps
        labels += [t] * len(ps)
    print(f"{len(prompts)} prompts, {len(TOPICS)} topic classes", flush=True)

    with torch.no_grad():
        enc = tok(prompts, return_tensors="pt", padding=True)
        outs = model(**enc, output_hidden_states=True)
    # last non-pad token per prompt, per layer
    lengths = enc["attention_mask"].sum(dim=1) - 1
    out = {"model": name, "n_prompts": len(prompts), "per_layer": []}
    rng = np.random.default_rng(42)
    for li, h in enumerate(outs.hidden_states):
        H = h[torch.arange(len(prompts)), lengths].numpy().astype(np.float64)
        n_shuf = N_SHUF if li == len(outs.hidden_states) - 1 else 0
        th_o, th_f, d, shufs = ladder_delta(H, labels, np.random.default_rng(42 + li),
                                            n_shuf=n_shuf)
        row = {"layer": li, "theta_obs": th_o, "theta_floor": th_f, "delta_topic": d}
        if shufs:
            row["shuffle_mean"] = float(np.mean(shufs))
            row["shuffle_sd"] = float(np.std(shufs))
        out["per_layer"].append(row)
        print(f"layer {li:2d}: delta_topic={d:+.4f}" +
              (f"  shuffle={row['shuffle_mean']:+.4f}±{row['shuffle_sd']:.4f}"
               if shufs else ""), flush=True)

    with OUT.open("w") as f:
        json.dump(out, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
