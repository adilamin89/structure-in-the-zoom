"""A known circuit as ground truth for the shift. GPT-2 small on indirect-object
identification (Wang et al. 2023, 'Interpretability in the wild'), whose name-mover heads (9.6, 9.9, 10.0) and backup name
movers (10.10, 10.6, 10.2, 10.1, 11.2, 9.7, 9.0, 11.9) were found by path patching to write the indirect object's name at
the final token.

Battery: 8 indirect-object (IO) names = 8 classes x 16 prompts; each prompt draws a subject name S from a disjoint pool,
a template (Wang et al.'s BABA forms and their ABBA mirrors, 8 + 8 per class), a place and an object (seed 0). Per head,
the head's write at the final token (its block of the input to attn.c_proj times its W_O rows; 144 heads x 768 dims).
Per head: the order-averaged shift along the IO axis (50 orders; floor 20 draws; llm_battery's estimator), its
permutation z (100 label draws, each averaged over 50 orders), a 5-fold logistic probe's held-out accuracy (span-exact fit,
llm_probes.span_fit), and the class-mean variance fraction Tr B / Tr C. Q10: AUC of the shift z for the 11 heads against the
other 133 (>= 0.8), beside the probe's and the variance fraction's AUC. Sanity: logit(IO) - logit(S) > 0 on most prompts.
Out: data_canonical/llm_ioi.json
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h2b = _load("h2b", "llm_battery.py")
h3p = _load("h3p", "llm_probes.py")
OUT = h2b.OUTDIR / "llm_ioi.json"
IO_NAMES = ["Mary", "John", "Tom", "James", "Sarah", "Paul", "Kate", "Dan"]
S_NAMES = ["Mike", "Lisa", "Anna", "David", "Emma", "Jack", "Laura", "Peter", "Rachel", "Steve", "Alice", "Mark"]
TEMPLATES = [
    "Then, {X} and {Y} went to the {P}. {S} gave a {O} to",
    "Then, {X} and {Y} had a lot of fun at the {P}. {S} gave a {O} to",
    "When {X} and {Y} got a {O} at the {P}, {S} decided to give it to",
    "After {X} and {Y} went to the {P}, {S} gave a {O} to",
    "While {X} and {Y} were working at the {P}, {S} gave a {O} to",
    "Friends {X} and {Y} went to the {P}. {S} gave a {O} to",
    "When {X} and {Y} went to the {P}, {S} gave a {O} to",
    "Then, {X} and {Y} were thinking about going to the {P}. {S} wanted to give a {O} to",
]
PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station"]
OBJECTS = ["ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack"]
NAME_MOVERS = ["9.6", "9.9", "10.0"]
BACKUP = ["10.10", "10.6", "10.2", "10.1", "11.2", "9.7", "9.0", "11.9"]
NEGATIVE = ["10.7", "11.10"]   # negative name movers: also write the IO name (with opposite sign); reported, not in the set
COUNTS = [1, 2, 3, 4, 6, 8]


def battery(tok):
    rng = np.random.default_rng(0)
    for n in IO_NAMES + S_NAMES:
        assert len(tok(" " + n).input_ids) == 1, n
    prompts, labels, io, s_names = [], [], [], []
    for c, a in enumerate(IO_NAMES):
        for i in range(16):
            t = TEMPLATES[i % 8]
            s = S_NAMES[rng.integers(len(S_NAMES))]
            x, y = (s, a) if i < 8 else (a, s)          # BABA (S first) for i < 8, ABBA (IO first) after
            prompts.append(t.format(X=x, Y=y, S=s, P=PLACES[rng.integers(8)], O=OBJECTS[rng.integers(8)]))
            labels.append(c)
            io.append(a)
            s_names.append(s)
    return prompts, np.array(labels), io, s_names


def encode(prompts, io, s_names):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype=torch.float32).eval()
    blocks = model.transformer.h
    nh, dh = model.config.n_head, model.config.n_embd // model.config.n_head
    Z = {}

    def pre_hook(i):
        def f(mod, args):
            Z[i] = args[0][0, -1].detach()          # merged head outputs (n_embd) at the final token
        return f
    for i, b in enumerate(blocks):
        b.attn.c_proj.register_forward_pre_hook(pre_hook(i))
    heads = np.zeros((len(blocks), nh, len(prompts), model.config.n_embd), np.float32)
    ld = []
    with torch.no_grad():
        for p_i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids
            logits = model(ids).logits[0, -1]
            ld.append(float(logits[tok(" " + io[p_i]).input_ids[0]] - logits[tok(" " + s_names[p_i]).input_ids[0]]))
            for i, b in enumerate(blocks):
                W = b.attn.c_proj.weight                # Conv1D: (in = n_embd, out = n_embd)
                z = Z[i]
                for h in range(nh):
                    heads[i, h, p_i] = (z[h * dh:(h + 1) * dh] @ W[h * dh:(h + 1) * dh]).numpy()
            # check: the heads' writes plus the bias reproduce c_proj's output
            if p_i == 0:
                full = blocks[0].attn.c_proj(Z[0][None])[0]
                rec = torch.from_numpy(heads[0, :, 0].sum(0)) + blocks[0].attn.c_proj.bias
                assert torch.allclose(full, rec, atol=1e-4), float((full - rec).abs().max())
    return heads, np.array(ld), tok


def auc(scores, positive):
    pos = scores[positive]
    neg = scores[~positive]
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    prompts, y, io, s_names = battery(tok)
    t0 = time.time()
    heads, ld, _ = encode(prompts, io, s_names)
    print(f"encoded {heads.shape} in {time.time() - t0:.0f}s; logit diff IO - S > 0 on {(ld > 0).mean():.3f} "
          f"(median {np.median(ld):+.2f})", flush=True)
    nl, nh, n = heads.shape[:3]
    nc = 8
    rng = np.random.default_rng(3801)
    orders = [rng.permutation(nc).tolist() for _ in range(50)]
    nrng = np.random.default_rng(4800)
    null_draws = [(h2b.null_labels("perm", y, nrng), [nrng.permutation(nc).tolist() for _ in range(50)]) for _ in range(100)]
    mem = h2b.members_of(y, nc)
    rows = []
    for l in range(nl):
        for h in range(nh):
            X = heads[l, h].astype(np.float64)
            K = h2b.gram(X, "linear")
            _, sizes = h2b.ladder(K, mem, list(range(nc)), COUNTS)
            th_f = h2b.floor(K, n, sizes, l * 100 + h + 1)
            dbar = h2b.order_avg(K, mem, orders, COUNTS, th_f)
            vals = np.array([h2b.order_avg(K, h2b.members_of(lab, nc), ords, COUNTS, th_f) for lab, ords in null_draws])
            z = float((dbar - vals.mean()) / vals.std()) if vals.std() > 0 else 0.0
            pred, _ = h3p.probe_cv(X, y, h3p.StratifiedKFold(5, shuffle=True, random_state=0), cv_c=False)
            Xc = X - X.mean(0)
            M = np.stack([X[y == c].mean(0) for c in range(nc)]) - X.mean(0)
            frac = float((np.bincount(y) @ (M ** 2).sum(1)) / (Xc ** 2).sum())
            rows.append({"head": f"{l}.{h}", "dbar": float(dbar), "null_mean": float(vals.mean()), "null_sd": float(vals.std()),
                         "z": z, "probe_acc": float((pred == y).mean()), "classmean_frac": frac,
                         "norm": float(np.sqrt((X ** 2).sum(1)).mean())})
        print(f"layer {l}: " + " ".join(f"{r['head']}:z{r['z']:+.1f}/acc{r['probe_acc']:.2f}" for r in rows[-nh:])
              + f" ({time.time() - t0:.0f}s)", flush=True)
    names = np.array([r["head"] for r in rows])
    pos = np.isin(names, NAME_MOVERS + BACKUP)
    out = {"n_prompts": int(n), "prompts": prompts,
           "labels": y.tolist(), "logit_diff": ld.tolist(), "frac_logit_diff_positive": float((ld > 0).mean()),
           "heads": rows, "positive_set": NAME_MOVERS + BACKUP}
    for key in ("z", "dbar", "probe_acc", "classmean_frac", "norm"):
        sc = np.array([r[key] for r in rows])
        out[f"AUC_{key}"] = auc(sc, pos)
        rank = (-sc).argsort().argsort() + 1
        out[f"rank_{key}"] = {h: int(rank[names == h][0]) for h in NAME_MOVERS + BACKUP + NEGATIVE}
        out[f"AUC_{key}_name_movers_only"] = auc(sc, np.isin(names, NAME_MOVERS))
    print("AUC", {k: round(v, 3) for k, v in out.items() if k.startswith("AUC")}, flush=True)
    print("ranks z", out["rank_z"], flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
