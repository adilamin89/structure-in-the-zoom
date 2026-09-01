"""Run 43 — Baroni syntactic-complexity contrasts through the delta battery (subsumption demo).

WHY: Baroni et al. 2026 (arXiv:2601.03779) show three psycholinguistic
complexity contrasts modulate LLM representations (via intrinsic dimension).
Their stimuli are classic minimal pairs — natural carriers — so the battery
can ask the sharper, floor-controlled question at BOTH null levels: does the
complexity contrast organize covariance accumulation beyond the shared
carrier content?

DATA (github.com/franfranz/syntactic_complexity_in_LLMs, cloned to /tmp):
  that_vs_and-c2.txt        subordinated \t coordinated
  center_right_clauses.txt  center-embedding \t right-branching
  ambiguity_dataset.txt     ambiguous \t high-attachment \t low-attachment
                            (member 2 = HIGH attachment as the unambiguous
                            partner; choice registered here)
First 16 pairs per contrast -> 6 classes x 16 = 96 prompts, 48 carriers.
Canonical accumulation order contrast-major (sub, coord, center, right,
ambig, unambig); ladder counts 1/2/3/4/6 (6-class cap).

NULLS per layer (500 each): full label permutation (label-free level) and
within-pair member swaps (nuisance-preserving level; carrier composition
invariant, only the contrast bit moves).

REGISTERED EXPECTATIONS (written before the run):
BA1: label-free permutation strongly exceeded at most layers (minimal-pair
     carrier composition, the run42-B physics; sign expected negative).
BA2: THE informative cell — unlike BLiMP's single-morpheme flip, these
     members differ in arrangement (connective/word order), and run31 shows
     arrangement carries structure: expect the strat null EXCEEDED
     (two-sided p<0.05) at mid/late layers on 2.8B; 160m weaker (Baroni:
     complexity effects strengthen with scale).
BA3: if BA2 passes, the battery subsumes their contrast findings with floor
     control added; if it fails, the contrasts live below the accumulation
     probe's resolution at 16 pairs — either way reported.

Out: ../data_canonical/run43_baroni_complexity.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run43_baroni_complexity.json"
SRC = Path("/tmp/syntactic_complexity_in_LLMs/data/complexity_datasets")

spec = importlib.util.spec_from_file_location(
    "r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r37)
r17 = r37.r17

N_PAIRS = 16
N_NULL = 500
LADDER = [1, 2, 3, 4, 6]
CLASS_NAMES = ["subordinated", "coordinated", "center_embedding",
               "right_branching", "ambiguous", "unambiguous_high"]
MODELS = [("EleutherAI/pythia-160m", "cpu"),
          ("EleutherAI/pythia-2.8b-deduped", "mps")]


def load_pairs():
    pairs = []
    for fname, cols in (("that_vs_and-c2.txt", (0, 1)),
                        ("center_right_clauses.txt", (0, 1)),
                        ("ambiguity_dataset.txt", (0, 1))):
        rows = []
        with open(SRC / fname) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    rows.append((parts[cols[0]], parts[cols[1]]))
                if len(rows) == N_PAIRS:
                    break
        pairs.append(rows)
    return pairs


def ladder_slope6(K, members, order):
    sizes, prs = [], []
    for c in LADDER:
        sel = np.concatenate([members[o] for o in order[:c]])
        sizes.append(len(sel))
        prs.append(r37.subset_pr(K, sel))
    return r17.slope(sizes, np.asarray(prs)), sizes


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    contrast_pairs = load_pairs()
    prompts, labels = [], []
    for c, rows in enumerate(contrast_pairs):
        for a, b in rows:
            prompts.extend([a, b])
            labels.extend([2 * c, 2 * c + 1])
    labels = np.array(labels)
    n_total, n_classes = len(prompts), 6
    canonical = list(range(n_classes))
    print(f"{n_total} prompts, {n_total//2} carriers", flush=True)

    prng = np.random.default_rng(4300)
    perms = [prng.permutation(n_total) for _ in range(N_NULL)]
    srng = np.random.default_rng(4301)
    swap_masks = [srng.integers(0, 2, n_total // 2) for _ in range(N_NULL)]

    out = {"n_pairs": N_PAIRS, "n_null": N_NULL,
           "class_names": CLASS_NAMES,
           "prompts": prompts,  # provenance: the exact stimuli used
           "labels": [int(x) for x in labels], "models": {}}
    if OUT.exists():
        out = json.load(open(OUT))
        print(f"resumed: {list(out['models'])}", flush=True)

    for model_name, device in MODELS:
        mkey = model_name.split("/")[-1]
        if mkey in out["models"]:
            continue
        print(f"\nloading {model_name} ({device})...", flush=True)
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        per_layer = r17.get_hidden_states(model, tokenizer, prompts,
                                          device=device)

        layers_out = []
        for l in range(len(per_layer)):
            X = per_layer[l].astype(np.float64)
            X = X / (X.std() + 1e-9)
            K = X @ X.T
            members = [np.where(labels == c)[0] for c in range(n_classes)]
            th_obs, sizes = ladder_slope6(K, members, canonical)
            frng = np.random.default_rng(l * 100 + 1)
            nl = np.zeros((r37.N_NULL_FLOOR, len(sizes)))
            for d in range(r37.N_NULL_FLOOR):
                for k, s in enumerate(sizes):
                    nl[d, k] = np.log(max(r37.subset_pr(
                        K, frng.choice(n_total, s, replace=False)), 1e-9))
            th_f = r17.slope(sizes, np.exp(nl.mean(axis=0)))
            d_obs = th_obs - th_f

            row = {"layer": l, "delta_plain": float(d_obs)}
            for nname in ("perm", "strat"):
                nd = []
                for i in range(N_NULL):
                    if nname == "perm":
                        pl = labels[perms[i]]
                    else:
                        pl = labels.copy()
                        for pair_idx, bit in enumerate(swap_masks[i]):
                            if bit:
                                a, b = 2 * pair_idx, 2 * pair_idx + 1
                                pl[a], pl[b] = pl[b], pl[a]
                    mem = [np.where(pl == c)[0] for c in range(n_classes)]
                    nd.append(ladder_slope6(K, mem, canonical)[0] - th_f)
                nd = np.asarray(nd)
                nm, ns = float(nd.mean()), float(nd.std())
                row[f"{nname}_null_mean"] = nm
                row[f"{nname}_null_sd"] = ns
                row[f"{nname}_z"] = (d_obs - nm) / ns if ns > 0 else 0.0
                row[f"{nname}_p_two"] = float(
                    (1 + (np.abs(nd - nm) >= abs(d_obs - nm)).sum())
                    / (N_NULL + 1))
            layers_out.append(row)
            print(f"    L{l:02d} d={d_obs:+.3f} | perm z={row['perm_z']:+.1f} "
                  f"p={row['perm_p_two']:.4f} | strat z={row['strat_z']:+.1f} "
                  f"p={row['strat_p_two']:.4f}", flush=True)
        out["models"][mkey] = {"layers": layers_out}
        json.dump(out, open(OUT, "w"), indent=1)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    print("\nDONE run43", flush=True)


if __name__ == "__main__":
    main()
