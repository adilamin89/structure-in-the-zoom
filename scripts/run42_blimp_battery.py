"""Run 42 — BLiMP syntactic axes with the NATIVE stratified floor.

WHY: the scrambled-order control (run31) answers the lexical alternative
with out-of-distribution text (a round-4 caveat). BLiMP minimal pairs are
natural carriers: each pair contributes one grammatical and one
ungrammatical sentence with near-identical vocabulary, so the run36
stratified floor applies IN-DISTRIBUTION — within-pair label swaps preserve
carrier composition exactly and randomize only the grammaticality bit.
Bonus control: every BLiMP sentence ends with a period, so the last-token
confound is uniform by construction.

DESIGN A — phenomenon axis (8 classes x 16 grammatical sentences):
  anaphor_gender_agreement, animate_subject_passive, principle_A_case_1,
  existential_there_object_raising, determiner_noun_agreement_1,
  wh_questions_object_gap, wh_island, npi_present_1 (one paradigm per
  linguistics term, first 16 items each). Analyzed with the run37 machinery
  (500 label permutations per layer + order-averaged deltabar, K=50).

DESIGN B — grammaticality x phenomenon (8 classes = 4 phenomena x good/bad,
  16 pairs each = 128 prompts): regular_plural_subject_verb_agreement_1,
  wh_island, npi_present_1, irregular_past_participle_verbs (four distinct
  mechanisms: agreement, movement, polarity, morphology). Canonical
  accumulation order is phenomenon-major (sv_good, sv_bad, isl_good, ...):
  rung 2 is a within-phenomenon minimal-pair rung. Three quantities/layer:
    delta_plain — observed minus random-subset floor (ordinary level);
    full-perm null — 500 label permutations (label-free level);
    strat null   — 500 independent within-pair good/bad swaps (the
                   nuisance-preserving level: phenomenon labels and carrier
                   composition are invariant; only grammaticality moves).

REGISTERED EXPECTATIONS (written before the run):
G1: Design A behaves CONTENT-LIKE (paradigm vocabulary is distinctive):
    positive delta certified per-layer (p<0.05, canonical or order-avg) at
    most layers, both models; IE declared negative (lexical dilution, the
    wk pattern).
G2: Design B delta_plain significant vs the full permutation at most layers
    (phenomenon separation dominates the partition).
G3: DECISION CELL (two-sided at late layers, 2.8B primary): observed theta
    outside the strat null reads grammaticality-linked covariance beyond
    carriers, in-distribution; a null result scopes the probe (single-bit
    pairwise contrasts below resolution at 64 pairs). Either outcome is
    informative; 160m expected weaker.

Out: ../data_canonical/run42_blimp_battery.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run42_blimp_battery.json"

spec = importlib.util.spec_from_file_location(
    "r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r37)
r17 = r37.r17
r37.DECLARED_DIRECTION["blimp_phenomenon"] = {"IE": -1, "PE": +1}

A_PARADIGMS = [
    "anaphor_gender_agreement", "animate_subject_passive",
    "principle_A_case_1", "existential_there_object_raising",
    "determiner_noun_agreement_1", "wh_questions_object_gap",
    "wh_island", "npi_present_1"]
B_PARADIGMS = [
    "regular_plural_subject_verb_agreement_1", "wh_island",
    "npi_present_1", "irregular_past_participle_verbs"]
N_ITEMS = 16
N_NULL_B = 500
MODELS = [("EleutherAI/pythia-160m", "cpu"),
          ("EleutherAI/pythia-2.8b-deduped", "mps")]


def load_blimp():
    from datasets import load_dataset
    a_classes, b_pairs = {}, {}
    for p in A_PARADIGMS:
        ds = load_dataset("nyu-mll/blimp", p, split="train")
        a_classes[p] = [ds[i]["sentence_good"] for i in range(N_ITEMS)]
    for p in B_PARADIGMS:
        ds = load_dataset("nyu-mll/blimp", p, split="train")
        b_pairs[p] = [(ds[i]["sentence_good"], ds[i]["sentence_bad"])
                      for i in range(N_ITEMS)]
    return a_classes, b_pairs


def design_b_analysis(per_layer, n_phen, n_pairs_per_phen):
    """Per-layer: delta_plain, full-perm null, within-pair strat null.

    Prompt layout: phenomenon-major, pair-major, good before bad —
    prompt index = ph*(2*n_pairs) + 2*pair + (0 good | 1 bad).
    Class id = 2*ph + (0 good | 1 bad); canonical order = class id order.
    """
    n_classes = 2 * n_phen
    n_total = n_phen * 2 * n_pairs_per_phen
    labels = np.array([2 * (i // (2 * n_pairs_per_phen)) + (i % 2)
                       for i in range(n_total)])
    canonical = list(range(n_classes))
    prng = np.random.default_rng(4200)
    perms = [prng.permutation(n_total) for _ in range(N_NULL_B)]
    srng = np.random.default_rng(4201)
    swap_masks = [srng.integers(0, 2, n_total // 2) for _ in range(N_NULL_B)]

    layers_out = []
    for l in range(len(per_layer)):
        X = per_layer[l].astype(np.float64)
        X = X / (X.std() + 1e-9)
        K = X @ X.T
        members = [np.where(labels == c)[0] for c in range(n_classes)]
        th_obs, sizes = r37.ladder_slope(K, members, canonical)
        th_f = r37.floor_slope(K, n_total, sizes,
                               np.random.default_rng(l * 100 + 1))
        d_obs = th_obs - th_f

        perm_d = []
        for perm in perms:
            pl = labels[perm]
            mem = [np.where(pl == c)[0] for c in range(n_classes)]
            perm_d.append(r37.ladder_slope(K, mem, canonical)[0] - th_f)
        strat_d = []
        for mask in swap_masks:
            sl = labels.copy()
            for pair_idx, bit in enumerate(mask):
                if bit:  # swap good/bad within this pair
                    i, j = 2 * pair_idx, 2 * pair_idx + 1
                    sl[i], sl[j] = sl[j], sl[i]
            mem = [np.where(sl == c)[0] for c in range(n_classes)]
            strat_d.append(r37.ladder_slope(K, mem, canonical)[0] - th_f)

        row = {"layer": l, "delta_plain": float(d_obs)}
        for nname, nd in (("perm", perm_d), ("strat", strat_d)):
            nd = np.asarray(nd)
            nm, ns = float(nd.mean()), float(nd.std())
            row[f"{nname}_null_mean"] = nm
            row[f"{nname}_null_sd"] = ns
            row[f"{nname}_z"] = (d_obs - nm) / ns if ns > 0 else 0.0
            row[f"{nname}_p_two"] = float(
                (1 + (np.abs(nd - nm) >= abs(d_obs - nm)).sum())
                / (len(nd) + 1))
        layers_out.append(row)
        print(f"    L{l:02d} d={d_obs:+.3f} | perm z={row['perm_z']:+.1f} "
              f"p={row['perm_p_two']:.4f} | strat z={row['strat_z']:+.1f} "
              f"p={row['strat_p_two']:.4f}", flush=True)
    return {"n_classes": n_classes, "n_prompts": n_total,
            "class_layout": "phenomenon-major, good before bad",
            "layers": layers_out}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a_classes, b_pairs = load_blimp()
    a_prompts, a_labels = [], []
    for ci, (cn, items) in enumerate(a_classes.items()):
        for s in items:
            a_prompts.append(s)
            a_labels.append(ci)
    a_labels = np.array(a_labels)
    b_prompts = []
    for p in B_PARADIGMS:
        for good, bad in b_pairs[p]:
            b_prompts.extend([good, bad])

    out = {"a_paradigms": A_PARADIGMS, "b_paradigms": B_PARADIGMS,
           "n_items": N_ITEMS, "models": {}}
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

        print(f"[{mkey} / A phenomenon] encoding {len(a_prompts)}...",
              flush=True)
        pl_a = r17.get_hidden_states(model, tokenizer, a_prompts,
                                     device=device)
        res_a = r37.analyze_axis(pl_a, a_labels, len(A_PARADIGMS),
                                 "blimp_phenomenon")
        print(f"[{mkey} / B grammaticality] encoding {len(b_prompts)}...",
              flush=True)
        pl_b = r17.get_hidden_states(model, tokenizer, b_prompts,
                                     device=device)
        res_b = design_b_analysis(pl_b, len(B_PARADIGMS), N_ITEMS)
        out["models"][mkey] = {"A_phenomenon": res_a,
                               "B_grammaticality": res_b}
        json.dump(out, open(OUT, "w"), indent=1)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    print("\nDONE run42", flush=True)


if __name__ == "__main__":
    main()
