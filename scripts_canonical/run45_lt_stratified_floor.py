"""Run 45 — topic-stratified floor for the language_type axis.

WHY: the run36 principle says a label-representation reading needs a
nuisance-preserving null. language_type's nuisance is TOPIC: the 128 prompts
mix topics within each construction type by design, but not uniformly —
topic composition could contribute to (or produce) the measured profile,
including the negative canonical embedding delta the paper uses in its
anti-lexical argument. Each prompt was hand-assigned one of 7 topic strata
(bio, phys, soc, tech, health, hum, evd) FROM PROMPT TEXT ALONE, blind to
any delta values; the assignment is frozen in this file. The stratified
null permutes construction labels WITHIN topic strata (500 draws),
preserving topic composition exactly; the full permutation (500) is the
label-free level. Prompts byte-identical to run17; canonical accumulation
order; floor as in run37.

REGISTERED EXPECTATIONS (written before the run):
S1: late-layer positive delta SURVIVES the stratified null on 2.8B
    (construction-type organization beyond topic; run31's scramble result
    makes this the expected outcome).
S2: DECISION CELL — embedding-layer negative canonical delta under
    stratification: if the stratified null reproduces it, the embedding
    negativity is topic composition (revising the anti-lexical argument to
    rest on run31/run37); if the observed value beats the stratified null,
    it is construction-linked. Either outcome informative.
S3: 160m same direction, weaker.

Out: ../data_canonical/run45_lt_stratified_floor.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run45_lt_stratified_floor.json"

spec = importlib.util.spec_from_file_location(
    "r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r37)
r17 = r37.r17

N_NULL = 500
MODELS = [("EleutherAI/pythia-160m", "cpu"),
          ("EleutherAI/pythia-2.8b-deduped", "mps")]

# Topic strata for the 128 language_type prompts, in run17 order
# (8 construction types x 16 prompts). Assigned from prompt text alone,
# blind to delta values.
TOPICS = {
    "question": ["phys", "bio", "bio", "hum", "phys", "bio", "phys", "bio",
                 "health", "phys", "health", "phys", "tech", "phys", "tech",
                 "phys"],
    "definition": ["phys", "soc", "bio", "soc", "bio", "bio", "phys", "tech",
                   "phys", "hum", "bio", "phys", "soc", "phys", "hum",
                   "tech"],
    "comparison": ["bio", "soc", "phys", "phys", "phys", "hum", "phys", "bio",
                   "phys", "bio", "hum", "soc", "health", "phys", "health",
                   "phys"],
    "narrative": ["hum", "hum", "hum", "hum", "hum", "phys", "tech", "hum",
                  "bio", "soc", "tech", "hum", "hum", "health", "hum", "evd"],
    "cause_effect": ["phys", "phys", "bio", "bio", "hum", "phys", "health",
                     "bio", "phys", "bio", "health", "health", "health",
                     "health", "soc", "tech"],
    "instruction": ["tech", "evd", "evd", "tech", "phys", "evd", "tech",
                    "bio", "phys", "evd", "tech", "health", "evd", "phys",
                    "evd", "tech"],
    "opinion": ["phys", "soc", "tech", "health", "soc", "phys", "soc", "bio",
                "hum", "hum", "hum", "health", "soc", "health", "soc",
                "health"],
    "negation": ["phys", "health", "phys", "phys", "bio", "health", "bio",
                 "health", "hum", "bio", "bio", "health", "bio", "hum",
                 "health", "hum"],
}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    classes = r17.build_axes()["language_type"]
    class_names = list(classes.keys())
    assert class_names == list(TOPICS.keys()), class_names
    prompts, labels, strata = [], [], []
    for ci, cn in enumerate(class_names):
        assert len(classes[cn]) == len(TOPICS[cn]) == 16, cn
        for p, t in zip(classes[cn], TOPICS[cn]):
            prompts.append(p)
            labels.append(ci)
            strata.append(t)
    labels = np.array(labels)
    strata = np.array(strata)
    n_total = len(prompts)
    n_classes = len(class_names)
    from collections import Counter
    print(f"strata counts: {dict(Counter(strata))}", flush=True)

    stratum_members = {t: np.where(strata == t)[0]
                       for t in np.unique(strata)}
    canonical = list(range(n_classes))

    prng = np.random.default_rng(4500)
    full_perms = [prng.permutation(n_total) for _ in range(N_NULL)]
    srng = np.random.default_rng(4501)

    def strat_perm():
        sl = labels.copy()
        for t, idx in stratum_members.items():
            sl[idx] = sl[idx[srng.permutation(len(idx))]]
        return sl

    strat_labels = [strat_perm() for _ in range(N_NULL)]
    for sl in strat_labels[:5]:  # class sizes must be preserved
        assert np.array_equal(np.bincount(sl), np.bincount(labels))

    out = {"n_null": N_NULL, "strata_counts":
           {t: int(len(v)) for t, v in stratum_members.items()},
           "models": {}}
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
            th_obs, sizes = r37.ladder_slope(K, members, canonical)
            th_f = r37.floor_slope(K, n_total, sizes,
                                   np.random.default_rng(l * 100 + 1))
            d_obs = th_obs - th_f
            row = {"layer": l, "delta": float(d_obs)}
            for nname, label_draws in (("perm", None), ("strat", strat_labels)):
                nd = []
                for i in range(N_NULL):
                    pl = (labels[full_perms[i]] if label_draws is None
                          else label_draws[i])
                    mem = [np.where(pl == c)[0] for c in range(n_classes)]
                    nd.append(r37.ladder_slope(K, mem, canonical)[0] - th_f)
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
                  f"p={row['strat_p_two']:.4f} "
                  f"(strat null {row['strat_null_mean']:+.3f})", flush=True)
        out["models"][mkey] = {"layers": layers_out}
        json.dump(out, open(OUT, "w"), indent=1)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    print("\nDONE run45", flush=True)


if __name__ == "__main__":
    main()
