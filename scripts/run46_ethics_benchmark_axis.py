"""Run 46 — benchmark-mined ethical axis (ETHICS domains) across the model suite.

WHY (user ruling): the hand-written ethical-concept axis could be attacked as
taxonomy-insensitive. This run rebuilds the ethical axis from the ETHICS
benchmark (Hendrycks et al. 2021): classes = four normative domains
(commonsense, deontology, justice, utilitarianism; the virtue subset is not
in the parquet mirror), 32 texts each = 128 prompts, each text the natural
scenario sentence prompt+chosen from wassname/ethics_expression_preferences
(a faithful text reformat of the original), filtered to 6-28 words, first 32
per domain, deterministic. DESIGN DIFFERENCE, disclosed: unlike the
hand-written axis, the ETHICS subsets differ in surface FORMAT, so
embedding-layer delta is expected nonzero here and the embedding remains the
format/lexical control; the base-instruct comparison is format-robust
because both members see identical prompts. Ladder [1,2,3,4]; run37
machinery (500 label permutations/layer, 50-order averages, IE/PE).

MODELS: the six battery models + the two base-instruct pairs (ten total),
so both taxonomies can be compared on every model in the paper.

REGISTERED EXPECTATIONS (written before the run):
H1: on base models the benchmark-derived axis reproduces the hand-written
    pattern under matched statistics: canonical certification near chance
    at most layers BEYOND the embedding-inherited component, order-averaged
    linkage weakly positive; replication closes the taxonomy objection.
H2: DECISION CELL: D_late(instruct - base) ~ 0 on both pairs replicates the
    RLHF null on an independent taxonomy; a clear positive shift would
    scope the run44 null as taxonomy-specific. Either outcome reported.
H3 (descriptive): model ranking by order-averaged certified fraction agrees
    loosely between the two ethical taxonomies.

Out: ../data_canonical/run46_ethics_benchmark_axis.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run46_ethics_benchmark_axis.json"

spec = importlib.util.spec_from_file_location(
    "r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r37)
r17 = r37.r17
r37.BIN_COUNTS = [1, 2, 3, 4]          # 4-class ladder
r37.DECLARED_DIRECTION["ethics_benchmark"] = {"IE": +1, "PE": +1}

DOMAINS = ["commonsense", "deontology", "justice", "utilitarianism"]
N_PER = 32
MODELS = [
    ("EleutherAI/pythia-160m", "cpu"),
    ("EleutherAI/pythia-410m-deduped", "mps"),
    ("EleutherAI/pythia-1b-deduped", "mps"),
    ("EleutherAI/pythia-2.8b-deduped", "mps"),
    ("allenai/OLMo-1B-hf", "mps"),
    ("EleutherAI/gpt-neo-1.3B", "mps"),
    ("allenai/OLMo-2-0425-1B", "cpu"),
    ("allenai/OLMo-2-0425-1B-Instruct", "cpu"),
    ("Qwen/Qwen2.5-0.5B", "cpu"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "cpu"),
]
PAIRS = [("OLMo-2-0425-1B", "OLMo-2-0425-1B-Instruct"),
         ("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct")]


def build_prompts():
    from datasets import load_dataset
    prompts, labels = [], []
    for ci, cfg in enumerate(DOMAINS):
        ds = load_dataset("wassname/ethics_expression_preferences",
                          data_files=f"{cfg}/train-00000-of-00001.parquet",
                          split="train")
        picked = 0
        for r in ds:
            text = " ".join((r["prompt"] + " " + r["chosen"]).split())
            nw = len(text.split())
            if 6 <= nw <= 28:
                prompts.append(text)
                labels.append(ci)
                picked += 1
            if picked == N_PER:
                break
        assert picked == N_PER, (cfg, picked)
    return prompts, np.array(labels)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts, labels = build_prompts()
    print(f"{len(prompts)} prompts, 4 domains", flush=True)

    out = {"domains": DOMAINS, "n_per_class": N_PER, "prompts": prompts,
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
        res = r37.analyze_axis(per_layer, labels, len(DOMAINS),
                               "ethics_benchmark")
        out["models"][mkey] = res
        json.dump(out, open(OUT, "w"), indent=1)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    print("\nD_late (instruct - base, order-averaged, final third):",
          flush=True)
    out["D_late"] = {}
    for bk, ik in PAIRS:
        db = [l["delta_orderavg"] for l in out["models"][bk]["layers"]]
        di = [l["delta_orderavg"] for l in out["models"][ik]["layers"]]
        k = len(db) - len(db) // 3
        d_late = float(np.mean(di[k:]) - np.mean(db[k:]))
        out["D_late"][bk] = d_late
        print(f"  {bk}: D_late={d_late:+.4f}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nDONE run46", flush=True)


if __name__ == "__main__":
    main()
