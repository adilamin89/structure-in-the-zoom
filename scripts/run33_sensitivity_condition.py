"""Run 33 — The sensitivity condition for the entry-coherence ordering rule
(round-3 referee M4).

The corrected account of the compass ordering null: the estimator centers
within every rung, so the earlier shared-mean explanation was wrong; the
operative variable is the between-class to within-class variance ratio,
which sets whether any ordering of class entries can steer early-rung PR.
This run measures that ratio per layer for the two planted axes and the
world-knowledge reference on Pythia-160m.

REGISTERED EXPECTATIONS:
S1: between/within ratio is small for BOTH planted axes (their delta is
    ~-0.2: within dominates), and smaller for compass than for clock at
    most layers — the stated sensitivity condition for why the ordering
    rule steers on clock (12/12) but is null on compass.
S2: world_knowledge's ratio exceeds both planted axes' (its ordering
    effects in run28 were an order of magnitude larger).

Out: ../data_canonical/run33_sensitivity_condition.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run33_sensitivity_condition.json"

spec = importlib.util.spec_from_file_location(
    "r28", HERE / "run28_cyclic_axis_llm.py")
r28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r28)
r17 = r28.r17


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = {
        "compass": {d: [t.format(d=d) for t in r28.COMPASS_TEMPLATES]
                    for d in r28.COMPASS},
        "clock": {t_: [tpl.format(t=t_) for tpl in r28.CLOCK_TEMPLATES]
                  for t_ in r28.CLOCK},
        "world_knowledge": r17.build_axes()["world_knowledge"],
    }
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m").to("mps")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    out = {"model": "EleutherAI/pythia-160m", "axes": {}}
    for axis_name, classes in axes.items():
        prompts, labels = [], []
        for ci, cn in enumerate(classes):
            for p in classes[cn]:
                prompts.append(p)
                labels.append(ci)
        labels = np.array(labels)
        per_layer = r17.get_hidden_states(model, tokenizer, prompts,
                                          device="mps")
        rows = []
        for l in range(1, len(per_layer)):
            X = per_layer[l].astype(np.float32)
            X = X / (X.std() + 1e-9)
            means = np.stack([X[labels == c].mean(axis=0) for c in range(8)])
            grand = means.mean(axis=0)
            between = float(((means - grand) ** 2).sum(axis=1).mean())
            within = float(np.mean([((X[labels == c] -
                                      means[c]) ** 2).sum(axis=1).mean()
                                    for c in range(8)]))
            rows.append({"layer": l, "between": between, "within": within,
                         "ratio": between / within})
        out["axes"][axis_name] = rows
        rat = [r["ratio"] for r in rows]
        print(f"[{axis_name}] between/within ratio: "
              f"median {np.median(rat):.4f}  range "
              f"[{min(rat):.4f}, {max(rat):.4f}]", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("DONE run33", flush=True)


if __name__ == "__main__":
    main()
