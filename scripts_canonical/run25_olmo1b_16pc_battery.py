"""Run 25 - OLMo-1B full battery at 16/class (the run17 design, unchanged).

WHY: run23 tested OLMo-1B only at 32/class, where added within-class prompt
diversity dilutes the structural axis (the V1 physics: diverse prompts raise
within-class dimensionality). That left the cross-family comparison confounded:
OLMo-32/class vs Pythia-16/class. This run removes the confound by running
OLMo-1B on the EXACT run17 prompt set (16/class, all six axes + random).

REGISTERED EXPECTATIONS (written before the run):
E1: world_knowledge positive at most layers (content axis replicates
    cross-family; run23 already showed this at 32/class).
E2: language_type at 16/class is the decisive cell: if OLMo shows a rising
    (negative-to-positive) profile like Pythia, architecture-dependence at
    32/class was a prompt-diversity artifact; if OLMo still shows
    positive-early/negative-late, the structural axis is genuinely
    architecture-dependent. Either outcome is reportable.
E3: ethical ~ 0 (base model, no alignment training).
E4: random ~ 0 (8-class exchangeability).

Model: allenai/OLMo-1B-hf (cached), fp16 on MPS. ~40 min.
Out: ../data_canonical/run25_olmo1b_16pc_battery.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run25_olmo1b_16pc_battery.json"

spec = importlib.util.spec_from_file_location(
    "r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r17)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = r17.build_axes()

    print("\nloading allenai/OLMo-1B-hf (fp16, MPS)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/OLMo-1B-hf", torch_dtype=torch.float16).to("mps")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, d={model.config.hidden_size}", flush=True)

    out = {"model": "allenai/OLMo-1B-hf", "n_layers": n_layers,
           "prompts_per_class": 16, "design": "run17", "axes": {}}

    all_prompts_pool = []
    for axis in axes.values():
        for cls in axis.values():
            all_prompts_pool.extend(cls)

    for axis_name, classes in axes.items():
        class_names = list(classes.keys())
        n_classes = len(class_names)
        all_prompts, labels = [], []
        for ci, cn in enumerate(class_names):
            for p in classes[cn]:
                all_prompts.append(p)
                labels.append(ci)
        labels = np.array(labels)
        print(f"\n[{axis_name}] {n_classes} classes, {len(all_prompts)} prompts",
              flush=True)

        per_layer = r17.get_hidden_states(model, tokenizer, all_prompts,
                                          device="mps")

        axis_results = {"n_classes": n_classes, "n_prompts": len(all_prompts),
                        "class_names": class_names, "layers": []}
        for l in range(len(per_layer)):
            X = per_layer[l].astype(np.float32)
            X = X / (X.std() + 1e-9)
            d, sh = r17.ladder_delta(X, labels, n_classes,
                                     np.random.default_rng(l * 100 + 1))
            if d is None:
                continue
            axis_results["layers"].append({
                "layer": l, "delta": d, "shuffle_mean": sh})

        if axis_results["layers"]:
            emb_delta = axis_results["layers"][0]["delta"]
            for lr in axis_results["layers"]:
                lr["delta_excess"] = lr["delta"] - emb_delta

        out["axes"][axis_name] = axis_results
        if axis_results["layers"]:
            ds = [lr["delta"] for lr in axis_results["layers"]]
            ss = [lr["shuffle_mean"] for lr in axis_results["layers"]]
            exs = [lr.get("delta_excess", 0) for lr in axis_results["layers"]]
            print(f"  delta: [{min(ds):+.3f}, {max(ds):+.3f}] | "
                  f"shuffle: [{min(ss):+.3f}, {max(ss):+.3f}] | "
                  f"excess: [{min(exs):+.3f}, {max(exs):+.3f}]", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)  # checkpoint after each axis

    print(f"\n[random] 8-class random on {len(all_prompts_pool)} prompts",
          flush=True)
    per_layer_rand = r17.get_hidden_states(model, tokenizer,
                                           all_prompts_pool[:128],
                                           device="mps")
    rng = np.random.default_rng(999)
    rand_labels = rng.integers(0, 8, 128)
    rand_results = {"n_prompts": 128, "n_classes": 8, "layers": []}
    for l in range(len(per_layer_rand)):
        X = per_layer_rand[l].astype(np.float32)
        X = X / (X.std() + 1e-9)
        d, sh = r17.ladder_delta(X, rand_labels, 8,
                                 np.random.default_rng(l * 100 + 99))
        rand_results["layers"].append({
            "layer": l, "delta": d, "shuffle_mean": sh if sh else 0.0})
    out["axes"]["random"] = rand_results
    ds = [lr["delta"] for lr in rand_results["layers"]
          if lr["delta"] is not None]
    print(f"  random mean: {np.mean(ds):+.3f} | "
          f"max |delta|: {max(abs(d) for d in ds):.3f}", flush=True)

    json.dump(out, open(OUT, "w"), indent=1)
    print("\nDONE run25", flush=True)


if __name__ == "__main__":
    main()
