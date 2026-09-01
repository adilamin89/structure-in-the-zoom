"""Run 26 - Pythia-2.8B production battery (run17 design), local MPS.

WHY: an earlier 2.8B measurement (delta=+0.302) used a 16-prompt
contrastive prototype with no shuffle control and no axis resolution. This run applies
the production 8-class battery (prompts byte-identical via run17.build_axes())
to EleutherAI/pythia-2.8b-deduped, completing the 4-scale chain
160m / 410m / 1B / 2.8B. Runs locally: fp16 on MPS, 128 GB unified memory.

REGISTERED EXPECTATIONS (written before the run):
E1: world_knowledge positive at (nearly) all layers, flat-to-declining
    profile (content axis, lexically inherited) - 4th scale in the chain.
E2: language_type negative-at-embedding rising to positive (structural
    axis, network-built) - the Pythia-family pattern at a 4th scale.
E3: ethical ~ 0 (base model); random ~ 0 (exchangeability).
E4: cross-model axis ranking extends run22's rho=1.0 to 2.8B.

Out: ../data_canonical/run26_pythia28b_battery.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run26_pythia28b_battery.json"

spec = importlib.util.spec_from_file_location(
    "r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r17)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = r17.build_axes()

    print("\nloading EleutherAI/pythia-2.8b-deduped (fp16, MPS)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-2.8b-deduped",
        torch_dtype=torch.float16).to("mps")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-deduped")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, d={model.config.hidden_size}", flush=True)

    out = {"model": "EleutherAI/pythia-2.8b-deduped", "n_layers": n_layers,
           "prompts_per_class": 16, "design": "run17", "axes": {}}
    if OUT.exists():  # resume after interruption
        out = json.load(open(OUT))
        print(f"  resumed: {list(out['axes'])} done", flush=True)

    all_prompts_pool = []
    for axis in axes.values():
        for cls in axis.values():
            all_prompts_pool.extend(cls)

    for axis_name, classes in axes.items():
        if axis_name in out["axes"]:
            continue
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
            ds = [lr["delta"] for lr in axis_results["layers"]]
            ss = [lr["shuffle_mean"] for lr in axis_results["layers"]]
            print(f"  delta: [{min(ds):+.3f}, {max(ds):+.3f}] | "
                  f"shuffle: [{min(ss):+.3f}, {max(ss):+.3f}]", flush=True)

        out["axes"][axis_name] = axis_results
        json.dump(out, open(OUT, "w"), indent=1)  # checkpoint after each axis

    if "random" not in out["axes"]:
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

    print("\nDONE run26", flush=True)


if __name__ == "__main__":
    main()
