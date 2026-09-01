"""Run 29 — GPT-Neo-1.3B battery (run17 design): data vs architecture
dissociation for the structural-axis profile.

WHY: run25 showed the structural (construction-type) rising profile does not
replicate on OLMo-1B at the matched 16/class design, while the content axis
does. OLMo differs from Pythia in BOTH architecture and training data, so the
"architecture fingerprint" reading is confounded with data. GPT-Neo-1.3B is
trained on the same corpus as Pythia (the Pile) with a different block
design (serial attention/MLP, learned positional embeddings, alternating
local attention, different tokenizer training), splitting the confound.

REGISTERED EXPECTATIONS (written before the run):
D1: world_knowledge positive (content = data-driven; the Pile is shared).
D2: THE DECISIVE CELL — if the structural rising profile is data-driven,
    GPT-Neo (Pile) should rise like Pythia; if architecture-driven, it
    should deviate like OLMo. Either outcome is reportable.
D3: ethical ~ 0; random ~ 0 away from the earliest layers.

Model: EleutherAI/gpt-neo-1.3B (fp16, MPS). Out:
../data_canonical/run29_gptneo_battery.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run29_gptneo_battery.json"

spec = importlib.util.spec_from_file_location(
    "r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r17)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = r17.build_axes()

    print("\nloading EleutherAI/gpt-neo-1.3B (fp16, MPS)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-1.3B", torch_dtype=torch.float16).to("mps")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_layers = model.config.num_layers
    print(f"  {n_layers} layers, d={model.config.hidden_size}", flush=True)

    out = {"model": "EleutherAI/gpt-neo-1.3B", "n_layers": n_layers,
           "prompts_per_class": 16, "design": "run17", "axes": {}}
    if OUT.exists():
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
            print(f"  delta: [{min(ds):+.3f}, {max(ds):+.3f}]", flush=True)
        out["axes"][axis_name] = axis_results
        json.dump(out, open(OUT, "w"), indent=1)

    if "random" not in out["axes"]:
        print(f"\n[random] 8-class random on 128 prompts", flush=True)
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
        json.dump(out, open(OUT, "w"), indent=1)

    print("\nDONE run29", flush=True)


if __name__ == "__main__":
    main()
