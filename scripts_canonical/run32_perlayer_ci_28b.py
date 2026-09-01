"""Run 32 — Per-layer prompt-resampling confidence intervals for the
content and structural axes on Pythia-2.8B (round-3 referee M2: the
quarter-depth sign crossing is the paper's main LLM claim and needs
per-layer intervals, not only the pooled shuffle band).

Hidden states are encoded once and stored; 200 prompt-resamples per layer
draw 12 of 16 prompts per class without replacement and recompute delta
(floor redrawn per replicate with a common seed per layer).

REGISTERED EXPECTATIONS:
C1: structural axis: the negative band at the embedding excludes zero, the
    positive band over the last quarter of depth excludes zero, and the
    crossing region spans zero — the sign structure of the crossover is
    resolved at this design.
C2: content axis: positive band excludes zero at the embedding and stays
    positive (possibly touching zero late).
Either miss is reportable and bounds the design instead.

Out: ../data_canonical/run32_perlayer_ci_28b.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run32_perlayer_ci_28b.json"

spec = importlib.util.spec_from_file_location(
    "r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r17)

N_BOOT = 200
KEEP = 12  # of 16 prompts per class per replicate


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = r17.build_axes()
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-2.8b-deduped",
        torch_dtype=torch.float16).to("mps")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-deduped")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    out = {"model": "EleutherAI/pythia-2.8b-deduped", "n_boot": N_BOOT,
           "keep_per_class": KEEP, "axes": {}}
    for axis_name in ["world_knowledge", "language_type"]:
        classes = axes[axis_name]
        prompts, labels = [], []
        for ci, cn in enumerate(classes):
            for p in classes[cn]:
                prompts.append(p)
                labels.append(ci)
        labels = np.array(labels)
        print(f"\n[{axis_name}] encoding...", flush=True)
        per_layer = r17.get_hidden_states(model, tokenizer, prompts,
                                          device="mps")
        rows = []
        for l in range(len(per_layer)):
            X = per_layer[l].astype(np.float32)
            X = X / (X.std() + 1e-9)
            point, _ = r17.ladder_delta(X, labels, 8,
                                        np.random.default_rng(l * 100 + 1))
            boots = []
            for b in range(N_BOOT):
                rng = np.random.default_rng(9000 + b)
                keep_idx = np.concatenate([
                    rng.choice(np.where(labels == c)[0], KEEP, replace=False)
                    for c in range(8)])
                d, _ = r17.ladder_delta(X[keep_idx], labels[keep_idx], 8,
                                        np.random.default_rng(l * 100 + 1),
                                        n_shuf=0)
                if d is not None:
                    boots.append(d)
                if b and b % 50 == 0:
                    print(f"    L{l} boot {b}/{N_BOOT}", flush=True)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"layer": l, "delta": point,
                         "ci_lo": float(lo), "ci_hi": float(hi),
                         "excludes_zero": bool(lo > 0 or hi < 0)})
            print(f"  L{l:2d} {point:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"{' *' if (lo > 0 or hi < 0) else ''}", flush=True)
        out["axes"][axis_name] = rows
        json.dump(out, open(OUT, "w"), indent=1)
    print("DONE run32", flush=True)


if __name__ == "__main__":
    main()
