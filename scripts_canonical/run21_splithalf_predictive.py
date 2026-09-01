"""Run 21 - Split-half predictive test on LLM axes (the V1 "rationalize→predict"
analog ported to Pythia-160m).

Design: for each 8-class axis, split the 16 prompts per class into two
disjoint halves (8 each). On HALF A: compute the class-mean correlation
kernel C(class_i, class_j) and fit entry coherence - predict which
accumulation orderings produce larger δ. On HALF B: measure δ under
each ordering and test the prediction.

Specifically:
1. Compute C_A(i,j) = correlation between class-mean vectors on half A.
2. For 20 random permutations of the 8 classes, predict δ ordering from
   first-rung coherence: mean C_A(classes in first rung) should predict
   larger δ.
3. Measure δ for each permutation on half B.
4. Report rank correlation between predicted coherence and measured δ.

REGISTERED: Spearman ρ > 0 on world_knowledge (content structure exists)
and language_type at late layers (structural organization exists there).

Model: Pythia-160m (local, CPU). ~15 min.
Out: feedback_runs/run21_splithalf_predictive.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r17)

N_NULL = 10
N_ORDERS = 20
BIN_COUNTS = [1, 2, 3, 4, 6, 8]


def ladder_delta_ordered(X, labels, order, n_classes, rng):
    bc = [c for c in BIN_COUNTS if c <= n_classes]
    members = [np.where(labels == order[c])[0] for c in range(n_classes)]
    if min(len(m) for m in members) < 3:
        return None
    sizes, prs = [], []
    for c in bc:
        sel = np.concatenate(members[:c])
        sizes.append(len(sel))
        prs.append(r17.pr_c(X[sel]))
    if len(sizes) < 3:
        return None
    th_o = r17.slope(sizes, np.asarray(prs))
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(r17.pr_c(X[rng.choice(len(X), s,
                                                         replace=False)]),
                                  1e-9))
    return th_o - r17.slope(sizes, np.exp(nl.mean(axis=0)))


def first_rung_coherence(class_means, order):
    """Mean pairwise correlation of the first 2 accumulated classes."""
    m0 = class_means[order[0]]
    m1 = class_means[order[1]]
    m0n = m0 / (np.linalg.norm(m0) + 1e-9)
    m1n = m1 / (np.linalg.norm(m1) + 1e-9)
    return float(m0n @ m1n)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.stats import spearmanr

    axes = r17.build_axes()
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    out = {}
    for axis_name in ["world_knowledge", "language_type"]:
        classes = axes[axis_name]
        class_names = list(classes.keys())
        n_classes = len(class_names)

        # Split prompts: first 8 = half A (fit), last 8 = half B (test)
        all_A, labels_A, all_B, labels_B = [], [], [], []
        for ci, cn in enumerate(class_names):
            prompts = classes[cn]
            for p in prompts[:8]:
                all_A.append(p)
                labels_A.append(ci)
            for p in prompts[8:16]:
                all_B.append(p)
                labels_B.append(ci)
        labels_A, labels_B = np.array(labels_A), np.array(labels_B)

        print(f"\n[{axis_name}] half A: {len(all_A)}, half B: {len(all_B)}",
              flush=True)

        per_layer_A = r17.get_hidden_states(model, tokenizer, all_A)
        per_layer_B = r17.get_hidden_states(model, tokenizer, all_B)
        n_layers = len(per_layer_A)

        # Generate 20 random class orderings
        rng = np.random.default_rng(7777)
        orders = [rng.permutation(n_classes).tolist() for _ in range(N_ORDERS)]

        axis_out = {"n_classes": n_classes, "layers": []}
        for l in range(n_layers):
            X_A = per_layer_A[l].astype(np.float32)
            X_A /= (X_A.std() + 1e-9)
            X_B = per_layer_B[l].astype(np.float32)
            X_B /= (X_B.std() + 1e-9)

            # Class means on half A
            class_means_A = np.stack([X_A[labels_A == ci].mean(axis=0)
                                      for ci in range(n_classes)])

            # For each order: predict coherence from A, measure delta on B
            coherences, deltas = [], []
            for order in orders:
                coh = first_rung_coherence(class_means_A, order)
                coherences.append(coh)
                d = ladder_delta_ordered(X_B, labels_B, order, n_classes,
                                         np.random.default_rng(l * 100 + 42))
                deltas.append(d if d is not None else np.nan)

            coherences = np.array(coherences)
            deltas = np.array(deltas)
            valid = ~np.isnan(deltas)
            if valid.sum() >= 5:
                rho, pval = spearmanr(coherences[valid], deltas[valid])
            else:
                rho, pval = np.nan, np.nan

            axis_out["layers"].append({
                "layer": l, "spearman_rho": float(rho),
                "spearman_p": float(pval),
                "n_valid": int(valid.sum()),
                "coherence_range": [float(coherences.min()),
                                    float(coherences.max())],
                "delta_range": [float(np.nanmin(deltas)),
                                float(np.nanmax(deltas))],
            })
            if l % 4 == 0 or l == n_layers - 1:
                print(f"  L{l:2d}: ρ={rho:+.3f} (p={pval:.3f}) | "
                      f"coh=[{coherences.min():+.3f},{coherences.max():+.3f}] "
                      f"δ=[{np.nanmin(deltas):+.3f},{np.nanmax(deltas):+.3f}]",
                      flush=True)

        out[axis_name] = axis_out

    json.dump(out, open(HERE / "run21_splithalf_predictive.json", "w"),
              indent=1)
    print("\nDONE run21", flush=True)


if __name__ == "__main__":
    main()
