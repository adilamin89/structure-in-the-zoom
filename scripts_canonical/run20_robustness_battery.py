"""Run 20 - All five robustness checks on the two flagship axes (world_knowledge
+ language_type), Pythia-160m. Results feed the arXiv paper's methodology
section and define the instrument's operating envelope.

Tests:
  (A) Prompt-bootstrap: resample 8 of 16 prompts per class, recompute δ,
      20 replicates → SD per layer per axis.
  (B) Mean-pooling variant: use mean over all tokens instead of last-token
      hidden states → compare depth profiles.
  (C) Accumulation-order permutation: permute the class order, 20 replicates
      → SD per layer per axis.
  (D) 2-class failure analysis: run world_knowledge as 2-class (STEM vs
      humanities) and language_type as 2-class (declarative vs interrogative)
      to show the 2-rung degradation quantitatively.

REGISTERED: (A)+(C) SDs should be small relative to the signal on both axes;
(B) should preserve the content/structure contrast; (D) should show markedly
weaker signal/shuffle than the 8-class versions.

Model: Pythia-160m (local, CPU). ~30 min.
Out: feedback_runs/run20_robustness_battery.json
"""
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
N_NULL = 10
N_SHUF = 5
BIN_COUNTS = [1, 2, 3, 4, 6, 8]

# Import prompts from run17
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r17)


def ladder_delta_with_order(X, labels, n_classes, class_order, rng):
    bc = [c for c in BIN_COUNTS if c <= n_classes]
    members = [np.where(labels == class_order[c])[0] for c in range(n_classes)]
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


def get_hidden_states_meanpool(model, tokenizer, prompts, max_len=128):
    all_states = []
    model.eval()
    with torch.no_grad():
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt", truncation=True,
                            max_length=max_len).input_ids
            out = model(ids, output_hidden_states=True)
            states = [h[0].mean(dim=0).cpu().numpy()
                      for h in out.hidden_states]
            all_states.append(states)
    n_layers = len(all_states[0])
    return [np.stack([s[l] for s in all_states]) for l in range(n_layers)]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = r17.build_axes()

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    out = {}
    target_axes = ["world_knowledge", "language_type"]

    for axis_name in target_axes:
        classes = axes[axis_name]
        class_names = list(classes.keys())
        n_classes = len(class_names)
        all_prompts, labels = [], []
        for ci, cn in enumerate(class_names):
            for p in classes[cn]:
                all_prompts.append(p)
                labels.append(ci)
        labels = np.array(labels)

        # Get both last-token and mean-pooled hidden states
        print(f"\n[{axis_name}] encoding {len(all_prompts)} prompts (last-token)...",
              flush=True)
        per_layer_last = r17.get_hidden_states(model, tokenizer, all_prompts)
        print(f"[{axis_name}] encoding {len(all_prompts)} prompts (mean-pool)...",
              flush=True)
        per_layer_mean = get_hidden_states_meanpool(model, tokenizer,
                                                     all_prompts)

        n_layers = len(per_layer_last)
        axis_out = {"n_classes": n_classes, "n_prompts": len(all_prompts)}

        # (A) Prompt-bootstrap: resample 8 of 16 prompts per class
        print(f"[{axis_name}] (A) prompt bootstrap...", flush=True)
        n_boot = 20
        boot_deltas = np.zeros((n_boot, n_layers))
        prompts_per_class = {ci: np.where(labels == ci)[0]
                             for ci in range(n_classes)}
        for b in range(n_boot):
            brng = np.random.default_rng(3000 + b)
            keep = []
            for ci in range(n_classes):
                idx = prompts_per_class[ci]
                keep.extend(brng.choice(idx, 8, replace=False))
            keep = np.array(keep)
            sub_labels = labels[keep]
            # re-index labels to 0..n-1
            for l in range(n_layers):
                X = per_layer_last[l][keep].astype(np.float32)
                X = X / (X.std() + 1e-9)
                d, _ = r17.ladder_delta(X, sub_labels, n_classes,
                                        np.random.default_rng(l * 100 + b))
                boot_deltas[b, l] = d if d is not None else np.nan
        axis_out["prompt_bootstrap"] = {
            "sd_per_layer": [float(np.nanstd(boot_deltas[:, l]))
                             for l in range(n_layers)],
            "mean_sd": float(np.nanmean(np.nanstd(boot_deltas, axis=0))),
        }
        print(f"  mean SD = {axis_out['prompt_bootstrap']['mean_sd']:.3f}",
              flush=True)

        # (B) Mean-pooling variant
        print(f"[{axis_name}] (B) mean-pool variant...", flush=True)
        meanpool_profile = []
        for l in range(n_layers):
            X = per_layer_mean[l].astype(np.float32)
            X = X / (X.std() + 1e-9)
            d, sh = r17.ladder_delta(X, labels, n_classes,
                                     np.random.default_rng(l * 100 + 50))
            meanpool_profile.append({"layer": l, "delta": d,
                                     "shuffle_mean": sh})
        axis_out["meanpool"] = meanpool_profile
        ds = [p["delta"] for p in meanpool_profile if p["delta"] is not None]
        print(f"  meanpool delta range: [{min(ds):+.3f}, {max(ds):+.3f}]",
              flush=True)

        # (C) Accumulation-order permutation
        print(f"[{axis_name}] (C) order permutation...", flush=True)
        n_perm = 20
        order_deltas = np.zeros((n_perm, n_layers))
        for p in range(n_perm):
            prng = np.random.default_rng(4000 + p)
            order = prng.permutation(n_classes)
            for l in range(n_layers):
                X = per_layer_last[l].astype(np.float32)
                X = X / (X.std() + 1e-9)
                d = ladder_delta_with_order(X, labels, n_classes, order,
                                            np.random.default_rng(l * 100 + p))
                order_deltas[p, l] = d if d is not None else np.nan
        axis_out["order_permutation"] = {
            "sd_per_layer": [float(np.nanstd(order_deltas[:, l]))
                             for l in range(n_layers)],
            "mean_sd": float(np.nanmean(np.nanstd(order_deltas, axis=0))),
        }
        print(f"  mean order SD = "
              f"{axis_out['order_permutation']['mean_sd']:.3f}", flush=True)

        out[axis_name] = axis_out

    # (D) 2-class failure demonstration
    print("\n[2-class failure demo]", flush=True)
    # world_knowledge: STEM vs humanities
    stem = ["mathematics", "physics", "technology", "medicine"]
    hum = ["geography", "history", "philosophy", "sports"]
    wk = axes["world_knowledge"]
    stem_prompts = [p for d in stem for p in wk[d]]
    hum_prompts = [p for d in hum for p in wk[d]]
    two_prompts = stem_prompts + hum_prompts
    two_labels = np.array([0] * len(stem_prompts) + [1] * len(hum_prompts))
    per_layer_2 = r17.get_hidden_states(model, tokenizer, two_prompts)
    two_profile = []
    for l in range(len(per_layer_2)):
        X = per_layer_2[l].astype(np.float32)
        X = X / (X.std() + 1e-9)
        d, sh = r17.ladder_delta(X, two_labels, 2,
                                 np.random.default_rng(l * 100 + 77))
        two_profile.append({"layer": l, "delta": d, "shuffle_mean": sh})
    ds2 = [p["delta"] for p in two_profile if p["delta"] is not None]
    ss2 = [abs(p["shuffle_mean"]) for p in two_profile if p["shuffle_mean"]]
    sig2 = np.mean([abs(d) for d in ds2]) / max(np.mean(ss2), 0.001)
    out["two_class_wk"] = {"profile": two_profile, "sig_sh": float(sig2)}
    print(f"  2-class world_knowledge sig/sh = {sig2:.1f}x "
          f"(8-class was 4.0x)", flush=True)

    json.dump(out, open(HERE / "run20_robustness_battery.json", "w"),
              indent=1)
    print("\nDONE run20", flush=True)


if __name__ == "__main__":
    main()
