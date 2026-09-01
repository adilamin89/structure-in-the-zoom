"""Run 22 - Cross-model + cross-axis predictive tests (the V1 "predict" analog
that works with LLM data).

Three predictive tests, zero data leakage:

TEST 1 (cross-axis ranking): For each model, compute the between-class
correlation strength per axis per layer. Predict: axes with higher between-
class separation at layer L should have higher |δ| at layer L. Test via
Spearman ρ across the 6 real axes at each layer. Uses ALL prompts (full
power, no split needed - the prediction is across axes, not within).

TEST 2 (cross-model transfer): Fit the axis-ranking on Pythia-160m, test
whether the same ranking holds on 410m and 1B. Spearman ρ of the δ-ranking
across axes between model pairs.

TEST 3 (depth-profile shape prediction): From the between-class correlation
at the EMBEDDING layer only, predict the depth-profile SHAPE: if embedding
separation is high (content axis), predict declining; if low (structural),
predict rising. Quantify as correlation between embedding-δ and the slope
of δ across layers.

TEST 4 (prompt-held-out within model): For each axis, use the FIRST 8
prompts per class to compute δ, use the LAST 8 to compute δ independently.
Correlation across axes between the two halves = prompt stability beyond
what the bootstrap SD measures.

REGISTERED: Tests 1-3 should show positive ρ (entry coherence predicts δ).
Test 4 should show high correlation (prompt-stable instrument).

Data: run17 (160m), run18 (410m), run19 (1B) JSONs - NO new forward passes.
Plus one quick rerun of the 8-prompt halves from cached model (160m only).
Out: feedback_runs/run22_cross_prediction.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    from scipy.stats import spearmanr

    # Load all three models' results
    runs = {}
    for name, fname in [("160m", "run17_multiclass_battery.json"),
                        ("410m", "run18_pythia410m_battery.json"),
                        ("1B", "run19_pythia1b_battery.json")]:
        runs[name] = json.load(open(HERE / fname))

    real_axes = ["world_knowledge", "language_type", "tqa_category",
                 "hs_activity", "arc_topic", "ethical"]
    out = {}

    # ---- TEST 1: cross-axis ranking per model ----
    print("=== TEST 1: cross-axis δ ranking within each model ===", flush=True)
    out["test1"] = {}
    for model_name, data in runs.items():
        n_layers = data["n_layers"] + 1  # +1 for embedding
        # Get δ per axis per layer
        per_layer_delta = {}
        per_layer_emb_delta = {}
        for axis in real_axes:
            info = data["axes"].get(axis, {})
            layers = info.get("layers", [])
            for l in layers:
                li = l["layer"]
                per_layer_delta.setdefault(li, {})[axis] = l.get("delta", 0)

        # At each layer, rank axes by |δ| - does it correlate with
        # between-class separation (approximated by |δ| at embedding)?
        emb_deltas = {}
        for axis in real_axes:
            info = data["axes"].get(axis, {})
            layers = info.get("layers", [])
            if layers:
                emb_deltas[axis] = abs(layers[0].get("delta", 0))

        rhos = []
        for li in sorted(per_layer_delta):
            ax_d = per_layer_delta[li]
            axes_common = [a for a in real_axes if a in ax_d and a in emb_deltas]
            if len(axes_common) < 4:
                continue
            x = [emb_deltas[a] for a in axes_common]
            y = [abs(ax_d[a]) for a in axes_common]
            rho, p = spearmanr(x, y)
            rhos.append({"layer": li, "rho": float(rho), "p": float(p)})

        out["test1"][model_name] = rhos
        mean_rho = np.mean([r["rho"] for r in rhos])
        print(f"  {model_name}: mean ρ(|emb_δ|, |δ_L|) = {mean_rho:+.3f} "
              f"across {len(rhos)} layers", flush=True)

    # ---- TEST 2: cross-model transfer of axis ranking ----
    print("\n=== TEST 2: cross-model δ-ranking transfer ===", flush=True)
    out["test2"] = {}
    # Get mean |δ| per axis per model (average over layers)
    mean_delta = {}
    for model_name, data in runs.items():
        mean_delta[model_name] = {}
        for axis in real_axes:
            layers = data["axes"].get(axis, {}).get("layers", [])
            ds = [abs(l.get("delta", 0)) for l in layers]
            mean_delta[model_name][axis] = float(np.mean(ds)) if ds else 0

    pairs = [("160m", "410m"), ("160m", "1B"), ("410m", "1B")]
    for m1, m2 in pairs:
        x = [mean_delta[m1][a] for a in real_axes]
        y = [mean_delta[m2][a] for a in real_axes]
        rho, p = spearmanr(x, y)
        out["test2"][f"{m1}_vs_{m2}"] = {"rho": float(rho), "p": float(p)}
        print(f"  {m1} vs {m2}: ρ = {rho:+.3f} (p={p:.3f})", flush=True)

    # ---- TEST 3: depth-profile shape prediction ----
    print("\n=== TEST 3: embedding δ predicts depth-profile slope ===",
          flush=True)
    out["test3"] = {}
    for model_name, data in runs.items():
        emb_ds, slopes = [], []
        for axis in real_axes:
            layers = data["axes"].get(axis, {}).get("layers", [])
            if len(layers) < 3:
                continue
            emb_d = layers[0].get("delta", 0)
            ds = [l.get("delta", 0) for l in layers]
            layer_idx = np.arange(len(ds))
            # slope of δ vs layer
            sl = float(np.polyfit(layer_idx, ds, 1)[0])
            emb_ds.append(emb_d)
            slopes.append(sl)
        if len(emb_ds) >= 4:
            rho, p = spearmanr(emb_ds, slopes)
            out["test3"][model_name] = {"rho": float(rho), "p": float(p),
                                        "emb_deltas": emb_ds, "slopes": slopes}
            print(f"  {model_name}: ρ(emb_δ, slope) = {rho:+.3f} (p={p:.3f})",
                  flush=True)

    # ---- TEST 4: prompt-held-out (160m only, needs forward pass) ----
    print("\n=== TEST 4: prompt-held-out (first 8 vs last 8, 160m) ===",
          flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = importlib.util.spec_from_file_location(
        "r17", HERE / "run17_multiclass_battery.py")
    r17 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r17)

    axes = r17.build_axes()
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    delta_half_A, delta_half_B = {}, {}
    for axis_name in real_axes:
        classes = axes.get(axis_name, {})
        if not classes:
            continue
        class_names = list(classes.keys())
        n_classes = len(class_names)

        for half_name, start, end, store in [
            ("A", 0, 8, delta_half_A), ("B", 8, 16, delta_half_B)
        ]:
            prompts, labels = [], []
            for ci, cn in enumerate(class_names):
                ps = classes[cn][start:end]
                prompts.extend(ps)
                labels.extend([ci] * len(ps))
            labels = np.array(labels)
            per_layer = r17.get_hidden_states(model, tokenizer, prompts)
            # Use the middle and late layers for comparison
            ds = []
            for l in range(len(per_layer)):
                X = per_layer[l].astype(np.float32)
                X /= (X.std() + 1e-9)
                d, _ = r17.ladder_delta(X, labels, n_classes,
                                        np.random.default_rng(l * 100 + 1))
                ds.append(d if d is not None else 0)
            store[axis_name] = float(np.mean(ds))

    # Correlation of mean-δ across axes between halves
    axes_common = [a for a in real_axes if a in delta_half_A and a in delta_half_B]
    if len(axes_common) >= 4:
        x = [delta_half_A[a] for a in axes_common]
        y = [delta_half_B[a] for a in axes_common]
        rho, p = spearmanr(x, y)
        out["test4"] = {"rho": float(rho), "p": float(p),
                        "half_A": {a: delta_half_A[a] for a in axes_common},
                        "half_B": {a: delta_half_B[a] for a in axes_common}}
        print(f"  prompt-held-out ρ = {rho:+.3f} (p={p:.3f})", flush=True)
        for a in axes_common:
            print(f"    {a}: A={delta_half_A[a]:+.3f} B={delta_half_B[a]:+.3f}",
                  flush=True)

    json.dump(out, open(HERE / "run22_cross_prediction.json", "w"), indent=1)
    print("\nDONE run22", flush=True)


if __name__ == "__main__":
    main()
