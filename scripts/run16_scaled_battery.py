"""Run 16 - Scaled multi-axis δ battery on Pythia-160m using REAL benchmark items.

Fixes run15's N=16 problem by mining actual benchmark datasets for prompts.
Each axis gets 64+ prompts, a proper multi-rung ladder, per-layer shuffles,
and embedding-excess computation.

AXES (from real benchmarks):
  topic_tqa:    TruthfulQA questions grouped by category (8 largest cats, 8+ each)
  factual_tqa:  TQA question + best_answer vs question + worst_incorrect (64 pairs)
  reasoning_hs: HellaSwag correct vs incorrect completions (64 items)
  domain_arc:   ARC-Challenge questions by science domain (mined from question text)
  activity_hs:  HellaSwag by activity category (8 largest, semantic ladder)
  random:       random splits of the factual prompts (floor calibration)

REGISTERED EXPECTATIONS (before run):
R1: random ≈ 0 at every layer.
R2: topic_tqa positive, declining with depth (lexical).
R3: different axes give different depth profiles - the paper's thesis.
R4: embedding-excess reveals which axes are lexical vs network-organized.
R5: axes with more prompts give tighter shuffle controls than run15.

Model: Pythia-160m (local). CPU. ~30 min.
Out: feedback_runs/run16_scaled_battery.json
"""
import csv
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
N_NULL = 10
N_SHUF = 5


def pr_c(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, float))
    y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def ladder_delta(X, labels, n_classes, rng, n_shuf=N_SHUF):
    if n_classes <= 2:
        bin_counts = [1, 2]
    elif n_classes <= 4:
        bin_counts = [1, 2, 3, 4][:n_classes]
    else:
        bin_counts = [1, 2, 3, 4, 6, 8][:n_classes]
    bin_counts = [c for c in bin_counts if c <= n_classes]
    members = [np.where(labels == c)[0] for c in range(n_classes)]
    # ensure each class has enough
    min_per = min(len(m) for m in members)
    if min_per < 3:
        return None, None
    sizes, prs = [], []
    for c in bin_counts:
        sel = np.concatenate(members[:c])
        if len(sel) < 4:
            continue
        sizes.append(len(sel))
        prs.append(pr_c(X[sel]))
    if len(sizes) < 2:
        return None, None
    th_o = slope(sizes, np.asarray(prs))
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(pr_c(X[rng.choice(len(X), s, replace=False)]),
                                  1e-9))
    th_f = slope(sizes, np.exp(nl.mean(axis=0)))
    shufs = []
    for s in range(n_shuf):
        srng = np.random.default_rng(500 + s)
        perm = labels[srng.permutation(len(labels))]
        m2 = [np.where(perm == c)[0] for c in range(n_classes)]
        sz2, pr2 = [], []
        for c in bin_counts:
            sel = np.concatenate(m2[:c])
            if len(sel) < 4:
                continue
            sz2.append(len(sel))
            pr2.append(pr_c(X[sel]))
        if len(sz2) >= 2:
            shufs.append(slope(sz2, np.asarray(pr2)) - th_f)
    return th_o - th_f, float(np.mean(shufs)) if shufs else 0.0


def get_hidden_states(model, tokenizer, prompts, device="cpu", max_len=128):
    all_states = []
    model.eval()
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tokenizer(p, return_tensors="pt", truncation=True,
                            max_length=max_len).input_ids.to(device)
            out = model(ids, output_hidden_states=True)
            states = [h[0, -1, :].cpu().numpy() for h in out.hidden_states]
            all_states.append(states)
            if i % 50 == 0 and i > 0:
                print(f"    encoded {i}/{len(prompts)}", flush=True)
    n_layers = len(all_states[0])
    per_layer = [np.stack([s[l] for s in all_states]) for l in range(n_layers)]
    return per_layer


def build_axes():
    """Mine real benchmark datasets for prompt batteries."""
    axes = {}

    # 1. TruthfulQA by category (topic axis with real questions)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
        "/tmp/tqa.csv")
    with open("/tmp/tqa.csv") as f:
        tqa = list(csv.DictReader(f))
    cats = {}
    for r in tqa:
        cats.setdefault(r["Category"], []).append(r["Question"])
    top8 = sorted(cats, key=lambda c: -len(cats[c]))[:8]
    axes["topic_tqa"] = {c: cats[c][:16] for c in top8}
    print(f"topic_tqa: {len(top8)} categories, "
          f"{sum(len(v) for v in axes['topic_tqa'].values())} prompts", flush=True)

    # 2. TruthfulQA factual: question+correct vs question+incorrect
    factual_true, factual_false = [], []
    for r in tqa[:128]:
        q = r["Question"]
        best = r.get("Best Answer", "")
        incorrects = r.get("Incorrect Answers", "").split("; ")
        if best and incorrects and incorrects[0]:
            factual_true.append(f"{q} {best}")
            factual_false.append(f"{q} {incorrects[0]}")
    n = min(len(factual_true), len(factual_false), 64)
    axes["factual_tqa"] = {"true": factual_true[:n],
                           "false": factual_false[:n]}
    print(f"factual_tqa: {n}+{n} prompts", flush=True)

    # 3. HellaSwag correct vs incorrect completions
    from datasets import load_dataset
    hs = load_dataset("Rowan/hellaswag", split="validation")
    hs_correct, hs_incorrect = [], []
    for item in hs:
        ctx = item["ctx"]
        label = int(item["label"])
        endings = item["endings"]
        if label < len(endings):
            hs_correct.append(f"{ctx} {endings[label]}")
            wrong = (label + 1) % len(endings)
            hs_incorrect.append(f"{ctx} {endings[wrong]}")
        if len(hs_correct) >= 64:
            break
    axes["reasoning_hs"] = {"correct": hs_correct[:64],
                            "incorrect": hs_incorrect[:64]}
    print(f"reasoning_hs: {len(hs_correct[:64])}+{len(hs_incorrect[:64])} prompts",
          flush=True)

    # 4. HellaSwag by activity category (8 largest → semantic multi-class axis)
    from collections import Counter
    act_counts = Counter(hs["activity_label"])
    top_acts = [a for a, _ in act_counts.most_common(8)]
    act_prompts = {a: [] for a in top_acts}
    for item in hs:
        a = item["activity_label"]
        if a in act_prompts and len(act_prompts[a]) < 16:
            act_prompts[a].append(item["ctx"])
    axes["activity_hs"] = act_prompts
    print(f"activity_hs: {len(top_acts)} activities, "
          f"{sum(len(v) for v in act_prompts.values())} prompts", flush=True)

    # 5. ARC-Challenge by inferred science domain
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    domain_kw = {
        "biology": ["cell", "organism", "species", "plant", "animal",
                     "dna", "gene", "body", "food chain", "ecosystem"],
        "physics": ["force", "energy", "light", "sound", "gravity",
                     "motion", "speed", "electric", "magnet", "wave"],
        "earth_sci": ["rock", "weather", "climate", "earth", "ocean",
                       "volcano", "fossil", "erosion", "plate", "mineral"],
        "chemistry": ["atom", "element", "chemical", "molecule", "react",
                       "solution", "acid", "metal", "gas", "compound"],
    }
    dom_prompts = {d: [] for d in domain_kw}
    for item in arc:
        q = item["question"].lower()
        for d, kws in domain_kw.items():
            if any(kw in q for kw in kws) and len(dom_prompts[d]) < 16:
                dom_prompts[d].append(item["question"])
                break
    dom_prompts = {d: v for d, v in dom_prompts.items() if len(v) >= 8}
    axes["domain_arc"] = dom_prompts
    print(f"domain_arc: {len(dom_prompts)} domains, "
          f"{sum(len(v) for v in dom_prompts.values())} prompts", flush=True)

    return axes


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = build_axes()

    print("\nloading Pythia-160m...", flush=True)
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"loaded: {n_layers} layers\n", flush=True)

    out = {"model": "EleutherAI/pythia-160m", "n_layers": n_layers, "axes": {}}

    # Collect all prompts for the random control (use factual pool)
    random_pool_prompts = (axes["factual_tqa"]["true"]
                           + axes["factual_tqa"]["false"])

    for axis_name, classes in axes.items():
        class_names = list(classes.keys())
        n_classes = len(class_names)
        all_prompts, labels = [], []
        for ci, cn in enumerate(class_names):
            for p in classes[cn]:
                all_prompts.append(p)
                labels.append(ci)
        labels = np.array(labels)
        print(f"[{axis_name}] {n_classes} classes, {len(all_prompts)} prompts",
              flush=True)

        per_layer = get_hidden_states(model, tokenizer, all_prompts)

        axis_results = {"n_classes": n_classes, "n_prompts": len(all_prompts),
                        "class_names": class_names, "layers": []}
        for l in range(len(per_layer)):
            X = per_layer[l].astype(np.float32)
            X = X / (X.std() + 1e-9)
            d, sh = ladder_delta(X, labels, n_classes,
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
        # Summary line
        if axis_results["layers"]:
            ds = [lr["delta"] for lr in axis_results["layers"]]
            ss = [lr["shuffle_mean"] for lr in axis_results["layers"]]
            print(f"  delta range: [{min(ds):+.3f}, {max(ds):+.3f}] | "
                  f"shuffle range: [{min(ss):+.3f}, {max(ss):+.3f}]",
                  flush=True)

    # Random control on the factual pool
    print(f"\n[random] {len(random_pool_prompts)} prompts", flush=True)
    per_layer_rand = get_hidden_states(model, tokenizer, random_pool_prompts)
    rand_results = {"n_prompts": len(random_pool_prompts), "layers": []}
    rng = np.random.default_rng(999)
    for l in range(len(per_layer_rand)):
        X = per_layer_rand[l].astype(np.float32)
        X = X / (X.std() + 1e-9)
        rand_labels = rng.integers(0, 2, len(X))
        d, sh = ladder_delta(X, rand_labels, 2,
                              np.random.default_rng(l * 100 + 99))
        rand_results["layers"].append({
            "layer": l, "delta": d, "shuffle_mean": sh if sh else 0.0})
    out["axes"]["random"] = rand_results
    ds = [lr["delta"] for lr in rand_results["layers"]
          if lr["delta"] is not None]
    print(f"  random mean: {np.mean(ds):+.3f} | "
          f"max |delta|: {max(abs(d) for d in ds):.3f}", flush=True)

    json.dump(out, open(HERE / "run16_scaled_battery.json", "w"), indent=1)
    print("\nDONE run16", flush=True)


if __name__ == "__main__":
    main()
