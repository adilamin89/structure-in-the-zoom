"""Run 15 - Multi-axis δ battery on Pythia-160m (all layers, full controls).

The instrument paper's generality demonstration: same model, same estimator,
different declared axes → different depth profiles. Each axis gets its own
floor, shuffle control (5 permutations per layer), and embedding-excess
computation (δ_excess = δ(layer) - δ(embedding)).

AXES:
  topic:     8 topic classes × 8 prompts (existing battery from topic_class_ladder.py)
  factual:   true-completion vs false-completion (8 pairs)
  reasoning: logically valid vs invalid continuations (8 pairs)
  sentiment: positive vs negative affect (8 pairs)
  syntax:    simple vs complex constructions (8 pairs)
  random:    random prompt subsets (floor calibration; must give δ ≈ 0)

REGISTERED EXPECTATIONS (before run):
R1: δ_random ≈ 0 at every layer (exchangeability; validates the floor).
R2: δ_topic positive and declining with depth (replicates existing result).
R3: different axes give different depth profiles - the paper's thesis.
R4: embedding-excess separates lexical from network-organized structure.

Model: EleutherAI/pythia-160m (local HF cache, CPU).
Out: feedback_runs/run15_multiaxis_pythia160m.json
"""
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# ---- Prompt batteries ----
AXES = {
    "topic": {
        "arithmetic": [
            "Two plus two equals", "Seven times eight is",
            "The square root of sixteen is", "Half of ninety is",
            "Ten minus three equals", "Five squared is",
            "The sum of one through ten is", "Twelve divided by four is"],
        "geography": [
            "The capital of France is", "The longest river in the world is",
            "Mount Everest is located in", "The largest ocean on Earth is",
            "The Sahara desert lies in", "Japan is an island nation in",
            "The Amazon rainforest spans", "Australia is a continent in the"],
        "history": [
            "The Roman Empire fell in", "World War Two ended in",
            "The French Revolution began", "The first moon landing was",
            "The Berlin Wall fell in", "The Renaissance started in",
            "Ancient Egypt built pyramids", "The Industrial Revolution began"],
        "biology": [
            "DNA stands for deoxyribonucleic", "Mitochondria are the powerhouse",
            "Photosynthesis converts sunlight", "The human heart has four",
            "Cells divide through mitosis", "Proteins are made of amino",
            "Evolution occurs through natural", "The brain contains billions of"],
        "physics": [
            "The speed of light is", "Gravity pulls objects toward",
            "Energy cannot be created", "Atoms are made of protons",
            "Electricity flows through conductors", "Sound travels as a wave",
            "Temperature measures kinetic energy", "Force equals mass times"],
        "literature": [
            "Shakespeare wrote Hamlet and", "The Odyssey was written by",
            "Moby Dick begins with", "Pride and Prejudice is about",
            "War and Peace was authored", "The Great Gatsby depicts",
            "To Kill a Mockingbird addresses", "One Hundred Years of Solitude"],
        "music": [
            "Beethoven composed nine symphonies", "Mozart was born in Salzburg",
            "The piano has eighty eight", "Jazz originated in New Orleans",
            "A guitar typically has six", "Bach wrote the Well Tempered",
            "The orchestra is led by", "Rhythm is the pattern of"],
        "cooking": [
            "Boiling water reaches one hundred", "Yeast makes bread dough rise",
            "Salt enhances the flavor of", "Olive oil is pressed from",
            "Sushi is a traditional Japanese", "Chocolate is made from cacao",
            "Garlic is used in many", "Baking requires precise measurements"],
    },
    "factual": {
        "true": [
            "The Earth orbits the Sun", "Water freezes at zero degrees",
            "Humans have forty six chromosomes", "The moon causes ocean tides",
            "Plants need sunlight to grow", "Iron is a magnetic metal",
            "Sound cannot travel through vacuum", "Diamonds are made of carbon"],
        "false": [
            "The Sun orbits the Earth", "Water freezes at fifty degrees",
            "Humans have seventy two chromosomes", "The sun causes ocean tides",
            "Plants need darkness to grow", "Wood is a magnetic metal",
            "Sound travels fastest through vacuum", "Diamonds are made of silicon"],
    },
    "reasoning": {
        "valid": [
            "All dogs are animals and all animals breathe so all dogs",
            "If it rains the ground gets wet and it rained so the ground",
            "Every prime greater than two is odd and seven is prime so seven is",
            "Mammals are warm blooded and whales are mammals so whales are",
            "All squares are rectangles and this shape is a square so it is a",
            "If the battery dies the phone stops and the battery died so",
            "Birds have feathers and eagles are birds so eagles have",
            "All planets orbit stars and Earth is a planet so Earth"],
        "invalid": [
            "All dogs are animals and all cats are animals so all dogs are",
            "If it rains the ground gets wet and the ground is wet so it",
            "Some primes are odd and seven is odd so seven is",
            "Mammals are warm blooded and lizards are warm so lizards are",
            "All squares are rectangles and this shape is a rectangle so it is a",
            "If the battery dies the phone stops and the phone stopped so",
            "Birds have feathers and this coat has feathers so this coat is a",
            "All planets orbit stars and the moon orbits Earth so the moon is a"],
    },
    "sentiment": {
        "positive": [
            "The celebration was joyful and everyone", "She smiled brightly at the wonderful",
            "The warm sunshine made everyone feel", "They cheered with excitement at the",
            "The garden bloomed beautifully in the", "His kindness inspired everyone around",
            "The music filled the room with", "A peaceful morning greeted the happy"],
        "negative": [
            "The disaster left everyone devastated and", "He frowned angrily at the terrible",
            "The cold darkness made everyone feel", "They cried with despair at the",
            "The wasteland decayed horribly in the", "His cruelty frightened everyone around",
            "The noise filled the room with", "A grim morning greeted the miserable"],
    },
    "syntax": {
        "simple": [
            "The cat sat on the mat", "She ate the red apple quickly",
            "The boy ran to the store", "It rained all day yesterday",
            "The bird sang in the tree", "He read the book at night",
            "The dog chased the small ball", "They walked along the river"],
        "complex": [
            "The cat which had been sleeping on the mat that was placed near the window",
            "She who had never eaten such a peculiarly colored apple so quickly before",
            "The boy whose mother had asked him to run to the store that was",
            "It having rained all day yesterday despite the forecast that had predicted",
            "The bird which had been singing in the tree that stood beside the",
            "He who had been reading the book at night while the others were",
            "The dog that had been chasing the small ball which had rolled under the",
            "They who had walked along the river that wound through the valley where"],
    },
}

BIN_COUNTS_MULTI = [1, 2]  # 2-class axes: accumulate 1 then 2
BIN_COUNTS_TOPIC = [1, 2, 3, 4, 6, 8]  # 8-class topic axis
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
    bin_counts = BIN_COUNTS_TOPIC if n_classes > 2 else BIN_COUNTS_MULTI
    bin_counts = [c for c in bin_counts if c <= n_classes]
    members = [np.where(labels == c)[0] for c in range(n_classes)]
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


def get_hidden_states(model, tokenizer, prompts, device="cpu"):
    """Extract last-token hidden states at every layer."""
    all_states = []
    model.eval()
    with torch.no_grad():
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
            out = model(ids, output_hidden_states=True)
            # out.hidden_states = (embedding, layer1, ..., layerN)
            states = [h[0, -1, :].cpu().numpy() for h in out.hidden_states]
            all_states.append(states)
    # Reorganize: per_layer[l] = (n_prompts, hidden_dim)
    n_layers = len(all_states[0])
    per_layer = [np.stack([s[l] for s in all_states]) for l in range(n_layers)]
    return per_layer


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("loading Pythia-160m...", flush=True)
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"loaded: {n_layers} layers", flush=True)

    out = {"model": "EleutherAI/pythia-160m", "n_layers": n_layers, "axes": {}}

    for axis_name, classes in AXES.items():
        class_names = list(classes.keys())
        n_classes = len(class_names)
        all_prompts = []
        labels = []
        for ci, cn in enumerate(class_names):
            for p in classes[cn]:
                all_prompts.append(p)
                labels.append(ci)
        labels = np.array(labels)
        print(f"\n[{axis_name}] {n_classes} classes, {len(all_prompts)} prompts",
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
            if l % 4 == 0 or l == len(per_layer) - 1:
                print(f"  layer {l}: delta={d:+.3f} shuffle={sh:+.3f}",
                      flush=True)

        # embedding excess
        emb_delta = axis_results["layers"][0]["delta"]
        for lr in axis_results["layers"]:
            lr["delta_excess"] = lr["delta"] - emb_delta

        out["axes"][axis_name] = axis_results

    # random-axis control (random 2-class splits of all topic prompts)
    all_topic = []
    for cn in AXES["topic"]:
        all_topic.extend(AXES["topic"][cn])
    per_layer_topic = get_hidden_states(model, tokenizer, all_topic)
    rand_results = {"n_prompts": len(all_topic), "layers": []}
    rng = np.random.default_rng(999)
    for l in range(len(per_layer_topic)):
        X = per_layer_topic[l].astype(np.float32)
        X = X / (X.std() + 1e-9)
        rand_labels = rng.integers(0, 2, len(X))
        d, sh = ladder_delta(X, rand_labels, 2,
                             np.random.default_rng(l * 100 + 99))
        rand_results["layers"].append({"layer": l, "delta": d,
                                       "shuffle_mean": sh if sh else 0.0})
    out["axes"]["random"] = rand_results
    print(f"\n[random] mean delta = "
          f"{np.mean([lr['delta'] for lr in rand_results['layers'] if lr['delta'] is not None]):+.3f}",
          flush=True)

    json.dump(out, open(HERE / "run15_multiaxis_pythia160m.json", "w"),
              indent=1)
    print("\nDONE run15", flush=True)


if __name__ == "__main__":
    main()
