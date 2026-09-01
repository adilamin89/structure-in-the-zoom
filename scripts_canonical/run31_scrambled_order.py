"""Run 31 — Scrambled-word-order control for the structural axis.

WHY: the structural (sentence-construction) axis could be organized by word
DISTRIBUTION (which function words appear) rather than by structure (how
they are arranged). Scrambling word order destroys syntax while preserving
the bag of words. Design subtlety: the readout is the last-token state, so
naive scrambling changes the reading position's identity and would destroy
the signal trivially; here every prompt is scrambled EXCEPT its final word,
preserving the completion position while destroying arrangement.

REGISTERED EXPECTATIONS (written before the run):
S1: content axis (world_knowledge) survives scrambling — its organization
    is lexical and order-free.
S2: THE TEST — if the structural axis's rising profile survives
    all-but-last scrambling, its organization is word-distributional; if it
    collapses toward the shuffle band, it requires intact arrangement.
    Either outcome is reportable; the prediction is substantial weakening,
    since construction types are defined by arrangement, but their function
    words (what/because/first/unlike) survive scrambling as markers.
S3: random relabeling ~ 0 on scrambled prompts.

Model: Pythia-160m (MPS). Out: ../data_canonical/run31_scrambled_order.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run31_scrambled_order.json"

spec = importlib.util.spec_from_file_location(
    "r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r17)


def scramble_keep_last(prompt, rng):
    words = prompt.split()
    if len(words) < 3:
        return prompt
    head = words[:-1]
    rng.shuffle(head)
    return " ".join(head + [words[-1]])


def run_axis(model, tokenizer, classes, labels_of, scramble, tag, rng):
    class_names = list(classes.keys())
    prompts, labels = [], []
    for ci, cn in enumerate(class_names):
        for p in classes[cn]:
            prompts.append(scramble_keep_last(p, rng) if scramble else p)
            labels.append(ci)
    labels = np.array(labels)
    per_layer = r17.get_hidden_states(model, tokenizer, prompts,
                                      device="mps")
    rows = []
    for l in range(len(per_layer)):
        X = per_layer[l].astype(np.float32)
        X = X / (X.std() + 1e-9)
        d, sh = r17.ladder_delta(X, labels, 8,
                                 np.random.default_rng(l * 100 + 1))
        if d is not None:
            rows.append({"layer": l, "delta": d, "shuffle_mean": sh})
    ds = [r["delta"] for r in rows]
    print(f"  [{tag}] delta [{min(ds):+.3f}, {max(ds):+.3f}] "
          f"emb {ds[0]:+.3f} final {ds[-1]:+.3f}", flush=True)
    return rows


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = r17.build_axes()
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m").to("mps")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    rng = np.random.default_rng(31)
    out = {"model": "EleutherAI/pythia-160m",
           "design": "all-but-last-word scrambling, seed 31", "axes": {}}
    for axis_name in ["world_knowledge", "language_type"]:
        for scram in [False, True]:
            tag = f"{axis_name}{'_scrambled' if scram else '_intact'}"
            print(f"\n[{tag}]", flush=True)
            out["axes"][tag] = run_axis(model, tokenizer, axes[axis_name],
                                        None, scram, tag,
                                        np.random.default_rng(31))
            json.dump(out, open(OUT, "w"), indent=1)
    print("\nDONE run31", flush=True)


if __name__ == "__main__":
    main()
