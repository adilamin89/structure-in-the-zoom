"""Write the minimal-pair benchmark sets of the paper's Section 8.4 as runnable axis files with their strata sidecars,
so that `rung llm --axis axes/baroni_complexity.json` (and blimp_grammaticality.json) reads them against both nulls.

baroni_complexity: six classes x 64 prompts (subordinated / coordinated, center_embedding / right_branching,
  ambiguous / unambiguous_high) from the run43b artifact, which stores the exact prompts; the stratum of a prompt is
  its minimal pair, so the within-pair label swap of the paper is the stratified null.
blimp_grammaticality: eight classes (four phenomena x good / bad) x 16 pairs, loaded from nyu-mll/blimp exactly as
  run42 did (the first 16 items of each paradigm); the stratum is the pair. Needs the `datasets` package and the
  network; skipped with a message if either is missing.
Out: ../axes/{baroni_complexity,blimp_grammaticality}.json + .strata.json
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent; AX = HERE.parent / "axes"; DC = HERE.parent / "data_canonical"

d = json.load(open(DC / "run43b_baroni_64pairs.json"))
names, prompts, labels = d["class_names"], d["prompts"], d["labels"]
n_pairs = d["n_pairs"]
axis = {c: [] for c in names}; strata = {c: [] for c in names}
for i, (p, l) in enumerate(zip(prompts, labels)):
    contrast = i // (2 * n_pairs); pair = (i % (2 * n_pairs)) // 2
    c = names[2 * contrast + (l % 2)] if len(names) == 6 else names[l]
    axis[c].append(p); strata[c].append(f"c{contrast}_p{pair}")
assert all(len(v) == n_pairs for v in axis.values()), {k: len(v) for k, v in axis.items()}
json.dump(axis, open(AX / "baroni_complexity.json", "w"), indent=1)
json.dump(strata, open(AX / "baroni_complexity.strata.json", "w"), indent=1)
print("baroni_complexity:", {k: len(v) for k, v in axis.items()})

B_PARADIGMS = ["regular_plural_subject_verb_agreement_1", "wh_island", "npi_present_1", "irregular_past_participle_verbs"]
N_ITEMS = 16
try:
    from datasets import load_dataset
    axis, strata = {}, {}
    for p in B_PARADIGMS:
        ds = load_dataset("nyu-mll/blimp", p, split="train")
        axis[p + "_good"] = [ds[i]["sentence_good"] for i in range(N_ITEMS)]
        axis[p + "_bad"] = [ds[i]["sentence_bad"] for i in range(N_ITEMS)]
        strata[p + "_good"] = [f"{p}_{i}" for i in range(N_ITEMS)]
        strata[p + "_bad"] = [f"{p}_{i}" for i in range(N_ITEMS)]
    json.dump(axis, open(AX / "blimp_grammaticality.json", "w"), indent=1)
    json.dump(strata, open(AX / "blimp_grammaticality.strata.json", "w"), indent=1)
    print("blimp_grammaticality:", {k: len(v) for k, v in axis.items()})
except Exception as e:
    print("blimp_grammaticality skipped:", e)
