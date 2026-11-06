"""H3 part B (registrations H3-R10..R12, R11): H2's analysis, unchanged, on the generality models and the within-block
components. Reads raw/llm_states/<key>__<battery>.npz (h3_encode.py, or H2's encodes for OLMo-2-0425-1B) and runs
h2_llm_battery.analyse_battery (H2's seeds: 200 orders, 200 label permutations x 50 orders, 20-draw floor, zscore / pc4
variants and spectrum per layer).

Usage: python -X faulthandler h3_generality.py <key> [battery ...]  -> data_canonical/h3_generality_<key>.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h2b", HERE / "h2_llm_battery.py")
h2b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h2b)


def main(key, batteries, gram_variant="linear"):
    if gram_variant != "linear":
        # R10b: run analyse_battery unchanged with its 'linear' Gram replaced by another estimator (ladder, floor, nulls
        # and orders all read the replaced Gram); the variant columns it adds are then variants of the replaced one
        base = h2b.gram
        h2b.gram = lambda X, v: base(X, gram_variant if v == "linear" else v)
    suffix = "" if gram_variant == "linear" else f"__{gram_variant}"
    out_path = h2b.OUTDIR / f"h3_generality_{key}{suffix}.json"
    out = json.load(open(out_path)) if out_path.exists() else {"key": key, "batteries": {}, "gram": gram_variant,
                                                               "registration": "H3-R10..R12, R10b (ex0_REGISTERED.md)"}
    for b in batteries:
        z = np.load(h2b.STATES / f"{key}__{b}.npz")
        prompts, labels, names, counts = h2b.battery_prompts(b)
        assert np.array_equal(labels, z["labels"]) and list(z["prompts"]) == prompts, "prompts differ from H2's"
        print(f"[{key} / {b}] H {z['H'].shape} {z['dtype']} model {z['model']}", flush=True)
        res = h2b.analyse_battery(z["H"], z["labels"], [str(x) for x in z["names"]], counts, b)
        res.update({"dtype": str(z["dtype"]), "device": str(z["device"]), "model": str(z["model"])})
        out["batteries"][b] = res
        json.dump(out, open(out_path, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    args = sys.argv[2:]
    gv = "linear"
    if "--gram" in args:
        i = args.index("--gram")
        gv = args[i + 1]
        args = args[:i] + args[i + 2:]
    main(sys.argv[1], args or ["world_knowledge", "language_type"], gv)
