"""The indirect-object battery of Appendix I as a rung axis: the 128 prompts and labels stored in data_canonical/llm_ioi.json (eight
indirect-object names x 16 prompts; llm_ioi.py's battery, seed 0), written as {name: [prompts]} in label order, so that
`rung llm --model openai-community/gpt2 --axis axes/ioi_names.json --heads --paper-seeds` replays the paper's per-head run.
Out: axes/ioi_names.json (+ its classes in axes/INDEX.json)"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
IO_NAMES = ["Mary", "John", "Tom", "James", "Sarah", "Paul", "Kate", "Dan"]   # llm_ioi.IO_NAMES: class c is IO_NAMES[c]
d = json.load(open(HERE.parent / "data_canonical" / "llm_ioi.json"))
axis = {nm: [p for p, y in zip(d["prompts"], d["labels"]) if y == c] for c, nm in enumerate(IO_NAMES)}
assert all(len(v) == 16 for v in axis.values()) and sum(len(v) for v in axis.values()) == d["n_prompts"] == 128
for nm, ps in axis.items():
    assert all(p.count(nm) >= 1 for p in ps), nm                                  # every prompt names its indirect object
json.dump(axis, open(HERE.parent / "axes" / "ioi_names.json", "w"), indent=1)
idx = json.load(open(HERE.parent / "axes" / "INDEX.json"))
idx["ioi_names"] = IO_NAMES
json.dump(idx, open(HERE.parent / "axes" / "INDEX.json", "w"), indent=1)
print("wrote axes/ioi_names.json and its INDEX.json entry")
