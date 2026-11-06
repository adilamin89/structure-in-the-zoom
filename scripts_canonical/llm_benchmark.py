"""The LM battery's benchmark axes at the largest balanced size their sources allow.

  build                       -> data_canonical/llm_benchmark/axes/<axis>.json (+ .strata.json, .provenance.json)
  encode <hf_model> <device> <dtype> <key>   -> raw/llm_states/<key>__<axis>.npz (run17.get_hidden_states, as llm_battery)
  analyse <key> [axis ...]    -> data_canonical/llm_benchmark/llm_benchmark_<key>.json (llm_battery.analyse_battery, unchanged)
  lasttoken <key> <hf_tokenizer> [axis ...]  -> data_canonical/llm_benchmark/llm_lasttoken*_<key>.json (the llm_lasttoken scripts, unchanged)

Axes (rung.build_axis, seed 0, texts kept whole): hs_activity_64 (HellaSwag validation ctx, run17's 8 most frequent
activities), arc_topic_64 (ARC-Challenge + ARC-Easy, all splits, run17's keyword rule, strata = source subset),
tqa_category_26 (TruthfulQA, run17's 8 largest categories, 26 per class = all of the two smallest).
"""
import csv
import importlib.util
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_battery as hb  # noqa: E402

OUT = hb.OUTDIR / "llm_benchmark"
AXDIR = OUT / "axes"
SUPP = HERE.parent   # the repository root (rung.py, axes/)
AXES = ("hs_activity_64", "arc_topic_64", "tqa_category_26", "ethics_domain_64")
COUNTS = [1, 2, 3, 4, 6, 8]
TQA_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"


def _rung():
    spec = importlib.util.spec_from_file_location("rung", SUPP / "rung.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arc_domains():
    """run17's keyword dictionary, read from its source so the rule is byte-identical."""
    src = open(HERE / "run17_multiclass_battery.py").read()
    start = src.index("    domains = {")
    end = src.index("    dom_prompts = {d: [] for d in domains}")
    ns = {}
    exec("domains = " + src[start:end].split("domains = ", 1)[1], ns)
    return ns["domains"]


def build():
    from datasets import load_dataset
    rung = _rung()
    AXDIR.mkdir(parents=True, exist_ok=True)
    old = {a: json.load(open(SUPP / "axes" / f"{a}.json")) for a in ("hs_activity", "arc_topic", "tqa_category")}
    kw = dict(min_chars=1, max_chars=10000, seed=0)

    # HellaSwag: run17's 8 most frequent validation activities, same names and order
    hs = load_dataset("Rowan/hellaswag", split="validation")
    names = [a for a, _ in Counter(hs["activity_label"]).most_common(8)]
    assert names == list(old["hs_activity"]), (names, list(old["hs_activity"]))
    rows = [{"text": r["ctx"], "label": r["activity_label"]} for r in hs]
    axis, _ = rung.build_axis(rows, "text", "label", n_per_class=64, classes=names, **kw)
    prov = {"source": "Rowan/hellaswag (Zellers et al. 2019), split validation, field ctx, label activity_label",
            "rule": "run17's 8 most frequent activities; 64 per class drawn by rung.build_axis(seed 0); texts whole",
            "pool_sizes": {a: int(sum(r["label"] == a for r in rows)) for a in names}}
    _save("hs_activity_64", axis, None, prov)

    # ARC: run17's keyword rule over ARC-Challenge + ARC-Easy, all splits
    domains = _arc_domains()
    assert list(domains) == list(old["arc_topic"]), list(domains)
    rows = []
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for sp in ("train", "validation", "test"):
            for item in load_dataset("allenai/ai2_arc", cfg, split=sp):
                q = item["question"].lower()
                for d, kws in domains.items():
                    if any(k in q for k in kws):
                        rows.append({"text": item["question"], "label": d, "source": f"{cfg}/{sp}"})
                        break
    axis, strata = rung.build_axis(rows, "text", "label", n_per_class=64, classes=list(domains),
                                   strata_field="source", **kw)
    prov = {"source": "allenai/ai2_arc (Clark et al. 2018), ARC-Challenge + ARC-Easy, splits train/validation/test",
            "rule": "run17's keyword rule (first domain whose keyword list occurs in the lower-cased question, dict order); "
                    "64 per class by rung.build_axis(seed 0); strata = config/split",
            "pool_sizes": dict(Counter(r["label"] for r in rows)),
            "why_pooled": "ARC-Challenge alone gives human_body 34 over all splits (test 14)",
            "challenge_fraction_per_class": {d: float(np.mean([s.startswith("ARC-Challenge") for s in strata[d]])) for d in domains}}
    _save("arc_topic_64", axis, strata, prov)

    # TruthfulQA: run17's 8 largest categories, the largest balanced size
    path = OUT / "TruthfulQA.csv"
    if not path.exists():
        urllib.request.urlretrieve(TQA_URL, path)
    tqa = list(csv.DictReader(open(path)))
    cats = Counter(r["Category"] for r in tqa)
    names = [c for c, _ in sorted(cats.items(), key=lambda kv: -kv[1])[:8]]
    assert names == list(old["tqa_category"]), (names, list(old["tqa_category"]))
    n = min(cats[c] for c in names)
    assert f"tqa_category_{n}" in AXES, n
    rows = [{"text": r["Question"], "label": r["Category"]} for r in tqa]
    axis, _ = rung.build_axis(rows, "text", "label", n_per_class=n, classes=names, **kw)
    prov = {"source": f"TruthfulQA.csv (Lin, Hilton, Evans 2022), {TQA_URL}", "n_per_class": n,
            "rule": f"run17's 8 largest categories; {n} per class (the smallest category's size) by rung.build_axis(seed 0)",
            "pool_sizes": {c: cats[c] for c in names}}
    _save(f"tqa_category_{n}", axis, None, prov)


def build_ethics():
    """ETHICS (Hendrycks et al. 2021) normative domains from the original release's CSVs, 64 per class."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    rung = _rung()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    AXDIR.mkdir(parents=True, exist_ok=True)
    rows, pool = [], {}
    for d in ("commonsense", "deontology", "justice", "utilitarianism", "virtue"):
        f = hf_hub_download("hendrycks/ethics", f"data/{d}/train.csv", repo_type="dataset")
        df = pd.read_csv(f, header=None if d == "utilitarianism" else 0)
        if d == "commonsense":
            texts = df.loc[df["is_short"].astype(str) == "True", "input"]
        elif d == "deontology":
            texts = df["scenario"].astype(str) + " " + df["excuse"].astype(str)
        elif d == "utilitarianism":
            texts = df[0]
        elif d == "virtue":
            texts = df["scenario"].astype(str).str.split(r"\s*\[SEP\]\s*", regex=True).str[0]
        else:
            texts = df["scenario"]
        seen, kept, long_ = set(), 0, 0
        for t in texts.astype(str):
            t = " ".join(t.split())
            if t in seen:
                continue
            seen.add(t)
            if len(tok(t).input_ids) > 128:
                long_ += 1
                continue
            rows.append({"text": t, "label": d})
            kept += 1
        pool[d] = {"unique": len(seen), "kept_le_128_tokens": kept, "dropped_over_128": long_}
    names = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]
    assert not any("[SEP]" in r["text"] for r in rows), "virtue trait not stripped"
    axis, _ = rung.build_axis(rows, "text", "label", n_per_class=64, classes=names, min_chars=1, max_chars=100000, seed=0)
    prov = {"source": "hendrycks/ethics (Hendrycks et al. 2021, ICLR), original release CSVs data/<domain>/train.csv",
            "rule": "commonsense: input of is_short items; deontology: scenario + ' ' + excuse; justice: scenario; "
                    "utilitarianism: first scenario (baseline); virtue: scenario before ' [SEP] ' (trait dropped); "
                    "whitespace-normalized, de-duplicated, <= 128 Pythia tokens; 64 per class by rung.build_axis(seed 0)",
            "pool": pool,
            "format_note": "the five subsets differ in surface format (deontology items are request + excuse pairs, virtue "
                           "items third-person sentences, commonsense and utilitarianism first-person), so the partition "
                           "carries format as well as norm type"}
    _save("ethics_domain_64", axis, None, prov)


def _save(name, axis, strata, prov):
    prov["n_classes"] = len(axis)
    prov["n_per_class"] = {k: len(v) for k, v in axis.items()}
    json.dump(axis, open(AXDIR / f"{name}.json", "w"), indent=1)
    if strata is not None:
        json.dump(strata, open(AXDIR / f"{name}.strata.json", "w"), indent=1)
    json.dump(prov, open(AXDIR / f"{name}.provenance.json", "w"), indent=1)
    print(f"wrote {name}: {prov['n_per_class']}", flush=True)


def axis_prompts(name):
    axis = json.load(open(AXDIR / f"{name}.json"))
    names = list(axis)
    prompts = [p for c in names for p in axis[c]]
    labels = np.array([i for i, c in enumerate(names) for _ in axis[c]])
    counts = [c for c in COUNTS if c <= len(names)]
    if counts[-1] != len(names):
        counts.append(len(names))
    return prompts, labels, names, counts


def _patch():
    """Let the h2_* code read the benchmark axes by name (only for names it does not know)."""
    base = hb.battery_prompts

    def bp(name):
        return axis_prompts(name) if name in AXES else base(name)
    hb.battery_prompts = bp


def encode(model_name, device, dtype_name, key):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    for a in (sys.argv[6:] or AXES):
        out = hb.STATES / f"{key}__{a}.npz"
        if out.exists():
            print("skip", out.name, flush=True)
            continue
        prompts, labels, names, _ = axis_prompts(a)
        H = np.stack(hb.r17.get_hidden_states(model, tok, prompts, device=device))
        np.savez(out, H=H, labels=labels, names=np.array(names), prompts=np.array(prompts), model=model_name,
                 device=device, dtype=dtype_name)
        print(f"{key} {a}: H {H.shape} finite={np.isfinite(H).all()}", flush=True)


def analyse(key, axes):
    out_path = OUT / f"llm_benchmark_{key}.json"
    out = json.load(open(out_path)) if out_path.exists() else {"key": key, "axes": {}}
    for a in axes:
        z = np.load(hb.STATES / f"{key}__{a}.npz")
        prompts, labels, names, counts = axis_prompts(a)
        assert np.array_equal(labels, z["labels"]) and list(z["prompts"]) == prompts
        print(f"[{key} / {a}] H {z['H'].shape} {z['dtype']}", flush=True)
        res = hb.analyse_battery(z["H"], labels, names, counts, a)
        sfile = AXDIR / f"{a}.strata.json"
        if sfile.exists():
            # within-source stratified null for the pooled ARC axis (200 permutations x 50 orders, as llm_battery)
            st = json.load(open(sfile))
            sv = np.array([s for c in names for s in st[c]])
            strata = {t: np.where(sv == t)[0] for t in np.unique(sv)}
            rng = np.random.default_rng(4802)
            draws = [(hb.null_labels("strat", labels, rng, strata), [rng.permutation(len(names)).tolist() for _ in range(50)])
                     for _ in range(200)]
            for row in res["layers"]:
                K = hb.gram(z["H"][row["layer"]], "linear")
                vals = np.array([hb.order_avg(K, hb.members_of(lab, len(names)), o, counts, row["floor"]) for lab, o in draws])
                d = row["dbar50_run37"]
                row["strat_source"] = {"null_mean": float(vals.mean()), "null_sd": float(vals.std()),
                                       "p_one": float((1 + (vals >= d).sum()) / 201)}
        res.update({"dtype": str(z["dtype"]), "device": str(z["device"])})
        out["axes"][a] = res
        json.dump(out, open(out_path, "w"), indent=1)
    print("DONE", flush=True)


def lasttoken(key, tok_name, axes):
    _patch()
    import llm_lasttoken as lt
    import llm_lasttoken_power as ltp
    hb.OUTDIR = OUT                       # the llm_lasttoken scripts write llm_lasttoken*_<key>.json into data_canonical/llm_benchmark/
    lt.run(key, tok_name, axes)
    ltp.run(key, tok_name, axes)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "build_ethics":
        build_ethics()
    elif cmd == "encode":
        encode(*sys.argv[2:6])
    elif cmd == "analyse":
        analyse(sys.argv[2], sys.argv[3:] or list(AXES))
    elif cmd == "lasttoken":
        lasttoken(sys.argv[2], sys.argv[3], sys.argv[4:] or list(AXES))
    print("EXIT_OK", flush=True)
