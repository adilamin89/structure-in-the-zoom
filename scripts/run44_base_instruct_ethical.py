"""Run 44 — base-vs-instruct pairs on the ethical-concept axis (the RLHF probe pilot).

WHY: the ethical-concept axis is ~0 on every BASE model tested (Pythia
160m/410m/1B/2.8B, OLMo-1B, GPT-Neo), which the paper states as a falsifiable
matched-comparison prediction: preference-tuned models should organize
moral-category structure that base models do not. This run makes the matched
comparison on TWO architecture families:
  allenai/OLMo-2-0425-1B  vs  allenai/OLMo-2-0425-1B-Instruct
  Qwen/Qwen2.5-0.5B       vs  Qwen/Qwen2.5-0.5B-Instruct
Prompts are byte-identical to the battery (run17.build_axes: ethical = 8
moral-foundation classes x 16; world_knowledge as the positive control) and
are fed RAW (no chat template) to both members of each pair — the probe
reads representation geometry under identical input, not chat behavior.
Analysis = run37 machinery (500 label permutations per layer + order-averaged
deltabar, K=50), so every cell carries an inferential null.

PRE-DECLARED COMPARISON STAT: D_late = mean over the final third of layers of
(deltabar_instruct - deltabar_base), per axis per pair.

REGISTERED EXPECTATIONS (written before the run):
E1 (control): world_knowledge certified per-layer (order-avg p<0.05 at most
    layers) in all four models; D_late(wk) small relative to D_late(ethical).
E2 (probe — the standing prediction): ethical shows more per-layer
    significant cells and D_late > 0 on instruct vs base, in both pairs.
    Either outcome is reported; a null result bounds the effect of
    preference tuning at these scales (1B / 0.5B).

Out: ../data_canonical/run44_base_instruct_ethical.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run44_base_instruct_ethical.json"

spec = importlib.util.spec_from_file_location(
    "r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r37)
r17 = r37.r17
r37.DECLARED_DIRECTION["ethical"] = {"IE": +1, "PE": +1}

PAIRS = [
    ("allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-Instruct"),
    ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
]
AXES = ["world_knowledge", "ethical"]


def encode_and_analyze(model_name, axes, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nloading {model_name} ({device})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    res = {}
    for axis_name in AXES:
        classes = axes[axis_name]
        prompts, labels = [], []
        for ci, cn in enumerate(classes):
            for p in classes[cn]:
                prompts.append(p)
                labels.append(ci)
        labels = np.array(labels)
        print(f"[{model_name.split('/')[-1]} / {axis_name}] encoding "
              f"{len(prompts)}...", flush=True)
        pl = r17.get_hidden_states(model, tokenizer, prompts, device=device)
        res[axis_name] = r37.analyze_axis(pl, labels, len(classes), axis_name)
    del model
    return res


def main():
    axes_all = r17.build_axes()
    axes = {a: axes_all[a] for a in AXES}

    out = {"pairs": PAIRS, "models": {}}
    if OUT.exists():
        out = json.load(open(OUT))
        print(f"resumed: {list(out['models'])}", flush=True)

    for base, inst in PAIRS:
        for model_name in (base, inst):
            mkey = model_name.split("/")[-1]
            if mkey in out["models"]:
                continue
            out["models"][mkey] = encode_and_analyze(model_name, axes)
            json.dump(out, open(OUT, "w"), indent=1)

    # pre-declared comparison stat
    print("\nD_late (instruct - base, order-averaged deltabar, final third):",
          flush=True)
    out["D_late"] = {}
    for base, inst in PAIRS:
        bk, ik = base.split("/")[-1], inst.split("/")[-1]
        for axis in AXES:
            db = [l["delta_orderavg"]
                  for l in out["models"][bk][axis]["layers"]]
            di = [l["delta_orderavg"]
                  for l in out["models"][ik][axis]["layers"]]
            k = len(db) - len(db) // 3
            d_late = float(np.mean(di[k:]) - np.mean(db[k:]))
            nsig_b = sum(1 for l in out["models"][bk][axis]["layers"]
                         if l["p_two_orderavg"] < 0.05)
            nsig_i = sum(1 for l in out["models"][ik][axis]["layers"]
                         if l["p_two_orderavg"] < 0.05)
            out["D_late"][f"{bk}/{axis}"] = {
                "D_late": d_late, "nsig_base": nsig_b, "nsig_instruct": nsig_i,
                "n_layers": len(db)}
            print(f"  {bk} {axis}: D_late={d_late:+.4f} | sig cells "
                  f"base {nsig_b}/{len(db)} -> instruct {nsig_i}/{len(di)}",
                  flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nDONE run44", flush=True)


if __name__ == "__main__":
    main()
