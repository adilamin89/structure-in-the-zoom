"""Hidden states for the generality models and the within-block components.

Prompts are the battery's, byte-identical (llm_battery.battery_prompts, asserted against the cached run17_axes.json); states go
to raw/llm_states/<key>__<battery>.npz in the battery's format (H: layers x prompts x features, labels, names, prompts), so
llm_battery.analyse_battery reads them unchanged.

Usage (python -X faulthandler llm_encode.py <mode> <device>):
  causal <hf_model> <device> [dtype] [key]  decoder / state-space LM: last token, every hidden state (= llm_battery encode)
  encoder <hf_model> <device> <key>         BERT-like: [CLS] (key_cls) and attention-mask mean-pooled tokens (key_mean)
  sentemb <hf_model> <device> <key>         sentence embedder: mean-pooled tokens per layer (key_mean) and the normalized
                                            sentence embedding (key_sent, one 'layer')
  components <hf_model> <device> <key>      GPT-NeoX parallel residual: attention output and MLP output at the last token
                                            per block (key_attn, key_mlp), asserting h_{l+1} = h_l + attn_l + mlp_l
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h2b", HERE / "llm_battery.py")
h2b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h2b)
STATES = h2b.STATES
BATTERIES = ("world_knowledge", "language_type")


def _save(key, b, H, labels, names, prompts, model_name, device, dtype):
    out = STATES / f"{key}__{b}.npz"
    np.savez(out, H=H, labels=labels, names=np.array(names), prompts=np.array(prompts), model=model_name,
             device=device, dtype=dtype)
    print(f"{key} {b}: H {H.shape} finite={np.isfinite(H).all()} max|h|={np.abs(H).max():.1f}", flush=True)


def encoder(model_name, device, key, sentemb=False):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    for b in BATTERIES:
        prompts, labels, names, _ = h2b.battery_prompts(b)
        cls, mean, sent = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for p in prompts:
                enc = tok(p, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = model(**enc, output_hidden_states=True)
                m = enc["attention_mask"][0].float()[:, None]
                hs = out.hidden_states
                cls.append([h[0, 0].cpu().numpy() for h in hs])
                mean.append([((h[0] * m).sum(0) / m.sum()).cpu().numpy() for h in hs])
                if sentemb:   # all-MiniLM-L6-v2's own pooling: mean over tokens of the last state, then L2 normalize
                    v = mean[-1][-1]
                    sent.append(v / np.linalg.norm(v))
        stack = lambda S: np.stack([np.stack([s[l] for s in S]) for l in range(len(S[0]))]).astype(np.float32)
        if not sentemb:
            _save(f"{key}_cls", b, stack(cls), labels, names, prompts, model_name, device, "fp32")
        _save(f"{key}_mean", b, stack(mean), labels, names, prompts, model_name, device, "fp32")
        if sentemb:
            _save(f"{key}_sent", b, np.stack(sent)[None].astype(np.float32), labels, names, prompts, model_name,
                  device, "fp32")
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


def components(model_name, device, key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    layers = model.gpt_neox.layers
    assert all(l.use_parallel_residual for l in layers), "components() assumes GPT-NeoX parallel residual"
    ACT = {}

    def hook(name):
        def f(mod, inp, out):
            ACT[name] = (out[0] if isinstance(out, tuple) else out)[0, -1].detach()
        return f
    for i, l in enumerate(layers):
        l.attention.register_forward_hook(hook(("attn", i)))
        l.mlp.register_forward_hook(hook(("mlp", i)))
    nl = len(layers)
    for b in BATTERIES:
        prompts, labels, names, _ = h2b.battery_prompts(b)
        A, M, worst = [], [], 0.0
        with torch.no_grad():
            for p in prompts:
                ids = tok(p, return_tensors="pt", truncation=True, max_length=128).input_ids.to(device)
                hs = model(ids, output_hidden_states=True).hidden_states
                a = [ACT[("attn", i)] for i in range(nl)]
                m = [ACT[("mlp", i)] for i in range(nl)]
                for i in range(nl - 1):   # the last hidden state is after the final layer norm: check blocks 0..nl-2
                    pred = hs[i][0, -1] + a[i] + m[i]
                    rel = float((pred - hs[i + 1][0, -1]).norm() / hs[i + 1][0, -1].norm())
                    worst = max(worst, rel)
                A.append([x.cpu().numpy() for x in a])
                M.append([x.cpu().numpy() for x in m])
        print(f"{b}: residual identity worst relative error {worst:.2e}", flush=True)
        assert worst < 1e-3, worst
        for tag, S in (("attn", A), ("mlp", M)):
            H = np.stack([np.stack([s[l] for s in S]) for l in range(nl)]).astype(np.float32)
            _save(f"{key}_{tag}", b, H, labels, names, prompts, model_name, device, "fp32")


if __name__ == "__main__":
    mode = sys.argv[1]
    # prompts byte-identical to the battery's cache
    import json
    cache = json.load(open(STATES / "run17_axes.json"))
    for b in BATTERIES:
        pr, _, _, _ = h2b.battery_prompts(b)
        assert pr == [p for c in cache[b].values() for p in c]
    if mode == "causal":
        dt = sys.argv[4] if len(sys.argv) > 4 else None
        h2b.encode(sys.argv[2], sys.argv[3], dt, batteries=BATTERIES, key=sys.argv[5] if len(sys.argv) > 5 else None)
    elif mode == "encoder":
        encoder(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "sentemb":
        encoder(sys.argv[2], sys.argv[3], sys.argv[4], sentemb=True)
    elif mode == "components":
        components(sys.argv[2], sys.argv[3], sys.argv[4])
    print("DONE", flush=True)
