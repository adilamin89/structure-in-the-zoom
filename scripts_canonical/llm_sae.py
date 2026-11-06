"""The shift with a sparse autoencoder. Pythia-160m, EleutherAI/sae-pythia-160m-32k
(TopK, k = 32, 32,768 latents) at layers 4 and 8 (hooked on gpt_neox.layers.L output = hidden_states[L + 1]; checked here by
reconstruction error against hidden_states[L] and [L + 2]). On the battery's cached last-token states (content and construction), the
order-averaged shift and its permutation null (llm_battery.analyse_battery, unchanged) for (i) the residual stream,
(ii) the SAE codes, (iii) the reconstruction error x - x_hat, at each SAE layer.

Out: data_canonical/llm_sae.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h2b", HERE / "llm_battery.py")
h2b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h2b)
LAYERS = (4, 8)
K = 32


def encode(w, x):
    pre = np.maximum((x - w["b_dec"]) @ w["encoder.weight"].T + w["encoder.bias"], 0.0)
    idx = np.argpartition(-pre, K - 1, axis=1)[:, :K]
    codes = np.zeros_like(pre)
    np.put_along_axis(codes, idx, np.take_along_axis(pre, idx, axis=1), axis=1)
    return codes, codes @ w["W_dec"] + w["b_dec"]


def fvu(x, xh):
    return float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())


def main():
    out = {"layers": {}}
    for b in ("world_knowledge", "language_type"):
        z = np.load(h2b.STATES / f"pythia-160m__{b}.npz")
        prompts, labels, names, counts = h2b.battery_prompts(b)
        assert np.array_equal(labels, z["labels"]) and list(z["prompts"]) == prompts
        H = z["H"].astype(np.float64)
        stack, tags = [], []
        for L in LAYERS:
            w = {k: v.astype(np.float64) for k, v in
                 load_file(hf_hub_download("EleutherAI/sae-pythia-160m-32k", f"layers.{L}/sae.safetensors")).items()}
            check = {f"hidden_states[{j}]": fvu(H[j], encode(w, H[j])[1]) for j in (L, L + 1, L + 2)}
            x = H[L + 1]
            codes, xh = encode(w, x)
            print(f"{b} SAE layers.{L}: FVU {check}; active latents per prompt {int((codes > 0).sum(1).mean())}, "
                  f"distinct active {int((codes > 0).any(0).sum())}", flush=True)
            out["layers"].setdefault(f"{b}/layers.{L}", {})["fvu_check"] = check
            out["layers"][f"{b}/layers.{L}"]["distinct_active_latents"] = int((codes > 0).any(0).sum())
            for tag, X in (("residual", x), ("codes", codes[:, (codes > 0).any(0)]), ("error", x - xh)):
                stack.append(X)
                tags.append(f"layers.{L}/{tag}")
        # analyse_battery takes a layers x prompts x features array: run each readout as its own one-'layer' stack
        base_floor = h2b.floor
        for X, tag in zip(stack, tags):
            # floor draws seeded as the battery's row for this hidden state (l * 100 + 1, l = L + 1), so the residual reproduces it
            lh = int(tag.split("/")[0].split(".")[1]) + 1
            h2b.floor = lambda K_, n_, sizes_, seed_, _lh=lh: base_floor(K_, n_, sizes_, _lh * 100 + 1)
            res = h2b.analyse_battery(X[None], labels, names, counts, b)
            h2b.floor = base_floor
            if tag.endswith("residual"):
                ref = json.load(open(h2b.OUTDIR / "llm_battery_pythia-160m.json"))["batteries"][b]["layers"][lh]
                assert abs(res["layers"][0]["dbar50_run37"] - ref["dbar50_run37"]) < 1e-9, "residual != H2 row"
            r = res["layers"][0]
            out["layers"][f"{b}/{tag.split('/')[0]}"][tag.split("/")[1]] = {
                "dbar50": r["dbar50_run37"], "dbar200": r["dbar200"], "delta_declared": r["delta_declared"],
                "null_mean": r["perm"]["null_mean"], "null_sd": r["perm"]["null_sd"], "p_one": r["perm"]["p_one"],
                "excess": r["dbar50_run37"] - r["perm"]["null_mean"], "zscore_dbar50": r["zscore"]["dbar50"],
                "floor": r["floor"], "d_eff": r["spectrum"]["d_eff"]}
            print(f"  {b} {tag}: dbar50 {r['dbar50_run37']:+.3f} null {r['perm']['null_mean']:+.3f}+-{r['perm']['null_sd']:.3f} "
                  f"p {r['perm']['p_one']:.3f} (z-scored {r['zscore']['dbar50']:+.3f}) d_eff {r['spectrum']['d_eff']:.1f}",
                  flush=True)
    json.dump(out, open(h2b.OUTDIR / "llm_sae.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
