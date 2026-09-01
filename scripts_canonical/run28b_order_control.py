"""Run 28b — order-permutation control for the planted-cycle circulant fit
(red-team item: is the compass kernel's stationarity tied to the planted
cyclic order, or would any ordering of these classes fit a circulant?).

For each layer's centered class-mean kernel, compute the circulant fit R2
under (i) the planted cyclic order, (ii) all orders reachable by rotation/
reflection of the planted order (the dihedral orbit, which stationarity
should respect), and (iii) 200 random non-dihedral orderings.

REGISTERED EXPECTATION: if the kernel genuinely tracks the planted cycle's
parity structure, dihedral-equivalent orders preserve R2 and random orders
collapse it; if any ordering fits, the stationarity is generic morphology
and the recovery claim fails.

Out: ../data_canonical/run28b_order_control.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run28b_order_control.json"

spec = importlib.util.spec_from_file_location(
    "r28", HERE / "run28_cyclic_axis_llm.py")
r28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r28)
r17 = r28.r17


def circ_r2(C, order):
    C = np.asarray(C)[np.ix_(order, order)]
    n = len(order)
    f = np.array([np.mean([C[i, (i + k) % n] for i in range(n)])
                  for k in range(n)])
    fs = (f + f[[(n - k) % n for k in range(n)]]) / 2
    Cc = np.array([[fs[(j - i) % n] for j in range(n)] for i in range(n)])
    m = ~np.eye(n, dtype=bool)
    resid = C[m] - Cc[m]
    tot = C[m] - C[m].mean()
    return float(1 - (resid ** 2).sum() / (tot ** 2).sum())


def dihedral_orbit(n=8):
    orders = []
    base = list(range(n))
    for s in range(n):
        rot = [(i + s) % n for i in base]
        orders.append(rot)
        orders.append(rot[::-1])
    return orders


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = {
        "compass": {d: [t.format(d=d) for t in r28.COMPASS_TEMPLATES]
                    for d in r28.COMPASS},
        "clock": {t_: [tpl.format(t=t_) for tpl in r28.CLOCK_TEMPLATES]
                  for t_ in r28.CLOCK},
    }
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m").to("mps")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    out = {"axes": {}}
    rng = np.random.default_rng(7)
    dih = dihedral_orbit()
    dih_set = {tuple(o) for o in dih}

    for axis_name, classes in axes.items():
        prompts, labels = [], []
        for ci, cn in enumerate(classes):
            for p in classes[cn]:
                prompts.append(p)
                labels.append(ci)
        labels = np.array(labels)
        per_layer = r17.get_hidden_states(model, tokenizer, prompts,
                                          device="mps")
        rows = []
        for l in range(1, len(per_layer)):  # skip degenerate embedding
            X = per_layer[l].astype(np.float32)
            X = X / (X.std() + 1e-9)
            means = np.stack([X[labels == c].mean(axis=0) for c in range(8)])
            cm = means - means.mean(axis=0)
            C = np.corrcoef(cm)
            planted = circ_r2(C, list(range(8)))
            dih_r2 = [circ_r2(C, o) for o in dih]
            rand_r2 = []
            while len(rand_r2) < 200:
                o = tuple(rng.permutation(8).tolist())
                if o in dih_set:
                    continue
                rand_r2.append(circ_r2(C, list(o)))
            rows.append({
                "layer": l, "planted_r2": planted,
                "dihedral_min": float(min(dih_r2)),
                "random_mean": float(np.mean(rand_r2)),
                "random_sd": float(np.std(rand_r2)),
                "random_max": float(np.max(rand_r2)),
                "n_random_ge_planted": int(sum(r >= planted
                                               for r in rand_r2))})
        out["axes"][axis_name] = rows
        mid = rows[len(rows) // 2]
        print(f"[{axis_name}] mid-layer planted R2 {mid['planted_r2']:.2f} | "
              f"dihedral min {mid['dihedral_min']:.2f} | "
              f"random {mid['random_mean']:.2f}±{mid['random_sd']:.2f} "
              f"(max {mid['random_max']:.2f}, "
              f">=planted: {mid['n_random_ge_planted']}/200)", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("DONE run28b", flush=True)


if __name__ == "__main__":
    main()
