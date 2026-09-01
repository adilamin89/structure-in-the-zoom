"""Run 41 (M6) — CNN ordering prediction rerun, storing PER-SEED values.

WHY: run12's artifact stores only mean/sd per (net, layer, order) cell; the
referee's M6 asks for a paired analysis, which needs the per-seed effect
values (each seed trains one net evaluated under both orders — a paired
design). Seeds and rngs are identical to run12 (torch.manual_seed(seed),
default_rng(seed), default_rng(10+seed)), so the cell means must REPRODUCE;
this run adds per-seed storage plus paired SE and the exact sign test.

REGISTERED EXPECTATIONS (written before the run):
Q1: cell means reproduce run12 (deterministic seeds; any drift = torch
    version nondeterminism, must be < 0.005 in the effect means).
Q2: effect (antipodal - adjacent) positive 5/5 seeds at conv1-2 in both
    nets (exact one-sided sign test p = 1/32 per cell), 4/5 at conv3
    (run12's recorded fractions).
Q3: paired SE small relative to the mean at conv1-2 (the effect is
    seed-stable, +0.002..+0.012 per run12).

Out: ../data_canonical/run41_cnn_ordering_perseed.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run41_cnn_ordering_perseed.json"

spec = importlib.util.spec_from_file_location("run5",
                                              HERE / "run5_equivariant_cnn.py")
r5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r5)

ORDERS = {"adjacent": [0, 1, 2, 3], "antipodal": [0, 2, 1, 3]}
N_ROT, nte, ntr = r5.N_ROT, 4000, 20000
m = r5.load_mnist()


def ladder_delta_order(X, labels, order, rng):
    members = [np.where(labels == g)[0] for g in order]
    sizes, prs = [], []
    for c in range(1, N_ROT + 1):
        sel = np.concatenate(members[:c])
        sizes.append(len(sel))
        prs.append(r5.pr_c(X[sel]))
    th_o = r5.slope(sizes, np.asarray(prs))
    nl = np.zeros((10, len(sizes)))
    for d in range(10):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(r5.pr_c(X[rng.choice(len(X), s,
                                                       replace=False)]), 1e-9))
    return th_o - r5.slope(sizes, np.exp(nl.mean(axis=0)))


agg = {}
for seed in range(5):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(m["train_x"]), ntr, replace=False)
    Xtr = m["train_x"][tr_idx].astype(np.float32) / 255.0
    ytr = m["train_y"][tr_idx].astype(np.int64)
    rot_tr = rng.integers(0, N_ROT, ntr)
    Xtr = np.stack([np.rot90(x, g) for x, g in zip(Xtr, rot_tr)])
    te_idx = rng.choice(len(m["test_x"]), nte, replace=False)
    Xte = m["test_x"][te_idx].astype(np.float32) / 255.0
    rot_te = np.repeat(np.arange(N_ROT), nte // N_ROT)
    rng.shuffle(rot_te)
    Xte = np.stack([np.rot90(x, g) for x, g in zip(Xte, rot_te)])
    Xtr_t = torch.tensor(np.ascontiguousarray(Xtr)).unsqueeze(1)
    Xte_t = torch.tensor(np.ascontiguousarray(Xte)).unsqueeze(1)
    ytr_t = torch.tensor(ytr)
    for name, Net in [("plain", r5.PlainNet), ("equivariant", r5.EquivNet)]:
        model = Net().to(r5.DEV)
        r5.train(model, Xtr_t, ytr_t)
        model.eval()
        feats = {}
        with torch.no_grad():
            for i in range(0, nte, 512):
                o = model(Xte_t[i:i + 512].to(r5.DEV))
                if name == "plain":
                    for l, f in enumerate(o[1]):
                        feats.setdefault(f"conv{l+1}", []).append(f.cpu().numpy())
                else:
                    for l, f in enumerate(o[1]):
                        feats.setdefault(f"conv{l+1}_equi", []).append(f.cpu().numpy())
        for layer, chunks in feats.items():
            X = np.concatenate(chunks)
            X = X / (X.std() + 1e-9)
            for oname, order in ORDERS.items():
                d = ladder_delta_order(X, rot_te, order,
                                       np.random.default_rng(10 + seed))
                agg.setdefault((name, layer, oname), []).append(float(d))
        print(f"seed {seed} {name} done", flush=True)

out = {"per_seed": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in agg.items()}}
print("\npaired ordering effect (antipodal - adjacent), five seeds:", flush=True)
for name in ("plain", "equivariant"):
    for layer in sorted({k[1] for k in agg if k[0] == name}):
        a = np.array(agg[(name, layer, "antipodal")])
        j = np.array(agg[(name, layer, "adjacent")])
        eff = a - j
        n_pos = int((eff > 0).sum())
        # exact one-sided sign test under H0: P(effect>0)=1/2
        from math import comb
        p_sign = sum(comb(5, k) for k in range(n_pos, 6)) / 2 ** 5
        out[f"{name}/{layer}/paired"] = {
            "effects": [float(e) for e in eff],
            "mean": float(eff.mean()),
            "paired_se": float(eff.std(ddof=1) / np.sqrt(len(eff))),
            "n_pos": n_pos, "p_sign_one_sided": p_sign}
        print(f"  {name}/{layer}: effect={eff.mean():+.4f} ± "
              f"{eff.std(ddof=1)/np.sqrt(5):.4f} (paired SE) | {n_pos}/5 pos, "
              f"sign p={p_sign:.4f} | per-seed "
              f"{np.array2string(eff, precision=4)}", flush=True)

json.dump(out, open(OUT, "w"), indent=1)
print("DONE run41", flush=True)
