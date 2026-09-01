"""Run 12 - PROSPECTIVE ordering prediction on the C4 network (round-5 gift).

run5c measured the network's rotation-class correlation: frequency-2 dominant,
C(180) ~ +0.94-0.99, C(90) strongly negative. Entry coherence therefore makes
a prospective prediction with a KNOWN answer-generating mechanism: an
antipodal-paired accumulation order (0, 180, 90, 270) has a highly coherent
second rung ({0,180}) and should climb FASTER than adjacent order
(0, 90, 180, 270) whose second rung ({0,90}) is anti-correlated. This is the
REVERSE of V1, where C(22.5) > C(180) made sequential adjacent the steeper
order.

REGISTERED PREDICTION (before run): delta_antipodal > delta_adjacent on the
rotation-carrying layers (plain conv2/conv3 and equivariant pre-pool
conv2/conv3), in the majority of seeds, five seeds.

Out: feedback_runs/run12_cnn_ordering.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run5", HERE / "run5_equivariant_cnn.py")
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
                agg.setdefault((name, layer, oname), []).append(d)
        print(f"seed {seed} {name} done", flush=True)

out = {}
for (name, layer, oname), vals in agg.items():
    out[f"{name}/{layer}/{oname}"] = {"mean": float(np.mean(vals)),
                                      "sd": float(np.std(vals))}
print("\nordering effect (antipodal - adjacent), five seeds:", flush=True)
for name in ("plain", "equivariant"):
    for layer in sorted({k[1] for k in agg if k[0] == name}):
        a = np.array(agg[(name, layer, "antipodal")])
        j = np.array(agg[(name, layer, "adjacent")])
        eff = a - j
        out[f"{name}/{layer}/effect"] = {"mean": float(eff.mean()),
                                         "sd": float(eff.std()),
                                         "frac_pos": float((eff > 0).mean())}
        print(f"  {name}/{layer}: anti={a.mean():+.3f} adj={j.mean():+.3f} "
              f"effect={eff.mean():+.3f}±{eff.std():.3f} "
              f"({(eff>0).mean()*100:.0f}% pos)", flush=True)

json.dump(out, open(HERE / "run12_cnn_ordering.json", "w"), indent=1)
print("DONE run12", flush=True)
