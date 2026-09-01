"""Run 5b - 5-seed robustness sweep of the equivariant-CNN double dissociation.

Sweeps torch + data seeds over the run5 design (C4 rotated MNIST, plain vs
orbit-shared equivariant net). Reports mean ± sd per layer for delta_rot and
delta_digit. REGISTERED EXPECTATION: the run5 single-seed pattern holds at
5 seeds - plain delta_rot grows with depth; equivariant pre-pool delta_rot
positive; invariant-layer delta_rot ~ 0 with delta_digit > 0 on conv3_inv.

Out: feedback_runs/run5b_cnn_seeds.json
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

m = r5.load_mnist()
N_ROT, nte, ntr = r5.N_ROT, 4000, 20000
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
    yte = m["test_y"][te_idx].astype(np.int64)
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
                    for l, f in enumerate(o[2]):
                        feats.setdefault(f"conv{l+1}_inv", []).append(f.cpu().numpy())
        for layer, chunks in feats.items():
            X = np.concatenate(chunks)
            X = X / (X.std() + 1e-9)
            d_rot, _ = r5.ladder_delta(X, rot_te, r5.ROT_RUNGS, N_ROT,
                                       np.random.default_rng(1), n_shuf=0)
            keep = yte < 8
            d_dig, _ = r5.ladder_delta(X[keep], yte[keep], r5.DIGIT_RUNGS, 8,
                                       np.random.default_rng(2), n_shuf=0)
            agg.setdefault((name, layer), []).append((d_rot, d_dig))
            print(f"seed{seed} {name}/{layer}: rot={d_rot:+.3f} dig={d_dig:+.3f}",
                  flush=True)

out = {}
for (name, layer), vals in agg.items():
    v = np.array(vals)
    out[f"{name}/{layer}"] = {
        "delta_rot_mean": float(v[:, 0].mean()), "delta_rot_sd": float(v[:, 0].std()),
        "delta_digit_mean": float(v[:, 1].mean()), "delta_digit_sd": float(v[:, 1].std()),
        "n_seeds": len(vals)}
    print(f"{name}/{layer}: rot={v[:,0].mean():+.3f}±{v[:,0].std():.3f} "
          f"dig={v[:,1].mean():+.3f}±{v[:,1].std():.3f}", flush=True)

json.dump(out, open(HERE / "run5b_cnn_seeds.json", "w"), indent=1)
print("DONE run5b", flush=True)
