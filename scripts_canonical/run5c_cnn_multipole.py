"""Run 5c - Sector resolution on the C4 network: multipole fit on per-layer
feature correlations (referee M3/Q4 upgrade).

The v2 CNN result shows presence/absence of rotation structure only (plain and
pre-pool delta_rot are equal, and the invariant-layer zero is forced by
exchangeability). The sector-resolved test: fit the C4 character expansion
  C(dg) = a0 + a1 cos(dg) + a2 cos(2 dg),   dg in {0, 90, 180, 270} deg,
to the correlation between rotation-class mean feature vectors, per layer.
The regular representation carried by the pre-pool equivariant stack contains
all C4 frequencies; group pooling projects onto frequency 0.

REGISTERED PREDICTIONS (before run):
Q1: plain CNN: nonzero harmonic content (|a1|+|a2| > 0) growing with depth.
Q2: equivariant pre-pool stack: nonzero a1 and/or a2 (regular representation).
Q3: invariant (group-pooled) features: a1, a2 ~ 0 - a flat profile; the
    probe resolves the architecture's prescribed sector content, frequency 0
    only.

Out: feedback_runs/run5c_cnn_multipole.json
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
torch.manual_seed(0)
rng = np.random.default_rng(0)
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


def c4_fit(X, labels):
    """Correlation between rotation-class mean vectors -> C4 character fit."""
    means = np.stack([X[labels == g].mean(axis=0) for g in range(N_ROT)])
    means = means - means.mean(axis=0)          # remove the class-common mode
    norm = np.linalg.norm(means, axis=1, keepdims=True) + 1e-9
    Cm = (means / norm) @ (means / norm).T
    # average over separations dg = 0, 90, 180, 270 (cyclic)
    C_dg = np.array([np.mean([Cm[i, (i + k) % N_ROT] for i in range(N_ROT)])
                     for k in range(N_ROT)])
    dg = np.arange(N_ROT) * np.pi / 2
    A = np.column_stack([np.ones_like(dg), np.cos(dg), np.cos(2 * dg)])
    coef = np.linalg.lstsq(A, C_dg, rcond=None)[0]
    return C_dg, coef


out = {}
for name, Net in [("plain", r5.PlainNet), ("equivariant", r5.EquivNet)]:
    print(f"[{name}] training", flush=True)
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
    out[name] = {}
    for layer, chunks in feats.items():
        X = np.concatenate(chunks)
        C_dg, coef = c4_fit(X, rot_te)
        out[name][layer] = {"C_dg": C_dg.tolist(),
                            "a0": float(coef[0]), "a1": float(coef[1]),
                            "a2": float(coef[2])}
        print(f"  {layer}: C(0,90,180,270)="
              + "/".join(f"{c:+.3f}" for c in C_dg)
              + f" | a1={coef[1]:+.3f} a2={coef[2]:+.3f}", flush=True)

json.dump(out, open(HERE / "run5c_cnn_multipole.json", "w"), indent=1)
print("DONE run5c", flush=True)
