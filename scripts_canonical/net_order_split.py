"""The equivariant network's accumulation order and the fluctuation overlaps that name it (Appendix J; Appendix E's odd/even ratio).

Same seeds, data and networks as run41_cnn_ordering_perseed.py (its helpers imported from run5_equivariant_cnn.py and
run63_deficit_decomposition.py). Per seed, network and layer, for the adjacent [0,1,2,3] and antipodal [0,2,1,3] orders of the four
rotation classes: the shift and its exact split into the within-class part log(w1^2/w2) and the class-mean part (Appendix K), each
against a ten-draw floor drawn from run41's random stream (default_rng(10 + seed) per order), so the shift replays run41's per-seed
values; and the split-half fluctuation-overlap kernel of the four classes by separation, with its harmonics and the ratio R of odd
to even harmonic power (Appendix E). Appendix J reads it: opposite rotations share more fluctuation shape than neighbours, R stays
below 3% in every layer that carries rotation structure, the antipodal order climbs faster, and at conv3 the within-class part
carries the effect.
Training on a GPU (MPS or CUDA) is not bit-reproducible: a rerun from a fresh clone moved seed 0's per-layer order effects by at most
0.00013 and left every sign and every ratio below 3%.
Run: python scripts_canonical/net_order_split.py   (MNIST is fetched into scripts_canonical/mnist_data/ by run5's loader)
Out: data_canonical/net_order_split.json"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SC = HERE
OUT = HERE.parent / "data_canonical" / "net_order_split.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


r5 = load("run5", SC / "run5_equivariant_cnn.py")
r63 = load("run63", SC / "run63_deficit_decomposition.py")
ORDERS = {"adjacent": [0, 1, 2, 3], "antipodal": [0, 2, 1, 3]}
N_ROT, nte, ntr = r5.N_ROT, 4000, 20000
m = r5.load_mnist()


def parts(Y, lab):
    b1, w1, b2, w2, x = r63.traces(Y, lab)
    return np.array([np.log((b1 + w1) ** 2 / (b2 + w2 + 2 * x)), np.log(w1 ** 2 / w2),
                     np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2))])


def ladder_split(X, labels, order, rng):
    members = [np.where(labels == g)[0] for g in order]
    sels = [np.concatenate(members[:c]) for c in range(1, N_ROT + 1)]
    sizes = np.log([len(s) for s in sels])
    lad = np.array([parts(X[s], labels[s]) for s in sels])
    flo = np.zeros_like(lad)
    nl = np.zeros((10, len(sels)))                       # run41's floor: mean log PR over 10 draws per rung
    for d in range(10):
        for k, s in enumerate(sels):
            i = rng.choice(len(X), len(s), replace=False)
            p = parts(X[i], labels[i])
            flo[k] += p / 10
            nl[d, k] = np.log(max(r5.pr_c(X[i]), 1e-9))
    sl = [float(np.polyfit(sizes, lad[:, j] - flo[:, j], 1)[0]) for j in range(3)]
    th_o = r5.slope([len(s) for s in sels], np.exp(lad[:, 0]))
    run41_delta = th_o - r5.slope([len(s) for s in sels], np.exp(nl.mean(axis=0)))
    return {"delta": sl[0], "within": sl[1], "class_means": sl[2], "run41_delta": float(run41_delta)}


def parity(X, labels, half):
    covs = []
    for h in (0, 1):
        R = []
        for c in range(N_ROT):
            Y = X[(labels == c) & (half == h)]
            R.append((Y - Y.mean(0)) / np.sqrt(len(Y)))
        covs.append(R)
    K = np.zeros((N_ROT, N_ROT))
    for c in range(N_ROT):
        for d in range(N_ROT):
            K[c, d] = 0.5 * (float(((covs[0][c] @ covs[1][d].T) ** 2).sum()) + float(((covs[1][c] @ covs[0][d].T) ** 2).sum()))
    dg = np.sqrt(np.outer(np.diag(K), np.diag(K))); Kn = K / dg
    a = np.arange(N_ROT); sep = (a[None, :] - a[:, None]) % N_ROT
    kappa = np.array([Kn[sep == s].mean() for s in range(N_ROT)])
    kl = np.array([np.mean(kappa * np.cos(2 * np.pi * l * np.arange(N_ROT) / N_ROT)) for l in range(N_ROT // 2 + 1)])
    return {"kappa": kappa.tolist(), "harmonics": kl.tolist(), "R": float(kl[1] ** 2 / kl[2] ** 2)}


rows = []
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
    half = np.zeros(nte, int)
    for c in range(N_ROT):
        idx = np.where(rot_te == c)[0]; half[idx[1::2]] = 1
    for name, Net in [("plain", r5.PlainNet), ("equivariant", r5.EquivNet)]:
        model = Net().to(r5.DEV)
        r5.train(model, Xtr_t, ytr_t)
        model.eval()
        feats = {}
        with torch.no_grad():
            for i in range(0, nte, 512):
                o = model(Xte_t[i:i + 512].to(r5.DEV))
                for l, f in enumerate(o[1]):
                    feats.setdefault(f"conv{l+1}" + ("" if name == "plain" else "_equi"), []).append(f.cpu().numpy())
        for layer, chunks in feats.items():
            X = np.concatenate(chunks).reshape(nte, -1).astype(np.float32)
            X = X / (X.std() + 1e-9)
            row = {"seed": seed, "net": name, "layer": layer, "parity": parity(X, rot_te, half)}
            for oname, order in ORDERS.items():
                row[oname] = ladder_split(X, rot_te, order, np.random.default_rng(10 + seed))
            rows.append(row)
            print(seed, name, layer, "R %.3f" % row["parity"]["R"],
                  "effect d %+.4f (run41-style %+.4f) within %+.4f cm %+.4f" % (
                      row["antipodal"]["delta"] - row["adjacent"]["delta"],
                      row["antipodal"]["run41_delta"] - row["adjacent"]["run41_delta"],
                      row["antipodal"]["within"] - row["adjacent"]["within"],
                      row["antipodal"]["class_means"] - row["adjacent"]["class_means"]), flush=True)
    json.dump({"rows": rows}, open(OUT, "w"), indent=1)
print("DONE net_order_split", flush=True)
