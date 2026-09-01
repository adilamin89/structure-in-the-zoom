"""Run 5 - Ground-truth symmetry test: C4-equivariant vs plain CNN on rotated
MNIST, declared-axis ladders per layer.

The architecture fixes the ground truth: the equivariant network's
group-pooled features are invariant to C4 rotations BY CONSTRUCTION, so the
rotation-axis ladder has a known right answer layer by layer. A
parameter-matched plain CNN is the contrast. This replaces the lexically
confounded Pythia appendix as the artificial-network demonstration.

Design: every image receives a random rotation from C4 = {0, 90, 180, 270}
(exact rot90, no interpolation). Both nets train on digit classification.
Equivariance via orbit weight-sharing: the same conv stack is applied to all
four rotations of the input; stacking the four copies gives an exactly
C4-equivariant representation, mean over the orbit gives the invariant layer.

REGISTERED PREDICTIONS (before run):
C1: plain CNN: delta_rot > 0 at every layer (rotation entangled in features).
C2: equivariant net, pre-pool (equivariant) features: delta_rot > 0
    (orbit structure explicit).
C3: equivariant net, group-pooled (invariant) features: delta_rot ~ 0
    while delta_digit stays > 0 on the same features - the architecture
    switches one axis off and leaves the other on (double dissociation).
C4: shuffled-label controls ~ 0 everywhere.

Out: feedback_runs/run5_equivariant_cnn.json
"""
import json
import gzip
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
MNIST_DIR = HERE / "mnist_data"
MNIST_DIR.mkdir(exist_ok=True)
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {"train_x": "train-images-idx3-ubyte.gz",
         "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz",
         "test_y": "t10k-labels-idx1-ubyte.gz"}

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_ROT = 4
ROT_RUNGS = [1, 2, 3, 4]
DIGIT_RUNGS = [1, 2, 3, 4, 6, 8]
N_NULL = 10
N_SHUF = 5


def load_mnist():
    arrs = {}
    for key, fname in FILES.items():
        p = MNIST_DIR / fname
        if not p.exists():
            print(f"downloading {fname}", flush=True)
            urllib.request.urlretrieve(BASE + fname, p)
        with gzip.open(p, "rb") as f:
            data = f.read()
        if "x" in key:
            arrs[key] = np.frombuffer(data, np.uint8, offset=16).reshape(-1, 28, 28)
        else:
            arrs[key] = np.frombuffer(data, np.uint8, offset=8)
    return arrs


class ConvStack(nn.Module):
    """Shared conv trunk; returns per-layer spatially pooled features."""

    def __init__(self, c=32):
        super().__init__()
        self.c1 = nn.Conv2d(1, c, 5, padding=2)
        self.c2 = nn.Conv2d(c, 2 * c, 3, padding=1)
        self.c3 = nn.Conv2d(2 * c, 2 * c, 3, padding=1)

    def forward(self, x):
        f1 = F.relu(self.c1(x))
        f2 = F.relu(self.c2(F.max_pool2d(f1, 2)))
        f3 = F.relu(self.c3(F.max_pool2d(f2, 2)))
        feats = [f.mean(dim=(2, 3)) for f in (f1, f2, f3)]
        return feats, f3


class PlainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = ConvStack()
        self.head = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        feats, f3 = self.trunk(x)
        return self.head(f3.flatten(1)), feats


class EquivNet(nn.Module):
    """Orbit weight-sharing: trunk applied to all 4 rotations of the input."""

    def __init__(self):
        super().__init__()
        self.trunk = ConvStack()
        self.head = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        per_rot = []          # equivariant stack of per-orientation features
        f3s = []
        for g in range(N_ROT):
            feats, f3 = self.trunk(torch.rot90(x, g, dims=(2, 3)))
            per_rot.append(feats)
            f3s.append(f3.flatten(1))
        equi = [torch.cat([per_rot[g][l] for g in range(N_ROT)], dim=1)
                for l in range(3)]                       # equivariant per layer
        inv = [torch.stack([per_rot[g][l] for g in range(N_ROT)]).mean(0)
               for l in range(3)]                        # invariant per layer
        pooled_f3 = torch.stack(f3s).mean(0)             # invariant head input
        return self.head(pooled_f3), equi, inv


def pr_c(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if tr2 > 0 else 1.0


def slope(sizes, prs):
    x, y = np.log(np.asarray(sizes, float)), np.log(np.maximum(prs, 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def ladder_delta(X, labels, rungs, n_classes, rng, n_shuf=N_SHUF):
    members = [np.where(labels == b)[0] for b in range(n_classes)]
    sizes, prs = [], []
    for c in rungs:
        sel = np.concatenate(members[:c])
        sizes.append(len(sel))
        prs.append(pr_c(X[sel]))
    th_o = slope(sizes, np.asarray(prs))
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(pr_c(X[rng.choice(len(X), s, replace=False)]), 1e-9))
    th_f = slope(sizes, np.exp(nl.mean(axis=0)))
    shufs = []
    for s in range(n_shuf):
        srng = np.random.default_rng(700 + s)
        perm = labels[srng.permutation(len(labels))]
        m2 = [np.where(perm == b)[0] for b in range(n_classes)]
        sz2 = [len(np.concatenate(m2[:c])) for c in rungs]
        pr2 = [pr_c(X[np.concatenate(m2[:c])]) for c in rungs]
        shufs.append(slope(sz2, np.asarray(pr2)) - th_f)
    return th_o - th_f, float(np.mean(shufs))


def train(model, Xtr, ytr, epochs=2, bs=256):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        tot, correct = 0, 0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx].to(DEV), ytr[idx].to(DEV)
            out = model(xb)
            logits = out[0]
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += len(idx); correct += (logits.argmax(1) == yb).sum().item()
        print(f"  epoch {ep}: train acc {correct / tot:.3f}", flush=True)


def main():
    torch.manual_seed(0)
    m = load_mnist()
    rng = np.random.default_rng(0)
    # rotate every image by a random C4 element (exact rot90)
    ntr = 20000
    tr_idx = rng.choice(len(m["train_x"]), ntr, replace=False)
    Xtr = m["train_x"][tr_idx].astype(np.float32) / 255.0
    ytr = m["train_y"][tr_idx].astype(np.int64)
    rot_tr = rng.integers(0, N_ROT, ntr)
    Xtr = np.stack([np.rot90(x, g) for x, g in zip(Xtr, rot_tr)])
    # test: 1000 per rotation class
    nte = 4000
    te_idx = rng.choice(len(m["test_x"]), nte, replace=False)
    Xte = m["test_x"][te_idx].astype(np.float32) / 255.0
    yte = m["test_y"][te_idx].astype(np.int64)
    rot_te = np.repeat(np.arange(N_ROT), nte // N_ROT)
    rng.shuffle(rot_te)
    Xte = np.stack([np.rot90(x, g) for x, g in zip(Xte, rot_te)])
    Xtr_t = torch.tensor(np.ascontiguousarray(Xtr)).unsqueeze(1)
    Xte_t = torch.tensor(np.ascontiguousarray(Xte)).unsqueeze(1)
    ytr_t = torch.tensor(ytr)

    out = {}
    for name, Net in [("plain", PlainNet), ("equivariant", EquivNet)]:
        print(f"[{name}] training on {DEV}", flush=True)
        model = Net().to(DEV)
        train(model, Xtr_t, ytr_t)
        model.eval()
        feats = {}
        with torch.no_grad():
            for i in range(0, nte, 512):
                xb = Xte_t[i:i + 512].to(DEV)
                o = model(xb)
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
            X = X / (X.std() + 1e-9)
            d_rot, sh_rot = ladder_delta(X, rot_te, ROT_RUNGS, N_ROT,
                                         np.random.default_rng(1))
            dig_lab = yte.copy()
            keep = dig_lab < 8
            d_dig, sh_dig = ladder_delta(X[keep], dig_lab[keep], DIGIT_RUNGS, 8,
                                         np.random.default_rng(2))
            out[name][layer] = {"delta_rot": d_rot, "shuf_rot": sh_rot,
                                "delta_digit": d_dig, "shuf_digit": sh_dig}
            print(f"  {layer}: delta_rot={d_rot:+.3f} (shuf {sh_rot:+.3f}) | "
                  f"delta_digit={d_dig:+.3f} (shuf {sh_dig:+.3f})", flush=True)

    json.dump(out, open(HERE / "run5_equivariant_cnn.json", "w"), indent=1)
    print("DONE run5", flush=True)


if __name__ == "__main__":
    main()
