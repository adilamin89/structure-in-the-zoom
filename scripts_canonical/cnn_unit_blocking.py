"""CNN analogue of the coarse-graining flow (S74, 2026-09-02): block-average conv
units of the C4 rotated-MNIST networks (run5 design, one seed each) and follow
the C4 harmonic content of the rotation-class kernel.

Kernel: correlation of rotation-class mean feature vectors (4 classes) per
layer; circular profile C(dg), dg in {0, 90, 180, 270}; harmonics a1 (frequency
1, the direction-like sector) and a2 (frequency 2, the orientation-like sector,
Nyquist for C4). The paper reports a2 carrying 97-99% of the tuned power.
Blocking: average K units per block, K = 1..64, blocks (i) random, (ii) sorted
by each unit's frequency-1 phase (direction-like columns), (iii) sorted by
frequency-2 phase (orientation-like columns). Group (orbit) pooling is the
K -> orbit limit where every nontrivial harmonic cancels (the invariant layer).

EXPECTATION (from the V1 result, registered here): random blocks preserve the
a2 share; frequency-1-sorted blocks raise the a1 share (the direction-like
sector), frequency-2-sorted blocks push a2 toward 1. Training: 2 epochs on
20k rotated MNIST, seed 0, plain and equivariant nets (run5 code).

Out: ../data_canonical/cnn_unit_blocking.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "cnn_unit_blocking.json"
cand = [HERE / "run5_equivariant_cnn.py", HERE.parent / "feedback_runs" / "run5_equivariant_cnn.py"]
src = next(c for c in cand if c.exists())
spec = importlib.util.spec_from_file_location("run5", src)
r5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
N_ROT = r5.N_ROT
KS = [1, 2, 4, 8, 16, 32, 64]


def c4_harmonics(M):
    """M: 4 classes x units (raw class means). Correlation kernel, circular profile, a1/a2 shares."""
    Cm = np.corrcoef(M)
    prof = np.array([np.mean([Cm[i, (i + k) % N_ROT] for i in range(N_ROT)]) for k in range(N_ROT)])
    dg = np.arange(N_ROT) * np.pi / 2
    a1 = 2 / N_ROT * np.sum(prof * np.cos(dg)); a2 = 2 / N_ROT * np.sum(prof * np.cos(2 * dg))
    tot = a1 ** 2 + a2 ** 2 + 1e-12
    return float(a1), float(a2), float(a2 ** 2 / tot)


def blocking(M):
    Mc = M - M.mean(0); dg = np.arange(N_ROT) * np.pi / 2
    ph1 = np.angle((Mc * np.exp(1j * dg[:, None])).sum(0)) % (2 * np.pi)
    ph2 = np.angle((Mc * np.exp(2j * dg[:, None])).sum(0)) % (2 * np.pi)
    rng = np.random.default_rng(0)
    out = {"K": KS, "random": [], "freq1_sorted": [], "freq2_sorted": []}
    for K in KS:
        nblk = M.shape[1] // K
        if nblk < 8:
            for key in ("random", "freq1_sorted", "freq2_sorted"):
                out[key].append(None)
            continue
        for key, order in (("random", rng.permutation(M.shape[1])), ("freq1_sorted", np.argsort(ph1)), ("freq2_sorted", np.argsort(ph2))):
            Mb = np.stack([M[:, order[i * K:(i + 1) * K]].mean(1) for i in range(nblk)], 1)
            a1, a2, share2 = c4_harmonics(Mb)
            out[key].append({"a1": a1, "a2": a2, "a2_share": share2})
    return out


def main():
    m = r5.load_mnist()
    nte, ntr = 4000, 20000
    torch.manual_seed(0); rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(m["train_x"]), ntr, replace=False)
    Xtr = m["train_x"][tr_idx].astype(np.float32) / 255.0; ytr = m["train_y"][tr_idx].astype(np.int64)
    rot_tr = rng.integers(0, N_ROT, ntr)
    Xtr = np.stack([np.rot90(x, g) for x, g in zip(Xtr, rot_tr)])
    Xte = m["test_x"][:nte].astype(np.float32) / 255.0
    rot_te = np.repeat(np.arange(N_ROT), nte // N_ROT)
    Xte = np.stack([np.rot90(x, g) for x, g in zip(Xte, rot_te)])
    Xtr_t = torch.tensor(Xtr)[:, None]; ytr_t = torch.tensor(ytr); Xte_t = torch.tensor(Xte)[:, None]
    out = {}
    for name, Net in [("plain", r5.PlainNet), ("equivariant", r5.EquivNet)]:
        print(f"[{name}] training", flush=True)
        model = Net().to(r5.DEV); r5.train(model, Xtr_t, ytr_t); model.eval()
        feats = {}
        with torch.no_grad():
            for i in range(0, nte, 512):
                o = model(Xte_t[i:i + 512].to(r5.DEV))
                for l, f in enumerate(o[1]):
                    feats.setdefault(f"conv{l+1}" + ("" if name == "plain" else "_equi"), []).append(f.cpu().numpy())
        out[name] = {}
        for layer, chunks in feats.items():
            X = np.concatenate(chunks).reshape(nte, -1)
            M = np.stack([X[rot_te == g].mean(0) for g in range(N_ROT)])
            a1, a2, share2 = c4_harmonics(M)
            fl = blocking(M)
            out[name][layer] = {"n_units": int(X.shape[1]), "a1": a1, "a2": a2, "a2_share": share2, "blocking": fl}
            last = lambda key: next((v for v in reversed(fl[key]) if v is not None), None)
            print(f"  {layer:12s} units {X.shape[1]:6d} a2 share {share2:.3f} | K=max: random {last('random')['a2_share']:.3f} "
                  f"freq1-sorted {last('freq1_sorted')['a2_share']:.3f} freq2-sorted {last('freq2_sorted')['a2_share']:.3f}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
