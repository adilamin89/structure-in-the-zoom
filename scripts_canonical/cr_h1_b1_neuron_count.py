"""H1-R21 (B1, Oct 22): neuron count, the control for Allen's positive class-mean part (H1-R6 / P4).

On the eight grating recordings (global SD scaling, the paper's input), random neuron subsets of 64, 128, 256, 512, 1024,
4096 and all neurons (ten draws each; all = one): the direction ladder of Table 9 (cr_order_and_withinbin_controls.ladder:
Gram trace, exact W + B split, ten-draw floor, rng 42, rungs 1,2,3,4,6,8 of eight 45-degree classes), and at the full stimulus
set PR(W), PR(B), Tr B/Tr W, the class-mean term with and without the overlap x, and the sign law (cr_c8_sign_condition):
class-mean term negative iff PR(W)/PR(B) > 1 + 2 Tr W/Tr B. Also in the within-class-SD frame (each neuron scaled by its pooled
within-class SD), the frame in which the Allen smoke test's PR(W) is no longer set by a few high-variance units.
Question: does the class-mean part turn positive near Allen's unit counts (51-421), and does the law predict the crossover N?
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler cr_h1_b1_neuron_count.py [n_workers]
Out: data_canonical/cr_h1_b1_neuron_count.json"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the eight Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_h1_b1_neuron_count.json"
SIZES = (64, 128, 256, 512, 1024, 4096)
DRAWS = 10


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def sign_law(X, lab):
    X = X - X.mean(0)
    cm = np.stack([X[lab == c].mean(0) for c in range(8)]); p = np.bincount(lab, minlength=8) / len(lab)
    n = len(X); Wh = (X - cm[lab]) / np.sqrt(n); Bh = np.sqrt(p)[:, None] * cm
    w1, b1 = (Wh * Wh).sum(), (Bh * Bh).sum()
    w2 = ((Wh @ Wh.T) ** 2).sum(); b2 = ((Bh @ Bh.T) ** 2).sum(); x = ((Wh @ Bh.T) ** 2).sum()
    prW, prB = w1 ** 2 / w2, b1 ** 2 / b2
    return {"trB_trW": float(b1 / w1), "PR_W": float(prW), "PR_B": float(prB), "ratio": float(prW / prB),
            "threshold": float(1 + 2 * w1 / b1), "c_full": float(np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2))),
            "c_without_x": float(np.log((1 + b1 / w1) ** 2 / (1 + b2 / w2)))}


def within_sd(X, lab):
    return np.sqrt(np.mean([((X[lab == c] - X[lab == c].mean(0)) ** 2).mean(0) for c in range(8)], 0))


def one(X, lab, sets, ow):
    r = {"raw": {**ow.ladder((X @ X.T).astype(np.float64), lab, sets, np.random.default_rng(42)), **sign_law(X, lab)}}
    sd = within_sd(X, lab); keep = sd > 1e-6 * np.median(sd[sd > 0]); Z = X[:, keep] / sd[keep]
    r["z"] = {**ow.ladder((Z @ Z.T).astype(np.float64), lab, sets, np.random.default_rng(42)), **sign_law(Z, lab)}
    return r


def run_one(fp):
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab = ow.bins(phi, 8)
    m8 = [np.where(lab == b)[0] for b in range(8)]
    sets = [np.concatenate(m8[:c]) for c in (1, 2, 3, 4, 6, 8)]
    n_neur = X.shape[1]; rows = []
    for N in [s for s in SIZES if s < n_neur] + [n_neur]:
        for d in range(DRAWS if N < n_neur else 1):
            idx = np.sort(np.random.default_rng(1000 * N + d).choice(n_neur, N, replace=False)) if N < n_neur else np.arange(n_neur)
            r = one(np.ascontiguousarray(X[:, idx], dtype=np.float64), lab, sets, ow)
            rows.append({"N": int(N), "draw": d, **r})
        cur = [r for r in rows if r["N"] == N]
        f = lambda fr, k: np.mean([r[fr][k] for r in cur])
        print(f"{fp.stem[9:30]:22s} N {N:6d}: delta {f('raw', 'delta'):+.3f} within {f('raw', 'within'):+.3f} class-means "
              f"{f('raw', 'class_means'):+.3f} | PR_W {f('raw', 'PR_W'):6.1f} PR_B {f('raw', 'PR_B'):4.2f} ratio "
              f"{f('raw', 'ratio'):5.1f} thr {f('raw', 'threshold'):5.1f} | z: cm {f('z', 'class_means'):+.3f} ratio "
              f"{f('z', 'ratio'):5.1f} thr {f('z', 'threshold'):5.1f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"{fp.stem} DONE", flush=True)
    return {"name": fp.stem, "n_neurons": int(n_neur), "n_presentations": int(len(lab)), "rows": rows}


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    files = sorted(RAW.glob("gratings*.npy"))
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        recs = list(ex.map(run_one, files))
    json.dump({"sizes": SIZES, "draws": DRAWS, "recordings": recs}, open(OUT, "w"))
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
