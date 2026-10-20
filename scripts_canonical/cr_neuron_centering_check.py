"""Referee control (ChatGPT referee, Oct 20, comment 7): the neuron ladders centre each presentation across the sampled
units, the stimulus ladders each unit across the sampled presentations, so the opening's three exponents differ in centering
as well as in axis. Sensitivity: the spatial neuron ladder of App D (spatial_neuron_ladder.py: kNN subsets around random
anchors vs random same-size subsets, sizes 50..5000) run with BOTH centerings on the same subsets (10 draws per size here,
50 in App D), on the ten recordings.

PREDICTION (written before the run): theta_obs changes with centering (C4 note: GT3 0.35 -> ~0.40 with unit centering), and
delta_spatial keeps its sign pattern (negative in most recordings, mean within 0.03 of App D's -0.033) because both arms
change alike.
Out: data_canonical/cr_neuron_centering_check.json"""
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the Stringer et al. 2019 grating recordings (figshare), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_neuron_centering_check.json"
SIZES, N_DRAWS = [50, 100, 200, 500, 1000, 2000, 5000], 10


def pr(X, axis):
    Xc = X - X.mean(axis=axis, keepdims=True)          # axis 0: each presentation across units; axis 1: each unit across presentations
    G = Xc @ Xc.T if Xc.shape[0] <= Xc.shape[1] else Xc.T @ Xc
    return float(np.trace(G) ** 2 / (G * G).sum())


def slope(sizes, prs):
    return float(np.polyfit(np.log(sizes), np.log(prs), 1)[0])


def run_one(fp):
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.asarray(dat["sresp"], dtype=np.float32)                          # neurons x presentations
    pos = np.array([s["med"] for s in dat["stat"]], dtype=float); del dat
    n = X.shape[0]; sizes = [s for s in SIZES if s <= min(n, 5000)]
    rng = np.random.default_rng(42); rng_f = np.random.default_rng(1000)
    sp, fl = [], []
    for ns in sizes:
        sp.append([np.argsort(((pos - pos[rng.integers(n)]) ** 2).sum(1))[:ns] for _ in range(N_DRAWS)])
        fl.append([rng_f.choice(n, ns, replace=False) for _ in range(N_DRAWS)])
    row = {"name": fp.stem}
    for tag, axis in (("presentation_centred", 0), ("unit_centred", 1)):
        ts = slope(sizes, [np.mean([pr(X[i], axis) for i in subs]) for subs in sp])
        tf = slope(sizes, [np.mean([pr(X[i], axis) for i in subs]) for subs in fl])
        row[tag] = {"theta_spatial": ts, "theta_floor": tf, "delta": ts - tf}
    print(f"{fp.stem[:30]:30s} " + "  ".join(f"{k}: theta {v['theta_spatial']:.3f} delta {v['delta']:+.3f}"
          for k, v in row.items() if k != "name") + f" ({time.time() - t0:.0f}s)", flush=True)
    return row


def main():
    files = sorted(RAW.glob("*.npy"))
    with mp.get_context("fork").Pool(4) as pool:
        rows = pool.map(run_one, files)
    for k in ("presentation_centred", "unit_centred"):
        d = [r[k]["delta"] for r in rows]
        print(f"{k}: mean delta {np.mean(d):+.4f}, negative {sum(x < 0 for x in d)}/{len(d)}", flush=True)
    json.dump({"rows": rows, "n_draws": N_DRAWS}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
