"""Referee control (fresh-Claude referee, Oct 20, concern 3): the direction-selective thirds of Sec 4 select and measure
on the same trials. Cross-fit: each class's trials split into alternating halves A and B; the tuning test (F, p < 0.01)
and the direction-selectivity index (vector strength of the eight class means, as run59) come from one half, the
eight-class direction ladder (run2.ladder_delta, ten-draw floor, seed 42) runs on the other half; both directions
(A -> B, B -> A). Halving the trials lowers every shift (App B), so the claim is tested in its own form: the most
direction-selective third shifts more than the least selective third on held-out trials.

PREDICTION (written before the run): DS > nonDS on held-out trials in 8 of 8 recordings in both directions (the
in-sample result of run59 is 8 of 8 with gaps of 0.13-0.28).
Out: data_canonical/cr_ds_thirds_crossfit.json"""
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the eight Stringer et al. 2019 grating recordings (figshare), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_ds_thirds_crossfit.json"
spec = importlib.util.spec_from_file_location("run2", HERE / "run2_calibrated_corotating.py")
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
NB = 8


def select(Xt, bl):
    n = len(bl); counts = np.bincount(bl, minlength=NB)
    M = np.stack([Xt[bl == k].mean(0) for k in range(NB)]); grand = Xt.mean(0)
    ssb = sum(counts[k] * (M[k] - grand) ** 2 for k in range(NB))
    ssw = sum(((Xt[bl == k] - M[k]) ** 2).sum(0) for k in range(NB))
    tuned = (1 - stats.f.cdf((ssb / (NB - 1)) / (ssw / (n - NB) + 1e-12), NB - 1, n - NB)) < 0.01
    R = M - M.min(0, keepdims=True) + 1e-9
    dsi = np.abs((R * np.exp(1j * 2 * np.pi * np.arange(NB)[:, None] / NB)).sum(0) / R.sum(0))
    tidx = np.where(tuned)[0]; nsub = len(tidx) // 3
    order = tidx[np.argsort(dsi[tidx])]
    return order[-nsub:], order[:nsub]


def main():
    rows = []
    for fp in sorted(RAW.glob("gratings*.npy")):
        t0 = time.time()
        dat = np.load(fp, allow_pickle=True).item()
        Xt = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); Xt /= Xt.std() + 1e-9
        phi = np.asarray(dat["istim"], float).ravel() % (2 * np.pi); del dat
        bl = np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, NB + 1)) - 1, 0, NB - 1)
        half = np.zeros(len(bl), int)
        for k in range(NB):
            idx = np.where(bl == k)[0]; half[idx[1::2]] = 1               # alternate trials within each class
        row = {"name": fp.stem}
        for sel_h, meas_h in ((0, 1), (1, 0)):
            ds, nds = select(Xt[half == sel_h], bl[half == sel_h])
            Xm, bm = Xt[half == meas_h], bl[half == meas_h]
            d_ds, _ = r2.ladder_delta(np.ascontiguousarray(Xm[:, ds]), bm, np.random.default_rng(42), n_shuf=0)
            d_nds, _ = r2.ladder_delta(np.ascontiguousarray(Xm[:, nds]), bm, np.random.default_rng(42), n_shuf=0)
            row[f"sel{sel_h}_meas{meas_h}"] = {"DS": float(d_ds), "nonDS": float(d_nds), "n_sub": int(len(ds))}
        print(f"{fp.stem[:34]:34s} " + "  ".join(f"{k}: DS {v['DS']:+.3f} nonDS {v['nonDS']:+.3f}"
              for k, v in row.items() if k != "name") + f"  ({time.time() - t0:.0f}s)", flush=True)
        rows.append(row)
    v = {k: int(sum(r[k]["DS"] > r[k]["nonDS"] for r in rows)) for k in ("sel0_meas1", "sel1_meas0")}
    print("VERDICT DS > nonDS on held-out trials:", v, flush=True)
    json.dump({"rows": rows, "verdict": v}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
