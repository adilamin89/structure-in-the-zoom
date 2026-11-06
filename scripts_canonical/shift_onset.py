"""The shift onset on all eight grating recordings against each one's own fluctuation dimension.
cr_finite_size_split.py ran GT3 only (Sec 6: "the shift appears once each class holds more trials than its own fluctuation
dimension, about 26"). Same engine (split_ladder: centred PR, rungs 1,2,3,4,6,8 of 8 bins, ten-draw floor, exact within /
class-mean split; same budgets and draws, seeds 20000 n + k) on each recording, plus d_w per class from Eq. 2 inverted at the
full set. Onset m* = presentations per class at which the draw-mean within-class part crosses zero for good (log-linear
interpolation between the last negative and the first positive budget after which it stays positive).
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler shift_onset.py [n_workers]   Out: data_canonical/shift_onset.json"""
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
OUT = HERE.parent / "data_canonical" / "shift_onset.json"


def _fs():
    spec = importlib.util.spec_from_file_location("fs", HERE / "cr_finite_size_split.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def onset(ms, w):
    ms, w = np.asarray(ms, float), np.asarray(w, float)
    neg = np.where(w <= 0)[0]
    if len(neg) == 0:
        return float(ms[0]), "positive_from_first"
    i = neg[-1]
    if i == len(w) - 1:
        return None, "never_positive"
    x0, x1 = np.log(ms[i]), np.log(ms[i + 1])
    return float(np.exp(x0 + (0 - w[i]) * (x1 - x0) / (w[i + 1] - w[i]))), "crossing"


def run_one(fp):
    fs = _fs(); t0 = time.time()
    d = np.load(fp, allow_pickle=True).item()
    X = np.asarray(d["sresp"], dtype=np.float32).T; X /= X.std() + 1e-9
    bins = np.clip(np.digitize(np.asarray(d["istim"], float).ravel() % (2 * np.pi), np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
    del d
    full = [fs.pr_centred(X[bins == b]) for b in range(8)]
    m_full = np.array([(bins == b).sum() for b in range(8)])
    d_w = [p * m / (m - 1 - p) for p, m in zip(full, m_full)]
    rows = []
    for n, draws in fs.BUDGETS.items():
        reps = []
        for k in range(draws if n < len(X) else 1):
            rng = np.random.default_rng(20_000 * n + k)
            idx = np.arange(len(X)) if n >= len(X) else np.sort(rng.choice(len(X), n, replace=False))
            r = fs.split_ladder(X[idx], bins[idx], rng)
            if r is not None:
                reps.append(r)
        agg = {k: float(np.mean([r[k] for r in reps])) for k in ("delta", "within", "class_means", "per_class")}
        agg["delta_sd"] = float(np.std([r["delta"] for r in reps], ddof=1)) if len(reps) > 1 else None
        rows.append(dict(n=min(n, len(X)), draws=len(reps), **agg))
        print(f"{fp.stem[9:30]:22s} n {min(n, len(X)):5d} m {agg['per_class']:6.1f} delta {agg['delta']:+.3f} within {agg['within']:+.3f} ({time.time() - t0:.0f}s)", flush=True)
        if n >= len(X):
            break
    ms = [r["per_class"] for r in rows]; m_star, how = onset(ms, [r["within"] for r in rows])
    out = {"name": fp.stem, "d_w": d_w, "d_w_mean": float(np.mean(d_w)), "single_class_pr_full": full, "rows": rows,
           "onset_m": m_star, "onset_how": how}
    print(f"{fp.stem} onset m* {m_star} ({how}) d_w mean {np.mean(d_w):.1f} DONE", flush=True)
    return out


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        recs = list(ex.map(run_one, sorted(RAW.glob("gratings*.npy"))))
    ok = [r for r in recs if r["onset_m"]]
    v = {"within_factor2": int(sum(0.5 <= r["onset_m"] / r["d_w_mean"] <= 2 for r in ok)), "n_with_onset": len(ok),
         "ratios": {r["name"]: (r["onset_m"] / r["d_w_mean"] if r["onset_m"] else None) for r in recs}}
    print("VERDICT", v, flush=True)
    json.dump({"recordings": recs, "verdict": v}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
