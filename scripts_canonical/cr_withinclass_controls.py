"""Referee control (fresh-Claude referee, Oct 20, concerns 1-2): is the within-class part of the direction shift
population fluctuation geometry (correlated modes), or single-neuron variance tracking each class's tuning?

The paper's direction ladder and exact split (run72.ladder: centred PR, 8 bins, rungs 1,2,3,4,6,8, ten-draw floor,
seed 42; within-class part = slope of log(w1^2/w2), class-mean part = the remainder) on the eight grating recordings,
under four inputs:
  orig    the paper's input (deconvolved responses / global SD); must reproduce Table 9's direction columns;
  shuf    each neuron's responses permuted across trials WITHIN each class, independently per neuron: every class
          mean and every neuron's per-class variance kept, every noise correlation destroyed (diagonal W_c);
  zscore  each neuron z-scored over all trials (correlation-matrix PR; concern 2);
  sqrt    square root of the responses, then global SD (variance-stabilizing; concern 1c).

COMPETING PREDICTIONS (written before the run):
  H_geom (the paper's Sec 7 reading): the shift rides on correlated within-class modes, so shuf lowers the within-class
    part and delta well below orig (delta_shuf < delta_orig / 2 in >= 6 of 8).
  H_diag (the referee's alternative): variance tracking the class mean suffices, so delta_shuf ~ delta_orig (within 25%
    in >= 6 of 8).
  zscore and sqrt: delta stays positive in 8 of 8 under either reading; their size is reported, not predicted.
Out: data_canonical/cr_withinclass_controls.json"""
import importlib.util
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the eight Stringer et al. 2019 grating recordings (figshare), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_withinclass_controls.json"
spec = importlib.util.spec_from_file_location("run72", HERE / "run72_paper_designs_split.py")
r72 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r72)


def shuffle_within_class(X, lab, rng):
    Y = X.copy()
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        perm = np.argsort(rng.random((len(idx), X.shape[1])), axis=0)       # independent permutation per neuron
        Y[idx] = np.take_along_axis(X[idx], perm, axis=0)
    return Y


def run_one(fp):
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    raw = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    lab = r72.bins(np.asarray(dat["istim"], float).ravel(), 8); del dat
    m8 = [np.where(lab == b)[0] for b in range(8)]
    sets = [np.concatenate(m8[:c]) for c in (1, 2, 3, 4, 6, 8)]
    base = raw / (raw.std() + 1e-9)
    sd = raw.std(0); keep = sd > 0
    inputs = {"orig": base,
              "shuf": shuffle_within_class(base, lab, np.random.default_rng(7)),
              "zscore": (raw[:, keep] - raw[:, keep].mean(0)) / sd[keep],
              "sqrt": np.sqrt(raw) / (np.sqrt(raw).std() + 1e-9)}
    row = {"name": fp.stem}
    for k, X in inputs.items():
        A = r72.ladder(np.ascontiguousarray(X, dtype=np.float32), lab, sets, np.random.default_rng(42))
        row[k] = {"delta": A["delta"], "within": A["slope_noise"], "class_means": A["slope_probe"]}
    print(f"{fp.stem[:34]:34s} " + "  ".join(f"{k} d {row[k]['delta']:+.3f} w {row[k]['within']:+.3f}" for k in inputs)
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    return row


def main():
    files = sorted(RAW.glob("gratings*.npy"))
    with mp.get_context("fork").Pool(4) as pool:
        rows = pool.map(run_one, files)
    d_o = [r["orig"]["delta"] for r in rows]; d_s = [r["shuf"]["delta"] for r in rows]
    verdict = {"H_geom_count": int(sum(s < o / 2 for o, s in zip(d_o, d_s))),
               "H_diag_count": int(sum(abs(s - o) <= 0.25 * o for o, s in zip(d_o, d_s))),
               "zscore_positive": int(sum(r["zscore"]["delta"] > 0 for r in rows)),
               "sqrt_positive": int(sum(r["sqrt"]["delta"] > 0 for r in rows))}
    print("VERDICT", json.dumps(verdict), flush=True)
    json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
