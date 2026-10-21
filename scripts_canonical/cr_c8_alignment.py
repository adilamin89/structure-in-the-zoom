"""C8 mining: why does the class-mean part of the shift come out negative in cortex, when the calibrated model's shared
within-class modes give delta > 0 (Sec 6: +0.080 on GT3)? With every class given the same pooled fluctuations
(cr_c8_identification 'resid_pool') the within-class part is ~0 and delta equals the class-mean part, -0.07 to -0.41.

Hypothesis (gain physics, Lin 2015 / Goris 2014): cortex's class means lie along the directions its fluctuations already
occupy, so accumulating classes adds little new variance direction relative to the floor; the model draws its class means
independently of its modes. Test on the eight grating recordings, engine and inputs of cr_c8_identification:
  pool         tuning-free residuals pooled across classes, re-centred, plus each class's own mean (A2 input).
  pool_perm    the same pooled residuals plus class means whose NEURONS are permuted by one permutation shared by all
               classes (seeds 7, 8, 9): every class-mean correlation C(dphi), Tr B and B's spectrum are kept exactly; only
               the alignment between the class means and the fluctuation subspace is broken.
Also printed: the alignment A = Tr(W B) / (Tr W Tr B) at the full set, for pool and pool_perm, and its ratio.
PREDICTIONS (written before the run):
  P1 A(pool) exceeds A(pool_perm) in 8 of 8 (the means sit in the fluctuation subspace).
  P2 delta(pool_perm) > delta(pool) in 8 of 8, and delta(pool_perm) > 0 in >= 6 of 8 (unaligned means add directions,
     as in the model).
  If P2 fails while P1 holds, alignment is not what sets the sign; the concentration of B itself (few large class-mean
  directions against W) is, and the Sec 6 sentence stands as written.
Out: data_canonical/cr_c8_alignment.json"""
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "raw"   # Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_c8_alignment.json"
spec = importlib.util.spec_from_file_location("ow", HERE / "cr_order_and_withinbin_controls.py")
ow = importlib.util.module_from_spec(spec); spec.loader.exec_module(ow)
spec2 = importlib.util.spec_from_file_location("c8", HERE / "cr_c8_identification.py")
c8 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(c8)
SEEDS = (7, 8, 9)


def alignment(Y, lab):
    cls = np.unique(lab)
    cm = np.stack([Y[lab == c].mean(0) for c in cls]); w = np.array([np.mean(lab == c) for c in cls])
    mu = (w[:, None] * cm).sum(0)
    Bh = np.sqrt(w)[:, None] * (cm - mu)                                   # B = Bh^T Bh
    R = Y - cm[np.searchsorted(cls, lab)]; Wh = R / np.sqrt(len(Y))          # W = Wh^T Wh
    trW, trB = (Wh * Wh).sum(), (Bh * Bh).sum()
    trWB = ((Wh @ Bh.T) ** 2).sum()                                         # Tr(W B) = ||Wh Bh^T||_F^2
    return float(trWB / (trW * trB))


def run_one(fp):
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab = ow.bins(phi, 8)
    m8 = [np.where(lab == b)[0] for b in range(8)]
    sets = [np.concatenate(m8[:c]) for c in (1, 2, 3, 4, 6, 8)]
    lad = lambda Z: ow.ladder((Z @ Z.T).astype(np.float64), lab, sets, np.random.default_rng(42))
    Y = ow.remove_withinbin(X, lab, phi, 6, 8)
    row = {"name": fp.stem, "pool": [], "pool_perm": []}
    for s in SEEDS:
        P = c8.pool_residuals(Y, lab, np.random.default_rng(s))
        cm = {c: P[lab == c].mean(0) for c in np.unique(lab)}
        R = P - np.stack([cm[c] for c in lab])
        perm = np.random.default_rng(100 + s).permutation(P.shape[1])
        Q = R + np.stack([cm[c][perm] for c in lab])
        a, b = lad(P), lad(Q)
        a["A"] = alignment(P, lab); b["A"] = alignment(Q, lab)
        row["pool"].append(a); row["pool_perm"].append(b)
    m = lambda k, f: float(np.mean([r[f] for r in row[k]]))
    print(f"{fp.stem[:28]:28s} pool d {m('pool','delta'):+.3f} A {m('pool','A'):.4f} | perm d {m('pool_perm','delta'):+.3f} "
          f"A {m('pool_perm','A'):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return row


def main():
    with ProcessPoolExecutor(3, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(run_one, sorted(DATA.glob("gratings*.npy"))))
    m = lambda r, k, f: float(np.mean([q[f] for q in r[k]]))
    v = {"P1_A_pool_gt_perm": int(sum(m(r, "pool", "A") > m(r, "pool_perm", "A") for r in rows)),
         "P2_delta_perm_gt_pool": int(sum(m(r, "pool_perm", "delta") > m(r, "pool", "delta") for r in rows)),
         "P2_delta_perm_positive": int(sum(m(r, "pool_perm", "delta") > 0 for r in rows)),
         "A_ratio_pool_over_perm": [m(r, "pool", "A") / m(r, "pool_perm", "A") for r in rows],
         "delta_pool": [m(r, "pool", "delta") for r in rows], "delta_perm": [m(r, "pool_perm", "delta") for r in rows]}
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
