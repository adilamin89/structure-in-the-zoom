"""Run 27 - Axis search on the static-grating sessions (TX40/TX42).

WHY: the paper excludes static_biased_TX40/TX42 from the primary claim
because their stimulus ensemble spans only 43-47 degrees - the 8-bin
orientation probe is degenerate there (probe-range note, App C). This run
turns the exclusion into a demonstration of the instrument's DISCOVERY mode:
given a session where the declared axis fails, which axis does the
population organize along? Candidate axes computable from the recorded
variables:

  fine_orientation: 8 bins over the 4-degree span (~0.5 deg/class) -
      the declared axis at its native (degenerate) resolution.
  time_block:       8 equal blocks of trial order - the slow behavioral-state
      axis (Stringer et al. 2019: state fluctuations dominate V1 variance).
  random:           8-class random relabeling (exchangeability control).

NOTE (plan correction, 2026-08-31): these files are static GRATINGS, not
natural images - istim (orientation) is the only stimulus variable, so
spatial-frequency and semantic-category axes are not computable here. Those
need the natimg2800 sessions (separate download).

REGISTERED EXPECTATIONS (written before the run):
S1: fine_orientation delta ~ 0 - 0.5-degree class pitch is below the code's
    orientation correlation scale (C(dphi) ~ C(0) at this separation, so
    within-class and between-class structure coincide).
S2: time_block delta > 0 - slow state drift organizes population activity;
    the axis that works in these sessions is temporal state, not orientation.
S3: random ~ 0.

Construction matches the paper's V1 pipeline (modal_bootstrap_all10.py):
X = sresp.T (trials x neurons), accumulate classes [1,2,3,4,6,8], floor from
random subsets (10-draw mean, run17-style), 5 label shuffles as control.

Out: ../data_canonical/run27_static_axis_search.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Stringer static-grating sessions (public; stringer_v1 release).
DATA = Path(__file__).resolve().parent.parent / "raw" / "stringer_natimg"
OUT = HERE.parent / "data_canonical" / "run27_static_axis_search.json"

N_NULL = 10
N_SHUF = 5
BIN_COUNTS = [1, 2, 3, 4, 6, 8]

FILES = ["static_biased_TX40_2019_06_12_2.npy",
         "static_biased_TX42_2019_06_13_1.npy"]


def pr_c(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, float))
    y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def ladder_delta(X, labels, n_classes, rng, n_shuf=N_SHUF):
    bc = [c for c in BIN_COUNTS if c <= n_classes]
    members = [np.where(labels == c)[0] for c in range(n_classes)]
    if min(len(m) for m in members) < 3:
        return None, None
    sizes, prs = [], []
    for c in bc:
        sel = np.concatenate(members[:c])
        sizes.append(len(sel))
        prs.append(pr_c(X[sel]))
    if len(sizes) < 3:
        return None, None
    th_o = slope(sizes, np.asarray(prs))
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(pr_c(X[rng.choice(len(X), s, replace=False)]),
                                  1e-9))
    th_f = slope(sizes, np.exp(nl.mean(axis=0)))
    shufs = []
    for s in range(n_shuf):
        srng = np.random.default_rng(500 + s)
        perm = labels[srng.permutation(len(labels))]
        m2 = [np.where(perm == c)[0] for c in range(n_classes)]
        sz2, pr2 = [], []
        for c in bc:
            sel = np.concatenate(m2[:c])
            sz2.append(len(sel))
            pr2.append(pr_c(X[sel]))
        if len(sz2) >= 3:
            shufs.append(slope(sz2, np.asarray(pr2)) - th_f)
    return th_o - th_f, float(np.mean(shufs)) if shufs else 0.0


def main():
    out = {"design": "axis search on static-grating sessions",
           "note": "istim spans 43-47 deg; orientation probe degenerate by "
                   "design. Axes: fine_orientation, time_block, random.",
           "sessions": {}}

    for fname in FILES:
        name = fname.replace(".npy", "")
        print(f"\n=== {name}", flush=True)
        dat = np.load(DATA / fname, allow_pickle=True).item()
        X = dat["sresp"].T.astype(np.float32)  # trials x neurons
        istim = dat["istim"]
        n_trials = X.shape[0]
        print(f"  {X.shape[1]} neurons, {n_trials} trials, "
              f"istim {np.degrees(istim.min()):.1f}-"
              f"{np.degrees(istim.max()):.1f} deg", flush=True)

        # Axis labelings
        edges = np.linspace(istim.min(), istim.max() + 1e-9, 9)
        lab_ori = np.clip(np.digitize(istim, edges) - 1, 0, 7)
        # trials are stored in stimulus-presentation order via stimtimes
        order = np.argsort(dat["stimtimes"])
        lab_time = np.zeros(n_trials, dtype=int)
        for b, chunk in enumerate(np.array_split(order, 8)):
            lab_time[chunk] = b
        rng = np.random.default_rng(999)
        lab_rand = rng.integers(0, 8, n_trials)

        sess = {"n_neurons": int(X.shape[1]), "n_trials": int(n_trials),
                "axes": {}}
        for axis_name, labels in [("fine_orientation", lab_ori),
                                  ("time_block", lab_time),
                                  ("random", lab_rand)]:
            counts = np.bincount(labels, minlength=8)
            d, sh = ladder_delta(X, labels, 8, np.random.default_rng(11))
            sess["axes"][axis_name] = {
                "delta": d, "shuffle_mean": sh,
                "class_counts": counts.tolist()}
            print(f"  [{axis_name}] delta {d:+.4f} | shuffle {sh:+.4f} | "
                  f"counts {counts.min()}-{counts.max()}", flush=True)

        out["sessions"][name] = sess
        json.dump(out, open(OUT, "w"), indent=1)

    print("\nDONE run27", flush=True)


if __name__ == "__main__":
    main()
