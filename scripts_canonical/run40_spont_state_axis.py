"""Run 40 (M5) — behavioral-state axes on the Stringer SPONT sessions, two null levels.

WHY: referee M5 asked for a behavioral-state axis; the static_biased originals
(with per-trial behavior aligned to gratings) are not local, but the 8 spont
sessions carry beh.runSpeed and beh.pupil.area aligned frame-by-frame with
Fsp. This run bins frames into octiles of each state variable and runs the
standard ladder (accumulate octiles in intensity order, counts 1/2/3/4/6/8,
floor = slope of mean null log-PR over 10 draws, rng(42)).

TWO NULL LEVELS (the run36 principle applied to real neural data): spont
frames are temporally autocorrelated, so a frame-level label permutation is
an easy null for any slowly varying axis. We therefore test against BOTH:
  perm  — 200 frame-level label permutations (label-free level);
  shift — 200 circular shifts of the label sequence (preserves label
          autocorrelation, breaks label-frame alignment; the
          nuisance-preserving level).

AXES (per session, on 4096 frames evenly subsampled from valid frames):
  runspeed8   — octiles of runSpeed (rank-based; ties split by time order —
                still-periods populate the bottom octiles)
  pupil8      — octiles of pupil area
  time_block8 — 8 contiguous blocks of the subsampled sequence (run27 analog)
  random8     — random 8-class labels (control)

REGISTERED EXPECTATIONS (written before the run):
B1: runspeed8 delta positive in most sessions; exceeds the frame-permutation
    null (p<0.05) in >=6/8 sessions; expected to also exceed the circular-
    shift null in the majority (state ALIGNMENT, not smoothness alone —
    Stringer 2019: behavior dominates spontaneous covariance).
B2: pupil8 same direction (arousal).
B3: time_block8 positive and exceeds the permutation null, but NOT expected
    to exceed the circular-shift null (slow drift is shift-invariant) — a
    designed real-data demonstration that the two null levels separate.
B4: random8 ~ 0 under both nulls.

GRAM TRICK: full 4096 x 4096 kernel per session; subset PR via double-
centered submatrix, validated in-session against direct pr (rel < 1e-3).
Fork-parallel over sessions (Pool(4)).

Out: ../data_canonical/run40_spont_state_axis.json
"""
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "basin_memory" / "data" / "stringer_v1" / "spont"
OUT = HERE.parent / "data_canonical" / "run40_spont_state_axis.json"

BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_BINS = 8
N_NULL = 10
N_PERM = 200
N_SHIFT = 200
N_FRAMES = 4096


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.maximum(np.asarray(prs, dtype=float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def subset_pr_gram(K, idx, full_cache=None):
    if full_cache is not None and len(idx) == K.shape[0]:
        return full_cache
    Ks = K[np.ix_(idx, idx)]
    rm = Ks.mean(axis=1)
    tm = rm.mean()
    Kc = Ks - rm[:, None] - rm[None, :] + tm
    tr = float(np.trace(Kc))
    tr2 = float((Kc * Kc).sum())
    if not np.isfinite(tr) or not np.isfinite(tr2) or tr2 <= 0:
        return 1.0
    return tr * tr / tr2


def pr_direct(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64) / (X.shape[0] - 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2


def octile_labels(v):
    ranks = np.argsort(np.argsort(v, kind="stable"))
    return (ranks * N_BINS // len(v)).astype(int)


def theta_obs(K, labels, full_cache):
    members = [np.where(labels == b)[0] for b in range(N_BINS)]
    sizes, prs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        sizes.append(len(sel))
        prs.append(subset_pr_gram(K, sel, full_cache))
    if len(sizes) < 3:
        return float("nan"), sizes
    return slope(sizes, prs), sizes


def process_session(fp):
    t0 = time.time()
    d = sio.loadmat(fp, squeeze_me=True, struct_as_record=False)
    Fsp = np.asarray(d["Fsp"], dtype=np.float32)
    beh = d["beh"]
    run = np.asarray(beh.runSpeed, dtype=float)
    pup = np.asarray(beh.pupil.area, dtype=float)
    valid = np.isfinite(run) & np.isfinite(pup)
    vidx = np.where(valid)[0]
    sub = vidx[np.unique(np.linspace(0, len(vidx) - 1, N_FRAMES).astype(int))]

    X = np.ascontiguousarray(Fsp[:, sub].T)  # frames x neurons
    X /= (X.std() + 1e-9)
    n = X.shape[0]
    K = (X @ X.T).astype(np.float64)

    vrng = np.random.default_rng(7)
    for s in (200, 1500):
        idx = vrng.choice(n, s, replace=False)
        a, b = subset_pr_gram(K, idx), pr_direct(X[idx])
        assert abs(a - b) / max(abs(b), 1e-9) < 1e-3, (fp.stem, a, b)
    full_cache = subset_pr_gram(K, np.arange(n))

    arng = np.random.default_rng(4000)
    axes = {
        "runspeed8": octile_labels(run[sub]),
        "pupil8": octile_labels(pup[sub]),
        "time_block8": (np.arange(n) * N_BINS // n).astype(int),
        "random8": arng.integers(0, N_BINS, n),
    }
    del Fsp, X

    row = {"name": fp.stem, "n_frames": n, "n_neurons": int(d["Fsp"].shape[0])}
    for tag, labels in axes.items():
        th_o, sizes = theta_obs(K, labels, full_cache)
        if not np.isfinite(th_o):
            row[tag] = {"status": "degenerate"}
            continue
        frng = np.random.default_rng(42)
        null_logs = np.zeros((N_NULL, len(sizes)))
        for dd in range(N_NULL):
            for k, s in enumerate(sizes):
                null_logs[dd, k] = np.log(max(
                    subset_pr_gram(K, frng.choice(n, s, replace=False),
                                   full_cache), 1e-9))
        th_f = slope(sizes, np.exp(null_logs.mean(axis=0)))
        d_obs = th_o - th_f

        nulls = {}
        prng = np.random.default_rng(4001)
        perm_d = [theta_obs(K, labels[prng.permutation(n)], full_cache)[0]
                  - th_f for _ in range(N_PERM)]
        srng = np.random.default_rng(4002)
        shift_d = [theta_obs(K, np.roll(labels, int(srng.integers(1, n))),
                             full_cache)[0] - th_f for _ in range(N_SHIFT)]
        for nname, nd in (("perm", perm_d), ("shift", shift_d)):
            nd = np.asarray(nd)
            nd = nd[np.isfinite(nd)]
            nm, ns = float(nd.mean()), float(nd.std())
            nulls[nname] = {
                "null_mean": nm, "null_sd": ns,
                "z": (d_obs - nm) / ns if ns > 0 else 0.0,
                "p_one": float((1 + (nd >= d_obs).sum()) / (len(nd) + 1))}
        row[tag] = {"delta": d_obs, "theta_obs": th_o, "theta_floor": th_f,
                    "rung_sizes": sizes, **{f"{k}_{kk}": vv
                                            for k, v in nulls.items()
                                            for kk, vv in v.items()}}
        print(f"{fp.stem} [{tag}]: delta={d_obs:+.4f} | perm z="
              f"{nulls['perm']['z']:+.1f} p={nulls['perm']['p_one']:.3f} | "
              f"shift z={nulls['shift']['z']:+.1f} "
              f"p={nulls['shift']['p_one']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)
    return row


def main():
    mp.set_start_method("fork", force=True)
    files = sorted(DATA.glob("spont_*.mat"))
    print(f"{len(files)} spont sessions", flush=True)
    with mp.Pool(4) as pool:
        rows = pool.map(process_session, files)
    json.dump({"n_perm": N_PERM, "n_shift": N_SHIFT, "n_frames": N_FRAMES,
               "construction": "octile state axes, ordered accumulation "
               "1/2/3/4/6/8, floor rng(42); two null levels: frame "
               "permutation + circular shift (autocorrelation-preserving)",
               "rows": rows}, open(OUT, "w"), indent=1)
    print("DONE run40", flush=True)


if __name__ == "__main__":
    main()
