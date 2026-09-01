"""Run 38 — V1 label-permutation inference: >=200 permutations per grating recording.

WHY: the committed shuffled-label control (stringer_mismatch_ablation.json)
uses 5 shuffle seeds per recording — descriptive, not inferential. This run
upgrades the same construction (dir8 + ori8 binnings, accumulate counts
1/2/3/4/6/8, floor = slope of mean null log-PR over 10 draws with rng(42),
rung sizes preserved under permutation so the floor is reused EXACTLY) to
N_PERM=200 label permutations per recording per binning, yielding real
permutation p-values and z-scores for the observed delta.

Scope: the 8 grating recordings (the paper's primary claim). The two
static_biased sessions stay excluded (degenerate probe range, ledger S68).

REGISTERED EXPECTATIONS (written before the run):
V1: dir8 — observed delta exceeds ALL 200 permuted deltas in 8/8 recordings
    (p = 1/201 each), consistent with the 5-seed control (obs +0.20..+0.45
    vs shuffled ~0).
V2: ori8 — same direction; expect 8/8 as well (ori8 deltas smaller but the
    5-seed nulls were equally tight).
V3: null SD comparable to the 5-seed SD already recorded (no widening).

GRAM TRICK: the full stimulus x stimulus linear kernel K = X X^T is computed
once per recording (float32 matmul on globally std-normalized responses,
float64 accumulation downstream); every subset PR is then O(n^2) via
double-centering the kernel submatrix — the 11K-22K neuron dimension is
touched exactly once. Validated in-recording against the committed pr_trace
(rel diff < 1e-3) before use. Fork-parallel over recordings (Pool(4)).

Out: ../data_canonical/run38_v1_label_permutations.json
"""
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = HERE.parent / "data_canonical" / "run38_v1_label_permutations.json"

BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_BINS = 8
N_NULL = 10
N_PERM = 200


def pr_trace(X):
    """Committed construction (shuffle_label_control.py) — validation only."""
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    if n < 3:
        return 1.0
    G = (Xc @ Xc.T).astype(np.float64) / (n - 1)
    tr = float(np.trace(G))
    tr2 = float((G * G).sum())
    if not np.isfinite(tr) or not np.isfinite(tr2) or tr2 <= 0:
        return 1.0
    return tr * tr / tr2


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.maximum(np.asarray(prs, dtype=float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def subset_pr_gram(K, idx):
    Ks = K[np.ix_(idx, idx)]
    rm = Ks.mean(axis=1)
    tm = rm.mean()
    Kc = Ks - rm[:, None] - rm[None, :] + tm
    tr = float(np.trace(Kc))
    tr2 = float((Kc * Kc).sum())
    if not np.isfinite(tr) or not np.isfinite(tr2) or tr2 <= 0:
        return 1.0
    return tr * tr / tr2


def obs_theta_gram(K, bin_idx):
    members = [np.where(bin_idx == b)[0] for b in range(N_BINS)]
    sizes, prs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        sizes.append(len(sel))
        prs.append(subset_pr_gram(K, sel))
    if len(sizes) < 3:
        return float("nan"), sizes
    return slope(sizes, prs), sizes


def floor_theta_gram(K, n, sizes, rng):
    null_logs = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            null_logs[d, k] = np.log(
                max(subset_pr_gram(K, rng.choice(n, s, replace=False)), 1e-9))
    return slope(sizes, np.exp(null_logs.mean(axis=0)))


def process_recording(fp):
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    Xt = np.ascontiguousarray(np.asarray(dat["sresp"], dtype=np.float32).T)
    Xt /= (Xt.std() + 1e-9)
    istim = np.asarray(dat["istim"], dtype=float)
    n = Xt.shape[0]

    K = (Xt @ Xt.T).astype(np.float64)
    # validate gram trick vs committed pr_trace on 2 subsets
    vrng = np.random.default_rng(7)
    for s in (200, 1500):
        idx = vrng.choice(n, s, replace=False)
        a, b = subset_pr_gram(K, idx), pr_trace(Xt[idx])
        assert abs(a - b) / max(abs(b), 1e-9) < 1e-3, (fp.stem, a, b)

    binnings = {
        "dir8": np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi,
                                                       N_BINS + 1)) - 1,
                        0, N_BINS - 1),
        "ori8": np.minimum(((istim % np.pi) / (np.pi / N_BINS)).astype(int),
                           N_BINS - 1),
    }
    row = {"name": fp.stem, "n_stimuli": n, "n_neurons": int(Xt.shape[1])}
    del Xt

    for tag, bin_idx in binnings.items():
        th_o, sizes = obs_theta_gram(K, bin_idx)
        if not np.isfinite(th_o):
            row[tag] = {"status": "degenerate_probe"}
            continue
        th_f = floor_theta_gram(K, n, sizes, np.random.default_rng(42))
        d_obs = th_o - th_f
        perm_deltas = []
        prng = np.random.default_rng(3800)
        for p in range(N_PERM):
            perm_idx = bin_idx[prng.permutation(n)]
            o, _ = obs_theta_gram(K, perm_idx)
            perm_deltas.append(o - th_f)
            if (p + 1) % 50 == 0:
                print(f"  {fp.stem} [{tag}] perm {p+1}/{N_PERM} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        pd = np.asarray(perm_deltas)
        nm, ns = float(pd.mean()), float(pd.std())
        row[tag] = {
            "rung_sizes": sizes, "theta_obs": th_o, "theta_floor": th_f,
            "delta": d_obs, "n_perm": N_PERM,
            "null_mean": nm, "null_sd": ns,
            "z": (d_obs - nm) / ns if ns > 0 else 0.0,
            "n_perm_geq_obs": int((pd >= d_obs).sum()),
            "p_one": float((1 + (pd >= d_obs).sum()) / (N_PERM + 1)),
            "perm_deltas": [float(x) for x in pd],
        }
        print(f"{fp.stem} [{tag}]: delta={d_obs:+.4f} null={nm:+.4f}±{ns:.4f} "
              f"z={row[tag]['z']:+.1f} p={row[tag]['p_one']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return row


def main():
    mp.set_start_method("fork", force=True)
    files = [fp for fp in sorted(DATA.glob("gratings_*.npy"))]
    print(f"{len(files)} grating recordings, {N_PERM} perms each", flush=True)
    with mp.Pool(4) as pool:
        rows = pool.map(process_recording, files)
    with OUT.open("w") as f:
        json.dump({"n_perm": N_PERM,
                   "construction": "identical to stringer_mismatch_ablation "
                   "(dir8+ori8, counts 1/2/3/4/6/8, floor rng(42) reused "
                   "across permutations — rung sizes preserved); gram-trick "
                   "subset PR validated per recording",
                   "rows": rows}, f, indent=1)
    print("DONE run38", flush=True)


if __name__ == "__main__":
    main()
