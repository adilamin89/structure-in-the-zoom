"""Shuffled-label control + ablations for the orientation zoom - COMMITTED artifact.

Regenerates the rescue-session mismatch analysis (ledger §G). Two binnings:
  dir8 - 8 equal-width bins over [0, 2pi) via digitize, matching the committed
         modal_bootstrap_all10.py construction (direction bins).
  ori8 - 8 equal-width bins over [0, pi) on istim mod pi (orientation bins,
         the construction the paper's RP^1 symmetry narrative describes).
Structured ladder: accumulate bins in index order at counts {1,2,3,4,6,8}.
Floor: slope of mean null log-PR over 10 random same-size stimulus subsets.

Arms (per recording, per binning):
  baseline  - real labels (dir8 must reproduce the committed point estimates)
  shuffled  - labels permuted across stimuli (5 seeds): the rung-matched
              shuffled-label control. Expected delta ~ 0.
Output: data/stringer_mismatch_ablation.json (kill-safe).
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stringer_v1" / "natimg"
OUT = HERE / "data" / "stringer_mismatch_ablation.json"

BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_BINS = 8
N_NULL = 10
N_SHUFFLE_SEEDS = 5


def pr_trace(X):
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    if n < 3:
        return 1.0
    # matmul in float32 for speed; trace arithmetic in float64 (float32
    # accumulation overflowed to inf on the 22K-neuron recordings)
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


def obs_theta(Xt, bin_idx):
    """Xt: stimuli x neurons. bin_idx: per-stimulus bin 0..7.
    Rungs with <10 stimuli are skipped (committed modal construction; also
    guards the static_biased sessions whose istim spans only ~4 degrees,
    leaving most bins empty)."""
    members = [np.where(bin_idx == b)[0] for b in range(N_BINS)]
    sizes, obs_prs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c]) if c else np.array([], dtype=int)
        if len(sel) < 10:
            continue
        sizes.append(len(sel))
        obs_prs.append(pr_trace(Xt[sel]))
    if len(sizes) < 3:
        return float("nan"), sizes
    return slope(sizes, obs_prs), sizes


def floor_theta(Xt, sizes, rng):
    n = Xt.shape[0]
    null_logs = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            null_logs[d, k] = np.log(max(pr_trace(Xt[rng.choice(n, s, replace=False)]), 1e-9))
    return slope(sizes, np.exp(null_logs.mean(axis=0)))


def main():
    rows = []
    for fp in sorted(DATA.glob("*.npy")):
        t0 = time.time()
        dat = np.load(fp, allow_pickle=True).item()
        Xt = np.ascontiguousarray(np.asarray(dat["sresp"], dtype=np.float32).T)
        Xt /= (Xt.std() + 1e-9)  # PR is scale-invariant; guards float32 overflow
        istim = np.asarray(dat["istim"], dtype=float)
        binnings = {
            "dir8": np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
                            0, N_BINS - 1),
            "ori8": np.minimum(((istim % np.pi) / (np.pi / N_BINS)).astype(int), N_BINS - 1),
        }

        row = {"name": fp.stem, "n_stimuli": int(Xt.shape[0]),
               "n_neurons": int(Xt.shape[1])}
        for tag, bin_idx in binnings.items():
            th_o, sizes = obs_theta(Xt, bin_idx)
            if not np.isfinite(th_o):
                row[tag] = {"status": "degenerate_probe", "rung_sizes": sizes}
                print(f"{fp.stem} [{tag}]: DEGENERATE (<3 usable rungs)", flush=True)
                continue
            # label shuffles preserve bin occupancies, so rung sizes are identical
            # and the floor can be computed once and reused exactly
            th_f = floor_theta(Xt, sizes, np.random.default_rng(42))
            shuf_deltas = []
            for s in range(N_SHUFFLE_SEEDS):
                srng = np.random.default_rng(500 + s)
                perm_idx = bin_idx[srng.permutation(len(bin_idx))]
                o, _ = obs_theta(Xt, perm_idx)
                shuf_deltas.append(o - th_f)
            row[tag] = {
                "rung_sizes": sizes,
                "theta_obs": th_o, "theta_floor": th_f, "delta": th_o - th_f,
                "shuffled_delta_mean": float(np.mean(shuf_deltas)),
                "shuffled_delta_sd": float(np.std(shuf_deltas)),
                "shuffled_deltas": [float(d) for d in shuf_deltas],
            }
            print(f"{fp.stem} [{tag}]: base delta={row[tag]['delta']:+.4f} "
                  f"shuffled={row[tag]['shuffled_delta_mean']:+.4f}"
                  f"±{row[tag]['shuffled_delta_sd']:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        rows.append(row)
        with OUT.open("w") as f:
            json.dump({"construction": "dir8=digitize [0,2pi) (committed modal "
                       "construction) + ori8=istim mod pi (paper narrative); "
                       "accumulate counts 1/2/3/4/6/8, floor=slope of mean null logPR (10 draws)",
                       "rows": rows}, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
