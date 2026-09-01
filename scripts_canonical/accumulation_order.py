"""Antipodal-pair vs sequential accumulation - the RP^1 quotient in the ladder domain.

Bins istim into 16 direction bins (22.5 deg each over [0, 2pi)). Two structured
ladders with IDENTICAL rung sizes (8 rungs, 2 bins per rung):
  paired     - rung k adds bin k and its antipode k+8 (orientation-respecting
               order: each rung completes one RP^1 class)
  sequential - rung k adds bins 2k, 2k+1 (S^1 order: adjacent directions;
               orientation classes complete only in the second half)
Floor: 10 random same-size stimulus subsets (slope of mean log PR).

Prediction registered before run: if the odd (direction) sector is a minority
of the aligned signal (harmonic even/odd variance ratio 4.8x), the paired
order should climb FASTER early (each rung adds a full orientation class =
maximal new even-sector content) and give delta_paired > delta_sequential.
The difference is a pure symmetry-order effect at matched sizes.

Output: data/stringer_antipodal_order.json
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stringer_v1" / "natimg"
OUT = HERE / "data" / "stringer_antipodal_order.json"

N_BINS = 16
N_NULL = 10


def pr_trace(X):
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


def ladder_theta(Xt, members, rung_bins):
    sizes, prs = [], []
    acc = []
    for bins_this_rung in rung_bins:
        for b in bins_this_rung:
            acc.append(members[b])
        sel = np.concatenate(acc)
        sizes.append(len(sel))
        prs.append(pr_trace(Xt[sel]))
    return slope(sizes, prs), sizes


def main():
    rows = []
    for fp in sorted(DATA.glob("gratings*.npy")):
        t0 = time.time()
        dat = np.load(fp, allow_pickle=True).item()
        Xt = np.ascontiguousarray(np.asarray(dat["sresp"], dtype=np.float32).T)
        Xt /= (Xt.std() + 1e-9)
        istim = np.asarray(dat["istim"], dtype=float)
        bin_idx = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
                          0, N_BINS - 1)
        members = [np.where(bin_idx == b)[0] for b in range(N_BINS)]

        paired_order = [(k, k + 8) for k in range(8)]
        sequential_order = [(2 * k, 2 * k + 1) for k in range(8)]

        th_paired, sizes_p = ladder_theta(Xt, members, paired_order)
        th_seq, sizes_s = ladder_theta(Xt, members, sequential_order)

        n = Xt.shape[0]

        def compute_floor(sizes, seed):
            rng = np.random.default_rng(seed)
            nl = np.zeros((N_NULL, len(sizes)))
            for d in range(N_NULL):
                for k, s in enumerate(sizes):
                    nl[d, k] = np.log(max(pr_trace(Xt[rng.choice(n, s, replace=False)]),
                                          1e-9))
            return slope(sizes, np.exp(nl.mean(axis=0)))

        th_floor_p = compute_floor(sizes_p, seed=42)
        th_floor_s = compute_floor(sizes_s, seed=43)

        row = {"name": fp.stem, "rung_sizes_paired": sizes_p,
               "rung_sizes_sequential": sizes_s,
               "theta_paired": th_paired, "theta_sequential": th_seq,
               "theta_floor_paired": th_floor_p,
               "theta_floor_sequential": th_floor_s,
               "delta_paired": th_paired - th_floor_p,
               "delta_sequential": th_seq - th_floor_s,
               "order_effect": (th_paired - th_floor_p) - (th_seq - th_floor_s)}
        rows.append(row)
        print(f"{fp.stem}: d_paired={row['delta_paired']:+.4f} "
              f"d_seq={row['delta_sequential']:+.4f} "
              f"order={row['order_effect']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        with OUT.open("w") as f:
            json.dump({"construction": "dir16, 8 rungs x 2 bins, paired=(k,k+8) "
                       "vs sequential=(2k,2k+1), identical rung sizes",
                       "rows": rows}, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
