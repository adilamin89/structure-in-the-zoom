"""Run 51 (S75) - Anatomical (spatial) blocking of the direction-aligned shift
and of the sector balance on all eight grating recordings.

Why (S75 literature sweep). Shmuel and Grinvald 1996: in cat area 18 direction
domains cluster two to four times more weakly than orientation domains because
each iso-orientation patch splits into opposite-direction halves. Lashgari et
al. 2012: in awake macaque V1 the LFP keeps the single units' preferences but
loses 45% of their direction selectivity against 15% of their tuning depth.
Ringach et al. 2016 / Kondo et al. 2016: mouse V1 is salt-and-pepper only
approximately; tuning similarity decays with cortical distance. The Discussion
now reads App E as a flow law: each harmonic sector's amplitude flows under
blocking at a rate set by the spatial correlation length of its preference
map; label-sorted blocks are the infinite-length limit, random blocks the
zero-length limit. Anatomical blocks in mouse are the intermediate point, and
the recordings carry imaging-plane centroids (dat["stat"][i]["med"]), so the
point can be measured instead of hedged (App E: "retention under anatomical
blocking depends on population size and tuning strength").

Arms at K in {1, 4, 16, 32}: spatial blocks (k-means on the imaging-plane
centroids with N // K clusters, each cluster averaged into one unit), random
blocks, orientation-sorted and direction-sorted blocks (run50b construction).
Two floor seeds averaged for delta_dir; b2/c1 of the blocked class means via
the exact 8-point profile (sector_balance_scale.profile_ratio).

REGISTERED PREDICTIONS (written before the run):
S1: spatial-block retention of delta_dir at K = 32 (relative to K = 1) is at
    least the random-block retention in >= 6 of 8 recordings (weak clustering
    acts as a short but nonzero correlation length).
S2: spatial blocks do not steer the sector balance: |b2/c1(K = 32) /
    b2/c1(K = 1) - 1| < 0.2 in all 8 recordings (mouse anatomy has no
    orientation columns to project onto the even sector). A rise above 1.2
    would mean orientation micro-clustering strong enough to act as a column.
S3 (descriptive): the spatial-minus-random retention gap is larger in the
    three localized recordings than in the five full-field ones (the aperture
    footprint clusters response range in the plane, which random blocks mix).

Out: ../data_canonical/run51_spatial_blocking.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
DATA = HERE.parent / "data_canonical"
OUT = DATA / "run51_spatial_blocking.json"
NB = 8; BIN_COUNTS = [1, 2, 3, 4, 6, 8]; N_NULL = 10; KS = [1, 4, 16, 32]


def pr_c(X):
    Xc = X - X.mean(0); G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum()); return tr * tr / tr2 if tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, float)); y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    return float(np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, y, rcond=None)[0][1])


def delta_dir(Xt, bl, rng):
    members = [np.where(bl == k)[0] for k in range(NB)]
    sizes, prs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c]); sizes.append(len(sel)); prs.append(pr_c(Xt[sel]))
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(pr_c(Xt[rng.choice(len(Xt), s, replace=False)]), 1e-9))
    return slope(sizes, prs) - slope(sizes, np.exp(nl.mean(0)))


def profile_ratio(M):
    Cm = np.corrcoef(M)
    prof = np.array([np.mean([Cm[i, (i + k) % NB] for i in range(NB)]) for k in range(NB)])
    ang = 2 * np.pi * np.arange(NB) / NB
    c1 = 2 / NB * np.sum(prof * np.cos(ang)); b2 = 2 / NB * np.sum(prof * np.cos(2 * ang))
    return float(b2 / c1)


def spatial_blocks(pos, K, seed=0):
    """Return a list of index arrays: N // K compact spatial clusters (k-means++)."""
    n = pos.shape[0]
    if K == 1:
        return [np.array([i]) for i in range(n)]
    nclu = max(n // K, 2)
    _, lab = kmeans2(pos.astype(np.float64), nclu, minit="++", seed=seed)
    return [np.where(lab == c)[0] for c in range(nclu) if np.any(lab == c)]


def sorted_blocks(order, K):
    n = len(order); nblk = n // K
    return [order[i * K:(i + 1) * K] for i in range(nblk)]


def main():
    rows = []
    for f in sorted(RAW.glob("gratings_*.npy")):
        dat = np.load(f, allow_pickle=True).item()
        X = np.asarray(dat["sresp"], np.float32); X /= X.std() + 1e-9
        istim = np.asarray(dat["istim"], float)
        bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, NB + 1)) - 1, 0, NB - 1)
        if np.bincount(bl, minlength=NB).min() < 10:
            rows.append({"name": f.stem, "status": "degenerate_bins"}); continue
        Xt = np.ascontiguousarray(X.T)
        pos = np.array([[s["med"][0], s["med"][1]] for s in dat["stat"]], float)
        assert pos.shape[0] == Xt.shape[1], (pos.shape, Xt.shape)
        ang = 2 * np.pi * np.arange(NB) / NB
        M = np.stack([Xt[bl == k].mean(0) for k in range(NB)]); Mc = M - M.mean(0)
        ph_ori = (np.angle((Mc * np.exp(2j * ang[:, None])).sum(0)) / 2) % np.pi
        ph_dir = np.angle((Mc * np.exp(1j * ang[:, None])).sum(0)) % (2 * np.pi)
        rand_order = np.random.default_rng(1).permutation(Xt.shape[1])
        res = {"K": KS}; bal = {"K": KS}
        for arm in ("spatial", "random", "ori_sorted", "dir_sorted"):
            vals, bals = [], []
            for K in KS:
                if arm == "spatial": blocks = spatial_blocks(pos, K)
                elif arm == "random": blocks = sorted_blocks(rand_order, K)
                elif arm == "ori_sorted": blocks = sorted_blocks(np.argsort(ph_ori), K)
                else: blocks = sorted_blocks(np.argsort(ph_dir), K)
                Xb = np.stack([Xt[:, b].mean(1) for b in blocks], 1)
                vals.append(float(np.mean([delta_dir(Xb, bl, np.random.default_rng(42 + s)) for s in (0, 1)])))
                Mb = np.stack([Xb[bl == k].mean(0) for k in range(NB)])
                bals.append(profile_ratio(Mb))
            res[arm] = vals; bal[arm] = bals
        ret = {a: res[a][-1] / res[a][0] for a in ("spatial", "random", "ori_sorted", "dir_sorted")}
        balflow = {a: bal[a][-1] / bal[a][0] for a in ("spatial", "random", "ori_sorted", "dir_sorted")}
        rows.append({"name": f.stem, "status": "ok", "n_neurons": int(Xt.shape[1]), "graining": res, "sector_balance": bal,
                     "retention_K32": ret, "balance_flow_K32": balflow})
        print(f"  {f.stem[:32]:32s} N {Xt.shape[1]:5d} | K=1 {res['random'][0]:+.3f} | ret K32 spatial {ret['spatial']:.2f} random {ret['random']:.2f} "
              f"ori {ret['ori_sorted']:.2f} dir {ret['dir_sorted']:.2f} | b2/c1 flow spatial {balflow['spatial']:.2f} random {balflow['random']:.2f} ori {balflow['ori_sorted']:.2f}", flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    loc = [r for r in ok if "local" in r["name"]]; ff = [r for r in ok if "local" not in r["name"]]
    s1 = sum(r["retention_K32"]["spatial"] >= r["retention_K32"]["random"] for r in ok)
    s2 = all(abs(r["balance_flow_K32"]["spatial"] - 1) < 0.2 for r in ok)
    gap = lambda r: r["retention_K32"]["spatial"] - r["retention_K32"]["random"]
    verdict = {"S1_spatial_ge_random_count": int(s1), "S1_pass": bool(s1 >= 6),
               "S2_balance_flow_spatial": [round(r["balance_flow_K32"]["spatial"], 3) for r in ok], "S2_pass": bool(s2),
               "S3_gap_localized": [round(gap(r), 3) for r in loc], "S3_gap_fullfield": [round(gap(r), 3) for r in ff],
               "S3_descriptive_localized_gap_larger": bool(np.median([gap(r) for r in loc]) > np.median([gap(r) for r in ff])),
               "spatial_retention": [round(r["retention_K32"]["spatial"], 2) for r in ok],
               "random_retention": [round(r["retention_K32"]["random"], 2) for r in ok]}
    print("VERDICT", json.dumps(verdict))
    json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1); print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
