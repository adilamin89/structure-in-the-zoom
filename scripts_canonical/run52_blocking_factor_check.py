"""Run 52 (S75) - Does the blocking factor B(K) account for the measured flow
of the sector balance, and what are the within-block coherences rho_l?

Section 4.1 now displays B(K) = [1/K + (1-1/K) rho_2] / [1/K + (1-1/K) rho_1]
with rho_l the within-block phase coherence of sector l, and calls it exact
bookkeeping. It is exact for block averaging of the class-mean matrix itself
(Parseval per neuron: each harmonic power is a sum over neurons). The paper's
profile, however, is the CORRELATION of class means, which re-standardizes each
class across the blocks after averaging. This run measures rho_l for every
blocking scheme (random, orientation-sorted, direction-sorted, spatial k-means)
and asks how far the correlation-profile flow the paper reports departs from
the raw-covariance flow that B(K) describes.

REGISTERED PREDICTIONS (before the run):
R1 (identity): for the raw class-mean covariance, the harmonic-power flow
   P_K(l)/P_1(l) equals 1/K + (1-1/K) rho_l with rho_l the power-weighted mean
   within-block coherence, to numerical precision (< 1e-8) in every cell.
R2 (renormalization): the correlation-profile flow of b2/c1 (Figure 4A) agrees
   with the raw B(K) within 15 percent at K <= 32 for random and
   orientation-sorted blocks in all eight recordings.
R3 (coherences): spatial (anatomical) blocks have |rho_l| < 0.05 in both
   sectors in all eight recordings (mouse anatomy is the random limit);
   orientation-sorted blocks have rho_2 >= 0.8 at K <= 32 in all eight.

Out: ../data_canonical/run52_blocking_factor_check.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
DATA = HERE.parent / "data_canonical"
OUT = DATA / "run52_blocking_factor_check.json"
NB = 8; KS = [1, 4, 16, 32, 64]
ANG = 2 * np.pi * np.arange(NB) / NB


def harmonics(M):
    """z[l, n] = sum_k Mc[k, n] exp(i l theta_k) for l = 1, 2 (Mc centred over classes)."""
    Mc = M - M.mean(0, keepdims=True)
    return np.stack([(Mc * np.exp(1j * l * ANG[:, None])).sum(0) for l in (1, 2)])


def profile_ratio_corr(M):
    Cm = np.corrcoef(M)
    prof = np.array([np.mean([Cm[i, (i + k) % NB] for i in range(NB)]) for k in range(NB)])
    c1 = 2 / NB * np.sum(prof * np.cos(ANG)); b2 = 2 / NB * np.sum(prof * np.cos(2 * ANG))
    return float(b2 / c1)


def raw_powers(M):
    z = harmonics(M)
    return np.array([float(np.sum(np.abs(z[l]) ** 2)) for l in (0, 1)])  # sectors 1, 2


def blocks_for(arm, K, order, pos):
    n = len(order)
    if K == 1:
        return [np.array([i]) for i in range(n)]
    if arm == "spatial":
        _, lab = kmeans2(pos.astype(np.float64), max(n // K, 2), minit="++", seed=0)
        return [np.where(lab == c)[0] for c in range(lab.max() + 1) if np.any(lab == c)]
    return [order[i * K:(i + 1) * K] for i in range(n // K)]


def main():
    rows = []
    for f in sorted(RAW.glob("gratings_*.npy")):
        dat = np.load(f, allow_pickle=True).item()
        X = dat["sresp"].T.astype(np.float64)
        phi = np.asarray(dat["istim"]).ravel() % (2 * np.pi)
        b = np.floor(phi / (2 * np.pi) * NB).astype(int) % NB
        if np.bincount(b, minlength=NB).min() < 10:
            rows.append({"name": f.stem, "status": "degenerate_bins"}); continue
        M = np.stack([X[b == k].mean(0) for k in range(NB)])
        pos = np.array([[s["med"][0], s["med"][1]] for s in dat["stat"]], float)
        z = harmonics(M)
        Mc = M - M.mean(0, keepdims=True)
        ph_ori = (np.angle(z[1]) / 2) % np.pi; ph_dir = np.angle(z[0]) % (2 * np.pi)
        orders = {"random": np.random.default_rng(1).permutation(M.shape[1]),
                  "ori_sorted": np.argsort(ph_ori), "dir_sorted": np.argsort(ph_dir), "spatial": np.arange(M.shape[1])}
        P1 = raw_powers(M); corr1 = profile_ratio_corr(M)
        res = {}
        for arm, order in orders.items():
            out = {"K": KS, "rho1": [], "rho2": [], "raw_flow": [], "B_from_rho": [], "corr_flow": [], "identity_dev": []}
            for K in KS:
                blocks = blocks_for(arm, K, order, pos)
                Mb = np.stack([Mc[:, blk].mean(1) for blk in blocks], 1)   # blocked centred class means
                Pk = raw_powers(Mb)
                # power-weighted within-block coherence per sector
                rho = []
                for l in (0, 1):
                    num = 0.0; den = 0.0
                    for blk in blocks:
                        kk = len(blk)
                        if kk < 2:
                            continue
                        zz = z[l][blk]; s_all = np.abs(zz.sum()) ** 2; s_self = np.sum(np.abs(zz) ** 2)
                        num += (s_all - s_self); den += (kk - 1) * s_self
                    rho.append(num / den if den > 0 else 0.0)
                # raw flow of each sector's total power, normalised per block count so that K=1 gives 1
                nb = len(blocks); kmean = M.shape[1] / nb
                raw_l = [Pk[l] * kmean / P1[l] for l in (0, 1)]           # = 1/K + (1-1/K) rho_l if the identity holds
                pred_l = [1 / kmean + (1 - 1 / kmean) * rho[l] for l in (0, 1)]
                dev = max(abs(raw_l[l] - pred_l[l]) for l in (0, 1)) if K > 1 else 0.0
                B = pred_l[1] / pred_l[0]
                corr_flow = profile_ratio_corr(np.stack([M[:, blk].mean(1) for blk in blocks], 1)) / corr1
                out["rho1"].append(rho[0]); out["rho2"].append(rho[1]); out["raw_flow"].append(raw_l[1] / raw_l[0])
                out["B_from_rho"].append(B); out["corr_flow"].append(corr_flow); out["identity_dev"].append(dev)
            res[arm] = out
            print(f"  {f.stem[:26]:26s} {arm:10s} " + " ".join(f"K{K}: rho1 {out['rho1'][i]:+.2f} rho2 {out['rho2'][i]:+.2f} B {out['B_from_rho'][i]:.2f} corr {out['corr_flow'][i]:.2f}" for i, K in enumerate(KS) if K in (4, 32)), flush=True)
        rows.append({"name": f.stem, "status": "ok", "arms": res})
    ok = [r for r in rows if r["status"] == "ok"]
    r1 = max(max(r["arms"][a]["identity_dev"]) for r in ok for a in r["arms"])
    def rel(a, r, i): return abs(r["arms"][a]["corr_flow"][i] / r["arms"][a]["B_from_rho"][i] - 1)
    r2 = max(rel(a, r, i) for r in ok for a in ("random", "ori_sorted") for i, K in enumerate(KS) if 1 < K <= 32)
    r3a = max(max(abs(v) for v in r["arms"]["spatial"]["rho1"][1:] + r["arms"]["spatial"]["rho2"][1:]) for r in ok)
    r3b = min(min(r["arms"]["ori_sorted"]["rho2"][i] for i, K in enumerate(KS) if 1 < K <= 32) for r in ok)
    verdict = {"R1_max_identity_dev": r1, "R1_pass": bool(r1 < 1e-8),
               "R2_max_rel_dev_corr_vs_B_K_le_32": r2, "R2_pass": bool(r2 < 0.15),
               "R3_spatial_max_abs_rho": r3a, "R3_ori_min_rho2_K_le_32": r3b, "R3_pass": bool(r3a < 0.05 and r3b >= 0.8),
               "per_recording_K32": {r["name"]: {a: {"rho1": round(r["arms"][a]["rho1"][3], 3), "rho2": round(r["arms"][a]["rho2"][3], 3),
                                                     "B": round(r["arms"][a]["B_from_rho"][3], 2), "corr_flow": round(r["arms"][a]["corr_flow"][3], 2)}
                                                 for a in r["arms"]} for r in ok}}
    print("VERDICT", json.dumps({k: v for k, v in verdict.items() if k != "per_recording_K32"}))
    json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1); print("wrote", OUT)


if __name__ == "__main__":
    main()
