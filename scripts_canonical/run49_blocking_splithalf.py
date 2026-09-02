"""Run 49 (S74) - Split-half control for the label-sorted coarse-graining flow.

sector_balance_scale.py sorted units by a preferred phase estimated from the
same trials used to evaluate the blocked profile. Sorting by an estimated phase
aligns that harmonic's estimation noise across the block, so the sorted flows
(especially the direction-sorted one, whose phases come from the weak odd
harmonic) are upper bounds. Here phases are estimated on odd-indexed trials and
the blocked class means are evaluated on even-indexed trials.

REGISTERED EXPECTATIONS (before the run): orientation-sorted amplification
of b2/c1 survives the split (the even phase is well estimated); the
direction-sorted decrease shrinks substantially (part of it was aligned noise)
but remains monotone; random blocks stay invariant.

Out: ../data_canonical/run49_blocking_splithalf.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = HERE.parent / "data_canonical" / "run49_blocking_splithalf.json"
NB = 8
KS = [1, 2, 4, 8, 16, 32, 64]


def ratio(M):
    Cm = np.corrcoef(M); prof = np.array([np.mean([Cm[i, (i + k) % NB] for i in range(NB)]) for k in range(NB)])
    ang = 2 * np.pi * np.arange(NB) / NB
    return float((2 / NB * np.sum(prof * np.cos(2 * ang))) / (2 / NB * np.sum(prof * np.cos(ang))))


def main():
    rows = []
    for f in sorted(RAW.glob("gratings_*.npy")):
        dat = np.load(f, allow_pickle=True).item()
        X = dat["sresp"].T.astype(np.float64); phi = np.asarray(dat["istim"]).ravel() % (2 * np.pi)
        b = np.floor(phi / (2 * np.pi) * NB).astype(int) % NB
        if np.bincount(b, minlength=NB).min() < 10:
            rows.append({"name": f.stem, "status": "degenerate_bins"}); continue
        odd = np.arange(len(b)) % 2 == 1; even = ~odd
        M_fit = np.stack([X[(b == k) & odd].mean(0) for k in range(NB)])
        M_eval = np.stack([X[(b == k) & even].mean(0) for k in range(NB)])
        ang = 2 * np.pi * np.arange(NB) / NB; Mc = M_fit - M_fit.mean(0)
        ph_ori = (np.angle((Mc * np.exp(2j * ang[:, None])).sum(0)) / 2) % np.pi
        ph_dir = np.angle((Mc * np.exp(1j * ang[:, None])).sum(0)) % (2 * np.pi)
        rng = np.random.default_rng(0)
        flow = {"K": KS, "ori_sorted": [], "dir_sorted": [], "random": []}
        for K in KS:
            nblk = M_eval.shape[1] // K
            for key, order in (("ori_sorted", np.argsort(ph_ori)), ("dir_sorted", np.argsort(ph_dir)), ("random", rng.permutation(M_eval.shape[1]))):
                Mb = np.stack([M_eval[:, order[i * K:(i + 1) * K]].mean(1) for i in range(nblk)], 1)
                flow[key].append(ratio(Mb))
        rows.append({"name": f.stem, "status": "ok", "b2_over_c1_even_half": flow["random"][0], "flow": flow})
        print(f"  {f.stem[:32]:32s} K=1 {flow['random'][0]:.2f} | K=64 ori {flow['ori_sorted'][-1]:.2f} dir {flow['dir_sorted'][-1]:.2f} rand {flow['random'][-1]:.2f}")
    ok = [r for r in rows if r["status"] == "ok"]
    verdict = {"ori_sorted_K64_over_K1": [round(r["flow"]["ori_sorted"][-1] / r["flow"]["ori_sorted"][0], 2) for r in ok],
               "dir_sorted_K64_over_K1": [round(r["flow"]["dir_sorted"][-1] / r["flow"]["dir_sorted"][0], 2) for r in ok],
               "random_K64_over_K1": [round(r["flow"]["random"][-1] / r["flow"]["random"][0], 2) for r in ok],
               "dir_sorted_monotone_decrease_count": sum(all(np.diff(r["flow"]["dir_sorted"]) <= 1e-9) for r in ok)}
    print("VERDICT", json.dumps(verdict))
    json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1); print("wrote", OUT)


if __name__ == "__main__":
    main()
