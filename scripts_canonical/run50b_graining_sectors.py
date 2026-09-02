"""Run 50b (S74) - Sector-resolved coarse-graining of the direction-aligned
shift on all eight grating recordings.

run50 showed that preference-sorted (orientation-phase) graining preserves
delta_dir on GT3 but decays it on GT2 and the localized recordings, and that
random blocks do not decay on GT1. The sector law of sector_balance_scale says
why: orientation-sorted blocks group antipodal preferences and project onto
the even sector, so they remove the shift wherever the shift rides on the odd
(direction) sector. This run adds a DIRECTION-sorted arm.

Arms at K in {1, 4, 16, 32}: orientation-sorted blocks, direction-sorted
blocks, random blocks; two floor seeds averaged.

REGISTERED PREDICTIONS (before the run):
G1: direction-sorted graining preserves delta_dir at K = 32 (>= 0.6 of K = 1)
    in every recording, including the three localized (dipole-dominant) ones.
G2: orientation-sorted graining preserves delta_dir (>= 0.6 of K = 1) in the
    quadrupole-dominant recordings with the smallest odd share (GT3, the two
    low-contrast) and decays it below 0.6 in the localized recordings.
G3: the K = 32 retention under orientation-sorted graining decreases with the
    recording's odd share c1/b2 (Spearman rho < 0 across the eight).

Out: ../data_canonical/run50b_graining_sectors.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
DATA = HERE.parent / "data_canonical"
OUT = DATA / "run50b_graining_sectors.json"
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


def main():
    mh = {r["name"]: r for r in json.load(open(DATA / "multipole_harmonics_8dir.json"))["rows"]}
    rows = []
    for f in sorted(RAW.glob("gratings_*.npy")):
        dat = np.load(f, allow_pickle=True).item()
        X = np.asarray(dat["sresp"], np.float32); X /= X.std() + 1e-9
        istim = np.asarray(dat["istim"], float)
        bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, NB + 1)) - 1, 0, NB - 1)
        if np.bincount(bl, minlength=NB).min() < 10:
            rows.append({"name": f.stem, "status": "degenerate_bins"}); continue
        Xt = np.ascontiguousarray(X.T)
        ang = 2 * np.pi * np.arange(NB) / NB
        M = np.stack([Xt[bl == k].mean(0) for k in range(NB)]); Mc = M - M.mean(0)
        ph_ori = (np.angle((Mc * np.exp(2j * ang[:, None])).sum(0)) / 2) % np.pi
        ph_dir = np.angle((Mc * np.exp(1j * ang[:, None])).sum(0)) % (2 * np.pi)
        arms = {"ori_sorted": np.argsort(ph_ori), "dir_sorted": np.argsort(ph_dir), "random": np.random.default_rng(1).permutation(Xt.shape[1])}
        res = {"K": KS}
        for arm, order in arms.items():
            vals = []
            for K in KS:
                nblk = Xt.shape[1] // K
                Xb = np.stack([Xt[:, order[i * K:(i + 1) * K]].mean(1) for i in range(nblk)], 1)
                vals.append(float(np.mean([delta_dir(Xb, bl, np.random.default_rng(42 + s)) for s in (0, 1)])))
            res[arm] = vals
        odd = mh[f.stem]["coef"]["c1"] / mh[f.stem]["coef"]["b2"]
        rows.append({"name": f.stem, "status": "ok", "odd_share": odd, "graining": res,
                     "retention_K32": {a: res[a][-1] / res[a][0] for a in arms}})
        print(f"  {f.stem[:32]:32s} odd {odd:.2f} | K=1 {res['random'][0]:+.3f} | K=32 ori {res['ori_sorted'][-1]:+.3f} dir {res['dir_sorted'][-1]:+.3f} rand {res['random'][-1]:+.3f} "
              f"| retention ori {res['ori_sorted'][-1]/res['ori_sorted'][0]:.2f} dir {res['dir_sorted'][-1]/res['dir_sorted'][0]:.2f} rand {res['random'][-1]/res['random'][0]:.2f}", flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    odd = [r["odd_share"] for r in ok]; ret_ori = [r["retention_K32"]["ori_sorted"] for r in ok]; ret_dir = [r["retention_K32"]["dir_sorted"] for r in ok]
    rho, p = stats.spearmanr(odd, ret_ori)
    loc = [r for r in ok if "local" in r["name"]]; even_heavy = [r for r in ok if ("drifting_GT3" in r["name"] or "low_contrast" in r["name"])]
    verdict = {"G1_dir_sorted_retention_ge_0p6_all": bool(all(v >= 0.6 for v in ret_dir)), "dir_sorted_retention": [round(v, 2) for v in ret_dir],
               "G2_ori_sorted_preserved_even_heavy": bool(all(r["retention_K32"]["ori_sorted"] >= 0.6 for r in even_heavy)),
               "G2_ori_sorted_decays_localized": bool(all(r["retention_K32"]["ori_sorted"] < 0.6 for r in loc)),
               "ori_sorted_retention": [round(v, 2) for v in ret_ori], "random_retention": [round(r["retention_K32"]["random"], 2) for r in ok],
               "G3_spearman_odd_share_vs_ori_retention": [round(float(rho), 2), round(float(p), 3)], "G3_pass": bool(rho < 0)}
    print("VERDICT", json.dumps(verdict))
    json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1); print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
