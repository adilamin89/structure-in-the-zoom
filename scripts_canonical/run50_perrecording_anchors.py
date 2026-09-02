"""Run 50 (S74) - GT3-anchored numbers on all eight grating recordings:
(a) preference-sorted vs random coarse-graining of the direction-aligned
shift, delta(K) for K in {1, 4, 16, 32} (A9.1 design, Table 10 extended);
(b) matched-floor delta vs the Chun et al. analytic bias correction on the
direction ladder (chun_comparison design).

REGISTERED EXPECTATIONS (before the run): (a) preference-sorted graining keeps
delta positive at every K in all eight recordings while random blocks decay
toward zero by K = 32; the K = 32 rise seen on GT3 is not assumed elsewhere.
(b) the two residuals agree within ~10% on every recording (GT3: 6.8%), with
the analytic value above the empirical one where the empirical floor retains
non-Gaussian structure.

Out: ../data_canonical/run50_perrecording_anchors.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = HERE.parent / "data_canonical" / "run50_perrecording_anchors.json"
spec = importlib.util.spec_from_file_location("chun", HERE / "chun_comparison.py")
chun = importlib.util.module_from_spec(spec); spec.loader.exec_module(chun)
NB = 8
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_NULL = 10
KS = [1, 4, 16, 32]


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
    return slope(sizes, prs) - slope(sizes, np.exp(nl.mean(0))), sizes, prs


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
        # (a) graining
        ang = 2 * np.pi * np.arange(NB) / NB
        M = np.stack([Xt[bl == k].mean(0) for k in range(NB)]); Mc = M - M.mean(0)
        pref = (np.angle((Mc * np.exp(2j * ang[:, None])).sum(0)) / 2) % np.pi
        grain = {"K": KS, "pref_sorted": [], "random": []}
        for K in KS:
            for key, order in (("pref_sorted", np.argsort(pref)), ("random", np.random.default_rng(1).permutation(Xt.shape[1]))):
                nblk = Xt.shape[1] // K
                Xb = np.stack([Xt[:, order[i * K:(i + 1) * K]].mean(1) for i in range(nblk)], 1)
                d, _, _ = delta_dir(Xb, bl, np.random.default_rng(42)); grain[key].append(float(d))
        # (b) chun on the K=1 ladder
        d_ours, sizes, prs = delta_dir(Xt, bl, np.random.default_rng(42))
        n_neur = Xt.shape[1]
        prs_chun = [chun.chun_correct(p, s, n_neur) for p, s in zip(prs, sizes)]
        d_chun = slope(sizes, prs_chun) - 0.0
        # the chun residual is the corrected-ladder slope minus the corrected random-ladder slope
        rng = np.random.default_rng(7); rand_prs = []
        for s in sizes:
            rand_prs.append(np.mean([chun.chun_correct(pr_c(Xt[rng.choice(len(Xt), s, replace=False)]), s, n_neur) for _ in range(N_NULL)]))
        d_chun = slope(sizes, prs_chun) - slope(sizes, rand_prs)
        rows.append({"name": f.stem, "status": "ok", "n_neurons": int(n_neur), "graining": grain,
                     "delta_matched_floor": float(d_ours), "delta_chun_corrected": float(d_chun),
                     "rel_diff": float((d_chun - d_ours) / d_ours)})
        print(f"  {f.stem[:32]:32s} graining pref {np.round(grain['pref_sorted'],3).tolist()} random {np.round(grain['random'],3).tolist()} | "
              f"delta ours {d_ours:+.3f} chun {d_chun:+.3f} ({100*(d_chun-d_ours)/d_ours:+.1f}%)", flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    verdict = {"a_pref_sorted_positive_all_K_count": sum(all(v > 0 for v in r["graining"]["pref_sorted"]) for r in ok),
               "a_random_K32_over_K1": [round(r["graining"]["random"][-1] / r["graining"]["random"][0], 2) for r in ok],
               "a_pref_K32_over_K1": [round(r["graining"]["pref_sorted"][-1] / r["graining"]["pref_sorted"][0], 2) for r in ok],
               "b_rel_diff_chun_vs_ours": [round(r["rel_diff"], 3) for r in ok],
               "b_within_10pct_count": sum(abs(r["rel_diff"]) <= 0.10 for r in ok)}
    print("VERDICT", json.dumps(verdict))
    json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1); print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
