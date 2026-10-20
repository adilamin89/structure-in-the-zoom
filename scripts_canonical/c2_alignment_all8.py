"""Run9's within-class subspace alignment on all eight grating recordings, and its relation to the
within-class part of the order effect (run72).

Same estimator as run9_alignment_225.py: sixteen direction classes (22.5 deg bins), rank-10 within-class subspaces from
120 trials per class (fewer if a class has fewer), alignment = mean cos^2 of the principal angles, per separation.
run9 covered the three drifting recordings; the camera-ready (App K) states the alignment-order relation on those three.

EXPECTATION, written before the run:
  E1: alignment is maximal at 22.5 deg in all eight recordings (adjacent classes share fluctuation directions).
  E2: adjacent-minus-antipodal alignment (22.5 deg minus 180 deg) is positive in all eight.
  E3: across the eight recordings that margin orders against the within-class order effect of run72 (more shared
      adjacent directions, more negative within-class order part): Spearman rho < 0.
A miss is reported as the result.
Out: data_canonical/c2_alignment_all8.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent / "raw"
N_BINS, RANK, N_PER = 16, 10, 120
FILES = {"D1": "gratings_drifting_GT1_2019_04_12_1", "D2": "gratings_drifting_GT2_2019_04_05_1",
         "D3": "gratings_drifting_GT3_2019_04_05_1", "L1": "gratings_local_GT1_2019_04_27_2",
         "L2": "gratings_local_GT2_2019_04_23_2", "L3": "gratings_local_GT3_2019_04_24_2",
         "C1": "gratings_low_contrast_GT1_2019_04_09_1", "C2": "gratings_low_contrast_GT2_2019_04_12_2"}


def align(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def profile(fname):
    dat = np.load(RAW / f"{fname}.npy", allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)
    rng = np.random.default_rng(0)
    n_per = min(int(np.bincount(bl, minlength=N_BINS).min()), N_PER)
    subs = []
    for b in range(N_BINS):
        idx = rng.choice(np.where(bl == b)[0], n_per, replace=False)
        V = Xt[idx]
        _, _, Vt = np.linalg.svd(V - V.mean(axis=0), full_matrices=False)
        subs.append(Vt[:RANK].T)
    by_sep = {}
    for i in range(N_BINS):
        for j in range(i + 1, N_BINS):
            sep = min((j - i) % N_BINS, (i - j) % N_BINS) * 22.5
            by_sep.setdefault(sep, []).append(align(subs[i], subs[j]))
    return n_per, {f"{k}": float(np.mean(v)) for k, v in sorted(by_sep.items())}


if __name__ == "__main__":
    split = json.load(open(DATA / "run72_paper_designs_split.json"))["rows"]
    out = {"design": {"bins": N_BINS, "rank": RANK, "n_per_class": N_PER}, "rows": {}}
    t0 = time.time()
    for tag, f in FILES.items():
        n_per, prof = profile(f)
        margin = prof["22.5"] - prof["180.0"]
        out["rows"][tag] = {"name": f, "n_per": n_per, "profile": prof, "adjacent_minus_antipodal": margin,
                            "max_at_22.5": max(prof, key=prof.get) == "22.5",
                            "order_within": split[tag]["all"]["order_noise"]}
        print(f"{tag}: n_per {n_per} 22.5 {prof['22.5']:.3f} 45 {prof['45.0']:.3f} 90 {prof['90.0']:.3f} "
              f"180 {prof['180.0']:.3f} margin {margin:+.3f} | order_within {split[tag]['all']['order_noise']:+.3f} "
              f"| {time.time()-t0:.0f}s", flush=True)
    R = out["rows"]
    m = [R[t]["adjacent_minus_antipodal"] for t in FILES]; w = [R[t]["order_within"] for t in FILES]
    rho, p = stats.spearmanr(m, w)
    out["verdict"] = {"E1_max_at_22.5": sum(R[t]["max_at_22.5"] for t in FILES), "E2_margin_positive": sum(x > 0 for x in m),
                      "E3_spearman": float(rho), "E3_p": float(p), "pearson": float(np.corrcoef(m, w)[0, 1])}
    print("VERDICT", out["verdict"], flush=True)
    json.dump(out, open(DATA / "c2_alignment_all8.json", "w"), indent=1)
