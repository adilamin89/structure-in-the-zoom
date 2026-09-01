"""Run 9 - Adjacent-class (22.5 deg) subspace alignment (referee M2's missing
point: the entry-coherence account expects alignment to be maximal at the
smallest separation).

Sixteen direction classes (22.5 deg bins), rank-10 within-class subspaces,
120 trials per class, alignment by mean cos^2 principal angles at separations
22.5 / 45 / 90 / 180 deg.

REGISTERED EXPECTATION: alignment at 22.5 deg exceeds 45 deg (maximum at the
smallest separation), with the pi-periodic recovery at 180 deg preserved.

Out: feedback_runs/run9_alignment_225.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
N_BINS = 16
RANK = 10
N_PER = 120

FILES = {
    "GT1": "gratings_drifting_GT1_2019_04_12_1.npy",
    "GT2": "gratings_drifting_GT2_2019_04_05_1.npy",
    "GT3": "gratings_drifting_GT3_2019_04_05_1.npy",
}


def align(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


out = {}
for tag, fname in FILES.items():
    dat = np.load(DATA / fname, allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
                 0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)
    rng = np.random.default_rng(0)
    subs = []
    n_min = int(np.bincount(bl, minlength=N_BINS).min())
    n_per = min(n_min, N_PER)
    for b in range(N_BINS):
        idx = rng.choice(np.where(bl == b)[0], n_per, replace=False)
        V = Xt[idx]
        Vc = V - V.mean(axis=0)
        _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
        subs.append(Vt[:RANK].T)
    by_sep = {}
    for i in range(N_BINS):
        for j in range(i + 1, N_BINS):
            sep = min((j - i) % N_BINS, (i - j) % N_BINS) * 22.5
            by_sep.setdefault(sep, []).append(align(subs[i], subs[j]))
    prof = {str(s): float(np.mean(v)) for s, v in sorted(by_sep.items())}
    out[tag] = {"n_per_class": n_per, "profile": prof,
                "adjacent_max": prof["22.5"] > prof["45.0"]}
    print(f"{tag} (n/class={n_per}): 22.5={prof['22.5']:.3f} "
          f"45={prof['45.0']:.3f} 90={prof['90.0']:.3f} "
          f"180={prof['180.0']:.3f} | 22.5>{'' if out[tag]['adjacent_max'] else '!'}45",
          flush=True)

json.dump(out, open(HERE / "run9_alignment_225.json", "w"), indent=1)
print("DONE run9", flush=True)
